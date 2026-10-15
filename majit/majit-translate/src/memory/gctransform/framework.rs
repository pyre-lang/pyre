//! `rpython/memory/gctransform/framework.py` — which operations can collect.
//!
//! Upstream answers this by construction: `BaseFrameworkGCTransformer` has one
//! `gct_*` handler per operation that can reach the collector, and each of them
//! brackets its operation with
//!
//! ```text
//! livevars = self.push_roots(hop)
//! ...
//! self.pop_roots(hop, livevars)
//! ```
//!
//! (`framework.py:790, 853, 901, 908, 971, 1022, 1050, 1150, …` — about thirty
//! pairs).  The live set is never written by hand either:
//! `get_livevars_for_roots` (`framework.py`) takes `hop.livevars_after_op()`
//! straight off the flow graph, and for a moving GC deliberately drops the
//! current operation's own arguments — *"moving GCs don't borrow, so the caller
//! does not need to keep the arguments alive"*.
//!
//! The flowspace rewrite below uses the already-ported `CollectAnalyzer` and
//! emits the same markers. The remainder of this file is the native-Rust audit
//! for source paths compiled directly by rustc: it answers the reachability
//! question over the *resolved* ULLBC call graph. A Call terminator names its
//! callee as `Fun::Regular(FunDeclId)`, so closing over callers cannot be
//! widened by two functions sharing a leaf name.
//!
//! # What actually collects
//!
//! Settled in source on 2026-08-19, because getting this backwards is what
//! makes an audit report windows that are not defects:
//!
//! - The **host** allocation path does **not** collect.  `try_gc_alloc`
//!   (`pyre-object/src/gc_hook.rs`) reaches the backend hook
//!   `dynasm_alloc_nursery_typed` (`majit-backend-dynasm/src/runner.rs`), whose
//!   own comment states it: *"host-side allocation must not trigger collection:
//!   the caller holds a raw `*mut u8` on the Rust stack that is NOT registered
//!   as a GC root."*  It routes to `try_alloc_nursery_no_collect_typed`, which
//!   bumps the nursery or spills to old-gen, and an object born old while a
//!   major cycle is marking is born black (`oldgen_birth_flags`).  So neither a
//!   minor nor a sweep can touch what such a call leaves behind.
//! - What collects is **application-level Python**: it can reach JIT-compiled
//!   code, which allocates inline in the nursery and runs a minor when it
//!   fills.  The explicit collection entry points and the collecting nursery
//!   hook (`alloc_nursery_collecting_typed`, reserved for the elidable bigint
//!   payload helpers) are the other two.
//!
//! So the seeds below are *dispatch* entry points, never allocators.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{GraphKey, GraphRef, Hlvalue, SpaceOperation, Variable};
use crate::translator::backendopt::collectanalyze::CollectAnalyzer;
use crate::translator::backendopt::graphanalyze::GraphAnalyzer;
use crate::translator::backendopt::support::var_needsgc;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::translator::TranslationContext;
use crate::translator::unsimplify::varoftype;

/// `framework.py::BaseFrameworkGCTransformer`'s marker-insertion half.
///
/// The collector-specific operation rewrites still live beside their runtime
/// helpers.  This carrier owns the part every moving framework GC shares:
/// `CollectAnalyzer` decides whether an operation can collect and
/// `get_livevars_for_roots` computes the GC variables that have a use after
/// that operation. `ShadowStackFrameworkGCTransformer.push_roots` then spells
/// the answer as `gc_push_roots` / `gc_pop_roots` for `shadowcolor.py`.
pub struct BaseFrameworkGCTransformer<'t> {
    collect_analyzer: CollectAnalyzer<'t>,
    seen_graphs: HashSet<GraphKey>,
    /// `BaseGCTransformer.num_pushs`: number of individual roots emitted in
    /// the naive marker form, before shadow colouring coalesces their slots.
    pub num_pushs: usize,
}

impl<'t> BaseFrameworkGCTransformer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            collect_analyzer: CollectAnalyzer::new(translator),
            seen_graphs: HashSet::new(),
            num_pushs: 0,
        }
    }

    /// `BaseFrameworkGCTransformer.get_livevars_for_roots` for a moving GC.
    ///
    /// `livevars` is the transform block's ordered list of GC variables that
    /// have become live so far.  Moving GCs do not borrow current-operation
    /// arguments, so the ordinary case keeps only variables used later.  The
    /// `weakref_create` handler is upstream's one `keep_current_args=True`
    /// caller: its referent must survive the allocation even when that is its
    /// last graph use.
    fn get_livevars_for_roots(
        livevars: &[Variable],
        var_last_needed_in: &HashMap<Variable, usize>,
        operation: &SpaceOperation,
        index: usize,
        keep_current_args: bool,
    ) -> Vec<Variable> {
        let mut roots: Vec<Variable> = livevars
            .iter()
            .filter(|var| {
                var_last_needed_in
                    .get(*var)
                    .is_some_and(|last| *last > index)
            })
            .cloned()
            .collect();
        if keep_current_args {
            for arg in &operation.args {
                let Hlvalue::Variable(var) = arg else {
                    continue;
                };
                if var_needsgc(var)
                    && var_last_needed_in
                        .get(var)
                        .is_some_and(|last| *last == index)
                    && !roots.contains(var)
                {
                    roots.push(var.clone());
                }
            }
        }
        roots
    }

    fn marker(opname: &str, roots: &[Variable]) -> SpaceOperation {
        SpaceOperation::new(
            opname,
            roots.iter().cloned().map(Hlvalue::Variable).collect(),
            Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
        )
    }

    /// Operations whose `framework.py gct_*` handler calls `push_roots`
    /// independently of `CollectAnalyzer.analyze(spaceop)`.
    ///
    /// Most rows also carry `LL_OPERATIONS[opname].canmallocgc` and therefore
    /// agree with the analyzer. The explicit list is still load-bearing for
    /// `gct_gc_obtain_free_space`, the `gct_gc_get_rpy_*` family and the
    /// type-id helpers: upstream lowers those LLOps to collecting helper calls
    /// even though their original descriptors have `canmallocgc=False`.
    /// The three `gc_thread_*` brackets are conditional on thread-hook
    /// function pointers that this transformer does not expose; they remain
    /// ordinary LLOps until that root-walker surface is ported rather than
    /// silently taking the enabled-hook branch.
    fn handler_pushes_roots(opname: &str) -> bool {
        matches!(
            opname,
            "gc__collect"
                | "gc__collect_step"
                | "gc_heap_stats"
                | "do_malloc_fixedsize"
                | "do_malloc_fixedsize_clear"
                | "do_malloc_varsize"
                | "do_malloc_varsize_clear"
                | "weakref_create"
                | "gc_identityhash"
                | "gc_id"
                | "gc_obtain_free_space"
                | "gc_get_rpy_roots"
                | "gc_get_rpy_referents"
                | "gc_get_rpy_memory_usage"
                | "gc_get_rpy_type_index"
                | "gc_is_rpy_instance"
                | "gc_dump_rpy_heap"
                | "gc_typeids_z"
                | "gc_typeids_list"
        )
    }

    /// `BaseGCTransformer.transform_block` plus
    /// `BaseFrameworkGCTransformer.gct_direct_call` and the collecting LLOp
    /// handlers, restricted to the root markers they all have in common.
    fn transform_block(&mut self, block: &crate::flowspace::model::BlockRef) {
        let (inputargs, operations, exits) = {
            let block = block.borrow();
            (
                block.inputargs.clone(),
                block.operations.clone(),
                block.exits.clone(),
            )
        };

        #[expect(
            clippy::mutable_key_type,
            reason = "RPython keys var_last_needed_in by Variable identity; its hash excludes annotation/concretetype cells"
        )]
        let mut var_last_needed_in: HashMap<Variable, usize> = HashMap::new();
        for (index, operation) in operations.iter().enumerate() {
            for arg in &operation.args {
                if let Hlvalue::Variable(var) = arg
                    && var_needsgc(var)
                {
                    var_last_needed_in.insert(var.clone(), index);
                }
            }
        }
        for link in exits {
            for arg in &link.borrow().args {
                if let Some(Hlvalue::Variable(var)) = arg
                    && var_needsgc(var)
                {
                    var_last_needed_in.insert(var.clone(), operations.len() + 1);
                }
            }
        }

        // Moving-GC `compute_borrowed_vars` returns false for every variable,
        // so all GC-pointer inputargs begin in `self.livevars`.
        let mut livevars: Vec<Variable> = inputargs
            .into_iter()
            .filter_map(|value| match value {
                Hlvalue::Variable(var) if var_needsgc(&var) => Some(var),
                _ => None,
            })
            .collect();
        let mut rewritten = Vec::with_capacity(operations.len());
        for (index, operation) in operations.into_iter().enumerate() {
            let can_collect = self.collect_analyzer.analyze(&operation, None, &())
                || Self::handler_pushes_roots(&operation.opname);
            if can_collect {
                let roots = Self::get_livevars_for_roots(
                    &livevars,
                    &var_last_needed_in,
                    &operation,
                    index,
                    operation.opname == "weakref_create",
                );
                self.num_pushs += roots.len();
                rewritten.push(Self::marker("gc_push_roots", &roots));
                rewritten.push(operation.clone());
                // Emitted even for an empty live set: shadowcolor.py's
                // move_pushes_earlier relies on the matching delimiter.
                rewritten.push(Self::marker("gc_pop_roots", &roots));
            } else {
                rewritten.push(operation.clone());
            }
            if let Hlvalue::Variable(result) = &operation.result
                && var_needsgc(result)
            {
                livevars.push(result.clone());
            }
        }
        block.borrow_mut().operations = rewritten;
    }

    /// `BaseGCTransformer.transform_graph`: insert the naive root markers once
    /// per reachable block. `shadowstack.py` invokes `shadowcolor.py` only
    /// after every graph has passed through this method.
    pub fn transform_graph(&mut self, graph: &GraphRef) {
        let key = GraphKey::of(graph);
        if !self.seen_graphs.insert(key) {
            return;
        }
        let blocks = graph.borrow().iterblocks();
        for block in blocks {
            self.transform_block(&block);
        }
    }
}

/// A resolved ULLBC call graph: which function calls which.
pub struct CallGraph {
    /// `def_id -> fully qualified name`
    pub names: HashMap<u64, String>,
    /// `def_id -> the spelling [`Joined`] identifies a function by`.  The name
    /// for everything charon spelled out, the name plus the item's source line
    /// where it did not.
    pub keys: HashMap<u64, String>,
    /// `def_id -> callees`, from `Fun::Regular` Call terminators only.
    pub callees: HashMap<u64, HashSet<u64>>,
    /// `def_id -> callers`, the reverse index.
    pub callers: HashMap<u64, HashSet<u64>>,
    /// Functions with at least one Call terminator this pass could not resolve
    /// to a `FunDeclId` — a call through a function pointer, a trait object or
    /// a closure.  Reported rather than dropped: an unresolved edge is exactly
    /// where "can this reach Python" stops being decidable from the graph.
    pub indirect: HashSet<u64>,
    /// What the unresolved calls actually were.  A taint that swallows a third
    /// of the graph is only actionable if you know whether it is `dyn Error`
    /// or something that can reach Python.
    pub opaque: OpaqueCensus,
}

/// The unresolved calls, bucketed by what could not be resolved.
#[derive(Default)]
pub struct OpaqueCensus {
    /// `dyn Trait` dispatch, counted per trait name.
    pub dyn_trait: HashMap<String, usize>,
    /// A call of a function-typed value whose trait could not be named — a
    /// `global_hook!` cell read, a function passed in as an argument.
    pub fn_value: usize,
    /// The unresolved calls split by which spelling charon used, so a bucket
    /// is never reasoned about as if it were all one thing.
    pub by_variant: HashMap<&'static str, usize>,
    /// Anything else the reader could not classify.
    pub unknown: usize,
}

/// Whether a `CallKind::Trait` payload's trait reference is a `dyn` one.
///
/// `[trait_ref, method_index, fun_decl_id]` — charon resolves the method and
/// puts its `FunDeclId` third, so a trait call is a real edge.  The exception
/// is a `Dyn` trait reference, where the third element names the trait's own
/// method declaration and the actual body is only known at run time.
/// The discriminant of a `CallKind::Trait` payload's trait reference.
pub fn trait_ref_kind(llbc: &majit_charon_reader::Llbc, tref: &serde_json::Value) -> String {
    let body = if let Some(id) = tref.get("Deduplicated").and_then(|x| x.as_u64()) {
        match llbc.dedup_trait_body(id) {
            Some(b) => b,
            None => return "unresolved-dedup".into(),
        }
    } else if let Some(inline) = tref.pointer("/Value/1") {
        inline
    } else {
        return "no-body".into();
    };
    match body.get("kind") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(m)) => m.keys().next().cloned().unwrap_or_default(),
        _ => "none".into(),
    }
}

/// The first `trait_decl_ref.skip_binder.id` anywhere under `v`.
///
/// Both spellings of a virtual call carry the trait the same way — the
/// `CallKind::Trait` payload and the fat pointer's `DynTrait` type — just at
/// different depths, so one search answers for both.
fn first_trait_id(v: &serde_json::Value) -> Option<u64> {
    if let Some(map) = v.as_object() {
        if let Some(r) = map.get("trait_decl_ref")
            && let Some(id) = r.pointer("/skip_binder/id").and_then(|x| x.as_u64())
        {
            return Some(id);
        }
        for sub in map.values() {
            if let Some(id) = first_trait_id(sub) {
                return Some(id);
            }
        }
    } else if let Some(arr) = v.as_array() {
        for sub in arr {
            if let Some(id) = first_trait_id(sub) {
                return Some(id);
            }
        }
    }
    None
}

/// The dispatch entry points application-level Python is reached through.
///
/// Matched on the trailing path segments so a crate rename cannot silently
/// empty the seed set; a seed that matches nothing is reported by
/// [`CallGraph::seed_report`].
pub const PYTHON_DISPATCH_SEEDS: &[&str] = &[
    // The generic callable dispatchers.
    "call::call_callable",
    "call::call_callable_in_ctx",
    "call::call_callable_with_mode",
    "call::call_user_function",
    "call::call_user_function_plain",
    "call::call_user_function_with_args",
    "call::call_user_function_resolved_frameless",
    "call::builtin_code_call_positional",
    "call::call_builtin_code_positional",
    // The gateway's indirect call into a builtin's Rust body.
    "gateway::builtin_code_call",
    // The frame executors.  `execute_frame` and its resumed twin are inherent
    // methods, so the `impl` block sits in the path as an opaque segment and
    // the bare `pyframe::execute_frame` spelling can never match.
    "eval::eval_frame_plain",
    "eval::eval_frame_plain_with_operr",
    "eval::eval_frame_plain_with_resume",
    "eval::eval_loop",
    "pyframe::<Impl>::execute_frame",
    "pyframe::<Impl>::resume_execute_frame",
    // The JIT layer's routes back into interpretation.  An artefact extracted
    // for `pyre-jit` carries only part of `pyre-interpreter`, so most of the
    // entries above are absent from it and the closure comes out at 0% -- a
    // scan over that is vacuous rather than clean.  These are the portal and
    // blackhole entries that artefact does carry.
    "call_jit::ll_portal_runner_shim",
    "call_jit::run_frame_through_portal",
    "call_jit::bh_portal_runner_c",
    "call_jit::bh_call_self_recursive_portal",
    "eval::portal_runner",
    "eval::portal_runner_dispatch",
    "eval::portal_body_result",
    "eval::portal_activation_result",
    "eval::portal_traced_activation_result",
    "eval::portal_activation_bracketed",
    "eval::enter_portal",
    "eval::eval_loop_jit",
    // The space-level helpers most builtins reach Python through.
    "baseobjspace::call_function",
    "baseobjspace::call_method",
    "baseobjspace::call_obj_args",
    "descroperation::try_call_special",
];

/// Explicit collection entry points and the collecting host allocator.
///
/// These are the same external symbols `majit-translate`'s call control marks
/// `canmallocgc` (`lib.rs`, the `mark_canmallocgc` loop): the `*_collecting_*`
/// allocators run a minor collection when the nursery cannot satisfy the
/// request, and the `collect_*` entries are requested collections outright.
/// They carry no lowered body, so nothing reaches them transitively — they have
/// to be named, or every direct caller (`pyre_object_gc_alloc_collecting_trampoline`
/// in `pyre-jit/src/eval.rs`, say) is classified non-collecting and so is
/// everything above it.
///
/// ⛔ The plain host allocator is deliberately **not** here.  `try_gc_alloc`
/// routes to `alloc_nursery_typed` and on through the backend's
/// `dynasm_alloc_nursery_typed`, which falls back to old-gen rather than
/// collect precisely because its caller holds an unregistered raw pointer; so
/// no host-side `w_*_new` is a collection point, and seeding one would report
/// every allocating body in the interpreter.  A minor runs from compiled code
/// allocating inline in the nursery, from the elidable bigint payload helpers,
/// and from a requested collection — which is why an artefact carrying only
/// interpreter bodies legitimately matches none of the allocator entries and
/// takes its whole closure from [`PYTHON_DISPATCH_SEEDS`].
pub const COLLECTING_SEEDS: &[&str] = &[
    "majit_gc::alloc_nursery_collecting_typed",
    "majit_gc::alloc_nursery_collecting_typed_rooted",
    "majit_gc::alloc_fast_nursery_collecting_typed_rooted",
    "majit_gc::alloc_fast_nursery_collecting_typed_roots",
    // Reported UNMATCHED, and that is the answer rather than a gap: these are
    // `#[inline]` and reached only from the two wrappers above them
    // (`majit-gc/src/lib.rs`) and from the dynasm runner's hook, which is not
    // in the analysed artefacts.  `reaching(standalone_X)` is therefore a
    // subset of `reaching(X)`, so naming them adds no closure -- they are kept
    // so a future extraction that does carry them is seeded, and the gate pins
    // the unmatched set so the pair cannot quietly grow.
    "majit_gc::standalone_alloc_nursery_collecting_typed_rooted",
    "majit_gc::standalone_alloc_fast_nursery_collecting_typed_rooted",
    "majit_gc::standalone_alloc_fast_nursery_collecting_typed_roots",
    // The generation-carrying entry every backend trampoline dispatches to;
    // `collect_full` is its `gen = 2` shim and no extracted crate calls it.
    "majit_gc::collect_generation",
    "majit_gc::collect_step",
    "majit_gc::collect_oldgen_nonmoving",
    // `external_malloc`'s `threshold_reached` collection, for a host object
    // whose payload is charged as memory pressure instead of allocated.
    "majit_gc::maybe_collect_for_external_malloc",
    // The host hook the interpreter reaches them through.
    "gc_hook::try_gc_alloc_collecting_rooted",
    // A requested collection, which is the one collection point an interpreter
    // body reaches without running Python at all (`gc.collect()`).
    "gc_hook::try_gc_collect",
    "gc_hook::try_gc_collect_step",
    "gc_hook::try_gc_collect_oldgen",
];

impl CallGraph {
    /// Every function that can transitively reach one of `seeds`.
    pub fn reaching(&self, seeds: &HashSet<u64>) -> HashSet<u64> {
        let mut out: HashSet<u64> = seeds.clone();
        let mut work: Vec<u64> = seeds.iter().copied().collect();
        while let Some(cur) = work.pop() {
            let Some(ups) = self.callers.get(&cur) else {
                continue;
            };
            for &up in ups {
                if out.insert(up) {
                    work.push(up);
                }
            }
        }
        out
    }

    /// A shortest call chain from `from` down to one of `seeds`, as names.
    ///
    /// A reachability verdict is only as good as the path behind it, and a
    /// path through an error-formatting helper is a very different claim from
    /// one through a dunder invocation.  Adjudicating a finding means reading
    /// this, not trusting the bit.
    pub fn path_to(&self, from: u64, seeds: &HashSet<u64>) -> Option<Vec<String>> {
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut seen: HashSet<u64> = HashSet::from([from]);
        let mut queue: std::collections::VecDeque<u64> = std::collections::VecDeque::from([from]);
        while let Some(cur) = queue.pop_front() {
            if seeds.contains(&cur) {
                let mut chain = vec![cur];
                let mut walk = cur;
                while let Some(&up) = prev.get(&walk) {
                    chain.push(up);
                    walk = up;
                }
                chain.reverse();
                return Some(
                    chain
                        .into_iter()
                        .map(|id| {
                            self.names
                                .get(&id)
                                .map(|n| n.rsplit("::").take(2).collect::<Vec<_>>().join("::"))
                                .unwrap_or_else(|| format!("#{id}"))
                        })
                        .collect(),
                );
            }
            for &down in self.callees.get(&cur).into_iter().flatten() {
                if seen.insert(down) {
                    prev.insert(down, cur);
                    queue.push_back(down);
                }
            }
        }
        None
    }

    /// Resolve the configured seed patterns against the real name table.
    pub fn seeds_for(&self, patterns: &[&str]) -> (HashSet<u64>, Vec<String>) {
        let mut ids = HashSet::new();
        let mut unmatched = Vec::new();
        for pat in patterns {
            let mut any = false;
            for (&id, name) in &self.names {
                // Anchored on a path separator: an unanchored `ends_with`
                // would let `fastcall::call_callable` answer for the
                // `call::call_callable` seed, and an over-matched seed widens
                // every verdict downstream of `reaching`.
                if name == pat || name.ends_with(&format!("::{pat}")) {
                    ids.insert(id);
                    any = true;
                }
            }
            if !any {
                unmatched.push((*pat).to_string());
            }
        }
        (ids, unmatched)
    }

    /// A seed pattern that matches nothing is a silently empty analysis, so
    /// name them rather than let the closure come back small and look clean.
    pub fn seed_report(&self, patterns: &[&str]) -> String {
        let (ids, unmatched) = self.seeds_for(patterns);
        let mut s = format!("{} seed function(s)", ids.len());
        if !unmatched.is_empty() {
            s.push_str(&format!("; UNMATCHED patterns: {unmatched:?}"));
        }
        s
    }
}

/// Several artefacts' call graphs, joined on fully qualified name.
///
/// A charon artefact carries only what rustc monomorphised into that crate, so
/// `pyre-jit.ullbc` holds the portal and the blackhole and almost none of the
/// interpreter they call into: its own closure comes out at 1%, and a scan over
/// it reports nothing because nothing in it reaches a seed — not because the
/// crate is clean.  Upstream never meets this: RPython analyses one translated
/// program, and `collect_analyzer` closes over the whole call graph at once.
/// So the artefacts are joined back into one graph here.
///
/// # The name does not always identify a function; the name and line do
///
/// `ItemMeta::name_path` renders an `impl` block as the opaque segment
/// `<Impl>`, so two inherent `new`s on types in the same module are both
/// spelled `pyframe::<Impl>::new`.  Joining on that spelling merges them into one node
/// and invents an edge from every caller of one to every callee of the other —
/// which is how a `getorcreatedebug` that only allocates came out reaching
/// `call_user_function_with_args`.  Widening is not automatically safe here:
/// `framework.py` over-brackets rather than under-brackets, but a fabricated
/// path is not conservatism, it is a wrong answer with a citation.
///
/// Holding every opaque spelling apart is not the answer either, and errs the
/// more dangerous way.  The join exists because one artefact declares what
/// another defines; refusing to join `pyframe::<Impl>::new` leaves the
/// declaration a childless leaf, and a caller that reaches a collection only
/// through it comes back "cannot reach" — an under-approximation, which is how
/// a required root bracket gets reported as unjustified.
///
/// So the key is [`CallGraph::keys`]: the name, plus the item's own source
/// line where the name went opaque.  Charon stamps that line from the
/// definition, so it survives into every artefact that carries the item, while
/// two items collapsing onto one spelling are written at two different lines.
/// A key **one artefact carries more than once** is still not a join key —
/// each occurrence keeps its own node — and [`Self::ambiguous_names`] reports
/// what that held back.
///
/// Measured on `pyre-interpreter` joined with `pyre-jit`: of the 1120 opaque
/// names both artefacts carry, 779 are a single item per artefact whose lines
/// agree — real joins, which keying on the bare name would have merged by luck
/// and blacklisting the spelling would have lost — 7 disagree, which is the
/// collision the key exists to catch, and 334 already repeat inside an
/// artefact.  Against the bare-name key that gives 1913 spellings merged
/// rather than 1337 and 215 held apart rather than 2417, and the scan answers
/// the same 5317/41572 reachable, 583 justified and 256 cannot-reach either
/// way.
pub struct Joined {
    /// The joined graph.  Its `opaque` census is left empty: an opaque *count*
    /// is per-artefact accounting, and summing two of them double-counts every
    /// body both artefacts carry.  `indirect` is joined, being a per-function
    /// fact.
    pub graph: CallGraph,
    /// `part index -> (that artefact's def id -> node id)`.
    canonical: Vec<HashMap<u64, u64>>,
    /// Unambiguous names carried by more than one artefact — what the join
    /// actually merged.
    pub joined_names: usize,
    /// Join keys held apart: carried more than once by some artefact, so the
    /// key names no single function there.
    pub ambiguous_names: usize,
}

impl Joined {
    pub fn build(parts: &[&CallGraph]) -> Self {
        let mut ambiguous: HashSet<&str> = HashSet::new();
        for part in parts {
            let mut seen: HashSet<&str> = HashSet::new();
            for key in part.keys.values() {
                if !seen.insert(key.as_str()) {
                    ambiguous.insert(key.as_str());
                }
            }
        }
        let mut shared: HashMap<&str, u64> = HashMap::new();
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut joined_keys: HashMap<u64, String> = HashMap::new();
        let mut canonical: Vec<HashMap<u64, u64>> = Vec::with_capacity(parts.len());
        let mut next = 0u64;
        let mut joined_names = 0usize;
        for part in parts {
            // Sorted, so the numbering does not depend on hash order and two
            // runs over the same inputs print the same ids.
            let mut sorted: Vec<(&u64, &String)> = part.keys.iter().collect();
            sorted.sort_unstable_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)));
            let mut mine: HashMap<u64, u64> = HashMap::new();
            for (&def_id, key) in sorted {
                let name = &part.names[&def_id];
                let id = if ambiguous.contains(key.as_str()) {
                    let id = next;
                    next += 1;
                    names.insert(id, name.clone());
                    joined_keys.insert(id, key.clone());
                    id
                } else if let Some(&id) = shared.get(key.as_str()) {
                    joined_names += 1;
                    id
                } else {
                    let id = next;
                    next += 1;
                    names.insert(id, name.clone());
                    joined_keys.insert(id, key.clone());
                    shared.insert(key.as_str(), id);
                    id
                };
                mine.insert(def_id, id);
            }
            canonical.push(mine);
        }
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut indirect: HashSet<u64> = HashSet::new();
        for (part, map) in parts.iter().zip(&canonical) {
            for (caller, cs) in &part.callees {
                let Some(&from) = map.get(caller) else {
                    continue;
                };
                let entry = callees.entry(from).or_default();
                for callee in cs {
                    if let Some(&to) = map.get(callee) {
                        entry.insert(to);
                    }
                }
            }
            for id in &part.indirect {
                if let Some(&c) = map.get(id) {
                    indirect.insert(c);
                }
            }
        }
        for (&caller, cs) in &callees {
            for &c in cs {
                callers.entry(c).or_default().insert(caller);
            }
        }
        Self {
            graph: CallGraph {
                names,
                keys: joined_keys,
                callees,
                callers,
                indirect,
                opaque: OpaqueCensus::default(),
            },
            canonical,
            joined_names,
            ambiguous_names: ambiguous.len(),
        }
    }

    /// Project a set of joined nodes back onto one artefact's def ids, so the
    /// liveness scan — which walks that artefact's bodies — can be asked the
    /// whole-program question.  `part` indexes the slice `build` was given.
    pub fn project(&self, part: usize, joined: &HashSet<u64>) -> HashSet<u64> {
        self.canonical[part]
            .iter()
            .filter(|(_, cid)| joined.contains(cid))
            .map(|(&def_id, _)| def_id)
            .collect()
    }
}

/// Where each function-pointer call gets its callee from, counted by source.
///
/// A count of unresolved calls says how much is opaque; this says *what* is
/// opaque, which is the part that decides whether modelling it is worth
/// anything.  The walk is flow-insensitive on purpose — it answers "what kind
/// of thing lands in this local", not "which value on this path".
pub fn dynamic_call_sources(llbc: &majit_charon_reader::Llbc) -> HashMap<String, usize> {
    use majit_charon_reader::ullbc::{
        CallFunc, CallKind, FunId, Operand, Place, PlaceKind, Rvalue, StmtKind, TermKind,
    };

    fn root(p: &Place) -> Option<u64> {
        match &p.kind {
            PlaceKind::Local(i) => Some(*i),
            PlaceKind::Projection(b, _) => root(b),
            _ => None,
        }
    }
    fn op_local(o: &Operand) -> Option<u64> {
        match o {
            Operand::Copy(p) | Operand::Move(p) => root(p),
            Operand::Const(_) => None,
        }
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        // Cleanup reached only through `on_unwind` is outside the flow
        // graph (`strip_cleanup_blocks`); a drop-glue call there is
        // not a definition this walk should chase. `forward_reachable_mask`
        // stops at `Unknown`, so a later non-cleanup block is still walked.
        let forward = crate::front::mir::forward_reachable_mask(llbc, &body);
        let mut defs: HashMap<u64, Rvalue> = HashMap::new();
        let mut from_call: HashMap<u64, String> = HashMap::new();
        let mut projected: HashMap<u64, String> = HashMap::new();
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !call_graph_walks_block(bb, forward[bb_idx]) {
                continue;
            }
            for st in &bb.statements {
                if let Ok(StmtKind::Assign(p, rv)) = st.stmt_kind_ref()
                    && let Some(l) = root(&p)
                {
                    if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = &rv
                        && let PlaceKind::Projection(_, elem) = &src.kind
                    {
                        projected.insert(l, elem.label());
                    }
                    defs.insert(l, rv.clone());
                }
            }
            if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
                && let Some(l) = root(&call.dest)
                && let CallFunc::Regular(reg) = &call.func
                && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
            {
                let name = llbc
                    .fn_by_id(*id)
                    .map(|f| f.item_meta.name_path())
                    .unwrap_or_default();
                from_call.insert(l, name.rsplit("::").take(2).collect::<Vec<_>>().join("::"));
            }
        }
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !call_graph_walks_block(bb, forward[bb_idx]) {
                continue;
            }
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let CallFunc::Dynamic(op) = &call.func else {
                continue;
            };
            let mut cur = op_local(op);
            let mut seen: HashSet<u64> = HashSet::new();
            let label = loop {
                let Some(l) = cur else {
                    break "operand is not a local".to_string();
                };
                if !seen.insert(l) {
                    break "cycle".to_string();
                }
                if let Some(callee) = from_call.get(&l) {
                    break format!("<- {callee}");
                }
                match defs.get(&l) {
                    None => {
                        break if l >= 1 && l <= body.locals.arg_count {
                            "<- this fn's own parameter".to_string()
                        } else {
                            "<- no definition".to_string()
                        };
                    }
                    Some(Rvalue::Use(o, _) | Rvalue::Cast(_, o, _)) => {
                        if let Some(nl) = op_local(o) {
                            cur = Some(nl);
                            continue;
                        }
                        break match projected.get(&l) {
                            Some(f) => format!("<- field {f}"),
                            None => "<- constant".to_string(),
                        };
                    }
                    Some(Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. }) => {
                        if let PlaceKind::Projection(_, elem) = &place.kind {
                            break format!("<- &field {}", elem.label());
                        }
                        match root(place) {
                            Some(nl) if nl != l => {
                                cur = Some(nl);
                                continue;
                            }
                            _ => break "<- borrow".to_string(),
                        }
                    }
                    Some(other) => {
                        break format!("<- rvalue {}", rvalue_label(other));
                    }
                }
            };
            *out.entry(label).or_insert(0) += 1;
        }
    }
    out
}

fn rvalue_label(r: &majit_charon_reader::ullbc::Rvalue) -> &'static str {
    use majit_charon_reader::ullbc::Rvalue as R;
    match r {
        R::Use(..) => "Use",
        R::BinaryOp(..) => "BinaryOp",
        R::UnaryOp(..) => "UnaryOp",
        R::Ref { .. } => "Ref",
        R::Aggregate(..) => "Aggregate",
        R::Discriminant(_) => "Discriminant",
        R::Cast(..) => "Cast",
        R::Len(_) => "Len",
        R::Repeat(..) => "Repeat",
        R::ShallowInitBox(..) => "ShallowInitBox",
        R::RawPtr { .. } => "RawPtr",
        R::NullaryOp(..) => "NullaryOp",
        R::Unknown => "Unknown",
    }
}

/// Blocks `GraphAnalyzer.analyze_direct_call` walks. Unwind-only
/// cleanup is outside `strip_cleanup_blocks`; every other block is in
/// the flow graph. `forward_reachable_mask` stops at `Unknown`, so a
/// later non-cleanup Call is still walked here.
fn call_graph_walks_block(bb: &majit_charon_reader::ullbc::BasicBlock, forward: bool) -> bool {
    forward || !bb.is_cleanup
}

/// `GraphAnalyzer.analyze` returns `top_result()` when call information
/// is unavailable (`DelayedPointer`, missing `graph`, `indirect_call`
/// with `graphs is None`). An unparsed or `Unknown` terminator is that
/// case for this walk.
fn call_graph_flow_is_incomplete(
    llbc: &majit_charon_reader::Llbc,
    bb: &majit_charon_reader::ullbc::BasicBlock,
) -> bool {
    match bb.term_ref(llbc) {
        Ok(majit_charon_reader::ullbc::TermKind::Unknown) | Err(_) => true,
        Ok(_) => false,
    }
}

/// Build the call graph of one charon artefact.
///
/// Only `Call` terminators contribute edges, and only when charon already
/// resolved the callee to a `FunDeclId`.  Everything else — `dyn` dispatch, a
/// function-pointer call, a trait call charon left unresolved — marks the
/// caller in [`CallGraph::indirect`] instead of inventing an edge.
pub fn build(llbc: &majit_charon_reader::Llbc) -> CallGraph {
    use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};

    let mut names = HashMap::new();
    let mut keys: HashMap<u64, String> = HashMap::new();
    let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut indirect = HashSet::new();
    let mut opaque = OpaqueCensus::default();
    let mut note_opaque = |raw: Option<&serde_json::Value>, variant: &'static str| {
        *opaque.by_variant.entry(variant).or_insert(0) += 1;
        match raw.and_then(first_trait_id) {
            Some(tid) => {
                let name = llbc
                    .trait_by_id(tid)
                    .map(|t| t.item_meta.name_path())
                    .unwrap_or_else(|| format!("trait#{tid}"));
                *opaque.dyn_trait.entry(name).or_insert(0) += 1;
            }
            None => opaque.fn_value += 1,
        }
    };

    for fd in llbc.iter_local_fns() {
        let id = fd.def_id;
        let name = fd.item_meta.name_path();
        // `name_path` renders an `impl` block as the opaque segment `<Impl>`,
        // so the spelling alone does not identify the function.  The item's
        // own source line restores what the rendering dropped: charon stamps
        // it from the definition, so every artefact carrying the item — the
        // one holding the body and the ones holding only a declaration —
        // stamps the same line, while two items collapsing onto one spelling
        // are written at two different lines.
        let key = if name.contains('<') {
            format!(
                "{name}@{}",
                llbc.span_data(&fd.item_meta.span)
                    .expect("item span missing from the artefact")
                    .beg
                    .line
            )
        } else {
            name.clone()
        };
        keys.insert(id, key);
        names.insert(id, name);
        let entry = callees.entry(id).or_default();
        let Some(body) = fd.unstructured() else {
            continue;
        };
        // A Call that lives only on an `on_unwind` cleanup chain is not a
        // call-graph edge: the flow graph never carries that edge, so
        // `strip_cleanup_blocks` dropped the block. `forward_reachable_mask`
        // stops at `Unknown`; a later non-cleanup Call is still an edge,
        // and that Unknown is `GraphAnalyzer.analyze` `top_result`.
        let forward = crate::front::mir::forward_reachable_mask(llbc, &body);
        let mut incomplete = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !call_graph_walks_block(bb, forward[bb_idx]) {
                continue;
            }
            if forward[bb_idx] && call_graph_flow_is_incomplete(llbc, bb) {
                incomplete = true;
            }
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id: callee }) => {
                        entry.insert(*callee);
                    }
                    CallKind::Fun(FunId::Other(v)) => {
                        indirect.insert(id);
                        note_opaque(Some(v), "Fun(Other)");
                    }
                    CallKind::Trait(v) => {
                        // `[trait_ref, method_index, fun_decl_id]`.  The third
                        // element is the **trait's method declaration**, not
                        // the impl that will run: measured over this artefact,
                        // 382 of 541 have no body at all, and every trait ref
                        // is a generic `Clause` / `ParentClause` rather than a
                        // `TraitImpl`, so the impl is not recoverable here
                        // without propagating the caller's instantiation.
                        // Turning this into an edge therefore buys false
                        // confidence — it retires the caller's undecidability
                        // by pointing it at a bodyless leaf.
                        //
                        // A default body is still real code the call may run,
                        // so it is added as an edge, which can only widen
                        // "reaches a collection".  The caller stays opaque
                        // either way, because an overriding impl is invisible
                        // from here.
                        if let Some(c) = v.get(2).and_then(|x| x.as_u64())
                            && llbc.fn_by_id(c).and_then(|f| f.unstructured()).is_some()
                        {
                            entry.insert(c);
                        }
                        indirect.insert(id);
                        note_opaque(Some(v), "Trait");
                    }
                    CallKind::Ptr(v) => {
                        indirect.insert(id);
                        note_opaque(Some(v), "Ptr");
                    }
                    CallKind::Unknown => {
                        indirect.insert(id);
                        opaque.unknown += 1;
                    }
                },
                CallFunc::Dynamic(op) => {
                    indirect.insert(id);
                    // A fat pointer's type is as often a dedup id as an inline
                    // body; resolving it is what keeps `dyn Trait` out of the
                    // function-pointer bucket.
                    let ty = match op {
                        majit_charon_reader::ullbc::Operand::Copy(p)
                        | majit_charon_reader::ullbc::Operand::Move(p) => match &p.ty {
                            majit_charon_reader::ullbc::TyRef::Inline { value: (_, v) } => {
                                Some(v.as_ref())
                            }
                            majit_charon_reader::ullbc::TyRef::Dedup { id } => llbc.dedup_body(*id),
                            majit_charon_reader::ullbc::TyRef::Other(v) => Some(v.as_ref()),
                        },
                        majit_charon_reader::ullbc::Operand::Const(v) => Some(v.as_ref()),
                    };
                    note_opaque(ty, "Dynamic");
                }
                CallFunc::Unknown => {
                    indirect.insert(id);
                    opaque.unknown += 1;
                }
            }
        }
        if incomplete {
            indirect.insert(id);
        }
    }
    for (&caller, cs) in &callees {
        for &c in cs {
            callers.entry(c).or_default().insert(caller);
        }
    }
    drop(note_opaque);
    CallGraph {
        names,
        keys,
        callees,
        callers,
        indirect,
        opaque,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::flowspace::model::{Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Link};
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget, Struct};

    fn gc_var(name: &str) -> Variable {
        let var = Variable::named(name);
        var.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc_with_hints(name, vec![], vec![])),
        }))));
        var
    }

    fn signed_var(name: &str) -> Variable {
        let var = Variable::named(name);
        var.set_concretetype(Some(LowLevelType::Signed));
        var
    }

    fn single_block_graph(
        name: &str,
        input: Variable,
        operations: Vec<SpaceOperation>,
        returned: Variable,
    ) -> GraphRef {
        let start = Block::shared(vec![Hlvalue::Variable(input)]);
        start.borrow_mut().operations = operations;
        let graph = FunctionGraph::new(name, start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(returned)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn collecting_operation_is_bracketed_with_gc_values_live_after_it() {
        let root = gc_var("root");
        let stats = signed_var("stats");
        let returned = gc_var("returned");
        let graph = single_block_graph(
            "live_after",
            root.clone(),
            vec![
                SpaceOperation::new("gc_heap_stats", vec![], Hlvalue::Variable(stats)),
                SpaceOperation::new(
                    "same_as",
                    vec![Hlvalue::Variable(root.clone())],
                    Hlvalue::Variable(returned.clone()),
                ),
            ],
            returned,
        );
        let translator = TranslationContext::new();
        let mut transformer = BaseFrameworkGCTransformer::new(&translator);

        transformer.transform_graph(&graph);

        let operations = graph.borrow().startblock.borrow().operations.clone();
        assert_eq!(
            operations
                .iter()
                .map(|operation| operation.opname.as_str())
                .collect::<Vec<_>>(),
            ["gc_push_roots", "gc_heap_stats", "gc_pop_roots", "same_as"]
        );
        assert_eq!(operations[0].args, vec![Hlvalue::Variable(root.clone())]);
        assert_eq!(operations[2].args, vec![Hlvalue::Variable(root)]);
        assert_eq!(transformer.num_pushs, 1);
    }

    #[test]
    fn handler_bracket_is_not_lost_when_llop_canmallocgc_is_false() {
        let root = gc_var("root");
        let space = signed_var("space");
        let returned = gc_var("returned");
        let graph = single_block_graph(
            "handler_override",
            root.clone(),
            vec![
                SpaceOperation::new(
                    "gc_obtain_free_space",
                    vec![Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(4096),
                        LowLevelType::Signed,
                    ))],
                    Hlvalue::Variable(space),
                ),
                SpaceOperation::new(
                    "same_as",
                    vec![Hlvalue::Variable(root.clone())],
                    Hlvalue::Variable(returned.clone()),
                ),
            ],
            returned,
        );
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        assert!(
            !analyzer.analyze(&graph.borrow().startblock.borrow().operations[0], None, &()),
            "the regression is specifically the handler/analyzer distinction"
        );
        let mut transformer = BaseFrameworkGCTransformer::new(&translator);

        transformer.transform_graph(&graph);

        let operations = graph.borrow().startblock.borrow().operations.clone();
        assert_eq!(operations[0].opname, "gc_push_roots");
        assert_eq!(operations[0].args, vec![Hlvalue::Variable(root.clone())]);
        assert_eq!(operations[2].opname, "gc_pop_roots");
        assert_eq!(operations[2].args, vec![Hlvalue::Variable(root)]);
    }

    #[test]
    fn moving_gc_drops_a_collecting_operations_last_use_argument() {
        let referent = gc_var("referent");
        let identity = signed_var("identity");
        let graph = single_block_graph(
            "moving_does_not_borrow",
            referent.clone(),
            vec![SpaceOperation::new(
                "gc_identityhash",
                vec![Hlvalue::Variable(referent)],
                Hlvalue::Variable(identity.clone()),
            )],
            identity,
        );
        let translator = TranslationContext::new();
        let mut transformer = BaseFrameworkGCTransformer::new(&translator);

        transformer.transform_graph(&graph);

        let operations = graph.borrow().startblock.borrow().operations.clone();
        assert_eq!(operations[0].opname, "gc_push_roots");
        assert!(operations[0].args.is_empty());
        assert_eq!(operations[2].opname, "gc_pop_roots");
        assert!(operations[2].args.is_empty());
    }

    #[test]
    fn weakref_create_keeps_its_last_use_argument_alive() {
        let referent = gc_var("referent");
        let weakref = gc_var("weakref");
        let graph = single_block_graph(
            "weakref_argument",
            referent.clone(),
            vec![SpaceOperation::new(
                "weakref_create",
                vec![Hlvalue::Variable(referent.clone())],
                Hlvalue::Variable(weakref.clone()),
            )],
            weakref,
        );
        let translator = TranslationContext::new();
        let mut transformer = BaseFrameworkGCTransformer::new(&translator);

        transformer.transform_graph(&graph);

        let operations = graph.borrow().startblock.borrow().operations.clone();
        assert_eq!(
            operations[0].args,
            vec![Hlvalue::Variable(referent.clone())]
        );
        assert_eq!(operations[2].args, vec![Hlvalue::Variable(referent)]);
    }

    #[test]
    fn graph_is_transformed_only_once() {
        let root = gc_var("root");
        let stats = signed_var("stats");
        let graph = single_block_graph(
            "once",
            root,
            vec![SpaceOperation::new(
                "gc_heap_stats",
                vec![],
                Hlvalue::Variable(stats.clone()),
            )],
            stats,
        );
        let translator = TranslationContext::new();
        let mut transformer = BaseFrameworkGCTransformer::new(&translator);

        transformer.transform_graph(&graph);
        transformer.transform_graph(&graph);

        assert_eq!(graph.borrow().startblock.borrow().operations.len(), 3);
    }

    /// One artefact's graph: `(def_id, name, line)` per function, plus edges.
    fn part(fns: &[(u64, &str, u64)], edges: &[(u64, u64)]) -> CallGraph {
        let mut names = HashMap::new();
        let mut keys = HashMap::new();
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        for &(id, name, line) in fns {
            names.insert(id, name.to_string());
            keys.insert(
                id,
                if name.contains('<') {
                    format!("{name}@{line}")
                } else {
                    name.to_string()
                },
            );
            callees.entry(id).or_default();
        }
        let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
        for &(from, to) in edges {
            callees.entry(from).or_default().insert(to);
            callers.entry(to).or_default().insert(from);
        }
        CallGraph {
            names,
            keys,
            callees,
            callers,
            indirect: HashSet::new(),
            opaque: OpaqueCensus::default(),
        }
    }

    fn node_of(j: &Joined, part: usize, def_id: u64) -> u64 {
        j.canonical[part][&def_id]
    }

    /// The join exists for this: one artefact declares `pyframe::<Impl>::new`
    /// and calls it, the other defines it.  The opaque spelling must not stop
    /// the caller from reaching what the body reaches.
    #[test]
    fn an_opaque_name_at_one_line_joins_across_artefacts() {
        let declaring = part(
            &[
                (1, crate::runtime_names::symbols::PYFRAME_NEW, 40),
                (2, "pyre::caller", 9),
            ],
            &[(2, 1)],
        );
        let defining = part(
            &[
                (1, crate::runtime_names::symbols::PYFRAME_NEW, 40),
                (7, "pyre::allocates", 80),
            ],
            &[(1, 7)],
        );
        let joined = Joined::build(&[&declaring, &defining]);
        assert_eq!(node_of(&joined, 0, 1), node_of(&joined, 1, 1));
        let caller = node_of(&joined, 0, 2);
        let allocates = node_of(&joined, 1, 7);
        let reached = joined.graph.reaching(&HashSet::from([allocates]));
        assert!(
            reached.contains(&caller),
            "the caller lost the body it declares"
        );
    }

    /// Two different functions rendering to one spelling are two nodes, so no
    /// edge is invented between a caller of one and a callee of the other.
    #[test]
    fn one_opaque_spelling_at_two_lines_stays_two_functions() {
        let a = part(
            &[
                (1, crate::runtime_names::symbols::PYFRAME_NEW, 40),
                (2, "pyre::caller", 9),
            ],
            &[(2, 1)],
        );
        let b = part(
            &[
                (1, crate::runtime_names::symbols::PYFRAME_NEW, 512),
                (7, "pyre::allocates", 80),
            ],
            &[(1, 7)],
        );
        let joined = Joined::build(&[&a, &b]);
        assert_ne!(node_of(&joined, 0, 1), node_of(&joined, 1, 1));
        let caller = node_of(&joined, 0, 2);
        let allocates = node_of(&joined, 1, 7);
        let reached = joined.graph.reaching(&HashSet::from([allocates]));
        assert!(
            !reached.contains(&caller),
            "the join invented a path between two different functions"
        );
    }

    /// A key one artefact carries twice names no single function there.
    #[test]
    fn a_key_repeated_inside_one_artefact_is_not_a_join_key() {
        let a = part(
            &[
                (1, crate::runtime_names::symbols::PYFRAME_NEW, 40),
                (2, crate::runtime_names::symbols::PYFRAME_NEW, 40),
            ],
            &[],
        );
        let b = part(&[(1, crate::runtime_names::symbols::PYFRAME_NEW, 40)], &[]);
        let joined = Joined::build(&[&a, &b]);
        assert_eq!(joined.joined_names, 0);
        assert_eq!(joined.ambiguous_names, 1);
        assert_ne!(node_of(&joined, 0, 1), node_of(&joined, 0, 2));
        assert_ne!(node_of(&joined, 0, 1), node_of(&joined, 1, 1));
    }

    /// What charon spelled out joins on the name alone, at any line — a
    /// declaration and its definition are not required to agree on one.
    #[test]
    fn a_fully_spelled_name_joins_on_the_name() {
        let a = part(&[(1, "pyre::call::call_function_impl_result", 100)], &[]);
        let b = part(&[(9, "pyre::call::call_function_impl_result", 100)], &[]);
        let joined = Joined::build(&[&a, &b]);
        assert_eq!(node_of(&joined, 0, 1), node_of(&joined, 1, 9));
        assert_eq!(joined.joined_names, 1);
    }

    /// `project` answers in the asked artefact's own def ids.
    #[test]
    fn project_maps_a_joined_node_back_onto_one_artefacts_ids() {
        let a = part(&[(1, crate::runtime_names::symbols::PYFRAME_NEW, 40)], &[]);
        let b = part(&[(5, crate::runtime_names::symbols::PYFRAME_NEW, 40)], &[]);
        let joined = Joined::build(&[&a, &b]);
        let node = node_of(&joined, 0, 1);
        assert_eq!(
            joined.project(0, &HashSet::from([node])),
            HashSet::from([1])
        );
        assert_eq!(
            joined.project(1, &HashSet::from([node])),
            HashSet::from([5])
        );
    }

    fn item_meta(path: &[&str]) -> serde_json::Value {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    }

    /// Body whose only Call to `cleanup_only` (id 2) sits in a cleanup
    /// block reached through `on_unwind`.
    fn llbc_with_cleanup_only_call() -> majit_charon_reader::Llbc {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let gens = serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        });
        let place = |id: u64| serde_json::json!({"kind": {"Local": id}, "ty": scalar});
        let call_bb = |fun: u64, target: u64, on_unwind: u64, is_cleanup: bool| {
            serde_json::json!({
                "statements": [],
                "is_cleanup": is_cleanup,
                "terminator": {"span": span.clone(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {"kind": {"Fun": fun}, "generics": gens.clone()}},
                        "args": [],
                        "dest": place(0)
                    },
                    "target": target,
                    "on_unwind": on_unwind
                }}}
            })
        };
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()}
                ]},
                "body": [
                    call_bb(1, 1, 2, false),
                    {"statements": [], "is_cleanup": false, "terminator": {
                        "span": span.clone(), "kind": "Return"
                    }},
                    call_bb(2, 3, 3, true),
                    {"statements": [], "is_cleanup": true, "terminator": {
                        "span": span.clone(), "kind": "UnwindResume"
                    }}
                ]
            }}
        });
        let opaque = |id: u64, name: &str| {
            serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(&["fixture", name]),
                "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
                "body": "Opaque"
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [holder, opaque(1, "normal_callee"), opaque(2, "cleanup_only")],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("cleanup-call fixture parses")
    }

    /// Regular and trait-default callees of every Call terminator in `body`.
    /// The oracle iterates every block, matching `build` before cleanup
    /// blocks were kept.
    fn all_block_callees(
        llbc: &majit_charon_reader::Llbc,
        body: &majit_charon_reader::ullbc::Unstructured,
    ) -> HashSet<u64> {
        use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};
        let mut entry = HashSet::new();
        for bb in &body.body {
            let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
                continue;
            };
            match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id }) => {
                        entry.insert(*id);
                    }
                    CallKind::Trait(v) => {
                        if let Some(c) = v.get(2).and_then(|x| x.as_u64())
                            && llbc.fn_by_id(c).and_then(|f| f.unstructured()).is_some()
                        {
                            entry.insert(c);
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        entry
    }

    /// A Call that exists only on the `on_unwind` cleanup chain is not a
    /// call-graph edge, matching `strip_cleanup_blocks` of the same body.
    #[test]
    fn cleanup_only_call_is_not_a_call_graph_edge() {
        use majit_charon_reader::ullbc::strip_cleanup_blocks;

        let llbc = llbc_with_cleanup_only_call();
        let cg = build(&llbc);
        let holder = cg.callees.get(&0).expect("holder is a local fn");
        assert!(holder.contains(&1), "forward Call to normal_callee");
        assert!(
            !holder.contains(&2),
            "cleanup-only Call must not be an edge: {holder:?}"
        );

        let fd = llbc.fn_by_id(0).expect("holder decl");
        let stripped = strip_cleanup_blocks(fd.unstructured().expect("Unstructured body"));
        assert_eq!(
            holder,
            &all_block_callees(&llbc, &stripped),
            "retained-body edges must equal strip_cleanup_blocks"
        );
    }

    /// Every Unstructured body in pyre-object yields the same call-graph
    /// edges as `strip_cleanup_blocks` of that body.
    #[test]
    fn object_ullbc_call_graph_matches_strip_cleanup_blocks() {
        use majit_charon_reader::ullbc::strip_cleanup_blocks;

        let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
        let llbc = majit_charon_reader::Llbc::load(path).expect("load pyre-object LLBC");
        let cg = build(&llbc);
        for fd in llbc.iter_fun_decls() {
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let stripped = strip_cleanup_blocks(body);
            let got = cg.callees.get(&fd.def_id).cloned().unwrap_or_default();
            let want = all_block_callees(&llbc, &stripped);
            assert_eq!(
                got,
                want,
                "call-graph edges of {} (id {}) must equal strip_cleanup_blocks",
                fd.item_meta.name_path(),
                fd.def_id
            );
        }
    }

    /// Body whose first terminator is unparsed (`Unknown`) and whose next
    /// non-cleanup block is a Call. `forward_reachable_mask` stops at
    /// the Unknown, so the Call is not in the mask.
    fn llbc_with_unknown_terminator_then_call() -> majit_charon_reader::Llbc {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let gens = serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        });
        let place = |id: u64| serde_json::json!({"kind": {"Local": id}, "ty": scalar});
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()}
                ]},
                "body": [
                    {"statements": [], "is_cleanup": false, "terminator": {
                        "span": span.clone(), "kind": "InlineAsm"
                    }},
                    {"statements": [], "is_cleanup": false, "terminator": {"span": span.clone(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {"kind": {"Fun": 1}, "generics": gens.clone()}},
                            "args": [],
                            "dest": place(0)
                        },
                        "target": 2,
                        "on_unwind": 2
                    }}}},
                    {"statements": [], "is_cleanup": false, "terminator": {
                        "span": span.clone(), "kind": "Return"
                    }}
                ]
            }}
        });
        let callee = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta(&["fixture", "downstream_callee"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": "Opaque"
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [holder, callee],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("unknown-then-call fixture parses")
    }

    /// A parseable Call after an `Unknown` terminator stays an edge, and
    /// the caller is `indirect` (`GraphAnalyzer.analyze` `top_result`).
    #[test]
    fn unknown_terminator_keeps_downstream_call_edge() {
        let llbc = llbc_with_unknown_terminator_then_call();
        let fd = llbc.fn_by_id(0).expect("holder decl");
        let body = fd.unstructured().expect("Unstructured body");
        let forward = crate::front::mir::forward_reachable_mask(&llbc, &body);
        assert!(forward[0], "Unknown block is reached");
        assert!(
            !forward[1],
            "mask stops at Unknown; Call block is not forward"
        );

        let cg = build(&llbc);
        let holder = cg.callees.get(&0).expect("holder is a local fn");
        assert!(
            holder.contains(&1),
            "Call after Unknown must remain an edge: {holder:?}"
        );
        assert!(
            cg.indirect.contains(&0),
            "Unknown terminator is GraphAnalyzer.analyze top_result"
        );
    }
}
