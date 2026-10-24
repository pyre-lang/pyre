//! RPython: `rpython/jit/codewriter/call.py` `CallControl`.
//!
//! pyre-specific analog of RPython's `CallControl`. RPython's CallControl
//! owns the JIT's per-warmspot state: the `jitcodes` dict mapping graphs
//! to compiled JitCodes and the `unfinished_graphs` queue that
//! `CodeWriter.make_jitcodes()` drains at warmspot init (codewriter.py).
//!
//! pyre's "graph" is a `CodeObject` pointer; otherwise the field-by-field
//! port matches upstream:
//!
//! | RPython `call.py`           | pyre `call.rs`              |
//! |-----------------------------|-----------------------------|
//! | `self.virtualref_info`      | `self.virtualref_info`      |
//! | `self.has_libffi_call`      | `self.has_libffi_call`      |
//! | `self.cpu`                  | `self.cpu`                  |
//! | `self.jitdrivers_sd`        | `self.jitdrivers_sd`        |
//! | `self.jitcodes`             | `self.jitcodes`             |
//! | `self.unfinished_graphs`    | `self.unfinished_graphs`    |
//! | `self.callinfocollection`   | `self.callinfocollection`   |
//! | `__init__(cpu, jitdrivers_sd)` | `new(cpu, jitdrivers_sd)`|
//! | `get_jitcode()`             | `get_jitcode()`             |
//! | `grab_initial_jitcodes()`   | `grab_initial_jitcodes()`   |
//! | `enum_pending_graphs()`     | `enum_pending_graphs()`     |
//! | (n/a — pure-lookup helper)  | `find_jitcode()`            |
//!
//! Note: RPython's CallControl is owned by a single
//! `CodeWriter` instance (`warmspot.py`) for the lifetime of the JIT.
//! pyre mirrors that with a per-thread singleton accessed via
//! `CodeWriter::instance()`; interior mutability of the queue/cache is
//! expressed with `UnsafeCell` on the owning `CodeWriter`.

use rustc_hash::FxHashMap;
use std::collections::HashMap;

use majit_jitcode::jitcode::BhCallDescr;
use pyre_interpreter::CodeObject;
use pyre_jit_trace::PyJitCode;

use super::cpu::Cpu;

/// RPython: `rpython/jit/codewriter/effectinfo.py` `class CallInfoCollection`.
///
/// Owns the `_callinfo_for_oopspec` table that maps `oopspec_name → (calldescr,
/// func_addr)` so `assembler.finished()` can register the raw helper addresses
/// for symbolic debugging.
///
/// Note: pyre has no `oopspec` system — every blackhole
/// helper goes through `bh_*` C ABI thunks declared on `CodeWriter` directly,
/// not through an indirected `_callinfo_for_oopspec` lookup. The struct is an
/// empty shell so `CallControl` and `Assembler::finished` keep their RPython
/// signatures intact; populating it would require an oopspec layer that
/// pyre's Python-bytecode-direct lowering does not need.
#[derive(Debug, Default)]
pub struct CallInfoCollection;

impl CallInfoCollection {
    pub fn new() -> Self {
        Self
    }
}

/// RPython: `rpython/jit/metainterp/warmspot.py` `JitDriverStaticData` —
/// the per-portal record `CallControl.jitdrivers_sd` holds and
/// `grab_initial_jitcodes` (call.py) iterates.  RPython's
/// `JitDriverStaticData` carries far more (`name`, `index`,
/// `mainjitcode`, `_green_args_spec`, …); pyre adds only the fields
/// the lazy portal-discovery path actually consumes today.
///
/// `Copy` was previously derived because every field was `Copy` /
/// `Option<Copy>`; restoring `mainjitcode` (`Option<Arc<PyJitCode>>`
/// per `call.py`) takes that away. Only `Clone` remains —
/// per-field reads inside `grab_initial_jitcodes` snapshot the
/// `Copy` fields explicitly to keep the iteration cheap.
#[derive(Clone)]
pub struct JitDriverStaticData {
    /// RPython: `JitDriverStaticData.portal_graph`. The CodeObject
    /// the portal_runner enters when `get_jitcode(jd.portal_graph)`
    /// fires inside `grab_initial_jitcodes`.
    pub portal_graph: *const CodeObject,
    /// RPython: `JitDriverStaticData.mainjitcode`
    /// (`call.py` `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`
    /// left-hand side, plus `call.py`
    /// `jd.mainjitcode.jitdriver_sd = jd`).  Populated by
    /// [`CallControl::grab_initial_jitcodes`] with the same Arc that
    /// `CallControl.jitcodes[graph]` holds, mirroring RPython's
    /// "same `JitCode` instance is shared between the cc.jitcodes dict
    /// and jd.mainjitcode" identity.  Stays `None` until
    /// `grab_initial_jitcodes` fires, matching RPython's
    /// `jd.mainjitcode = None` before call.py.
    ///
    /// Note (type-only): RPython types this as
    /// `JitCode`; pyre stores `Arc<PyJitCode>`, where `PyJitCode` is a
    /// thin wrapper around `Arc<RuntimeJitCode>` plus pyre-only
    /// metadata (`PyJitCodeMetadata`, `code_ptr`, `w_code`, `has_abort`).
    /// The wrapper exists because pyre's codewriter
    /// publication path needs interior mutability to fill an empty
    /// skeleton in place (so that the same Arc identity sits in both
    /// `cc.jitcodes[graph]` and `jd.mainjitcode` while the assembler
    /// drains pending graphs into populated payloads).  The runtime
    /// `JitCode` (`RuntimeJitCode`) is reachable via
    /// `mainjitcode.jitcode` and is the object `jitdriver_sd` is
    /// stamped onto, matching upstream's
    /// `jd.mainjitcode.jitdriver_sd = jd` line-for-line.  Production
    /// access drills explicitly through `.jitcode` rather than relying
    /// on `Deref`, so the wrapper boundary stays visible at every site.
    ///
    /// **Deletion criterion**: removable once the codewriter publication
    /// path is reworked so the runtime `JitCode` itself can be
    /// late-stamped without an outer skeleton (the per-jitcode pyre-only
    /// metadata moves to a side table keyed by `RuntimeJitCode` Arc
    /// identity).
    pub mainjitcode: Option<std::sync::Arc<PyJitCode>>,
}

/// RPython: `rpython/jit/codewriter/call.py` `class CallControl(object)`.
pub struct CallControl {
    /// call.py `virtualref_info = None` — class-level default,
    /// populated by `CodeWriter.setup_vrefinfo` (codewriter.py).
    ///
    /// Note: pyre has no `virtualref` machinery yet
    /// (no `@jit.virtual_ref` annotations, no `vref_info` lookup); the
    /// slot is `None` and the setter is a no-op shell.
    pub virtualref_info: Option<()>,

    /// call.py `has_libffi_call = False` — class-level default.
    ///
    /// Note: pyre has no `_call_aroundstate_target_`
    /// rewriting in `getcalldescr`, so this stays `false`.
    pub has_libffi_call: bool,

    /// call.py `self.cpu = cpu`.
    ///
    /// pyre's `Cpu` (see [`super::cpu::Cpu`]) bundles the blackhole
    /// helper function pointers used by `transform_graph_to_jitcode`
    /// to populate the `JitCode` fn-ptr table. The same `cpu` is also
    /// reachable via `CodeWriter::cpu(&self)` (codewriter.py
    /// `self.cpu = cpu`).
    pub cpu: Cpu,

    /// call.py `self.jitdrivers_sd = jitdrivers_sd`.
    ///
    /// `grab_initial_jitcodes()` (call.py) iterates this list
    /// and calls `get_jitcode(jd.portal_graph)` on each entry. pyre
    /// populates the list lazily via `CodeWriter::setup_jitdriver`
    /// (codewriter.py), once per unique portal CodeObject the
    /// runtime decides to JIT — see [`JitDriverStaticData`] for the
    /// pyre-only adapter fields.
    pub jitdrivers_sd: Vec<JitDriverStaticData>,

    /// call.py `self.jitcodes = {}` — map `{graph: jitcode}`.
    ///
    /// pyre keys on the canonical raw `CodeObject*` graph identity.
    /// Callers normalize wrapper-backed runtime inputs before they
    /// reach `CallControl`, so the cache itself can stay a plain
    /// `{graph: jitcode}` map like RPython's.
    ///
    /// Values are `Arc<PyJitCode>` so the same allocation can sit in
    /// `MetaInterpStaticData.jitcodes` too. RPython's two stores hold
    /// references to identical Python `JitCode` objects through
    /// refcount semantics; pyre mirrors that by sharing the `Arc` —
    /// the `find_jitcode_arc` helper hands the trace-side
    /// `state::jitcode_for` callback the same `Arc` it just installed
    /// here. `get_jitcode` returns a `&PyJitCode` elaborated into
    /// `'static` to match RPython's Python semantics (see the SAFETY
    /// comment there); the `Arc` keeps the address stable across
    /// `HashMap` rehashes.
    pub jitcodes: HashMap<usize, std::sync::Arc<PyJitCode>>,

    /// call.py `self.unfinished_graphs = []` — LIFO queue of graphs
    /// pending compilation. Populated by `get_jitcode()` when it sees a
    /// new graph and drained by `enum_pending_graphs()` inside
    /// `make_jitcodes()` (codewriter.py).
    ///
    /// pyre now mirrors RPython's bare-graph queue directly. The
    /// pyre-only `w_code` adapter state lives on the cached `PyJitCode`
    /// shell (`payload.w_code`, stamped at enqueue) instead of in the
    /// queue tuple so `enum_pending_graphs()` can match `call.py` again.
    ///
    /// The queue itself is not a GC root. The wrapper lives on the
    /// skeleton `payload.w_code` at enqueue. That slot is forwarded by
    /// `walk_jitcode_code_roots_in` through `CallControl.jitcodes` (the
    /// owner while the drain compiles) and again through `sd.jitcodes`
    /// after install. `install_jitcodes` rejects skeletons, so a queued
    /// payload is not in `sd.jitcodes` until the drain finishes.
    pub unfinished_graphs: Vec<*const CodeObject>,

    /// call.py `self.callinfocollection = CallInfoCollection()`.
    ///
    /// Fed to `Assembler::finished()` at the end of
    /// `CodeWriter::make_jitcodes` (codewriter.py). See
    /// [`CallInfoCollection`] for the pyre-side shell rationale.
    pub callinfocollection: CallInfoCollection,

    /// Pyre-only per-graph result of `eval::unsupported_jit_shape`.
    ///
    /// RPython has no runtime frame-shape gate: policy and encodability are
    /// decided once while `CallControl.jitcodes` is populated.  Until pyre's
    /// remaining gate arms are removed, keep their likewise immutable result
    /// under the same per-graph owner and raw graph key as `jitcodes`, rather
    /// than re-walking a user code object's constants and bytecode on every
    /// invocation.  The owner is the same graph-keyed table as
    /// `CallControl.jitcodes`; FxHash keeps a pointer-keyed probe off
    /// SipHash.  Values are the private `UnsupportedJitShape` discriminants,
    /// kept as `u8` to avoid making the codewriter layer depend on the
    /// portal evaluator.
    pub graph_jit_shapes: FxHashMap<usize, u8>,
}

/// Slot of `PyPyJitDriver` in `MetaInterpStaticData.jitdrivers_sd`.
///
/// `pypyjit_driver_descriptor` is registered first (`register_jitdriver_sd`),
/// then `unpackiterable_driver_descriptor`, then
/// `generatorentry_driver_descriptor`. A per-code body that is the pypyjit
/// portal's `mainjitcode` for one green `pycode` stamps this slot
/// (`call.py` `jd.mainjitcode.jitdriver_sd = jd`). The position in
/// `CallControl.jitdrivers_sd` is a different list and is not a driver index.
pub(crate) const PYPYJIT_DRIVER_SD_INDEX: usize = 0;

/// The [`CallControl::graph_jit_shapes`] value recorded for a graph whose
/// jitcode fails to assemble.
///
/// The map's values are `eval::UnsupportedJitShape` discriminants, and the
/// codewriter must not name that private portal-evaluator type, so the one
/// discriminant it has to write lives beside the map it writes into. `eval.rs`
/// pins this to the enum with a `const _: () = assert!(..)`, so renumbering or
/// removing the variant is a build error rather than a silently misclassified
/// frame shape.
pub(crate) const JIT_SHAPE_CONST_ENCODING_OVERFLOW: u8 = 3;

impl CallControl {
    /// RPython: `CallControl.__init__(cpu=None, jitdrivers_sd=[])`
    /// (call.py).
    pub fn new(cpu: Cpu, jitdrivers_sd: Vec<JitDriverStaticData>) -> Self {
        // call.py `assert isinstance(jitdrivers_sd, list)`.
        // Rust's type system enforces this at compile time.
        Self {
            virtualref_info: None,
            has_libffi_call: false,
            cpu,
            jitdrivers_sd,
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
            callinfocollection: CallInfoCollection::new(),
            graph_jit_shapes: FxHashMap::default(),
        }
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py).
    ///
    /// ```python
    /// for jd in self.jitdrivers_sd:
    ///     jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///     jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// PARITY: `JitDriverStaticData.mainjitcode` (call.py left-hand
    /// side) is assigned immediately from `get_jitcode`. The
    /// back-reference at call.py (`jd.mainjitcode.jitdriver_sd = jd`)
    /// is the pypyjit driver slot [`PYPYJIT_DRIVER_SD_INDEX`], not this
    /// list's position. `CallControl.jitdrivers_sd` records which code
    /// objects are portal graphs; `MetaInterpStaticData.jitdrivers_sd`
    /// is the table blackhole and `is_main_jitcode` index.
    pub fn grab_initial_jitcodes(&mut self) {
        // Index loop because get_jitcode borrows `self.jitcodes`
        // mutably, which would conflict with an immutable borrow over
        // `self.jitdrivers_sd`. The field snapshotted below
        // (`portal_graph`) is `Copy`, so the per-iteration
        // field read stays cheap.
        for i in 0..self.jitdrivers_sd.len() {
            let portal_graph = self.jitdrivers_sd[i].portal_graph;
            let code = unsafe { &*portal_graph };
            // call.py `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`.
            // Inserts an empty PyJitCode skeleton into `jitcodes`, pushes
            // the graph onto `unfinished_graphs`. Drop the returned
            // clone immediately so the cached slot is uniquely owned for
            // the call.py stamp below.
            drop(self.get_jitcode(code, std::ptr::null()));
            // call.py `jd.mainjitcode.jitdriver_sd = jd` — stamp the
            // skeleton's `jitdriver_sd` while the outer `Arc<PyJitCode>`
            // and inner `Arc<JitCode>` still have refcount 1 (only the
            // jitcodes slot holds them). Then bind `jd.mainjitcode` to
            // that same Arc so consumers reading `cc.jitcodes[key]` and
            // `jd.mainjitcode` see byte-identical Arc identity per the
            // upstream invariant.
            let key = Self::jitcode_key(portal_graph);
            if let Some(slot) = self.jitcodes.get_mut(&key) {
                // RPython call.py:148 — `jd.mainjitcode.jitdriver_sd =
                // jd`.  Canonical's `jitdriver_sd` is an OnceLock with
                // interior mutability, so the write succeeds through
                // `&Arc<JitCode>` even when the surrounding Arc is
                // shared (e.g. `cc.jitcodes` and `jd.mainjitcode`).
                let pyjitcode = std::sync::Arc::clone(slot);
                if pyjitcode.jitcode.jitdriver_sd().is_none() {
                    pyjitcode.jitcode.set_jitdriver_sd(PYPYJIT_DRIVER_SD_INDEX);
                }
                self.jitdrivers_sd[i].mainjitcode = Some(std::sync::Arc::clone(slot));
            }
        }
    }

    /// Pure lookup of an already-compiled JitCode. Matches RPython's
    /// runtime pattern where callers read `self.jitcodes[graph]`
    /// directly after `make_jitcodes()` has populated the dict; no
    /// compilation side effect. Returns `None` if `code` has not been
    /// compiled yet.
    pub fn find_jitcode(&self, code: *const CodeObject) -> Option<&PyJitCode> {
        self.jitcodes.get(&(code as usize)).map(|arc| arc.as_ref())
    }

    /// `Arc`-cloning variant used by the trace-side `state::jitcode_for`
    /// callback: returns the same `Arc<PyJitCode>` the SD will store,
    /// so both stores reference one allocation. RPython's
    /// `MetaInterpStaticData.jitcodes`, `CallControl.jitcodes`, and
    /// `JitDriverStaticData.mainjitcode` hold the same Python `JitCode`
    /// objects via refcount semantics; this helper is the Rust analog.
    pub fn find_jitcode_arc(&self, code: *const CodeObject) -> Option<std::sync::Arc<PyJitCode>> {
        self.jitcodes
            .get(&(code as usize))
            .map(std::sync::Arc::clone)
    }

    /// Compiled-entry variant used by setup-time publishers and tests:
    /// return the shared `Arc<PyJitCode>` only when the entry is fully
    /// populated by the `make_jitcodes()` drain (`PyJitCode::is_populated`
    /// — `pc_map` non-empty). RPython never exposes skeleton entries to
    /// runtime readers; a `None` here means the caller is still in setup
    /// and must finish draining before publishing to trace-side SD.
    pub fn find_compiled_jitcode_arc(
        &self,
        code: *const CodeObject,
    ) -> Option<std::sync::Arc<PyJitCode>> {
        let arc = self.jitcodes.get(&(code as usize))?;
        if !arc.is_populated() {
            return None;
        }
        Some(std::sync::Arc::clone(arc))
    }

    /// Reverse lookup: find the `PyJitCode` whose inner `JitCode` matches
    /// the given raw pointer. Pyre-only adaptation: RPython's
    /// `JitCode` has its portal-register assignments embedded in the
    /// regular inputargs, so the blackhole interpreter can fill them
    /// via `_prepare_next_section` with no side channel. Pyre carries
    /// `portal_frame_reg`/`portal_ec_reg` on `PyJitCodeMetadata`
    /// instead, so the blackhole chain needs this reverse map to
    /// populate them after `blackhole_from_resumedata` builds the chain.
    pub fn find_pyjitcode_by_jitcode_ptr(
        &self,
        jitcode_ptr: *const majit_metainterp::jitcode::JitCode,
    ) -> Option<&PyJitCode> {
        self.jitcodes
            .values()
            .find(|pyjit| std::sync::Arc::as_ptr(&pyjit.jitcode) == jitcode_ptr)
            .map(|arc| arc.as_ref())
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py).
    ///
    /// ```python
    /// while self.unfinished_graphs:
    ///     graph = self.unfinished_graphs.pop()
    ///     yield graph, self.jitcodes[graph]
    /// ```
    ///
    pub fn enum_pending_graphs(&mut self) -> Option<*const CodeObject> {
        self.unfinished_graphs.pop()
    }

    /// RPython: `CallControl.jitdriver_sd_from_portal_graph(graph)` —
    /// returns the `JitDriverStaticData` whose `portal_graph` matches
    /// `code`, or `None` if `code` is not a registered portal. Used by
    /// `CodeWriter::transform_graph_to_jitcode` (codewriter.py) to
    /// learn whether a graph is a portal **before** running jtransform,
    /// matching the RPython `portal_jd = self.callcontrol.…` lookup.
    ///
    /// pyre returns the jdindex (`usize`) instead of a borrowed `&jd`
    /// so callers can keep mutating `CallControl` afterwards without
    /// fighting the borrow checker; the index is stable across the
    /// lifetime of `jitdrivers_sd` because `setup_jitdriver` only
    /// appends and dedups.
    pub fn jitdriver_sd_from_portal_graph(&self, code: *const CodeObject) -> Option<usize> {
        self.jitdrivers_sd
            .iter()
            .position(|jd| jd.portal_graph == code)
    }

    /// RPython: `CallControl.get_jitcode_calldescr(graph)` (call.py).
    ///
    /// ```python
    /// def get_jitcode_calldescr(self, graph):
    ///     fnptr = getfunctionptr(graph)
    ///     FUNC = lltype.typeOf(fnptr).TO
    ///     fnaddr = llmemory.cast_ptr_to_adr(fnptr)
    ///     NON_VOID_ARGS = [ARG for ARG in FUNC.ARGS if ARG is not lltype.Void]
    ///     calldescr = self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS),
    ///                                      FUNC.RESULT, EffectInfo.MOST_GENERAL)
    ///     return (fnaddr, calldescr)
    /// ```
    ///
    /// The pair is constant across every graph, where RPython has one per
    /// `FUNC` type because lltype-typed `direct_call` ops carry varying
    /// signatures. Keeping the method shape preserves the call.py
    /// `(fnaddr, calldescr) = self.get_jitcode_calldescr(graph)` flow even so.
    ///
    /// Nothing dispatches the address it returns today. `bhimpl_recursive_call_*`
    /// reaches the portal through `get_portal_runner`, and where upstream's
    /// `bhimpl_inline_call_*` calls `cpu.bh_call_*(jitcode.fnaddr, …)`
    /// (`blackhole.py bhimpl_inline_call_*`), pyre reads its callee out of a build-time
    /// descr pool whose `fnaddr` comes from `JitCodeBuilder::set_native_entry`,
    /// and a CodeObject jitcode minted here is never in that pool.
    ///
    /// The pair is still the one that describes the portal runner, rather than
    /// a placeholder address beside an arbitrary descr.  A published `fnaddr`
    /// whose descr does not describe it is only latent while nothing dispatches
    /// it: the first caller that does reads the arg banks through the descr and
    /// calls the address, so the mismatch surfaces as a wrong-ABI call rather
    /// than as an error.  `bh_portal_runner_c` is the C-ABI runner
    /// `get_portal_runner` already answers with, and `"iirrr"` is the
    /// `portalfunc_ARGS` layout it consumes (`warmspot.py handle_jitexception`):
    /// greens `[next_instr:i, is_being_profiled:i, pycode:r]` + reds
    /// `[frame:r, ec:r]` (`interp_jit.py PyPyJitDriver`).
    pub fn get_jitcode_calldescr(&self, _graph: *const CodeObject) -> (i64, BhCallDescr) {
        let fnaddr = crate::call_jit::bh_portal_runner_c as *const () as usize as i64;
        let calldescr = BhCallDescr::from_arg_classes(
            "iirrr".to_string(),
            'r',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        (fnaddr, calldescr)
    }

    /// RPython: `CallControl.get_jitcode(self, graph, called_from=None)`
    /// (call.py).
    ///
    /// ```python
    /// def get_jitcode(self, graph, called_from=None):
    ///     try:
    ///         return self.jitcodes[graph]
    ///     except KeyError:
    ///         ...
    ///         fnaddr, calldescr = self.get_jitcode_calldescr(graph)
    ///         jitcode = JitCode(graph.name, fnaddr, calldescr,
    ///                           called_from=called_from)
    ///         self.jitcodes[graph] = jitcode
    ///         self.unfinished_graphs.append(graph)
    ///         return jitcode
    /// ```
    ///
    /// pyre creates an empty `PyJitCode` skeleton (call.py — RPython
    /// `JitCode(name, fnaddr, calldescr)` without bytecode) and queues
    /// the graph for the drain in `CodeWriter::make_jitcodes`, which
    /// re-runs `transform_graph_to_jitcode` and replaces the slot with
    /// the populated entry (codewriter.py
    /// `transform_graph_to_jitcode(graph, jitcode, ...)`).
    ///
    /// `w_code` is the owner `PyCode` wrapper. RPython's graph *is* that
    /// object; pyre threads it as a value from the caller that already
    /// holds it and stamps `payload.w_code` on the skeleton at enqueue.
    /// Null is only for a graph with no Python code object, or a test
    /// that never installs into `sd.jitcodes`.
    pub fn get_jitcode(
        &mut self,
        code: &CodeObject,
        w_code: *const (),
    ) -> std::sync::Arc<PyJitCode> {
        // RPython's `get_jitcode(graph)` receives the exact graph object
        // the dict is keyed by. Match that by requiring callers to pass
        // the canonical raw graph here; portal setup canonicalizes
        // `jd.portal_graph` once, and the lazy trace-side callback
        // unwraps the wrapper to raw before calling into `CallControl`.
        let code_ptr = code as *const CodeObject;
        let key = code_ptr as usize;
        // call.py get_jitcode `if graph in self.jitcodes: return self.jitcodes[graph]`
        // — the skeleton is built once. An already-registered portal is not
        // rebuilt when a later registration carries a different merge-point
        // PC: pyre observes merge-point PCs at runtime, but the first
        // registration's skeleton stands (no recompile-on-refine).
        let needs_rebuild = !self.jitcodes.contains_key(&key);
        if needs_rebuild {
            // The payload keeps this raw body address and trace-side readers
            // dereference it holding no wrapper, so the graph it belongs to
            // must outlive its wrappers
            // (`pyre_interpreter::pycode::retire_code_unit`).
            pyre_interpreter::pycode::pin_code_unit(code_ptr as *const ());
            // call.py:168-171 — create JitCode skeleton, insert, append
            // to unfinished_graphs. The body fill (jtransform / regalloc
            // / flatten / assemble) is deferred to
            // `CodeWriter::make_jitcodes`'s drain loop at
            // codewriter.py `transform_graph_to_jitcode(graph,
            // jitcode, verbose, len(all_jitcodes))`.
            self.reset_jitcode_skeleton(key, code_ptr, w_code);
            // call.py `self.unfinished_graphs.append(graph)`.
            self.unfinished_graphs.push(code_ptr);
        }
        let pyjit = std::sync::Arc::clone(self.jitcodes.get(&key).unwrap());
        pyjit.stamp_w_code(w_code);
        pyjit
    }

    /// Insert an empty skeleton for a freshly-registered graph — the state
    /// that follows `JitCode(graph.name, fnaddr, calldescr, ...)` at
    /// `call.py:168` before the drain runs `assembler.assemble`
    /// (codewriter.py:67) on it.
    ///
    /// Inserts a fresh `Arc<PyJitCode>` rather than mutating an existing
    /// payload in place: `publish_jitcode` fills the slot in place after
    /// the drain, but this entry point only runs when the key is absent
    /// (`get_jitcode`'s `call.py` guard), so there is no populated
    /// holder to clobber.
    fn reset_jitcode_skeleton(
        &mut self,
        key: usize,
        code_ptr: *const CodeObject,
        w_code: *const (),
    ) {
        // Only the absent-key path (`get_jitcode`'s `needs_rebuild`) installs a
        // skeleton. Enforce it so a stray call cannot roll a populated jitcode
        // back to a skeleton and break the runtime-reader publication invariant
        // documented above.
        debug_assert!(
            !self.jitcodes.contains_key(&key),
            "reset_jitcode_skeleton must only create fresh skeleton entries"
        );
        self.jitcodes.insert(
            key,
            std::sync::Arc::new(PyJitCode::skeleton_with_w_code(code_ptr, w_code)),
        );
    }

    /// Publish the populated jitcode into `self.jitcodes[graph]`.
    ///
    /// RPython mutates the same `JitCode` object in place — the drain
    /// at `codewriter.py` calls `transform_graph_to_jitcode(graph,
    /// jitcode, ...)` where `jitcode` is the skeleton that was already
    /// stored at `call.py` (`self.jitcodes[graph] = jitcode`), and
    /// inside that call `self.assembler.assemble(ssarepr, jitcode,
    /// num_regs)` (codewriter.py) fills the skeleton's fields.
    ///
    /// Pyre's `transform_graph_to_jitcode` returns a fresh `PyJitCode`
    /// instead of mutating the skeleton, so this helper bridges the
    /// split by filling the cached object in place. Both the outer
    /// `Arc<PyJitCode>` and the inner runtime `Arc<JitCode>` allocation
    /// stay stable; the latter is required before the orthodox
    /// `jtransform.handle_regular_call -> inline_call_*` port can store
    /// callee JitCode descriptors in callers.
    pub fn publish_jitcode(
        &mut self,
        key: usize,
        pyjitcode: PyJitCode,
    ) -> std::sync::Arc<PyJitCode> {
        if let Some(slot) = self.jitcodes.get_mut(&key) {
            // SAFETY: `publish_jitcode` runs on the JIT setup thread
            // during the codewriter drain (codewriter.py:79-85). The
            // cached `Arc<PyJitCode>` is only shared with `jd.mainjitcode`
            // and any pre-bound `MetaInterpStaticData.jitcodes` clone,
            // which were established by the same setup thread just
            // before this call. No runtime tracing or blackhole resume
            // can be observing the slot at this point — those paths
            // only run after `make_jitcodes` returns.
            unsafe {
                slot.replace_with(pyjitcode);
            }
            return std::sync::Arc::clone(slot);
        }
        let arc = std::sync::Arc::new(pyjitcode);
        self.jitcodes.insert(key, arc);
        std::sync::Arc::clone(self.jitcodes.get(&key).unwrap())
    }

    pub(crate) fn jitcode_key(code: *const CodeObject) -> usize {
        code as usize
    }

    /// Forward `payload.w_code` of every cached jitcode, including a
    /// skeleton queued by `get_jitcode` and not yet installed into
    /// `sd.jitcodes`. `walk_jitcode_code_roots_in` reaches this through
    /// `CallJitCallbacks.walk_writer_jitcode_code_roots`.
    pub fn walk_jitcode_code_roots(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        for payload in self.jitcodes.values() {
            pyre_jit_trace::state::walk_payload_w_code(payload, visitor);
        }
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new(Cpu::default(), Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitcode_calldescr_matches_its_published_portal_runner() {
        let control = CallControl::default();
        let (fnaddr, calldescr) = control.get_jitcode_calldescr(std::ptr::null());

        assert_eq!(
            fnaddr,
            crate::call_jit::bh_portal_runner_c as *const () as usize as i64
        );
        assert_eq!(calldescr.arg_classes, "iirrr");
        assert_eq!(calldescr.result_type, 'r');
    }
}
