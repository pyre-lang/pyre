//! Per-instantiation call-graph specialization.
//!
//! rustc compiles `map::<closure#N>` as its own function.
//! `rpython/translator/backendopt/graphanalyze.py` `analyze_direct_call`
//! walks that specialized graph; `top_result` is only for a callee that
//! is truly unknown (`indirect_call` with `graphs is None`).
//!
//! Charon extracts with `monomorphize: false`, so a generic body arrives
//! once and its `CallKind::Trait` obligations are `Clause` /
//! `ParentClause` refs. A Regular call whose `generics.trait_refs` already
//! name a `TraitImpl` supplies the binding: the copy substitutes each
//! depth-0 `Clause` with that ref, and a `ParentClause` of a resolved impl
//! reads that impl's `implied_trait_refs`. Each concrete instantiation is
//! its own node (`FunctionDesc.cachedgraph`). The unspecialized body is
//! left unchanged and stays `indirect`.
//!
//! A Regular `Fun` or resolved trait method whose declaration has no
//! Unstructured body is `GraphAnalyzer.analyze_external_call` when the
//! decl's crate is not a workspace crate: `bottom_result` (a leaf, not
//! unresolved) unless `_callbacks` names workspace `TraitImpl` methods.

use std::collections::{HashMap, HashSet};

use majit_charon_reader::ullbc::{
    CallFunc, CallKind, FunId, TermKind, TypeDeclSrc, Unstructured, is_closure_leaf,
};
use majit_charon_reader::{FunDecl, Llbc};
use serde_json::Value;

use super::framework::{
    CallGraph, OpaqueCensus, call_graph_flow_is_incomplete, call_graph_walks_block,
};

/// Nesting bound on Regular → spec → Regular instantiation.
/// A clause the caller cannot resolve, a `dyn`, a bodyless workspace
/// method, or a walk past this bound falls back to the generic node.
pub(crate) const MAX_INSTANTIATION_DEPTH: usize = 8;

/// Outcome of substituting a `CallKind::Trait` payload through the
/// current instantiation's clause bindings.
#[derive(Debug)]
pub(crate) enum TraitCallRes {
    /// `TraitImpl` method with an unstructured body in this artefact.
    Method { fn_id: u64, generics: Value },
    /// `BuiltinOrAuto` — a selected empty implementation (`NoopDestruct`
    /// and the other auto traits), not an unresolved clause.
    Empty,
    /// `GraphAnalyzer.analyze_external_call`: a bodiless callee whose crate
    /// is not a workspace crate. `callbacks` are `_callbacks` method
    /// graphs. The leaf is not unresolved.
    External {
        fn_id: u64,
        callbacks: Vec<(u64, Value)>,
    },
    /// Still a clause, a `Dyn`, a bodyless workspace method, a blocked
    /// external call, or a ref this caller cannot resolve.
    Unresolved,
}

/// Shared helper: the call-graph node a Regular call site names.
///
/// `caller_bindings` are the caller's resolved clause refs (empty when
/// walking a FunDecl as itself). `callee` is the `FunId::Regular` id and
/// `generics` is the call site's generic arguments (`trait_refs`).
///
/// During [`build_call_graph`] this creates a specialized node when the
/// callee body has a `Clause` / `ParentClause` Trait call — or a Regular
/// `Fun` that forwards such a clause in `generics.trait_refs` — that
/// resolves to a `TraitImpl` method. After the graph is built, it is
/// lookup-only on `spec_memo` and falls back to `callee` when the
/// instantiation was not recorded — never narrower than the generic node.
pub(crate) fn callee_node_for_call(
    graph: &CallGraph,
    llbc: &Llbc,
    caller_bindings: &[Value],
    callee: u64,
    generics: &Value,
) -> u64 {
    match instantiation_memo_key(llbc, caller_bindings, callee, generics) {
        Some(key) => graph.spec_memo.get(&key).copied().unwrap_or(callee),
        None => callee,
    }
}

/// Memo key for one callee instantiation, or `None` when the call site
/// cannot supply a `TraitImpl` binding (today's generic node).
pub(crate) fn instantiation_memo_key(
    llbc: &Llbc,
    caller_bindings: &[Value],
    callee: u64,
    generics: &Value,
) -> Option<(u64, String)> {
    let bindings = subst_call_trait_refs(llbc, caller_bindings, generics)?;
    let key = binding_key(llbc, &bindings)?;
    Some((callee, key))
}

/// Substitute each of a call site's `generics.trait_refs` through
/// `caller_bindings`. `None` when any ref is unresolvable (`dyn`, a
/// clause the caller cannot fill). BuiltinOrAuto refs pass through;
/// they do not block specialization of a neighbouring `TraitImpl`.
pub(crate) fn subst_call_trait_refs(
    llbc: &Llbc,
    caller_bindings: &[Value],
    generics: &Value,
) -> Option<Vec<Value>> {
    let generics = peel_generic_args(generics);
    let refs = generics
        .get("trait_refs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::with_capacity(refs.len());
    for tref in &refs {
        out.push(bind_trait_ref(llbc, tref, caller_bindings, 0)?);
    }
    Some(out)
}

/// Unwrap `Deduplicated` / `Value` / `skip_binder`, then:
/// - `Clause` Bound[0, i] (after counting binders) → `bindings[i]`
/// - `ParentClause` → that impl's `implied_trait_refs[index]`
/// - `TraitImpl` / `BuiltinOrAuto` → the unwrapped object
/// - `Dyn` → `None`
pub(crate) fn bind_trait_ref(
    llbc: &Llbc,
    tref: &Value,
    bindings: &[Value],
    depth: usize,
) -> Option<Value> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return None;
    }
    let (obj, extra) = unwrap_ref_binders(tref, llbc, 0)?;
    let tag = kind_tag(obj)?;
    match tag.as_str() {
        "Dyn" => None,
        "Clause" => {
            let (clause_depth, index) = clause_var(obj.get("kind")?.get("Clause")?)?;
            if clause_depth != extra {
                return None;
            }
            let bound = bindings.get(index)?;
            bind_trait_ref(llbc, bound, bindings, depth + 1)
        }
        "ParentClause" => {
            let pair = obj.get("kind")?.get("ParentClause")?.as_array()?;
            let parent = bind_trait_ref(llbc, pair.first()?, bindings, depth + 1)?;
            let index = pair.get(1)?.as_u64()? as usize;
            let impl_id = trait_impl_id(&parent, llbc)?;
            let implied = implied_trait_refs(llbc, impl_id)?;
            let child = implied.get(index)?;
            bind_trait_ref(llbc, child, bindings, depth + 1)
        }
        "TraitImpl" | "BuiltinOrAuto" => Some(obj.clone()),
        _ => Some(obj.clone()),
    }
}

/// Deterministic rendering of the `TraitImpl` slots in `bindings`.
/// BuiltinOrAuto is skipped so Sized/Drop auto traits do not split a
/// key. `None` when no slot is a `TraitImpl` — that call is not a
/// specialization.
pub(crate) fn binding_key(llbc: &Llbc, bindings: &[Value]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for (i, tref) in bindings.iter().enumerate() {
        let Some(impl_id) = trait_impl_id(tref, llbc) else {
            continue;
        };
        let rendered = crate::front::clause_spec::render_trait_impl(llbc, impl_id)
            .unwrap_or_else(|| format!("impl#{impl_id}"));
        parts.push(format!("{i}={rendered}"));
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("|"))
}

/// Resolve a `CallKind::Trait` payload under `bindings`.
pub(crate) fn resolve_trait_call(
    llbc: &Llbc,
    payload: &Value,
    bindings: &[Value],
    call_generics: &Value,
    workspace_crates: &HashSet<String>,
) -> TraitCallRes {
    let Some(arr) = payload.as_array() else {
        return TraitCallRes::Unresolved;
    };
    let Some(tref) = arr.first() else {
        return TraitCallRes::Unresolved;
    };
    let Some(bound) = bind_trait_ref(llbc, tref, bindings, 0) else {
        return TraitCallRes::Unresolved;
    };
    if kind_tag(&bound).as_deref() == Some("Dyn") {
        return TraitCallRes::Unresolved;
    }
    if kind_tag(&bound).as_deref() == Some("BuiltinOrAuto") {
        return TraitCallRes::Empty;
    }
    let mut subst = payload.clone();
    if let Some(slot) = subst.get_mut(0) {
        *slot = bound.clone();
    }
    let resolved = crate::front::clause_spec::trait_impl_method(&subst, llbc).or_else(|| {
        let impl_id = trait_impl_id(&bound, llbc)?;
        let method_idx = arr.get(1)?.as_u64()?;
        let fn_id = crate::front::clause_spec::impl_method_fn_id(llbc, impl_id, method_idx)
            .or_else(|| impl_method_from_row(llbc, impl_id, method_idx))?;
        let generics = bound
            .pointer("/kind/TraitImpl/generics")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        Some((fn_id, generics))
    });
    match resolved {
        Some((fn_id, generics)) => {
            if llbc
                .fn_by_id(fn_id)
                .is_some_and(|fd| fd.has_unstructured_body())
            {
                TraitCallRes::Method { fn_id, generics }
            } else {
                analyze_external_call(llbc, fn_id, call_generics, &[bound], workspace_crates)
                    .map(|callbacks| TraitCallRes::External { fn_id, callbacks })
                    .unwrap_or(TraitCallRes::Unresolved)
            }
        }
        None => TraitCallRes::Unresolved,
    }
}

/// `impl_method_fn_id` indexes `trait_impls_raw` by def id. A hole or a
/// shifted row misses; [`Llbc::trait_impl_by_id`] finds the row by id.
fn impl_method_from_row(llbc: &Llbc, impl_id: u64, method_idx: u64) -> Option<u64> {
    llbc.trait_impl_methods_of_def(impl_id)?
        .iter()
        .find_map(|method| {
            let method = method.as_ref()?;
            if method.method_idx == Some(method_idx) {
                method.fn_id
            } else {
                None
            }
        })
}

/// Build the resolved ULLBC call graph, specializing Regular calls whose
/// callee body has a clause-bound Trait call that the call site binds to
/// a `TraitImpl` method.
///
/// `workspace_crates` are cargo workspace package names (`-` → `_`). An
/// empty set classifies no callee as external (`GraphAnalyzer.analyze_external_call`
/// is not applied).
pub(crate) fn build_call_graph(llbc: &Llbc, workspace_crates: &HashSet<String>) -> CallGraph {
    let mut builder = Builder::new(llbc, workspace_crates);
    for fd in llbc.iter_fun_decls() {
        builder.install_generic(fd);
    }
    let generic_ids: Vec<u64> = builder.graph.names.keys().copied().collect();
    for id in generic_ids {
        builder.walk_generic(id);
    }
    builder.finish()
}

struct Builder<'a> {
    llbc: &'a Llbc,
    graph: CallGraph,
    next_id: u64,
    open_trait: HashMap<u64, bool>,
    workspace_crates: &'a HashSet<String>,
}

impl<'a> Builder<'a> {
    fn new(llbc: &'a Llbc, workspace_crates: &'a HashSet<String>) -> Self {
        let next_id = llbc
            .iter_fun_decls()
            .map(|fd| fd.def_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        Self {
            llbc,
            graph: CallGraph {
                names: HashMap::new(),
                keys: HashMap::new(),
                callees: HashMap::new(),
                callers: HashMap::new(),
                indirect: HashSet::new(),
                opaque: OpaqueCensus::default(),
                spec_memo: HashMap::new(),
            },
            next_id,
            open_trait: HashMap::new(),
            workspace_crates,
        }
    }

    fn install_generic(&mut self, fd: &FunDecl) {
        let id = fd.def_id;
        let name = fd.item_meta.name_path();
        let key = if name.contains('<') {
            format!(
                "{name}@{}",
                self.llbc
                    .span_data(&fd.item_meta.span)
                    .expect("item span missing from the artefact")
                    .beg
                    .line
            )
        } else {
            name.clone()
        };
        self.graph.keys.insert(id, key);
        self.graph.names.insert(id, name);
        self.graph.callees.entry(id).or_default();
    }

    fn walk_generic(&mut self, id: u64) {
        let Some(body) = self.llbc.fn_by_id(id).and_then(|fd| fd.unstructured()) else {
            return;
        };
        self.walk_body(id, &body, &[], 0, true);
    }

    /// Node for a Regular call from a body walked under `caller_bindings`.
    fn callee_node_for_call(
        &mut self,
        caller_bindings: &[Value],
        callee: u64,
        generics: &Value,
        depth: usize,
    ) -> u64 {
        if depth >= MAX_INSTANTIATION_DEPTH {
            return callee;
        }
        let Some(key) = instantiation_memo_key(self.llbc, caller_bindings, callee, generics) else {
            return callee;
        };
        if let Some(&id) = self.graph.spec_memo.get(&key) {
            return id;
        }
        if !self.body_has_open_trait(callee) {
            self.graph.spec_memo.insert(key, callee);
            return callee;
        }
        let Some(bindings) = subst_call_trait_refs(self.llbc, caller_bindings, generics) else {
            return callee;
        };
        let Some(body) = self.llbc.fn_by_id(callee).and_then(|fd| fd.unstructured()) else {
            self.graph.spec_memo.insert(key, callee);
            return callee;
        };
        // Identity while this instantiation is in progress so a Regular
        // forward that mutually recurses terminates (`node == callee`).
        self.graph.spec_memo.insert(key.clone(), callee);
        if !self.has_resolvable_trait_call(&body, &bindings, depth) {
            return callee;
        }
        let spec_id = self.next_id;
        self.next_id += 1;
        self.graph.spec_memo.insert(key.clone(), spec_id);
        let generic_name = self
            .graph
            .names
            .get(&callee)
            .cloned()
            .unwrap_or_else(|| format!("#{callee}"));
        let generic_key = self
            .graph
            .keys
            .get(&callee)
            .cloned()
            .unwrap_or_else(|| generic_name.clone());
        // Joined key: generic node's key + the resolved impl renders, so
        // the same instantiation in two artefacts joins. The name is the
        // same spelling plus that suffix so a seed pattern cannot match
        // the spec node as if it were the generic item.
        let spec_name = format!("{generic_name}#{}", key.1);
        let spec_key = format!("{generic_key}#{}", key.1);
        self.graph.names.insert(spec_id, spec_name);
        self.graph.keys.insert(spec_id, spec_key);
        self.graph.callees.entry(spec_id).or_default();
        self.walk_body(spec_id, &body, &bindings, depth + 1, false);
        spec_id
    }

    fn walk_body(
        &mut self,
        id: u64,
        body: &Unstructured,
        bindings: &[Value],
        depth: usize,
        note_opaque: bool,
    ) {
        let forward = crate::front::mir::forward_reachable_mask(self.llbc, body);
        let mut incomplete = false;
        let mut unresolved = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !call_graph_walks_block(bb, forward[bb_idx]) {
                continue;
            }
            if forward[bb_idx] && call_graph_flow_is_incomplete(self.llbc, bb) {
                incomplete = true;
            }
            let Ok(TermKind::Call { call, .. }) = bb.term(self.llbc) else {
                continue;
            };
            match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id: callee }) => {
                        self.add_regular_fun_call(id, *callee, &reg.generics, bindings, depth);
                    }
                    CallKind::Fun(FunId::Other(v)) => {
                        unresolved = true;
                        if note_opaque {
                            self.note_opaque(Some(v), "Fun(Other)");
                        }
                    }
                    CallKind::Trait(v) => {
                        // The generic FunDecl walk matches HEAD: every Trait
                        // terminator is opaque, plus a default-method edge
                        // when payload[2] has a body. Only a specialized
                        // copy may drop `indirect`, and only when every
                        // Trait call in that copy resolved to a body.
                        if note_opaque {
                            if let Some(c) = v.get(2).and_then(Value::as_u64)
                                && self
                                    .llbc
                                    .fn_by_id(c)
                                    .and_then(|f| f.unstructured())
                                    .is_some()
                            {
                                self.graph.callees.entry(id).or_default().insert(c);
                            }
                            unresolved = true;
                            self.note_opaque(Some(v), "Trait");
                        } else {
                            match resolve_trait_call(
                                self.llbc,
                                v,
                                bindings,
                                &reg.generics,
                                self.workspace_crates,
                            ) {
                                TraitCallRes::Method { fn_id, generics } => {
                                    let node =
                                        self.callee_node_for_call(&[], fn_id, &generics, depth);
                                    self.graph.callees.entry(id).or_default().insert(node);
                                }
                                TraitCallRes::Empty => {}
                                TraitCallRes::External { fn_id, callbacks } => {
                                    self.add_external_edges(id, fn_id, callbacks, depth);
                                }
                                TraitCallRes::Unresolved => {
                                    if let Some(c) = v.get(2).and_then(Value::as_u64)
                                        && self
                                            .llbc
                                            .fn_by_id(c)
                                            .and_then(|f| f.unstructured())
                                            .is_some()
                                    {
                                        self.graph.callees.entry(id).or_default().insert(c);
                                    }
                                    unresolved = true;
                                }
                            }
                        }
                    }
                    CallKind::Ptr(v) => {
                        // A `global_hook!` wrapper with a closed trampoline
                        // family is recovered by
                        // `CallGraph::apply_hook_call_families`
                        // (`GraphAnalyzer.analyze_indirect_call`).
                        unresolved = true;
                        if note_opaque {
                            self.note_opaque(Some(v), "Ptr");
                        }
                    }
                    CallKind::Unknown => {
                        unresolved = true;
                        if note_opaque {
                            self.graph.opaque.unknown += 1;
                        }
                    }
                },
                CallFunc::Dynamic(op, _) => {
                    // Same closed-family recovery as `CallKind::Ptr`.
                    unresolved = true;
                    if note_opaque {
                        let ty = match op {
                            majit_charon_reader::ullbc::Operand::Copy(p)
                            | majit_charon_reader::ullbc::Operand::Move(p) => match &p.ty {
                                majit_charon_reader::ullbc::TyRef::Inline { value: (_, v) } => {
                                    Some(v)
                                }
                                majit_charon_reader::ullbc::TyRef::Dedup { id } => {
                                    self.llbc.dedup_body(*id)
                                }
                                majit_charon_reader::ullbc::TyRef::Other(v) => Some(v),
                            },
                            majit_charon_reader::ullbc::Operand::Const(v) => Some(v),
                        };
                        self.note_opaque(ty, "Dynamic");
                    }
                }
                CallFunc::Unknown => {
                    unresolved = true;
                    if note_opaque {
                        self.graph.opaque.unknown += 1;
                    }
                }
            }
        }
        if incomplete || unresolved {
            self.graph.indirect.insert(id);
        }
    }

    fn note_opaque(&mut self, raw: Option<&Value>, variant: &'static str) {
        *self.graph.opaque.by_variant.entry(variant).or_insert(0) += 1;
        match raw.and_then(first_trait_id) {
            Some(tid) => {
                let name = self
                    .llbc
                    .trait_by_id(tid)
                    .map(|t| t.item_meta.name_path())
                    .unwrap_or_else(|| format!("trait#{tid}"));
                *self.graph.opaque.dyn_trait.entry(name).or_insert(0) += 1;
            }
            None => self.graph.opaque.fn_value += 1,
        }
    }

    fn body_has_open_trait(&mut self, id: u64) -> bool {
        if let Some(&known) = self.open_trait.get(&id) {
            return known;
        }
        let known = self
            .llbc
            .fn_by_id(id)
            .and_then(|fd| fd.unstructured())
            .is_some_and(|body| body_has_clause_trait_call(self.llbc, &body));
        self.open_trait.insert(id, known);
        known
    }

    /// Regular `Fun` to a bodiless decl: `GraphAnalyzer.analyze_external_call`
    /// when the callee is external and has no blocker (leaf plus `_callbacks`).
    /// A blocker, or a workspace-crate bodiless decl, keeps today's edge:
    /// a leaf for a bodiless callee, and the caller is not unresolved.
    fn add_regular_fun_call(
        &mut self,
        caller: u64,
        callee: u64,
        generics: &Value,
        bindings: &[Value],
        depth: usize,
    ) {
        if is_external_callee(self.llbc, callee, self.workspace_crates)
            && let Some(refs) = subst_call_trait_refs(self.llbc, bindings, generics)
            && let Some(callbacks) =
                analyze_external_call(self.llbc, callee, generics, &refs, self.workspace_crates)
        {
            self.add_external_edges(caller, callee, callbacks, depth);
            return;
        }
        let node = self.callee_node_for_call(bindings, callee, generics, depth);
        self.graph.callees.entry(caller).or_default().insert(node);
    }

    fn add_external_edges(
        &mut self,
        caller: u64,
        fn_id: u64,
        callbacks: Vec<(u64, Value)>,
        depth: usize,
    ) {
        self.graph.callees.entry(caller).or_default().insert(fn_id);
        for (cb, generics) in callbacks {
            let node = self.callee_node_for_call(&[], cb, &generics, depth);
            self.graph.callees.entry(caller).or_default().insert(node);
        }
    }

    fn finish(mut self) -> CallGraph {
        self.graph.apply_hook_call_families();
        for (&caller, cs) in &self.graph.callees {
            for &c in cs {
                self.graph.callers.entry(c).or_default().insert(caller);
            }
        }
        self.graph
    }

    /// True when a clause-bound Trait terminator resolves to a method body,
    /// or a Regular `Fun` that still carries a `Clause` / `ParentClause`
    /// substitutes to a `TraitImpl` whose callee itself specializes.
    fn has_resolvable_trait_call(
        &mut self,
        body: &Unstructured,
        bindings: &[Value],
        depth: usize,
    ) -> bool {
        let forward = crate::front::mir::forward_reachable_mask(self.llbc, body);
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !call_graph_walks_block(bb, forward[bb_idx]) {
                continue;
            }
            let Ok(TermKind::Call { call, .. }) = bb.term(self.llbc) else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            match &reg.kind {
                CallKind::Trait(v) => {
                    if !payload_is_clause_or_parent(self.llbc, v) {
                        continue;
                    }
                    if matches!(
                        resolve_trait_call(
                            self.llbc,
                            v,
                            bindings,
                            &reg.generics,
                            self.workspace_crates,
                        ),
                        TraitCallRes::Method { .. } | TraitCallRes::External { .. }
                    ) {
                        return true;
                    }
                }
                CallKind::Fun(FunId::Regular { id: callee }) => {
                    if !generics_has_clause_or_parent(self.llbc, &reg.generics) {
                        continue;
                    }
                    if instantiation_memo_key(self.llbc, bindings, *callee, &reg.generics).is_none()
                    {
                        continue;
                    }
                    let node = self.callee_node_for_call(bindings, *callee, &reg.generics, depth);
                    if node != *callee {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
}

fn body_has_clause_trait_call(llbc: &Llbc, body: &Unstructured) -> bool {
    let forward = crate::front::mir::forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !call_graph_walks_block(bb, forward[bb_idx]) {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        match &reg.kind {
            CallKind::Trait(v) => {
                if payload_is_clause_or_parent(llbc, v) {
                    return true;
                }
            }
            CallKind::Fun(FunId::Regular { .. }) => {
                if generics_has_clause_or_parent(llbc, &reg.generics) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn generics_has_clause_or_parent(llbc: &Llbc, generics: &Value) -> bool {
    let generics = peel_generic_args(generics);
    let Some(refs) = generics.get("trait_refs").and_then(Value::as_array) else {
        return false;
    };
    refs.iter().any(|tref| ref_is_clause_or_parent(llbc, tref))
}

fn payload_is_clause_or_parent(llbc: &Llbc, payload: &Value) -> bool {
    let Some(tref) = payload.as_array().and_then(|a| a.first()) else {
        return false;
    };
    ref_is_clause_or_parent(llbc, tref)
}

fn ref_is_clause_or_parent(llbc: &Llbc, tref: &Value) -> bool {
    let Some((obj, _)) = unwrap_ref_binders(tref, llbc, 0) else {
        return false;
    };
    matches!(kind_tag(obj).as_deref(), Some("Clause" | "ParentClause"))
}

fn peel_generic_args(generics: &Value) -> &Value {
    let Some(obj) = generics.as_object() else {
        return generics;
    };
    if obj.contains_key("skip_binder")
        && (obj.contains_key("regions") || obj.contains_key("params"))
    {
        obj.get("skip_binder").unwrap_or(generics)
    } else {
        generics
    }
}

fn unwrap_ref_binders<'a>(v: &'a Value, llbc: &'a Llbc, depth: usize) -> Option<(&'a Value, u64)> {
    unwrap_ref_binders_at(v, llbc, depth, 0)
}

fn unwrap_ref_binders_at<'a>(
    v: &'a Value,
    llbc: &'a Llbc,
    depth: usize,
    binders: u64,
) -> Option<(&'a Value, u64)> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
        return unwrap_ref_binders_at(llbc.dedup_trait_body(id)?, llbc, depth + 1, binders);
    }
    if let Some(arr) = obj.get("Value").and_then(Value::as_array)
        && arr.len() == 2
    {
        return unwrap_ref_binders_at(&arr[1], llbc, depth + 1, binders);
    }
    if let Some(inner) = obj.get("skip_binder") {
        return unwrap_ref_binders_at(inner, llbc, depth + 1, binders + 1);
    }
    Some((v, binders))
}

fn kind_tag(obj: &Value) -> Option<String> {
    match obj.get("kind") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(m)) => m.keys().next().cloned(),
        _ => None,
    }
}

fn clause_var(clause: &Value) -> Option<(u64, usize)> {
    if let Some(arr) = clause.as_array() {
        return debruijn_var(arr.first()?);
    }
    debruijn_var(clause)
}

fn debruijn_var(var: &Value) -> Option<(u64, usize)> {
    if let Some(bound) = var.get("Bound").and_then(Value::as_array) {
        return Some((bound.first()?.as_u64()?, bound.get(1)?.as_u64()? as usize));
    }
    var.get("Free")
        .and_then(Value::as_u64)
        .map(|index| (0, index as usize))
}

fn trait_impl_id(v: &Value, llbc: &Llbc) -> Option<u64> {
    let (obj, _) = unwrap_ref_binders(v, llbc, 0)?;
    obj.get("kind")?.get("TraitImpl")?.get("id")?.as_u64()
}

fn implied_trait_refs(llbc: &Llbc, impl_id: u64) -> Option<&Vec<Value>> {
    let row = llbc
        .trait_impl_by_id(impl_id)
        .or_else(|| llbc.trait_impls_raw().get(impl_id as usize))?;
    row.get("implied_trait_refs")?.as_array()
}

fn first_trait_id(v: &Value) -> Option<u64> {
    if let Some(map) = v.as_object() {
        if let Some(r) = map.get("trait_decl_ref")
            && let Some(id) = r
                .get("skip_binder")
                .unwrap_or(r)
                .get("id")
                .and_then(Value::as_u64)
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

fn crate_of_path(name: &str) -> Option<&str> {
    let seg = name.split("::").next()?;
    if seg.is_empty() { None } else { Some(seg) }
}

fn crate_of_fun(fd: &FunDecl) -> Option<&str> {
    crate_of_path(fd.item_meta.name_path_str())
}

/// A fun decl with no Unstructured body whose crate is not a workspace crate.
/// An empty workspace list classifies nothing as external.
fn is_external_callee(llbc: &Llbc, fn_id: u64, workspace: &HashSet<String>) -> bool {
    if workspace.is_empty() {
        return false;
    }
    let Some(fd) = llbc.fn_by_id(fn_id) else {
        return false;
    };
    if fd.has_unstructured_body() {
        return false;
    }
    crate_of_fun(fd).is_some_and(|crate_name| !workspace.contains(crate_name))
}

/// `GraphAnalyzer.analyze_external_call` / `_callbacks`.
///
/// `None` is a blocker (`Clause` / `ParentClause` / `Dyn`, a dyn type, a
/// fn pointer, a closure type, or a by-value workspace ADT). `Some` is
/// `bottom_result` joined with `analyze_direct_call` over each callback
/// method graph.
fn analyze_external_call(
    llbc: &Llbc,
    fn_id: u64,
    call_generics: &Value,
    trait_refs: &[Value],
    workspace: &HashSet<String>,
) -> Option<Vec<(u64, Value)>> {
    if !is_external_callee(llbc, fn_id, workspace) {
        return None;
    }
    let types = generic_arg_types(call_generics);
    collect_external_effects(llbc, &types, trait_refs, workspace, 0).ok()
}

fn generic_arg_types(generics: &Value) -> Vec<Value> {
    peel_generic_args(generics)
        .get("types")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn collect_external_effects(
    llbc: &Llbc,
    types: &[Value],
    trait_refs: &[Value],
    workspace: &HashSet<String>,
    depth: usize,
) -> Result<Vec<(u64, Value)>, ()> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return Err(());
    }
    let mut callbacks = Vec::new();
    for ty in types {
        type_has_blocker(ty, llbc, workspace, false, depth)?;
    }
    for tref in trait_refs {
        walk_trait_ref_effects(tref, llbc, workspace, depth, Some(&mut callbacks))?;
    }
    Ok(callbacks)
}

fn walk_trait_ref_effects(
    tref: &Value,
    llbc: &Llbc,
    workspace: &HashSet<String>,
    depth: usize,
    mut callbacks: Option<&mut Vec<(u64, Value)>>,
) -> Result<(), ()> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return Err(());
    }
    let Some(bound) = bind_trait_ref(llbc, tref, &[], 0) else {
        return Err(());
    };
    let Some(tag) = kind_tag(&bound) else {
        return Err(());
    };
    match tag.as_str() {
        "Dyn" | "Clause" | "ParentClause" => Err(()),
        "BuiltinOrAuto" => Ok(()),
        "TraitImpl" => {
            let generics = bound
                .pointer("/kind/TraitImpl/generics")
                .cloned()
                .unwrap_or_else(|| Value::Object(Default::default()));
            let g = peel_generic_args(&generics);
            if let Some(types) = g.get("types").and_then(Value::as_array) {
                for ty in types {
                    type_has_blocker(ty, llbc, workspace, false, depth + 1)?;
                }
            }
            if let Some(refs) = g.get("trait_refs").and_then(Value::as_array) {
                for nested in refs {
                    walk_trait_ref_effects(
                        nested,
                        llbc,
                        workspace,
                        depth + 1,
                        callbacks.as_deref_mut(),
                    )?;
                }
            }
            let Some(impl_id) = trait_impl_id(&bound, llbc) else {
                return Ok(());
            };
            if let Some(out) = callbacks.as_deref_mut()
                && impl_in_workspace(llbc, impl_id, workspace)
            {
                if let Some(methods) = llbc.trait_impl_methods_of_def(impl_id) {
                    for method in methods.iter().flatten() {
                        if let Some(fn_id) = method.fn_id {
                            out.push((fn_id, generics.clone()));
                        }
                    }
                }
            }
            Ok(())
        }
        _ => Err(()),
    }
}

fn impl_in_workspace(llbc: &Llbc, impl_id: u64, workspace: &HashSet<String>) -> bool {
    let Some(methods) = llbc.trait_impl_methods_of_def(impl_id) else {
        return false;
    };
    methods.iter().flatten().any(|method| {
        method.fn_id.is_some_and(|fn_id| {
            llbc.fn_by_id(fn_id)
                .and_then(crate_of_fun)
                .is_some_and(|crate_name| workspace.contains(crate_name))
        })
    })
}

fn type_has_blocker(
    ty: &Value,
    llbc: &Llbc,
    workspace: &HashSet<String>,
    behind_pointer: bool,
    depth: usize,
) -> Result<(), ()> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return Err(());
    }
    let Some(node) = peel_ty(ty, llbc) else {
        return Err(());
    };
    if node.as_str().is_some() || node.get("Scalar").is_some() || node.get("TypeVar").is_some() {
        return Ok(());
    }
    if node.get("FnPtr").is_some()
        || node.get("DynTrait").is_some()
        || node.get("Dynamic").is_some()
    {
        return Err(());
    }
    if node.get("FnDef").is_some() {
        return Ok(());
    }
    if let Some(arr) = node.get("Ref").and_then(Value::as_array) {
        return match arr.get(1) {
            Some(inner) => type_has_blocker(inner, llbc, workspace, true, depth + 1),
            None => Ok(()),
        };
    }
    if let Some(arr) = node.get("RawPtr").and_then(Value::as_array) {
        return match arr.first() {
            Some(inner) => type_has_blocker(inner, llbc, workspace, true, depth + 1),
            None => Ok(()),
        };
    }
    if let Some(adt) = node.get("Adt") {
        return adt_has_blocker(adt, llbc, workspace, behind_pointer, depth);
    }
    if let Some(parts) = node.get("Tuple").and_then(Value::as_array) {
        for part in parts {
            type_has_blocker(part, llbc, workspace, behind_pointer, depth + 1)?;
        }
        return Ok(());
    }
    if let Some(element) = node.get("Array").or_else(|| node.get("Slice")) {
        if let Some(parts) = element.as_array() {
            if let Some(inner) = parts.first() {
                type_has_blocker(inner, llbc, workspace, behind_pointer, depth + 1)?;
            }
        } else {
            type_has_blocker(element, llbc, workspace, behind_pointer, depth + 1)?;
        }
        return Ok(());
    }
    Ok(())
}

fn adt_has_blocker(
    adt: &Value,
    llbc: &Llbc,
    workspace: &HashSet<String>,
    behind_pointer: bool,
    depth: usize,
) -> Result<(), ()> {
    if let Some(id) = adt_def_id(adt)
        && let Some(td) = llbc.type_by_id(id)
    {
        if matches!(td.src, TypeDeclSrc::Closure(_)) {
            return Err(());
        }
        let path = td.item_meta.name_path_str();
        let leaf = path.rsplit("::").next().unwrap_or(path);
        if is_closure_leaf(leaf) {
            return Err(());
        }
        if !behind_pointer && crate_of_path(path).is_some_and(|c| workspace.contains(c)) {
            return Err(());
        }
    }
    let generics = adt.get("generics").unwrap_or(adt);
    if let Some(types) = generics.get("types").and_then(Value::as_array) {
        for ty in types {
            type_has_blocker(ty, llbc, workspace, behind_pointer, depth + 1)?;
        }
    }
    if let Some(refs) = generics.get("trait_refs").and_then(Value::as_array) {
        for tref in refs {
            walk_trait_ref_effects(tref, llbc, workspace, depth + 1, None)?;
        }
    }
    Ok(())
}

fn adt_def_id(adt: &Value) -> Option<u64> {
    adt.get("id")
        .and_then(Value::as_u64)
        .or_else(|| adt.as_array()?.first()?.as_u64())
}

fn peel_ty<'a>(ty: &'a Value, llbc: &'a Llbc) -> Option<&'a Value> {
    peel_ty_at(ty, llbc, 0)
}

fn peel_ty_at<'a>(ty: &'a Value, llbc: &'a Llbc, depth: usize) -> Option<&'a Value> {
    if depth > MAX_INSTANTIATION_DEPTH {
        return None;
    }
    if let Some(id) = ty.get("Deduplicated").and_then(Value::as_u64) {
        return peel_ty_at(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(pair) = ty.get("Value").and_then(Value::as_array) {
        return peel_ty_at(pair.get(1)?, llbc, depth + 1);
    }
    if let Some(inner) = ty.get("skip_binder") {
        return peel_ty_at(inner, llbc, depth + 1);
    }
    Some(ty)
}
