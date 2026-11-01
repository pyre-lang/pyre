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

use std::collections::{HashMap, HashSet};

use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind, Unstructured};
use majit_charon_reader::{FunDecl, Llbc};
use serde_json::Value;

use super::framework::{
    CallGraph, OpaqueCensus, call_graph_flow_is_incomplete, call_graph_walks_block,
};

/// Nesting bound on Regular → spec → Regular instantiation.
/// A clause the caller cannot resolve, a `dyn`, a bodyless impl method,
/// or a walk past this bound falls back to the generic node.
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
    /// Still a clause, a `Dyn`, a bodyless impl method, or a ref this
    /// caller cannot resolve.
    Unresolved,
}

/// Shared helper: the call-graph node a Regular call site names.
///
/// `caller_bindings` are the caller's resolved clause refs (empty when
/// walking a FunDecl as itself). `callee` is the `FunId::Regular` id and
/// `generics` is the call site's generic arguments (`trait_refs`).
///
/// During [`build_call_graph`] this creates a specialized node when the
/// callee body has a `Clause` / `ParentClause` Trait call that resolves
/// to a `TraitImpl` method. After the graph is built, it is lookup-only
/// on `spec_memo` and falls back to `callee` when the instantiation was
/// not recorded — never narrower than the generic node.
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
pub(crate) fn resolve_trait_call(llbc: &Llbc, payload: &Value, bindings: &[Value]) -> TraitCallRes {
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
                .is_some_and(crate::front::clause_spec::decl_has_unstructured_body)
            {
                TraitCallRes::Method { fn_id, generics }
            } else {
                TraitCallRes::Unresolved
            }
        }
        None => TraitCallRes::Unresolved,
    }
}

/// `impl_method_fn_id` indexes `trait_impls_raw` by def id. A hole or a
/// shifted row misses; [`Llbc::trait_impl_by_id`] finds the row by id.
fn impl_method_from_row(llbc: &Llbc, impl_id: u64, method_idx: u64) -> Option<u64> {
    llbc.trait_impl_by_id(impl_id)?
        .get("methods")?
        .as_array()?
        .iter()
        .find(|method| {
            method
                .pointer("/kind/TraitMethod/1")
                .and_then(Value::as_u64)
                == Some(method_idx)
        })?
        .pointer("/skip_binder/id")?
        .as_u64()
}

/// Build the resolved ULLBC call graph, specializing Regular calls whose
/// callee body has a clause-bound Trait call that the call site binds to
/// a `TraitImpl` method.
pub(crate) fn build_call_graph(llbc: &Llbc) -> CallGraph {
    let mut builder = Builder::new(llbc);
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
}

impl<'a> Builder<'a> {
    fn new(llbc: &'a Llbc) -> Self {
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
        if !has_resolvable_trait_call(self.llbc, &body, &bindings) {
            self.graph.spec_memo.insert(key, callee);
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
                        let node =
                            self.callee_node_for_call(bindings, *callee, &reg.generics, depth);
                        self.graph.callees.entry(id).or_default().insert(node);
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
                            match resolve_trait_call(self.llbc, v, bindings) {
                                TraitCallRes::Method { fn_id, generics } => {
                                    let node =
                                        self.callee_node_for_call(&[], fn_id, &generics, depth);
                                    self.graph.callees.entry(id).or_default().insert(node);
                                }
                                TraitCallRes::Empty => {}
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

    fn finish(mut self) -> CallGraph {
        for (&caller, cs) in &self.graph.callees {
            for &c in cs {
                self.graph.callers.entry(c).or_default().insert(caller);
            }
        }
        let (py, _) = self
            .graph
            .seeds_for(super::framework::PYTHON_DISPATCH_SEEDS);
        let (col, _) = self.graph.seeds_for(super::framework::COLLECTING_SEEDS);
        let mut seeds = py;
        seeds.extend(col);
        let collecting = self.graph.reaching(&seeds);
        self.graph.demote_specializations_in(&collecting);
        self.graph
    }
}

fn has_resolvable_trait_call(llbc: &Llbc, body: &Unstructured, bindings: &[Value]) -> bool {
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
        let CallKind::Trait(v) = &reg.kind else {
            continue;
        };
        if !payload_is_clause_or_parent(llbc, v) {
            continue;
        }
        if matches!(
            resolve_trait_call(llbc, v, bindings),
            TraitCallRes::Method { .. }
        ) {
            return true;
        }
    }
    false
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
        let CallKind::Trait(v) = &reg.kind else {
            continue;
        };
        if payload_is_clause_or_parent(llbc, v) {
            return true;
        }
    }
    false
}

fn payload_is_clause_or_parent(llbc: &Llbc, payload: &Value) -> bool {
    let Some(tref) = payload.as_array().and_then(|a| a.first()) else {
        return false;
    };
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
