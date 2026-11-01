//! `<[T]>::get(slice, i)` → bounds-checked `Option<&T>` diamond.
//!
//! ## Positioning
//!
//! `core::slice::<Impl>::get` and `get_mut` are foreign leaves whose bodies
//! are Opaque in the LLBC (Charon cannot extract `core`), so the caller emits
//! a residual `get` / `get_mut` call — an unregistered callee the rtyper
//! census Skips, which Skips the CALLING graph with it.  Its `Self` is the
//! primitive slice `[T]` (not an ADT), so `lower_call` keeps the raw
//! `FunctionPath` segments `["core","slice","<Impl>","get"]` (or `get_mut`),
//! receiver in `args[0]` and index in `args[1]`.  Both return `Some(&slice[i])`
//! / `Some(&mut slice[i])` iff `i` is in bounds, so this pass *synthesizes*
//! the guard `i < len(slice)`:
//!
//! ```text
//!     opt = get(slice, i)                // residual `get` call
//! becomes
//!     if i < len(slice) { opt = Some(&slice[i]) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the element read must not run when `i` is out of
//! range (`slice[i]` would read past the block), so a single-block always-read
//! encoding is unsound.  This is exactly why an *inline* fold (the
//! `emit_tagged_pair_aggregate` path, which writes `__pos_0` unconditionally in
//! the call's own block before the consumer's discriminant switch) cannot
//! express `get`; a post-pass that splits block A into a guarded Some/None
//! diamond can.  It is the same "diamond is unavoidable" shape as
//! [`crate::front::slice_first`], of which `get` is the general case: `first`
//! is `get` at a fixed index 0, where `i < len` collapses to `len > 0`.
//!
//! Only the upper bound is tested.  The index is a `usize` — the scalar
//! `SliceIndex` instantiation is the only one this pass fires for (see below)
//! — so it is non-negative by type and needs no lower-bound compare.
//!
//! ## The `SliceIndex` instantiation gate
//!
//! `<[T]>::get` is generic over `SliceIndex`, and only its scalar
//! instantiation has the shape this diamond encodes.  `get(0..2)` returns
//! `Option<&[T]>` — a sub-slice, not an element — and an `ArrayRead` at the
//! range's start would hand the consumer a `T` where a `[T]` is expected.
//! `front::mir` therefore pins the instantiation through
//! `is_slice_get_scalar_call`: a local index uses the operand's declared
//! `usize`, while a literal uses the method's `I` generic.  Every range form
//! falls through to the range lowering instead of being mistaken for an item
//! read.
//!
//! ## Payload representation
//!
//! `get` returns `Option<&T>`, but the `Some` payload is materialised by
//! `OpKind::ArrayRead`, which yields the element VALUE, not a pointer-to-slot.
//! In the list model a `&T` and a `T` use the same item register — a GC pointer
//! for object items and the scalar bank for `u8`/`i64` items — and no consumer
//! derefs a slot-pointer.  The site's payload type therefore comes from the
//! destination `Option` consumer shape rather than being hard-coded to a
//! reference.  The receiver is a `SomeList`-modelled slice (its
//! `Input` op carries the list-container `class_root`), so `__len` and the
//! element read repr-dispatch to `arraylen_gc` / `getarrayitem` on the
//! underlying length-prefixed GcArray.  A primitive `&T` payload needs no
//! class narrowing: the rewrite's bare-call arm carries the element's scalar
//! [`ValueType`] directly.
//!
//! ## The rewrite (`rewire_one_slice_get_site`)
//!
//! Block A holds the residual `get` call producing `opt`.  Because
//! `Option<&PyObjectRef>` triggers `option_residual_narrow_root`, `lower_call`
//! appends a trailing `__cast_instance_intrinsic` after the call, so the call is NOT
//! A's last op — the block-A skeleton absorbs that optional cast, exactly as
//! [`crate::front::option_map_or`] does.  The rewrite:
//! 1. drops the `get` call (+ absorbed cast) and closes A with a
//!    `bool(i < len(slice))` branch to two fresh arms;
//! 2. the `then_bb` arm reads `slice[i]` and wraps it in `Some`; a one-word
//!    reference niche forwards the typed payload directly, while value
//!    payloads build `__discriminant = 1` / `__pos_0 = elem`;
//! 3. the `else_bb` arm emits a typed null for that niche, or an aggregate
//!    `None` (`__discriminant = 0`) for value payloads;
//! 4. both arms re-apply the absorbed narrowing and forward to B, reproducing
//!    A's original exit args with the `opt` slot sourced from the arm's
//!    `Some`/`None` value and every other live value threaded through.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `get` callee keeps
//! the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::option_map_or::emit_narrow;
use crate::front::option_variant::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// `true` iff `path` is the FunDecl `name_path` of `<[T]>::get` or `<[T]>::get_mut`.
///
/// Two crate prefixes reach the residual, the same pair `to_vec` already
/// distinguishes (`alloc::slice::<Impl>::to_vec` vs the `core` inherent impl):
/// `core::slice::<Impl>::get{,_mut}` and `alloc::slice::<Impl>::get{,_mut}`.
/// `core::slice::index::<Impl>::get` is a different function: its `self` is
/// the index, so args[0]/args[1] are swapped and this diamond would read
/// `len(index)`.
pub(crate) fn is_slice_get_name_path(path: &str) -> bool {
    slice_get_leaf_from_path(path).is_some()
}

/// `true` iff `path` is `<[T]>::get_mut` (not `get`).
pub(crate) fn is_slice_get_mut_name_path(path: &str) -> bool {
    slice_get_leaf_from_path(path) == Some(SliceGetLeaf::GetMut)
}

/// `true` iff `segments` name `<[T]>::get` or `<[T]>::get_mut`.
///
/// Anchored on `slice / <Impl> / get{,_mut}` so a crate-stripped spelling
/// (`slice::<Impl>::get`) and the FunDecl spelling (`core::slice::<Impl>::get`)
/// both match, while `slice::index::<Impl>::get` (last three: `index`,
/// `<Impl>`, `get`) does not.
pub(crate) fn is_slice_get_segments(segments: &[String]) -> bool {
    slice_get_leaf_from_segments(segments).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SliceGetLeaf {
    Get,
    GetMut,
}

fn slice_get_leaf_from_path(path: &str) -> Option<SliceGetLeaf> {
    slice_get_leaf_from_parts(&path.split("::").collect::<Vec<_>>())
}

fn slice_get_leaf_from_segments(segments: &[String]) -> Option<SliceGetLeaf> {
    slice_get_leaf_from_parts(&segments.iter().map(String::as_str).collect::<Vec<_>>())
}

fn slice_get_leaf_from_parts(parts: &[&str]) -> Option<SliceGetLeaf> {
    let n = parts.len();
    if n < 3 || parts[n - 3] != "slice" || parts[n - 2] != "<Impl>" {
        return None;
    }
    // `slice::index::<Impl>::get` has `index` immediately before `<Impl>`.
    if n >= 4 && parts[n - 4] == "index" {
        return None;
    }
    match parts[n - 1] {
        "get" => Some(SliceGetLeaf::Get),
        "get_mut" => Some(SliceGetLeaf::GetMut),
        _ => None,
    }
}

/// `get` may lower a scalar or a thin-pointer element. `get_mut` returns
/// `&mut T` of the element slot. `ArrayRead` copies the element, so a later
/// store through that reference misses the slot whether `T` is a scalar or
/// a pointer. `get_mut` stays residual.
///
/// `array_type_id` is `Some("[u8]")` for a scalar spelling and `None` only
/// for a proven thin-pointer element (`SliceGetSite::array_type_id`).
/// An `AtomicU64` or `Option<Entry>` never produces a proof, so it never
/// reaches this gate.
pub(crate) fn slice_get_element_may_record(
    is_get_mut: bool,
    array_type_id: Option<&String>,
) -> bool {
    if is_get_mut {
        let _ = array_type_id;
        false
    } else {
        true
    }
}

/// `true` iff `target` is `Result::ok` — the Method-hint form the capture
/// used to require, or the FunDecl `FunctionPath` the rewriter already
/// accepts (`from_size_align::is_result_ok_target`).
pub(crate) fn is_result_ok_call_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            name == "ok"
                && receiver_root
                    .as_deref()
                    .map(majit_ir::descr::class_root_leaf)
                    == Some("Result")
        }
        CallTarget::FunctionPath { segments, .. } => {
            let n = segments.len();
            n >= 3
                && segments[n - 1] == "ok"
                && matches!(segments[n - 2].as_str(), "Result" | "<Impl>")
                && segments[n - 3] == "result"
        }
        _ => false,
    }
}

/// `Result::expect` — Method hint or FunDecl `result::<Impl>::expect`.
pub(crate) fn is_result_expect_call_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            name == "expect"
                && receiver_root
                    .as_deref()
                    .map(majit_ir::descr::class_root_leaf)
                    == Some("Result")
        }
        CallTarget::FunctionPath { segments, .. } => {
            let n = segments.len();
            n >= 3
                && segments[n - 1] == "expect"
                && matches!(segments[n - 2].as_str(), "Result" | "<Impl>")
                && segments[n - 3] == "result"
        }
        _ => false,
    }
}

/// A recognized `<[T]>::get(slice, i)` call site captured during body lowering
/// (`front::mir` `recognize_slice_get_site`).  The owner strings are resolved
/// at the recording site where the destination `Option<&T>` type is in hand;
/// the post-pass only needs them to spell the `Some`/`None` aggregates.
#[derive(Clone)]
pub(crate) struct SliceGetSite {
    /// The `get` call result (the `Option<&T>` value) — locates block A.  The
    /// slice and index operands are read from that located call's
    /// `args[0]`/`args[1]`.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` (per-instantiation, suffixed) — the
    /// ctor owner for the `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The slice element `T` projected to a [`ValueType`] — the devirtualized
    /// `ArrayRead` and `Some::__pos_0` field kind. Rust spells the source
    /// result `Option<&T>`, but RPython's getarrayitem yields the value `T`.
    pub payload_ty: ValueType,
    /// Concrete ARRAY identity carrying the scalar/RPython item spelling to
    /// the descr, or `None` only for a proven thin-pointer element whose
    /// identity-less descr already has the correct one-word stride.
    pub array_type_id: Option<String>,
    /// Object gcarray identity when the receiver is an object-pointer slice
    /// (`*mut PyObject`). `None` for `[u8]`, `[i64]`, `[f64]`, `[str]` and
    /// every other element. Distinct from [`Self::array_type_id`]: that one
    /// stays the element read's identity.
    pub object_array_type_id: Option<String>,
    /// True when `Option<&T>` is Rust's one-word nullable-reference niche.
    /// RPython carries this as the payload annotation with `can_be_None=True`,
    /// not as an Option object with discriminant and payload fields.
    pub niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Concrete payload class for a niche reference.  In particular,
    /// `<[*mut PyObject]>::get` returns `Option<&PyObjectRef>`: the successful
    /// list item and the null arm must both retain the `PyObject` ClassDef.
    pub payload_narrow_root: Option<String>,
}

/// Rewrite every recorded `<[T]>::get` call site into the bounds-checked
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never regresses
/// a graph the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_slice_get_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceGetSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_get_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `get` call; the unregistered callee keeps
                // the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_slice_get_site(graph: &mut FunctionGraph, site: &SliceGetSite) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `get` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: slice::get result var has no producer block"))?;

    // Locate the `get` call op by its result (not assuming it is the block
    // tail — the trailing `__cast_instance_intrinsic` may follow, see below).
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: slice::get call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();

    // `Option<&PyObjectRef>` gains a trailing `__cast_instance_intrinsic` narrowing
    // op (`result_narrow_root`) whose output is what the block forwards on.
    // Absorb that optional cast: `flow_result` is the value B consumes,
    // `narrow_root` re-applies the narrowing per arm, `remove_upto` bounds the
    // ops to drop.  Any other trailing shape declines (fail-safe).
    let (flow_result, narrow_root, remove_upto) = if ci + 1 == ops_len {
        (site.result_var.clone(), None, ci)
    } else if ci + 2 == ops_len {
        let cast = &graph.blocks[a].operations[ci + 1];
        match cast.result.as_ref() {
            Some(narrowed)
                if let Some(root) =
                    crate::model::cast_instance_of(&cast.kind, &site.result_var) =>
            {
                (narrowed.clone(), Some(root.to_string()), ci + 1)
            }
            _ => {
                return Err(format!(
                    "{name}: slice::get call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: slice::get call is not the last op of block {a}"
        ));
    };

    // Capture the slice receiver and the index operand.
    let (slice, index) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (
            args[0].clone().into_variable(),
            args[1].clone().into_variable(),
        ),
        other => {
            return Err(format!(
                "{name}: slice::get producer op is not a 2-arg call: {other:?}"
            ));
        }
    };
    // `get_mut` yields `&mut` of the element slot. `ArrayRead` copies the
    // element, so a store through the reference does not update the slice.
    let is_get_mut = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { target, .. } => match target {
            CallTarget::FunctionPath { segments, .. } => {
                segments.last().is_some_and(|leaf| leaf == "get_mut")
            }
            CallTarget::Method { name, .. } => name == "get_mut",
            _ => false,
        },
        _ => false,
    };
    if is_get_mut {
        return Err(format!(
            "{name}: get_mut addresses the element slot; ArrayRead would copy it"
        ));
    }

    // A's single exit → B (the continuation consuming the Option).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: slice::get call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: slice::get call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    // `carried` = the distinct live Values A forwards to B other than the
    // Option itself (`flow_result`); each must be threaded through the diamond
    // arms to reach B (a fresh block cannot see A-scope Variables directly).
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural validation passed; mutate the graph.

    // `then_bb` (`Some`) carries `carried` plus `slice` and `index` (the base
    // and subscript of the element read); `else_bb` (`None`) carries only
    // `carried`.  The source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    for v in [&slice, &index] {
        if !then_sources.contains(v) {
            then_sources.push(v.clone());
        }
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: elem = slice[i]; opt = Some(elem).
    let slice_in_then = map_source(&then_sources, &then_inputs, &slice)
        .ok_or_else(|| format!("{name}: slice not threaded into Some arm"))?;
    let index_in_then = map_source(&then_sources, &then_inputs, &index)
        .ok_or_else(|| format!("{name}: index not threaded into Some arm"))?;
    let elem = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(elem.clone()),
        kind: OpKind::ArrayRead {
            base: slice_in_then,
            // The element read and the `Some::__pos_0` field write below
            // consume the same `elem`, so the read's declared element type
            // must be the value type the devirtualized `T` carries
            // (`site.payload_ty`, `Ref(None)` for a GC-pointer element) — a
            // hardcoded `Ref(None)` would disagree for scalar elements.
            item_ty: site.payload_ty.clone(),
            index: index_in_then,
            array_type_id: site.array_type_id.clone(),
            nolength: crate::front::typestr::nolength_from_array_type_id(
                site.array_type_id.as_deref(),
            ),
            gckind: crate::front::typestr::gckind_from_array_type_id(site.array_type_id.as_deref()),
            pure: false,
        },
    });
    let then_result = if site.niche {
        emit_narrow(graph, then_bb, elem, &site.payload_narrow_root)
    } else {
        let some_var = emit_option_variant(
            graph,
            then_bb,
            &site.option_owner,
            1,
            Some((&site.some_owner, elem, site.payload_ty.clone())),
        );
        emit_narrow(graph, then_bb, some_var, &narrow_root)
    };
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_result,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    // `else_bb`: opt = None.
    let else_result = if site.niche {
        let null = graph.push_niche_null(else_bb, site.niche_null_cast.as_ref());
        emit_narrow(graph, else_bb, null, &site.payload_narrow_root)
    } else {
        let none_var = emit_option_variant(graph, else_bb, &site.option_owner, 0, None);
        emit_narrow(graph, else_bb, none_var, &narrow_root)
    };
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_result,
        &carried,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    // A: drop the residual `get` call (+ absorbed cast), synthesize the guard
    // `i < len(slice)`, branch on it.  `__len` on the `SomeList`-modelled slice
    // routes through the rtyper `len` op → `AbstractBaseListRepr.rtype_len` →
    // `arraylen_gc`.  The compare's result is a Bool, so it declares `Int` (as
    // `slice_first` does) and never the `Unsigned` that
    // would wrap the flag in `r_uint`; the annotator derives the `lt` llop from
    // the OPERAND annotations, where the `usize` index and the non-negative
    // `arraylen_gc` both admit the unsigned compare.  `set_branch` appends the
    // idempotent `bool(cond)` hop and installs the Bool(false)/Bool(true) arm
    // links, so the true (in-bounds) arm is `then_bb`.
    let a_id = graph.blocks[a].id;
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let len = graph.alloc_value_var();
    let len_kind = crate::front::option_variant::slice_len_op(
        slice,
        site.object_array_type_id.as_deref(),
        // A byte-view receiver never reaches here: the site capture in
        // `front::mir` declines one, because this rewriter's element read is
        // always an `ArrayRead` while `first`/`last` also serve `as_bytes()`.
        false,
    );
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(len.clone()),
        kind: len_kind,
    });
    let cond = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(cond.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: index.clone(),
            rhs: len,
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_get_site(result_var: Variable) -> SliceGetSite {
        SliceGetSite {
            result_var,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Ref(None),
            array_type_id: None,
            object_array_type_id: None,
            niche: false,
            niche_null_cast: None,
            payload_narrow_root: None,
        }
    }

    fn scalar_slice_get_site(result_var: Variable) -> SliceGetSite {
        SliceGetSite {
            payload_ty: ValueType::Int,
            ..slice_get_site(result_var)
        }
    }

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, args: Vec<Variable>) -> Variable {
        emit_named_call(g, a, args, "get")
    }

    fn emit_named_call(
        g: &mut FunctionGraph,
        a: crate::model::BlockId,
        args: Vec<Variable>,
        leaf: &str,
    ) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "slice".into(), "<Impl>".into(), leaf.into()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(args),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap()
    }

    fn residual_get_survives(g: &FunctionGraph, a: crate::model::BlockId) -> bool {
        g.blocks[a.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if matches!(
                        segments.last().map(String::as_str),
                        Some("get" | "get_mut")
                    )
            )
        })
    }

    /// Build the minimal `opt = get(slice, i)` shape — block A = the residual
    /// call closed by a single goto to B (which consumes the Option) — and
    /// assert the rewrite drops the call, synthesizes `i < len(slice)`, and
    /// branches to a `Some` arm (`slice[i]` → `Some`) and a `None` arm, both
    /// merging to B.
    #[test]
    fn rewrite_lifts_get_to_bounds_checked_option() {
        let mut g = FunctionGraph::new("test_slice_get");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice.clone(), index.clone()]);

        // B: the continuation consuming the get() result.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[scalar_slice_get_site(opt)]);
        assert_eq!(rewritten, 1, "the slice::get site must be rewritten");

        // The residual `get` call is gone from block A.
        assert!(
            !residual_get_survives(&g, a),
            "residual get call removed from A"
        );
        // A synthesizes the length guard and an `lt` compare, then branches.
        // No object-gcarray identity: the receiver is not an object-pointer
        // slice, so the guard stays the `__len` marker.
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().map(String::as_str) == Some("__len")
            )),
            "A keeps the __len marker"
        );
        assert!(
            !g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. })),
            "a non-object receiver does not emit arraylen_gc"
        );
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt")),
            "A compares i < len"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        // Exactly one arm reads an array element (the Some payload), and it
        // subscripts with the arm's threaded index — NOT a synthesized 0, which
        // is what would make this pass a mis-indexed `slice_first`.
        let elem_reads: Vec<&Variable> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ArrayRead { index, .. } => Some(index),
                _ => None,
            })
            .collect();
        assert_eq!(elem_reads.len(), 1, "the Some arm reads slice[i]");
        assert!(
            g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::ArrayRead {
                        item_ty: ValueType::Int,
                        ..
                    }
                )
            }),
            "a primitive-reference payload is read in the scalar bank"
        );
        let then_inputs = &g.block(g.blocks[a.0].exits[1].target).inputargs;
        assert!(
            then_inputs.contains(elem_reads[0]),
            "the element read subscripts the arm's threaded index"
        );
        // Both arms write an Option `__discriminant` (Some=1, None=0).
        let disc_writes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "__discriminant")
            })
            .count();
        assert_eq!(disc_writes, 2, "both arms write a discriminant");
    }

    /// `slice: &[*mut PyObject]` makes `get` return
    /// `Option<&PyObjectRef>`, a nullable shared reference.  Its RPython shape
    /// is one maybe-null `SomeInstance(PyObject)`, so neither arm may mint the
    /// Rust Option aggregate used by scalar/value payloads.
    #[test]
    fn rewrite_niche_get_forwards_typed_payload_or_null() {
        let mut g = FunctionGraph::new("test_slice_get_niche");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        let narrowed = emit_narrow(&mut g, a, opt.clone(), &Some("PyObject".into()));
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed]);

        let mut site = slice_get_site(opt);
        site.niche = true;
        site.payload_narrow_root = Some("PyObject".into());
        assert_eq!(rewire_slice_get_call_sites(&mut g, &[site]), 1);

        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldWrite { field, .. }
                            if field.name == "__discriminant" || field.name == "__pos_0"
                    )
                }),
            "a niche Option has no aggregate fields"
        );
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments == &["core", "ptr", "null_mut"]
                    )
                })
        );
        let payload_narrows = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if crate::model::cast_instance_root(&op.kind) == Some("PyObject")
                )
            })
            .count();
        assert_eq!(payload_narrows, 2, "payload and null keep one ClassDef");
    }

    /// A scalar slice item uses the integer register bank throughout the
    /// guarded diamond.  This is the ordinary RPython `getitem` result shape,
    /// not a pointer-to-slot or an object-pointer-only special case.
    #[test]
    fn rewrite_preserves_unsigned_item_payload() {
        let mut g = FunctionGraph::new("test_slice_get_u8");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(2), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let mut site = slice_get_site(opt);
        site.payload_ty = ValueType::Unsigned;
        assert_eq!(rewire_slice_get_call_sites(&mut g, &[site]), 1);
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::ArrayRead {
                        item_ty: ValueType::Unsigned,
                        ..
                    }
                ))
        );
        assert!(
            g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::FieldWrite {
                        field,
                        ty: ValueType::Unsigned,
                        ..
                    } if field.name == "__pos_0"
                ))
        );
    }

    /// The production shape: an `Option<&RegisteredStruct>` result appends a
    /// trailing `__cast_instance_intrinsic` narrowing op, so the call is NOT the
    /// block tail.  The rewrite absorbs the cast, fires, and re-applies the
    /// narrowing in both arms.
    #[test]
    fn rewrite_absorbs_trailing_narrow_cast() {
        let mut g = FunctionGraph::new("test_slice_get_narrow");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("PyObject", opt.clone()),
                true,
            )
            .unwrap();

        // B consumes the NARROWED value (what the block actually forwards).
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![narrowed]);

        // Site records the CALL result; the rewrite discovers the trailing cast.
        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 1, "the slice::get site must be rewritten");

        // The residual call and the trailing cast are gone from block A.
        assert!(
            !residual_get_survives(&g, a),
            "residual get call removed from A"
        );
        // Two arms each re-emit a `__cast_instance_intrinsic` narrowing.
        let narrow_casts = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.first().map(String::as_str) == Some(crate::runtime_names::shims::CAST_INSTANCE)
                )
            })
            .count();
        assert_eq!(narrow_casts, 2, "each diamond arm re-applies the narrowing");
    }

    /// A call block whose trailing shape is neither the bare call nor a single
    /// absorbed cast declines (fail-safe): the residual call survives untouched.
    #[test]
    fn rewrite_declines_on_unexpected_trailing_shape() {
        let mut g = FunctionGraph::new("test_slice_get_decline");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice, index]);
        // Two trailing ops break both the "call is last" and "single cast" shapes.
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 0, "an unexpected trailing shape declines");
        assert!(
            residual_get_survives(&g, a),
            "residual call survives on decline"
        );
    }

    /// The arity guard: `get` is recorded with two operands, so a located
    /// producer that is a 1-arg call (the `first` shape) cannot supply an
    /// index.  It declines rather than reading `slice[?]`.
    #[test]
    fn rewrite_declines_on_one_arg_producer() {
        let mut g = FunctionGraph::new("test_slice_get_arity");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = emit_call(&mut g, a, vec![slice]);

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_get_call_sites(&mut g, &[slice_get_site(opt)]);
        assert_eq!(rewritten, 0, "a 1-arg producer declines");
        assert!(
            residual_get_survives(&g, a),
            "residual call survives on decline"
        );
    }

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    /// FunDecl / FunctionPath spellings the capture must accept, and the
    /// `SliceIndex::get` form it must not (args[0] would be the index).
    #[test]
    fn slice_get_name_accepts_get_and_get_mut_and_rejects_index_trait() {
        assert!(is_slice_get_name_path("core::slice::<Impl>::get"));
        assert!(is_slice_get_name_path("core::slice::<Impl>::get_mut"));
        assert!(is_slice_get_name_path("alloc::slice::<Impl>::get"));
        assert!(is_slice_get_name_path("alloc::slice::<Impl>::get_mut"));
        assert!(is_slice_get_mut_name_path("core::slice::<Impl>::get_mut"));
        assert!(!is_slice_get_mut_name_path("core::slice::<Impl>::get"));
        assert!(!is_slice_get_name_path("core::slice::index::<Impl>::get"));
        assert!(!is_slice_get_name_path(
            "core::slice::index::<Impl>::get_mut"
        ));
        assert!(!is_slice_get_name_path(
            "core::slice::index::SliceIndex::get"
        ));
        assert!(!is_slice_get_name_path("core::slice::<Impl>::first"));
        assert!(is_slice_get_segments(&segs(&[
            "core", "slice", "<Impl>", "get"
        ])));
        assert!(is_slice_get_segments(&segs(&[
            "core", "slice", "<Impl>", "get_mut"
        ])));
        assert!(is_slice_get_segments(&segs(&["slice", "<Impl>", "get"])));
        assert!(!is_slice_get_segments(&segs(&[
            "core", "slice", "index", "<Impl>", "get"
        ])));
    }

    /// `get_mut` of a pointer element still returns `&mut` of the slot.
    /// Copying the pointer with `ArrayRead` drops a store of a new pointer.
    #[test]
    fn get_mut_of_thin_pointer_does_not_copy_the_slot() {
        let mut g = FunctionGraph::new("test_slice_get_mut");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let opt = emit_named_call(&mut g, a, vec![slice, index], "get_mut");
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let mut site = slice_get_site(opt);
        site.niche = true;
        assert_eq!(rewire_slice_get_call_sites(&mut g, &[site]), 0);
        assert!(
            residual_get_survives(&g, a),
            "residual get_mut call stays: the result is a slot reference"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. })),
            "get_mut must not copy the pointer element"
        );
    }

    /// `get_mut` of a non-pointer copies the element. The address of the slot
    /// is an interior pointer; a copy then a store misses the write, so the
    /// rewrite declines instead of emitting `ArrayRead`.
    #[test]
    fn get_mut_of_non_pointer_does_not_copy_the_element() {
        let mut g = FunctionGraph::new("test_slice_get_mut_scalar");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let index = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let opt = emit_named_call(&mut g, a, vec![slice, index], "get_mut");
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let mut site = scalar_slice_get_site(opt);
        site.array_type_id = Some("[u8]".into());
        assert_eq!(
            rewire_slice_get_call_sites(&mut g, &[site]),
            0,
            "get_mut of a non-pointer must not lower to a copied ArrayRead"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. })),
            "no copied element read"
        );
    }

    /// `get_mut` of a scalar spelling is not recorded: `ArrayRead` would copy
    /// the integer and a write through `&mut T` would miss the slot.
    /// `portal_diag_bump`'s `[AtomicU64]::get` and `RDict` `Option<Entry>`
    /// never produce a proof at all (`slice_get_element` is scalar /
    /// thin-pointer only), so they stay residual even though the callee
    /// spelling is `core::slice::<Impl>::get`.
    #[test]
    fn get_mut_scalar_and_aggregate_elements_stay_unrecorded() {
        let u8_id = "[u8]".to_string();
        assert!(
            slice_get_element_may_record(false, Some(&u8_id)),
            "get of a scalar is the existing diamond"
        );
        assert!(
            slice_get_element_may_record(false, None),
            "get of a thin pointer is the existing diamond"
        );
        assert!(
            !slice_get_element_may_record(true, Some(&u8_id)),
            "get_mut of a scalar would copy the slot"
        );
        assert!(
            !slice_get_element_may_record(true, None),
            "get_mut of a pointer is still a slot reference"
        );
        // No element proof ⇒ not recorded.  That is how `[AtomicU64]::get`
        // (`portal_diag_bump`) and `Option<Entry>` stay residual: they are
        // ADTs, neither a scalar spelling nor a thin pointer.
        assert!(is_slice_get_name_path("core::slice::<Impl>::get"));
        assert!(is_slice_get_name_path("core::slice::<Impl>::get_mut"));
    }

    /// The rewriter already accepts `FunctionPath` `Result::ok`; the capture
    /// used to require `CallTarget::Method` and dropped the FunDecl spelling.
    #[test]
    fn result_ok_capture_accepts_method_and_functionpath() {
        let method = CallTarget::Method {
            name: "ok".into(),
            receiver_root: Some("Result".into()),
            resolved_path: None,
            fun_decl_id: None,
            branch_payloads: None,
        };
        let impl_path = CallTarget::FunctionPath {
            segments: segs(&["core", "result", "<Impl>", "ok"]),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        };
        let owner_path = CallTarget::FunctionPath {
            segments: segs(&["core", "result", "Result", "ok"]),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        };
        let option_ok = CallTarget::Method {
            name: "ok".into(),
            receiver_root: Some("Option".into()),
            resolved_path: None,
            fun_decl_id: None,
            branch_payloads: None,
        };
        let unrelated = CallTarget::FunctionPath {
            segments: segs(&["core", "option", "<Impl>", "ok"]),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        };
        assert!(is_result_ok_call_target(&method));
        assert!(is_result_ok_call_target(&impl_path));
        assert!(is_result_ok_call_target(&owner_path));
        assert!(!is_result_ok_call_target(&option_ok));
        assert!(!is_result_ok_call_target(&unrelated));
    }
}
