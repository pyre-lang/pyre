//! `<[T]>::first(slice)` and checked `get(start..)` → `Option` diamonds.
//!
//! ## Positioning
//!
//! `core::slice::<Impl>::first` is a foreign leaf whose body is Opaque in the
//! LLBC (Charon cannot extract `core`), so the caller emits a residual `first`
//! call — an unregistered callee the rtyper census Skips.  Its `Self` is the
//! primitive slice `[T]` (not an ADT), so `lower_call` keeps the raw
//! `FunctionPath` segments `["core","slice","<Impl>","first"]` (same shape as
//! `bool::then`), receiver in `args[0]`.  Unlike `bool::then`, the condition is
//! not an operand — `first` returns `Some(&slice[0])` iff `slice` is non-empty,
//! so this pass *synthesizes* the guard `len(slice) > 0`:
//!
//! ```text
//!     opt = first(slice)                 // residual `first` call
//! becomes
//!     if len(slice) > 0 { opt = Some(&slice[0]) } else { opt = None }
//! ```
//!
//! The branch is mandatory — the element read must not run when `slice` is
//! empty (`slice[0]` would be an out-of-bounds read), so a single-block
//! always-read encoding is unsound.  This is exactly why an *inline* fold (the
//! `emit_tagged_pair_aggregate` path, which writes `__pos_0` unconditionally in
//! the call's own block before the consumer's discriminant switch) cannot
//! express `first`; a post-pass that splits block A into a guarded Some/None
//! diamond can.
//!
//! The same control-flow skeleton handles the RangeFrom `get(start..)` form
//! retained by the #346 slice-range port: its successful arm emits the
//! ordinary RPython `getslice(start, None)` marker and its guard is
//! `start <= len(slice)`.  Scalar `get(i)` belongs to the dedicated
//! [`crate::front::slice_get`] pass from origin/main.
//!
//! ## Payload representation
//!
//! `first` returns `Option<&T>`, but the `Some` payload is materialised by
//! `OpKind::ArrayRead` (element 0), which yields the element VALUE, not a
//! pointer-to-slot.  RPython's list model likewise returns the item value from
//! `getitem`; no surviving consumer dereferences a Rust slot pointer.  This is
//! true for both GC-reference items and scalar items (`u8`, `i64`, …), and the
//! site's `payload_ty` keeps their register bank exact.  The receiver is a
//! `SomeList`-modelled slice, so `__len` and the element read repr-dispatch to
//! `arraylen_gc` / `getarrayitem` on the underlying length-prefixed GcArray.
//!
//! ## The rewrite (`rewire_one_slice_first_site`)
//!
//! Block A holds the residual `first` call producing `opt`.  Because
//! A reference-payload `Option<&PyObjectRef>` may append a trailing
//! `__cast_instance_intrinsic` after the call, so the call is not A's last op; a
//! scalar-payload `Option<u8>` has no such cast.  The block-A skeleton accepts
//! both shapes and absorbs the optional cast, exactly as
//! [`crate::front::option_map_or`] does.  The rewrite:
//! 1. drops the `first` call (+ absorbed cast) and closes A with a
//!    `bool(len(slice) > 0)` branch to two fresh arms;
//! 2. the `then_bb` arm reads `slice[0]` and wraps it in `Some`; a one-word
//!    reference niche forwards the typed payload directly, while value
//!    payloads build `__discriminant = 1` / `__pos_0 = elem`;
//! 3. the `else_bb` arm emits a typed null for that niche, or an aggregate
//!    `None` (`__discriminant = 0`) for value payloads;
//! 4. both arms re-apply the absorbed narrowing and forward to B, reproducing
//!    A's original exit args with the `opt` slot sourced from the arm's
//!    `Some`/`None` value and every other live value threaded through.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `first` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::option_map_or::emit_narrow;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `<[T]>::first(slice)` call site captured during body lowering
/// (`front::mir` `recognize_slice_first_site`).  The owner strings are resolved
/// at the recording site where the destination `Option<&T>` type is in hand;
/// the post-pass only needs them to spell the `Some`/`None` aggregates.
#[derive(Clone)]
pub(crate) struct SliceFirstSite {
    /// The `first` call result (the `Option<&T>` value) — locates block A.  The
    /// slice operand is read from that located call's `args[0]`.
    pub result_var: Variable,
    /// The access performed in the successful arm.  `RangeFrom` retains both
    /// the transparent range shell (for its exclusive-consumer proof/removal)
    /// and its scalar start bound (for the RPython `getslice(start, None)`).
    pub access: SliceAccess,
    /// The `Option` enum root `name_path` (per-instantiation, suffixed) — the
    /// ctor owner for the `Some`/`None` aggregates.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload field
    /// owner (matching the variant-qualified `resolve_adt_field` read owner).
    pub some_owner: String,
    /// The `Option`'s payload `&T` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind (`Ref(None)` for `Option<&PyObjectRef>`).
    pub payload_ty: ValueType,
    /// True when the result is the one-word nullable-reference Option niche.
    /// RPython represents that as the payload instance with
    /// `can_be_None=True`, without an Option aggregate.
    pub niche: bool,
    /// Repr projection of this receiver's niche null; see `FunctionGraph::push_niche_null`.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Concrete payload class retained on both the successful pointer and
    /// the null arm of a niche result.
    pub payload_narrow_root: Option<String>,
    /// Concrete ARRAY identity for the element read, or `None` only for a
    /// proven thin-pointer element.  Threaded the same way
    /// [`crate::front::slice_get::SliceGetSite::array_type_id`] is.
    pub array_type_id: Option<String>,
    /// Object gcarray identity when the receiver is an object-pointer slice
    /// (`*mut PyObject`). `None` for `[u8]`, `[i64]`, `[f64]`, `[str]` and
    /// every other element. Distinct from [`Self::array_type_id`]: that one
    /// stays the element read's identity.
    pub object_array_type_id: Option<String>,
    /// True when the receiver is a `String|str|Wtf8::as_bytes` view.  That
    /// base is a `StringRepr`, not a GC array, so the successful arm must
    /// emit `__string_byte_getitem` rather than `ArrayRead`.
    pub string_byte_view: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum SliceAccess {
    First,
    /// `<[T]>::last(slice)` — `Some(&slice[len-1])` when non-empty.
    Last,
    RangeFrom {
        range: Variable,
        start: Variable,
    },
}

/// Rewrite every recorded `<[T]>::first` call site into the length-checked
/// `Option` diamond.  Fail-safe: a site whose block does not fit the
/// residual-call shape is left untouched (Skip), so a mismatch never regresses
/// a graph the legacy walker already handled.  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_slice_first_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceFirstSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_first_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `first` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_slice_first_site(
    graph: &mut FunctionGraph,
    site: &SliceFirstSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the `first` residual call producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: slice::first result var has no producer block"))?;

    // Locate the `first` call op by its result (not assuming it is the block
    // tail — the trailing `__cast_instance_intrinsic` may follow, see below).
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: slice::first call op not found in block {a}"))?;
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
                    "{name}: slice::first call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: slice::first call is not the last op of block {a}"
        ));
    };

    // Capture the slice receiver and validate the recorded access operand.
    let slice = match (&graph.blocks[a].operations[ci].kind, &site.access) {
        (OpKind::Call { args, .. }, SliceAccess::First | SliceAccess::Last) if args.len() == 1 => {
            args[0].clone().into_variable()
        }
        (OpKind::Call { args, .. }, SliceAccess::RangeFrom { range, .. })
            if args.len() == 2 && args[1] == *range =>
        {
            args[0].clone().into_variable()
        }
        other => {
            return Err(format!(
                "{name}: slice first/get producer does not match its recorded operands: {other:?}"
            ));
        }
    };
    if let SliceAccess::RangeFrom { range, .. } = &site.access
        && !crate::front::slice_index::range_feeds_only_index(graph, range, &site.result_var, false)
    {
        return Err(format!(
            "{name}: RangeFrom value has a non-get consumer — declining"
        ));
    }

    // A's single exit → B (the continuation consuming the Option).  Must be a
    // plain goto — `lower_call` closes with exactly this shape.
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: slice::first call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: slice::first call block {a} exit is not a plain goto"
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

    // `then_bb` (`Some`) carries `carried` plus `slice` (the base for the
    // element read); `else_bb` (`None`) carries only `carried`.  The
    // source-var lists double as the branch link args.
    let mut then_sources = carried.clone();
    if !then_sources.contains(&slice) {
        then_sources.push(slice.clone());
    }
    let success_operand = match &site.access {
        SliceAccess::First | SliceAccess::Last => None,
        SliceAccess::RangeFrom { start, .. } => Some(start),
    };
    if let Some(operand) = success_operand
        && !then_sources.contains(operand)
    {
        then_sources.push(operand.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(carried.len());

    // `then_bb`: item = slice[0] or slice[start..]; opt = Some(item).
    let slice_in_then = map_source(&then_sources, &then_inputs, &slice)
        .ok_or_else(|| format!("{name}: slice not threaded into Some arm"))?;
    let elem = match &site.access {
        SliceAccess::First | SliceAccess::Last => {
            let item_index = graph.alloc_value_var();
            if matches!(site.access, SliceAccess::Last) {
                let len = graph.alloc_value_var();
                let len_kind = crate::front::bool_then::slice_len_op(
                    slice_in_then.clone(),
                    site.object_array_type_id.as_deref(),
                    site.string_byte_view,
                );
                graph.block_mut(then_bb).operations.push(SpaceOperation {
                    result: Some(len.clone()),
                    kind: len_kind,
                });
                let one = graph.alloc_value_var();
                graph.block_mut(then_bb).operations.push(SpaceOperation {
                    result: Some(one.clone()),
                    kind: OpKind::ConstInt(1),
                });
                graph.block_mut(then_bb).operations.push(SpaceOperation {
                    result: Some(item_index.clone()),
                    kind: OpKind::BinOp {
                        op: "sub".to_string(),
                        lhs: len,
                        rhs: one,
                        result_ty: ValueType::Int,
                    },
                });
            } else {
                graph.block_mut(then_bb).operations.push(SpaceOperation {
                    result: Some(item_index.clone()),
                    kind: OpKind::ConstInt(0),
                });
            }
            let elem = graph.alloc_value_var();
            let kind = if site.string_byte_view {
                // `as_bytes()` aliases the slice to the `StringRepr`; a
                // generic ArrayRead would address the string object as a
                // GC array.  The scalar-index path emits this marker so
                // the adapter can build `ord(s[i])`.
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__string_byte_getitem".to_string()],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice_in_then, item_index]),
                    result_ty: ValueType::Int,
                }
            } else {
                OpKind::ArrayRead {
                    base: slice_in_then,
                    // The element read and the `Some::__pos_0` field write
                    // consume the same value, so their register banks match.
                    item_ty: site.payload_ty.clone(),
                    index: item_index,
                    array_type_id: site.array_type_id.clone(),
                    nolength: crate::front::typestr::nolength_from_array_type_id(
                        site.array_type_id.as_deref(),
                    ),
                    pure: false,
                }
            };
            graph.block_mut(then_bb).operations.push(SpaceOperation {
                result: Some(elem.clone()),
                kind,
            });
            elem
        }
        SliceAccess::RangeFrom { start, .. } => {
            let start_in_then = map_source(&then_sources, &then_inputs, start)
                .ok_or_else(|| format!("{name}: RangeFrom start not threaded into Some arm"))?;
            let subslice = graph.alloc_value_var();
            // List getslice of `Vec<u8>` / `&mut [u8]` into a niche
            // `Option<&[u8]>` is `rstr.py` `LLHelpers.ll_join_chars`:
            // the Option payload is rstr `STR`. A `string_byte_view`
            // receiver is already that string; identity getslice.
            let copy_list_to_str =
                site.niche && !site.string_byte_view && matches!(site.payload_ty, ValueType::Str);
            graph.block_mut(then_bb).operations.push(SpaceOperation {
                result: Some(subslice.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__getslice_rangefrom".to_string()],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice_in_then, start_in_then]),
                    result_ty: if copy_list_to_str {
                        ValueType::Ref(None)
                    } else {
                        site.payload_ty.clone()
                    },
                },
            });
            if copy_list_to_str {
                let dest = graph.alloc_value_var();
                graph.block_mut(then_bb).operations.push(SpaceOperation {
                    result: Some(dest.clone()),
                    kind: OpKind::Call {
                        target: CallTarget::function_path([
                            crate::front::join_chars::LL_JOIN_CHARS,
                        ]),
                        args: crate::model::call_args(vec![subslice]),
                        result_ty: ValueType::Str,
                    },
                });
                dest
            } else {
                subslice
            }
        }
    };
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

    // A: drop the residual `first` call (+ absorbed cast), synthesize the guard
    // `len(slice) > 0`, branch on it.  `__len` on the `SomeList`-modelled slice
    // routes through the rtyper `len` op → `AbstractBaseListRepr.rtype_len` →
    // `arraylen_gc`; the `gt` of two `Int` operands lowers to `int_gt`.
    // `set_branch` appends the idempotent `bool(cond)` hop and installs the
    // Bool(false)/Bool(true) arm links, so the true (non-empty) arm is `then_bb`.
    let a_id = graph.blocks[a].id;
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let len = graph.alloc_value_var();
    let len_kind = crate::front::bool_then::slice_len_op(
        slice.clone(),
        site.object_array_type_id.as_deref(),
        site.string_byte_view,
    );
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(len.clone()),
        kind: len_kind,
    });
    let cond = graph.alloc_value_var();
    if let SliceAccess::RangeFrom { start: index, .. } = &site.access {
        // A RangeFrom start is `usize`.  Compare it with an unsigned view of
        // the RPython list length; `start == len` is the valid empty suffix.
        let len_u = graph.alloc_value_var();
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(len_u.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "rpython".to_string(),
                        "rlib".to_string(),
                        "rarithmetic".to_string(),
                        "r_uint".to_string(),
                    ],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![len]),
                result_ty: ValueType::Unsigned,
            },
        });
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(cond.clone()),
            kind: OpKind::BinOp {
                op: "le".to_string(),
                lhs: index.clone(),
                rhs: len_u,
                result_ty: ValueType::Int,
            },
        });
    } else {
        let zero = graph.alloc_value_var();
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(zero.clone()),
            kind: OpKind::ConstInt(0),
        });
        graph.block_mut(a_id).operations.push(SpaceOperation {
            result: Some(cond.clone()),
            kind: OpKind::BinOp {
                op: "gt".to_string(),
                lhs: len,
                rhs: zero,
                result_ty: ValueType::Int,
            },
        });
    }
    if let SliceAccess::RangeFrom { range, .. } = &site.access {
        for block in &mut graph.blocks {
            block.operations.retain(|op| {
                let is_field_write =
                    matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == range);
                let is_ctor = op.result.as_ref() == Some(range)
                    && matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::SyntheticTransparentCtor { .. },
                            ..
                        }
                    );
                !(is_field_write || is_ctor)
            });
        }
    }
    graph.set_branch(a_id, cond, then_bb, then_sources, else_bb, carried);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn slice_first_site(result_var: Variable) -> SliceFirstSite {
        SliceFirstSite {
            result_var,
            access: SliceAccess::First,
            option_owner: "core::option::Option".into(),
            some_owner: "core::option::Option::Some".into(),
            payload_ty: ValueType::Ref(None),
            niche: false,
            niche_null_cast: None,
            payload_narrow_root: None,
            array_type_id: None,
            object_array_type_id: None,
            string_byte_view: false,
        }
    }

    /// Build the minimal `opt = first(slice)` shape — block A = the residual
    /// call closed by a single goto to B (which consumes the Option) — and
    /// assert the rewrite drops the call, synthesizes `len(slice) > 0`, and
    /// branches to a `Some` arm (`slice[0]` → `Some`) and a `None` arm, both
    /// merging to B.
    #[test]
    fn rewrite_lifts_first_to_length_checked_option() {
        let mut g = FunctionGraph::new("test_slice_first");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        // B: the continuation consuming the first() result.
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);

        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt.clone())]);
        assert_eq!(rewritten, 1, "the slice::first site must be rewritten");

        // The residual `first` call is gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual first call removed from A"
        );
        // A synthesizes the length guard and a `gt` compare, then branches.
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
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "gt")),
            "A compares len > 0"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to Some/None arms");
        let elem_reads: Vec<Option<String>> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ArrayRead { array_type_id, .. } => Some(array_type_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(elem_reads, vec![None], "the Some arm reads slice[0]");
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

    #[test]
    fn rewrite_lifts_last_to_length_checked_option() {
        let mut g = FunctionGraph::new("test_slice_last");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "last".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::Last;
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        assert!(g.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "sub"))
        }));
        assert!(g.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
        }));
        // Exactly one arm reads an array element (the Some payload).
        let elem_reads = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ArrayRead { .. }))
            .count();
        assert_eq!(elem_reads, 1, "the Some arm reads slice[len-1]");
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

    #[test]
    fn last_threads_array_type_id_onto_the_element_read() {
        let mut g = FunctionGraph::new("test_slice_last_id");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "last".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::Last;
        site.payload_ty = ValueType::Int;
        site.array_type_id = Some("[u8]".into());
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        let ids: Vec<Option<String>> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ArrayRead { array_type_id, .. } => Some(array_type_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![Some("[u8]".into())]);
        // `[u8]` names the element read. The receiver is a borrowed view, not
        // the object gcarray, so both lengths — A's guard and `Last`'s
        // `len - 1` — stay the `__len` marker.
        assert_eq!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .filter(|op| matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.first().map(String::as_str) == Some("__len")
                ))
                .count(),
            2,
            "a [u8] receiver keeps the __len marker"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. })),
            "no arraylen_gc on a [u8] view"
        );
    }

    /// An object-pointer slice is the GC array `arraylen_gc` reads.
    /// `slice_object_array_type_id` names that array; both length reads carry
    /// it with `nolength: false`.
    #[test]
    fn last_on_an_object_pointer_slice_emits_arraylen() {
        let mut g = FunctionGraph::new("test_slice_last_object");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "last".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let id = crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID;
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::Last;
        site.array_type_id = Some(id.into());
        site.object_array_type_id = Some(id.into());
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        let len_ids: Vec<(Option<String>, bool)> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ArrayLen {
                    array_type_id,
                    nolength,
                    ..
                } => Some((array_type_id.clone(), *nolength)),
                _ => None,
            })
            .collect();
        assert_eq!(
            len_ids,
            vec![(Some(id.into()), false), (Some(id.into()), false)]
        );
    }

    /// A `Vec` indexes a headerless buffer, so its length is the third word of
    /// the `Vec` value and not an ARRAY header.  `arraylen_gc` would want a
    /// `lendescr` that identity has no reason to carry, so the guard keeps the
    /// `__len` marker the rtyper answers through `ll_length`.
    #[test]
    fn last_on_a_headerless_buffer_keeps_the_len_marker() {
        let mut g = FunctionGraph::new("test_slice_last_vec");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "last".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::Last;
        site.payload_ty = ValueType::Int;
        site.array_type_id = Some("Vec<u8>".into());
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        assert_eq!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .filter(|op| matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.first().map(String::as_str) == Some("__len")
                ))
                .count(),
            2,
            "a headerless buffer keeps the __len marker"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. })),
            "no arraylen_gc on a headerless buffer"
        );
    }

    #[test]
    fn last_on_a_string_byte_view_emits_the_string_byte_marker() {
        let mut g = FunctionGraph::new("test_slice_last_bytes");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "last".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::Last;
        site.payload_ty = ValueType::Int;
        site.string_byte_view = true;
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        assert!(g.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["__string_byte_getitem"]
                )
            })
        }));
        assert!(!g.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
        }));
        // The bounds guard and the `len - 1` index read the same receiver, so
        // both take `ll_strlen` and neither reads an ARRAY header.
        let strlens = g
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["__strlen"]
                )
            })
            .count();
        assert_eq!(strlens, 2, "guard and index length both use __strlen");
        assert!(!g.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        }));
    }

    /// The production shape: an `Option<&RegisteredStruct>` result appends a
    /// trailing `__cast_instance_intrinsic` narrowing op, so the call is NOT the
    /// block tail.  The rewrite absorbs the cast, fires, and re-applies the
    /// narrowing in both arms.
    #[test]
    fn rewrite_absorbs_trailing_narrow_cast() {
        let mut g = FunctionGraph::new("test_slice_first_narrow");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
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
        g.set_goto(a, b, vec![narrowed.clone()]);

        // Site records the CALL result; the rewrite discovers the trailing cast.
        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt.clone())]);
        assert_eq!(rewritten, 1, "the slice::first site must be rewritten");

        // The residual call and the trailing cast are gone from block A.
        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual first call removed from A"
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
        let mut g = FunctionGraph::new("test_slice_first_decline");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "first".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // Two trailing ops break both the "call is last" and "single cast" shapes.
        g.push_op_var(a, OpKind::ConstInt(9), true).unwrap();
        g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        g.set_return(a, None);

        let rewritten = rewire_slice_first_call_sites(&mut g, &[slice_first_site(opt)]);
        assert_eq!(rewritten, 0, "an unexpected trailing shape declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("first")
            )),
            "residual call survives on decline"
        );
    }

    #[test]
    fn rangefrom_niche_list_getslice_copies_into_str() {
        let mut g = FunctionGraph::new("test_slice_rangefrom_join_chars");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let range = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let start = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "get".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Str,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::RangeFrom { range, start };
        site.niche = true;
        site.payload_ty = ValueType::Str;
        site.string_byte_view = false;
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments == &["__getslice_rangefrom"]
                    )
                })
            }),
            "successful arm still emits getslice"
        );
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.first().map(String::as_str)
                            == Some(crate::front::join_chars::LL_JOIN_CHARS)
                    )
                })
            }),
            "list getslice into a niche STR Option copies via ll_join_chars"
        );
    }

    #[test]
    fn rangefrom_string_byte_view_skips_join_chars() {
        let mut g = FunctionGraph::new("test_slice_rangefrom_str_identity");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let range = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let start = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "slice".into(),
                            "<Impl>".into(),
                            "get".into(),
                        ],
                        fun_decl_id: None,
                        generic_rust_args: Vec::new(),
                    },
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Str,
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![opt.clone()]);
        let mut site = slice_first_site(opt);
        site.access = SliceAccess::RangeFrom { range, start };
        site.niche = true;
        site.payload_ty = ValueType::Str;
        site.string_byte_view = true;
        assert_eq!(rewire_slice_first_call_sites(&mut g, &[site]), 1);
        assert!(
            !g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.first().map(String::as_str)
                            == Some(crate::front::join_chars::LL_JOIN_CHARS)
                    )
                })
            }),
            "a STR receiver's getslice is already STR"
        );
    }
}
