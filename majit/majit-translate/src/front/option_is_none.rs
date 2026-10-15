//! `Option::is_none` / `Option::is_some` → representation test (#131).
//!
//! `is_none`/`is_some` are pure predicates on the receiver's representation.
//! An aggregate Option tests its tag: `is_none` is
//! `opt.__discriminant == 0` and `is_some` is `opt.__discriminant != 0`
//! (`None`=0, `Some`=1). A nullable RPython reference (`SomeInstance`,
//! `SomeString`, or `SomeList` with `can_be_None=True`) has no enum object or
//! tag field, so it uses RPython's pointer-identity `is None` operation
//! (negated for `is_some`). Unlike
//! `unwrap_or`/`map_or` these predicates read no payload and take no default,
//! so there is no diamond and no block split; the residual one-arg `Method`
//! call is replaced **in place** while preserving its result `Variable`.
//!
//! `is_none`/`is_some` bodies are Opaque (foreign `core`), so the receiver is
//! the `Option` ADT and `first_is_self` routes the call to a
//! `CallTarget::Method` (receiver in `args[0]`).  The `Option` enum root
//! owning `__discriminant` is resolved at the recording site
//! (`front::mir` `recognize_is_none_site`) where the receiver type is in hand.
//!
//! It is **fail-safe**: any structural mismatch leaves the residual call
//! untouched, so the unregistered callee keeps the rtyper census Skip and no
//! graph the legacy walker already handled regresses.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueType};

/// A recognized `Option::is_none(opt)` / `Option::is_some(opt)` call site
/// captured during body lowering (`front::mir` `recognize_is_none_site`).
#[derive(Clone)]
pub(crate) struct IsNoneSite {
    /// The predicate call result (the `bool`) — locates the producer op.
    pub result_var: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner.
    pub option_owner: String,
    /// `true` for `is_some` (`__discriminant != 0`), `false` for `is_none`
    /// (`__discriminant == 0`).
    pub is_some: bool,
    /// The receiver is the translated nullable-pointer representation. Its
    /// predicate is an identity comparison with null; no `__discriminant`
    /// exists.
    pub niche: bool,
    /// Optional `__cast_instance_intrinsic` target and result kind for a Rust
    /// container projected to an RPython string/list repr. The null operand
    /// must enter that same repr before comparison.
    pub niche_null_cast: Option<(String, ValueType)>,
    /// Int-banked niche (`Option<fn>` / raw Address): the null arm is
    /// `core.ptr.null_fn` (`ValueType::Int`). `null_mut` is a GC ref and
    /// would emit `is_` on mixed `ir` operands (`rnone.py` `rtype_is_None`).
    pub fn_ptr: bool,
    /// `Option<NonZero<_>>`: the word itself is the tag. `None` is 0, so
    /// the predicate compares that integer and does not read `__discriminant`.
    pub scalar_niche: bool,
}

/// Rewrite every recorded `is_none`/`is_some` call into the discriminant
/// comparison.  Fail-safe: a site whose producer op does not fit the residual
/// one-arg-call shape is left untouched (Skip).  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_is_none_call_sites(graph: &mut FunctionGraph, sites: &[IsNoneSite]) -> usize {
    let mut rewritten = 0;
    for site in sites {
        if rewire_one_is_none_site(graph, site).is_ok() {
            rewritten += 1;
        }
    }
    rewritten
}

fn rewire_one_is_none_site(graph: &mut FunctionGraph, site: &IsNoneSite) -> Result<(), String> {
    let name = graph.name.clone();
    // The block + op index producing `result_var`.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: is_none result var has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .expect("producer op located above");

    // The producer must be the one-arg residual call; capture the receiver.
    let opt = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: is_none producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // Structural validation passed; mutate the graph.
    let new_ops = if site.niche {
        let mut ops = Vec::new();
        let rhs = if site.fn_ptr {
            let null = graph.alloc_value_var();
            ops.push(SpaceOperation {
                result: Some(null.clone()),
                kind: OpKind::Call {
                    target: CallTarget::function_path(["core", "ptr", "null_fn"]),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Int,
                },
            });
            if let Some((root, result_ty)) = &site.niche_null_cast {
                let narrowed = graph.alloc_value_var();
                ops.push(SpaceOperation {
                    result: Some(narrowed.clone()),
                    kind: crate::model::cast_instance_call_result(
                        root.clone(),
                        null,
                        result_ty.clone(),
                    ),
                });
                narrowed
            } else {
                null
            }
        } else {
            let block_id = graph.blocks[a].id;
            let before = graph.block(block_id).operations.len();
            let rhs = graph.push_niche_null(block_id, site.niche_null_cast.as_ref());
            ops.extend(graph.block_mut(block_id).operations.split_off(before));
            rhs
        };
        let is_none = if site.is_some {
            graph.alloc_value_var()
        } else {
            site.result_var.clone()
        };
        ops.push(SpaceOperation {
            result: Some(is_none.clone()),
            kind: OpKind::BinOp {
                op: "is_".to_string(),
                lhs: opt.into_variable(),
                rhs,
                result_ty: ValueType::Int,
            },
        });
        if site.is_some {
            let false_var = graph.alloc_value_var();
            ops.push(SpaceOperation {
                result: Some(false_var.clone()),
                kind: OpKind::ConstBool(false),
            });
            ops.push(SpaceOperation {
                result: Some(site.result_var.clone()),
                kind: OpKind::BinOp {
                    op: "eq".to_string(),
                    lhs: is_none,
                    rhs: false_var,
                    result_ty: ValueType::Int,
                },
            });
        }
        ops
    } else if site.scalar_niche {
        let op = if site.is_some { "ne" } else { "eq" };
        let zero = graph.alloc_value_var();
        vec![
            SpaceOperation {
                result: Some(zero.clone()),
                kind: crate::front::mir::nonzero_option_zero(&site.option_owner),
            },
            SpaceOperation {
                result: Some(site.result_var.clone()),
                kind: OpKind::BinOp {
                    op: op.to_string(),
                    lhs: opt.into_variable(),
                    rhs: zero,
                    result_ty: ValueType::Int,
                },
            },
        ]
    } else {
        let op = if site.is_some { "ne" } else { "eq" };
        let disc = graph.alloc_value_var();
        let zero = graph.alloc_value_var();
        vec![
            SpaceOperation {
                result: Some(disc.clone()),
                kind: OpKind::FieldRead {
                    base: opt.into_variable(),
                    field: FieldDescriptor {
                        name: "__discriminant".to_string(),
                        owner_root: Some(site.option_owner.clone()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Int,
                    pure: true,
                },
            },
            SpaceOperation {
                result: Some(zero.clone()),
                kind: OpKind::ConstInt(0),
            },
            SpaceOperation {
                result: Some(site.result_var.clone()),
                kind: OpKind::BinOp {
                    op: op.to_string(),
                    lhs: disc,
                    rhs: zero,
                    result_ty: ValueType::Int,
                },
            },
        ]
    };
    graph.blocks[a]
        .operations
        .splice(call_idx..=call_idx, new_ops);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_none_target(name: &str) -> CallTarget {
        CallTarget::method(name, Some("core::option::Option".to_string()))
    }

    fn site(result_var: Variable, is_some: bool) -> IsNoneSite {
        IsNoneSite {
            result_var,
            option_owner: "core::option::Option".into(),
            is_some,
            niche: false,
            niche_null_cast: None,
            fn_ptr: false,
            scalar_niche: false,
        }
    }

    /// Build `result = is_none(opt)` — a one-arg residual call — and assert the
    /// rewrite drops the call and replaces it with `opt.__discriminant == 0`
    /// bound to the same result var.
    fn build_and_rewrite(method: &str, is_some: bool) -> (FunctionGraph, Variable, Variable) {
        let mut g = FunctionGraph::new("test_is_none");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: is_none_target(method),
                    args: crate::model::call_args(vec![opt.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);
        let rewritten = rewire_is_none_call_sites(&mut g, &[site(result.clone(), is_some)]);
        assert_eq!(rewritten, 1, "the is_none/is_some site must be rewritten");
        (g, opt, result)
    }

    #[test]
    fn rewrite_lifts_is_none_to_discriminant_eq_zero() {
        let (g, opt, result) = build_and_rewrite("is_none", false);
        let a = g.startblock.0;
        // Residual call is gone.
        assert!(
            !g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual is_none call removed"
        );
        // The discriminant is read off `opt` exactly once.
        let disc = g.blocks[a]
            .operations
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, base, .. }
                    if field.name == "__discriminant" && base == &opt)
            })
            .expect("reads opt.__discriminant");
        let disc_var = disc.result.clone().unwrap();
        // The result var is a `BinOp("eq", disc, 0)`.
        let binop = g.blocks[a]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&result))
            .expect("result var has a producer");
        match &binop.kind {
            OpKind::BinOp { op, lhs, .. } => {
                assert_eq!(op, "eq", "is_none compares == 0");
                assert_eq!(lhs, &disc_var, "compares the discriminant read");
            }
            other => panic!("result is not a BinOp: {other:?}"),
        }
    }

    #[test]
    fn rewrite_lifts_is_some_to_discriminant_ne_zero() {
        let (g, _opt, result) = build_and_rewrite("is_some", true);
        let a = g.startblock.0;
        let binop = g.blocks[a]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&result))
            .expect("result var has a producer");
        match &binop.kind {
            OpKind::BinOp { op, .. } => assert_eq!(op, "ne", "is_some compares != 0"),
            other => panic!("result is not a BinOp: {other:?}"),
        }
    }

    #[test]
    fn rewrite_lifts_nullable_is_some_to_pointer_null_test() {
        let mut g = FunctionGraph::new("test_nullable_is_some");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstRefNull, true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: is_none_target("is_some"),
                    args: crate::model::call_args(vec![opt.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);
        let rewritten = rewire_is_none_call_sites(
            &mut g,
            &[IsNoneSite {
                result_var: result.clone(),
                option_owner: "core::option::Option".into(),
                is_some: true,
                niche: true,
                niche_null_cast: None,
                fn_ptr: false,
                scalar_niche: false,
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            !g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant")),
            "nullable RPython references have no enum tag field"
        );
        let null = g.blocks[a.0]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. }
                    if target.to_string() == "core::ptr::null_mut" && args.is_empty() =>
                {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("one repr-adaptive null pointer");
        let is_none = g.blocks[a.0]
            .operations
            .iter()
            .find_map(|operation| match &operation.kind {
                OpKind::BinOp {
                    op: name, lhs, rhs, ..
                } if name == "is_" && lhs == &opt && rhs == &null => operation.result.clone(),
                _ => None,
            })
            .expect("opt is null identity test");
        let false_var = g.blocks[a.0]
            .operations
            .iter()
            .find_map(|operation| match operation.kind {
                OpKind::ConstBool(false) => operation.result.clone(),
                _ => None,
            })
            .expect("is_some negation false constant");
        let predicate = g.blocks[a.0]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&result))
            .expect("is_some result producer");
        assert!(
            matches!(&predicate.kind, OpKind::BinOp { op, lhs, rhs, .. }
                if op == "eq" && lhs == &is_none && rhs == &false_var),
            "is_some on a nullable ref negates `opt is null`"
        );
    }

    /// A producer op that is not a 1-arg call declines (fail-safe): the residual
    /// op survives untouched.
    #[test]
    fn rewrite_declines_on_wrong_arity() {
        let mut g = FunctionGraph::new("test_is_none_decline");
        let a = g.startblock;
        let opt = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let default = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: is_none_target("is_none"),
                    args: crate::model::call_args(vec![opt, default]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);
        let rewritten = rewire_is_none_call_sites(&mut g, &[site(result, false)]);
        assert_eq!(rewritten, 0, "a 2-arg call declines");
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::Call { .. })),
            "residual call survives on decline"
        );
    }
}
