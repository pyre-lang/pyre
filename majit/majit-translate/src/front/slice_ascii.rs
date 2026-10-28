//! `<[u8]>::is_ascii(slice)` → `rutf8.check_ascii`: `first_non_ascii_char(s) < 0`.
//!
//! ## Positioning
//!
//! `core::slice::ascii::<Impl>::is_ascii` is a foreign leaf whose body is
//! Opaque in the LLBC (Charon cannot extract `core`), so the caller emits a
//! residual `is_ascii` call — an unregistered callee the rtyper census Skips,
//! which Skips the calling graph with it.  Its `Self` is the primitive slice
//! `[u8]` (not an ADT), so `lower_call` keeps the raw `FunctionPath` segments
//! `["core","slice","ascii","<Impl>","is_ascii"]`, receiver in `args[0]`.
//!
//! Upstream does not residualize this check and does not inline the scan
//! into the caller. `rpython/rlib/rutf8.py` `first_non_ascii_char` is an
//! `@jit.elidable` loop over the byte string (`ord(s[i]) > 0x7F`), and
//! `check_ascii` is `first_non_ascii_char(s) < 0`.  The codewriter turns
//! that call into `call_pure` / `EF_ELIDABLE_*`.  This pass plants the
//! same shape: one helper graph holds the loop, each call site holds a
//! call to it plus `lt 0`.
//!
//! The helper is a translator-built rich-model graph, the same construction
//! [`crate::codewriter::minmax::build_ll_minmax_graph`] / [`crate::codewriter::minmax::minmax_path`]
//! use for `rbuiltin.py ll_min` / `ll_max` (mint a `FunctionGraph`, register
//! it on `CallControl`, specialize the name by repr: `ll_min__int` vs
//! `ll_min__uint`).  The MixLevelHelperAnnotator cousin is
//! `RPythonTyper::lowlevel_helper_function_with_builder` /
//! `build_ll_str2unicode_helper_graph`; those emit a rtyper `PyGraph` for a
//! hop that already reached the rtyper.  `is_ascii` is rewritten on the
//! front spine before prepass, so the helper has to be a `FunctionGraph`
//! in `CallControl` the way `minmax_path` plants `ll_min`.
//!
//! Elidable effect is `CallControl::mark_elidable` (`call.py`
//! `func._elidable_function_`), the same stamp harvested
//! `#[majit_macros::elidable]` functions receive.  The helper has no
//! source-level attribute for the harvester; `ll_str2unicode` attaches
//! its extra-effect through `mark_oopspec` because upstream
//! `LLHelpers.ll_str2unicode` owns `@oopspec`.  `first_non_ascii_char`
//! owns `@jit.elidable` and no oopspec, so the stamp is `mark_elidable`.
//!
//! Two helper graphs, one per receiver representation, matching how
//! `ll_min__int` / `ll_min__uint` specialize `ll_*` helpers by bank:
//! a `String|str|Wtf8::as_bytes` view is a `StringRepr`
//! (`__strlen` / `__string_byte_getitem`); any other `[u8]` receiver is
//! a length-prefixed GcArray (`__len` / `ArrayRead`).  Pair-slice
//! `{ptr,len}` never applies: `rust_slice_item_kind_for_spelling("&[u8]", 8)`
//! is `None`.
//!
//! The rewrite is **fail-safe**: a site whose producer is not a 1-arg
//! call returns `Err`, the caller leaves the residual call untouched,
//! and the unregistered `is_ascii` callee keeps the rtyper census Skip.

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;

/// `rutf8.first_non_ascii_char` on a `StringRepr` byte view.
pub(crate) const LL_FIRST_NON_ASCII_CHAR: &str = "ll_first_non_ascii_char";
/// Same helper specialized to a `[u8]` GcArray receiver
/// (`ll_min__int` / `ll_min__uint` bank split).
pub(crate) const LL_FIRST_NON_ASCII_CHAR_ARRAY: &str = "ll_first_non_ascii_char__u8";

/// A recognized `<[u8]>::is_ascii(slice)` call site captured during body
/// lowering.  The post-pass reads the slice operand from the located call's
/// `args[0]`.
#[derive(Clone)]
pub(crate) struct SliceIsAsciiSite {
    /// The `is_ascii` call result (the `bool`) — locates the producer op.
    pub result_var: Variable,
    /// True when the receiver is a `String|str|Wtf8::as_bytes` view.  That
    /// base is a `StringRepr`, so the call is to [`LL_FIRST_NON_ASCII_CHAR`]
    /// rather than [`LL_FIRST_NON_ASCII_CHAR_ARRAY`].
    pub string_byte_view: bool,
}

/// `true` iff `segments` name `<[u8]>::is_ascii`.
///
/// The FunDecl lives in the `ascii` submodule
/// (`core::slice::ascii::<Impl>::is_ascii`).  A crate-stripped spelling
/// (`slice::ascii::<Impl>::is_ascii`) and the inherent
/// `slice::<Impl>::is_ascii` form both match; `slice::index::<Impl>::is_ascii`
/// does not.
pub(crate) fn is_slice_is_ascii_segments(segments: &[String]) -> bool {
    let n = segments.len();
    if n < 3 {
        return false;
    }
    let leaf = crate::front::clause_spec::unspecialized_leaf(&segments[n - 1]);
    if leaf != "is_ascii" || segments[n - 2] != "<Impl>" {
        return false;
    }
    if n >= 4 && segments[n - 3] == "ascii" && segments[n - 4] == "slice" {
        return true;
    }
    segments[n - 3] == "slice" && (n < 4 || segments[n - 4] != "index")
}

/// Mint and register the two `first_non_ascii_char` helper graphs, then
/// stamp `func._elidable_function_` (`CallControl::mark_elidable`).
///
/// Called from the translator setup next to the other synthesized-helper
/// marks (`ll_str2unicode` oopspec), before `find_all_graphs`, so prepass
/// lookups of the rewritten call sites resolve.
pub(crate) fn register_first_non_ascii_char_helpers(cc: &mut CallControl) {
    for (name, string_byte_view) in [
        (LL_FIRST_NON_ASCII_CHAR, true),
        (LL_FIRST_NON_ASCII_CHAR_ARRAY, false),
    ] {
        let path = CallPath::from_segments([name]);
        if cc.has_function_graph(&path) {
            continue;
        }
        let graph =
            build_ll_first_non_ascii_char_graph(name, string_byte_view).with_return_type("i64");
        cc.register_function_graph(path.clone(), graph);
        cc.mark_elidable(path);
    }
}

/// `rutf8.first_non_ascii_char` in the rich model, following
/// [`crate::codewriter::minmax::build_ll_minmax_graph`]: one argument `s`,
/// loop `for i in range(len(s))`, return the first `i` with `ord(s[i]) > 0x7F`
/// or `-1`.
pub(crate) fn build_ll_first_non_ascii_char_graph(
    name: &str,
    string_byte_view: bool,
) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name);
    let start = graph.startblock;

    let s = graph.alloc_value_var();
    graph.push_inputarg_var(start, s.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s".to_string(),
            ty: ValueType::Ref(None),
            class_root: None,
        },
        s.clone(),
    );

    let n = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(n.clone()),
        kind: crate::front::option_variant::slice_len_op(s.clone(), None, string_byte_view),
    });
    let i = graph.alloc_value_var();
    graph.block_mut(start).operations.push(SpaceOperation {
        result: Some(i.clone()),
        kind: OpKind::ConstInt(0),
    });

    let (loop_bb, loop_args) = graph.create_block_with_arg_vars(3);
    let (body_bb, body_args) = graph.create_block_with_arg_vars(3);
    let (found_bb, found_args) = graph.create_block_with_arg_vars(1);
    let (miss_bb, _) = graph.create_block_with_arg_vars(0);

    graph.set_goto(start, loop_bb, vec![s, n, i]);

    let s_loop = loop_args[0].clone();
    let n_loop = loop_args[1].clone();
    let i_loop = loop_args[2].clone();
    let lt = graph.alloc_value_var();
    graph.block_mut(loop_bb).operations.push(SpaceOperation {
        result: Some(lt.clone()),
        kind: OpKind::BinOp {
            op: "lt".to_string(),
            lhs: i_loop.clone(),
            rhs: n_loop.clone(),
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(
        loop_bb,
        lt,
        body_bb,
        vec![s_loop, n_loop, i_loop],
        miss_bb,
        vec![],
    );

    let s_body = body_args[0].clone();
    let n_body = body_args[1].clone();
    let i_body = body_args[2].clone();
    let byte = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(byte.clone()),
        kind: byte_at(string_byte_view, s_body.clone(), i_body.clone()),
    });
    let limit = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(limit.clone()),
        kind: OpKind::ConstInt(0x7F),
    });
    let gt = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(gt.clone()),
        kind: OpKind::BinOp {
            op: "gt".to_string(),
            lhs: byte,
            rhs: limit,
            result_ty: ValueType::Int,
        },
    });
    let one = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(one.clone()),
        kind: OpKind::ConstInt(1),
    });
    let i_next = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(i_next.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: i_body.clone(),
            rhs: one,
            result_ty: ValueType::Int,
        },
    });
    graph.set_branch(
        body_bb,
        gt,
        found_bb,
        vec![i_body],
        loop_bb,
        vec![s_body, n_body, i_next],
    );

    let i_found = found_args[0].clone();
    graph.set_return(found_bb, Some(i_found));

    let minus_one = graph.alloc_value_var();
    graph.block_mut(miss_bb).operations.push(SpaceOperation {
        result: Some(minus_one.clone()),
        kind: OpKind::ConstInt(-1),
    });
    graph.set_return(miss_bb, Some(minus_one));

    graph
}

/// Rewrite every recorded `<[u8]>::is_ascii` call site into
/// `first_non_ascii_char(s) < 0`.  Fail-safe: a site whose producer is not
/// a 1-arg call is left untouched (Skip).  Returns the number of sites
/// rewritten.
pub(crate) fn rewire_slice_is_ascii_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIsAsciiSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_is_ascii_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `is_ascii` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_slice_is_ascii_site(
    graph: &mut FunctionGraph,
    site: &SliceIsAsciiSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: slice::is_ascii result var has no producer block"))?;

    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: slice::is_ascii call op not found in block {a}"))?;

    let slice = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone().into_variable(),
        other => {
            return Err(format!(
                "{name}: slice::is_ascii producer is not a 1-arg call: {other:?}"
            ));
        }
    };

    let helper = if site.string_byte_view {
        LL_FIRST_NON_ASCII_CHAR
    } else {
        LL_FIRST_NON_ASCII_CHAR_ARRAY
    };
    let idx = graph.alloc_value_var();
    let zero = graph.alloc_value_var();
    graph.blocks[a].operations[ci].kind = OpKind::Call {
        target: CallTarget::function_path([helper]),
        args: crate::model::call_args(vec![slice]),
        result_ty: ValueType::Int,
    };
    graph.blocks[a].operations[ci].result = Some(idx.clone());
    graph.blocks[a].operations.insert(
        ci + 1,
        SpaceOperation {
            result: Some(zero.clone()),
            kind: OpKind::ConstInt(0),
        },
    );
    graph.blocks[a].operations.insert(
        ci + 2,
        SpaceOperation {
            result: Some(site.result_var.clone()),
            kind: OpKind::BinOp {
                op: "lt".to_string(),
                lhs: idx,
                rhs: zero,
                result_ty: ValueType::Bool,
            },
        },
    );
    Ok(())
}

fn byte_at(string_byte_view: bool, slice: Variable, index: Variable) -> OpKind {
    if string_byte_view {
        // `as_bytes()` aliases the slice to the `StringRepr`; a generic
        // ArrayRead would address the string object as a GC array.  The
        // adapter builds `ord(s[i])` from this marker.
        OpKind::Call {
            target: CallTarget::function_path(["__string_byte_getitem"]),
            args: crate::model::call_args(vec![slice, index]),
            result_ty: ValueType::Int,
        }
    } else {
        OpKind::ArrayRead {
            base: slice,
            item_ty: ValueType::Int,
            index,
            array_type_id: Some("[u8]".to_string()),
            nolength: crate::front::typestr::nolength_from_array_type_id(Some("[u8]")),
            pure: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::{AnalysisCache, CallControl};
    use crate::model::CallTarget;
    use majit_ir::descr::{ExtraEffect, OopSpecIndex};
    use majit_ir::value::Type;

    fn is_ascii_segments() -> Vec<String> {
        vec![
            "core".into(),
            "slice".into(),
            "ascii".into(),
            "<Impl>".into(),
            "is_ascii".into(),
        ]
    }

    fn residual_is_ascii(g: &mut FunctionGraph, slice: Variable) -> Variable {
        let a = g.startblock;
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: is_ascii_segments(),
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![slice]),
                result_ty: ValueType::Bool,
            },
            true,
        )
        .unwrap()
    }

    fn has_path(g: &FunctionGraph, leaf: &str) -> bool {
        g.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some(leaf)
                )
            })
        })
    }

    fn has_binop(g: &FunctionGraph, name: &str) -> bool {
        g.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == name))
        })
    }

    fn has_loop_back_edge(g: &FunctionGraph) -> bool {
        g.blocks.iter().any(|block| {
            block.exits.iter().any(|exit| {
                g.blocks[exit.target.0]
                    .exits
                    .iter()
                    .any(|back| back.target == block.id)
                    || exit.target == block.id
            })
        })
    }

    fn rewrite_site(string_byte_view: bool) -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_slice_is_ascii");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let ok = residual_is_ascii(&mut g, slice);
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![ok.clone()]);
        let site = SliceIsAsciiSite {
            result_var: ok.clone(),
            string_byte_view,
        };
        assert_eq!(rewire_slice_is_ascii_call_sites(&mut g, &[site]), 1);
        (g, ok)
    }

    /// Pin the lowering on a minimal `<[u8]>::is_ascii` residual — the
    /// `as_bytes()` string-byte-view shape — without naming a pyre function.
    #[test]
    fn rewrite_lifts_is_ascii_on_a_string_byte_view() {
        let (g, ok) = rewrite_site(true);

        assert!(!has_path(&g, "is_ascii"), "residual is_ascii call removed");
        assert!(
            has_path(&g, LL_FIRST_NON_ASCII_CHAR),
            "call site calls ll_first_non_ascii_char"
        );
        assert!(!has_path(&g, LL_FIRST_NON_ASCII_CHAR_ARRAY));
        assert!(
            has_binop(&g, "lt"),
            "check_ascii is first_non_ascii_char(s) < 0"
        );
        assert!(
            !has_path(&g, "__strlen"),
            "the scan loop does not live in the caller"
        );
        assert!(!has_path(&g, "__string_byte_getitem"));
        assert!(!has_binop(&g, "add"), "caller has no loop increment");
        assert!(
            !g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(127) | OpKind::ConstInt(-1)))),
            "caller does not hold the scan threshold or miss sentinel"
        );
        assert!(
            !g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. } | OpKind::ArrayLen { .. }))),
            "a string-byte-view does not read a GC array"
        );
        // The original bool result is the `lt` compare, not the helper call.
        let producer = g
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .find(|op| op.result.as_ref() == Some(&ok))
            .expect("bool result still produced");
        assert!(
            matches!(&producer.kind, OpKind::BinOp { op, .. } if op == "lt"),
            "is_ascii result is idx < 0, got {:?}",
            producer.kind
        );
    }

    #[test]
    fn rewrite_lifts_is_ascii_on_a_u8_gcarray() {
        let (g, _) = rewrite_site(false);
        assert!(!has_path(&g, "is_ascii"));
        assert!(has_path(&g, LL_FIRST_NON_ASCII_CHAR_ARRAY));
        assert!(!has_path(&g, LL_FIRST_NON_ASCII_CHAR));
        assert!(has_binop(&g, "lt"));
        assert!(
            !has_path(&g, "__len"),
            "the scan loop does not live in the caller"
        );
        assert!(
            !g.blocks.iter().any(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
            }),
            "GcArray byte read lives in the helper, not the caller"
        );
        assert!(!has_path(&g, "__strlen"));
        assert!(!has_path(&g, "__string_byte_getitem"));
    }

    #[test]
    fn helper_graph_holds_the_scan_loop_on_a_string_byte_view() {
        let g = build_ll_first_non_ascii_char_graph(LL_FIRST_NON_ASCII_CHAR, true);
        assert!(
            has_path(&g, "__strlen"),
            "string-byte-view length is __strlen"
        );
        assert!(
            has_path(&g, "__string_byte_getitem"),
            "string-byte-view byte is __string_byte_getitem"
        );
        assert!(
            !g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. } | OpKind::ArrayLen { .. }))),
            "a string-byte-view does not read a GC array"
        );
        assert!(has_binop(&g, "lt"), "loop condition is i < n");
        assert!(has_binop(&g, "gt"), "byte test is ord(s[i]) > 0x7F");
        assert!(has_binop(&g, "add"), "loop increments i");
        assert!(
            g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(127)))),
            "threshold is 0x7F (rutf8.first_non_ascii_char)"
        );
        assert!(
            g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(-1)))),
            "miss path returns -1"
        );
        assert!(
            has_loop_back_edge(&g),
            "the scan loops back to the condition"
        );
    }

    #[test]
    fn helper_graph_holds_the_scan_loop_on_a_u8_gcarray() {
        let g = build_ll_first_non_ascii_char_graph(LL_FIRST_NON_ASCII_CHAR_ARRAY, false);
        assert!(has_path(&g, "__len"), "GcArray [u8] length is __len");
        assert!(
            g.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::ArrayRead {
                            array_type_id: Some(id),
                            ..
                        } if id == "[u8]"
                    )
                })
            }),
            "GcArray [u8] byte is ArrayRead"
        );
        assert!(!has_path(&g, "__strlen"));
        assert!(!has_path(&g, "__string_byte_getitem"));
        assert!(has_loop_back_edge(&g));
    }

    #[test]
    fn helper_call_is_elidable_in_getcalldescr() {
        let mut cc = CallControl::new();
        register_first_non_ascii_char_helpers(&mut cc);
        cc.find_all_graphs_for_tests();

        for name in [LL_FIRST_NON_ASCII_CHAR, LL_FIRST_NON_ASCII_CHAR_ARRAY] {
            let target = CallTarget::function_path([name]);
            assert!(
                cc.is_elidable(&target),
                "{name} carries func._elidable_function_"
            );
            let mut cache = AnalysisCache::default();
            let descriptor = cc.getcalldescr(
                &SpaceOperation {
                    result: None,
                    kind: OpKind::Call {
                        target: target.clone(),
                        args: crate::model::call_args(vec![Variable::new()]),
                        result_ty: ValueType::Int,
                    },
                },
                vec![Type::Ref],
                Type::Int,
                OopSpecIndex::None,
                None,
                &mut cache,
                None,
            );
            assert!(
                matches!(
                    descriptor.extra_info.extraeffect,
                    ExtraEffect::ElidableCannotRaise
                        | ExtraEffect::ElidableOrMemoryError
                        | ExtraEffect::ElidableCanRaise
                ),
                "{name} extraeffect was {:?}",
                descriptor.extra_info.extraeffect
            );
        }
    }

    #[test]
    fn is_slice_is_ascii_segments_matches_the_ascii_impl() {
        fn segs(parts: &[&str]) -> Vec<String> {
            parts.iter().map(|s| (*s).to_string()).collect()
        }
        assert!(is_slice_is_ascii_segments(&segs(&[
            "core", "slice", "ascii", "<Impl>", "is_ascii"
        ])));
        assert!(is_slice_is_ascii_segments(&segs(&[
            "slice", "ascii", "<Impl>", "is_ascii"
        ])));
        assert!(is_slice_is_ascii_segments(&segs(&[
            "core", "slice", "<Impl>", "is_ascii"
        ])));
        assert!(!is_slice_is_ascii_segments(&segs(&[
            "core", "slice", "index", "<Impl>", "is_ascii"
        ])));
        assert!(!is_slice_is_ascii_segments(&segs(&[
            "core", "slice", "<Impl>", "is_empty"
        ])));
        assert!(!is_slice_is_ascii_segments(&segs(&["is_ascii"])));
    }

    #[test]
    fn mismatch_leaves_the_residual() {
        let mut g = FunctionGraph::new("test_slice_is_ascii_mismatch");
        let a = g.startblock;
        let ok = g.push_op_var(a, OpKind::ConstBool(true), true).unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![ok.clone()]);

        let site = SliceIsAsciiSite {
            result_var: ok,
            string_byte_view: true,
        };
        assert_eq!(rewire_slice_is_ascii_call_sites(&mut g, &[site]), 0);
        assert!(
            g.blocks.iter().any(|block| block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ConstBool(true)))),
            "fail-safe keeps the non-call producer"
        );
    }
}
