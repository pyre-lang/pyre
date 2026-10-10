//! End-to-end smoke tests for the MIR-driven flowspace driver.
//!
//! The corpus snapshot at `majit/charon-corpus/corpus.ullbc` is the
//! input and the regression fixture for the production MIR frontend.

mod common;

use common::{INTERPRETER_LLBC, load_llbc, lower_context_for, lower_named};
use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, build_semantic_program_from_llbc, lower_function};
use std::sync::OnceLock;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc",);

/// Load `corpus.ullbc` once and share it across every test. `Llbc` is
/// read-only after `load`, so a single parse behind a `OnceLock` is
/// sufficient: `get_or_init` runs the load exactly once even under the
/// concurrent test threads, and the lowering entry points only borrow it.
fn load_corpus() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(CORPUS).expect("load corpus.ullbc"))
}

/// BFS the set of blocks reachable from the graph's startblock. The
/// `bool_then` / `option_question_mark` rewrites can leave the pre-split
/// framestate merge block unreachable, so the reachable-only assertions
/// filter against this set.
fn reachable_blocks(
    graph: &majit_translate::model::FunctionGraph,
) -> std::collections::HashSet<majit_translate::model::BlockId> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![graph.startblock];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        for l in &graph.block(id).exits {
            stack.push(l.target);
        }
    }
    seen
}

#[test]
fn lowers_straight_line_add() {
    let llbc = load_corpus();
    let graph = lower_function(llbc, "straight_line_add").expect("lowering");
    // FunctionGraph.name keeps the full Charon-qualified path
    // because it identifies the LLBC source — only the
    // SemanticFunction.name has the crate-prefix stripping applied
    // at SemanticProgram build time.
    assert_eq!(graph.name, "charon_corpus::straight_line_add");

    let startblock = graph.block(graph.startblock);
    assert_eq!(
        startblock.inputargs.len(),
        3,
        "straight_line_add takes three i64 args"
    );
    // Charon emits 7 MIR BBs: three overflow Asserts, a Return, and one
    // cleanup UnwindResume per Assert. `FlowContext.build_flow` records a
    // block only when `pendingblocks` reaches it, and the lowering never
    // follows `on_unwind`, so the three cleanup blocks are not recorded.
    // bb0 maps onto startblock; returnblock and exceptblock are the other
    // sentinels. 4 reachable MIR blocks + those two sentinels = 6.
    assert_eq!(
        graph.blocks.len(),
        6,
        "4 reachable MIR bbs + returnblock + exceptblock"
    );

    // At least one of the MIR blocks should carry a BinOp operation
    // (the AddChecked / MulChecked / AddChecked sequence collapses to
    // three BinOp ops once the overflow asserts are stripped).
    use majit_translate::model::OpKind;
    let mut binop_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            if matches!(op.kind, OpKind::BinOp { .. }) {
                binop_count += 1;
            }
        }
    }
    assert_eq!(
        binop_count, 3,
        "expected 3 BinOps for the a + b * 2 + c chain"
    );
}

#[test]
fn lowers_branch_loop_sum_with_calls_and_discriminant() {
    // `branch_loop_sum` exercises three surfaces together: `Call`
    // terminators (`slice.iter()` / `Iterator::next`), `Drop`
    // terminators, and `Rvalue::Discriminant` on the iterator's
    // `Option<&i64>` step result.
    let llbc = load_corpus();
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::branch_loop_sum");

    use majit_translate::model::{CallTarget, OpKind};
    let mut call_count = 0usize;
    let mut discr_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                // An Abort terminator lowers the `exc_from_raise` op
                // pair (`simple_call(const(exc_class))` + `type(evalue)`)
                // into its block; exclude those raise-machinery ops so
                // the count characterizes the body's own calls.
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if matches!(
                    segments.first().map(String::as_str),
                    Some("simple_call" | "type")
                ) => {}
                OpKind::Call { .. } => call_count += 1,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    discr_count += 1
                }
                _ => {}
            }
        }
    }
    // `branch_loop_sum` iterates the `(items, length)` pair of its `&[i64]`:
    // `range(0, intmask(length))` and its `iter` op, `Iterator::next` lifted
    // to the `[__iter_next]` op, the item read `ll_slice_getitem_fast_i`
    // at `r_uint(index)`, and `intmask` of that `usize` helper result
    // (`cast_uint_to_int`) so the signed item stays Signed.
    assert_eq!(
        call_count, 7,
        "expected 7 body Call ops (intmask, range, iter, next, r_uint, getitem, intmask)"
    );
    // The `next`-diamond rewrite (`front::iter_next`) replaces the
    // `Option` step's `__discriminant` switch with the `next` op's
    // StopIteration exception edge, so the discriminant read is consumed
    // and its (now-unreachable) block dropped.
    assert_eq!(
        discr_count, 0,
        "the Option __discriminant read is consumed by the next rewrite"
    );
}

#[test]
fn lowers_strategy_len_with_discriminant_switch() {
    let llbc = load_corpus();
    let graph = lower_function(llbc, "strategy_len").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::strategy_len");
    // bb0 Discriminant + Switch, bb1/bb2/bb3 arm bodies + Return.
    // bb4 is the Abort default of that switch, outside the edges the
    // lowering emits, so it is not recorded. 4 reachable MIR bbs +
    // returnblock + exceptblock = 6.
    assert_eq!(graph.blocks.len(), 6);
}

/// Charon `TerminatorKind::Panic` lowers the same implicit
/// `AssertionError` raise as the older `Abort` terminator.
#[test]
fn a_panic_terminator_raises_implicitly_like_abort() {
    use majit_charon_reader::ullbc::TermKind;
    use majit_translate::front::mir::{LowerContext, lower_fun_decl};
    use majit_translate::model::LinkArg;

    let span = serde_json::json!({
        "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
        "generated_from_span": null
    });
    let unit = serde_json::json!({"Tuple": []});
    let local = serde_json::json!({
        "index": 0,
        "name": null,
        "span": span,
        "ty": unit
    });
    let panic_name = serde_json::json!([
        {"Ident": ["core", 0]},
        {"Ident": ["panicking", 0]},
        {"Ident": ["panic_fmt", 0]}
    ]);
    let fun = |id: u64, leaf: &str, term: serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": {
                "name": [{"Ident": ["probe", 0]}, {"Ident": [leaf, 0]}],
                "span": span,
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": unit},
            "body": {"Unstructured": {
                "span": span,
                "locals": {"arg_count": 0, "locals": [local]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span, "kind": term},
                    "is_cleanup": false
                }]
            }}
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.281",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": [],
            "fun_decls": [
                fun(0, "abort_raise", serde_json::json!({"Abort": {"Panic": panic_name.clone()}})),
                fun(1, "panic_raise", serde_json::json!({
                    "Panic": {"name": panic_name, "on_unwind": 0}
                }))
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let panic_body = llbc
        .fn_by_id(1)
        .and_then(|fd| fd.unstructured())
        .expect("panic body");
    assert!(
        matches!(panic_body.body[0].term(&llbc), Ok(TermKind::Panic { .. })),
        "the Panic JSON shape must decode as TermKind::Panic"
    );

    let context = LowerContext::new(&llbc);
    let abort_graph =
        lower_fun_decl(&context, llbc.fn_by_id(0).expect("abort_raise")).expect("lower abort");
    let panic_graph =
        lower_fun_decl(&context, llbc.fn_by_id(1).expect("panic_raise")).expect("lower panic");

    let implicit_raises = |graph: &majit_translate::model::FunctionGraph| {
        graph
            .blocks
            .iter()
            .filter(|b| {
                b.exits.iter().any(|l| {
                    l.target == graph.exceptblock
                        && l.args.len() == 2
                        && matches!(l.args[0], LinkArg::Const(_))
                        && matches!(l.args[1], LinkArg::Const(_))
                })
            })
            .count()
    };
    assert_eq!(implicit_raises(&abort_graph), 1);
    assert_eq!(implicit_raises(&panic_graph), 1);
}

#[test]
fn lowers_desugar_mix_with_aggregate_and_question_mark() {
    // `desugar_mix` exercises every surface the corpus carries: `?`
    // desugaring (Call + Match + Discriminant on `Result`), enum
    // construction (`Rvalue::Aggregate` for `PyResult::Ok`), iterator
    // calls, and `break`.
    let llbc = load_corpus();
    let graph = lower_function(llbc, "desugar_mix").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::desugar_mix");

    use majit_translate::model::{CallTarget, OpKind};
    let mut ctor_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            } = &op.kind
            {
                ctor_count += 1;
            }
        }
    }
    assert!(
        ctor_count >= 1,
        "expected at least one SyntheticTransparentCtor for PyResult::Ok"
    );
}

#[test]
fn lowers_tuple_roundtrip_with_symmetric_positional_field_reads() {
    // `tuple_roundtrip` constructs a real tuple `(a + b, a - b)` and
    // reads both `.0` / `.1` in the same function.  The lowering must
    // emit a `FieldRead __pos_<idx>` for those reads — symmetric with
    // the construction-side `FieldWrite __pos_<idx>` chain and carrying
    // the *same* `owner_root` — rather than collapsing every `.N` to
    // the synthetic-ctor base Variable.
    //
    // The same function also exercises the case that MUST still
    // collapse: each `a + b` / `a - b` / `pair.0 * pair.1` lowers
    // through a `*Checked` `(value, bool)` `BinaryOp`, whose `.0` reads
    // are `Field` projections of a `(i64, bool)` local.  Those locals
    // are bound by `Rvalue::BinaryOp`, never an `Aggregate`, so they
    // are absent from `positional_aggregate_locals` and their `.0`
    // reads fall through.  Asserting the FieldRead count is exactly the
    // two genuine tuple reads (not five) pins that boundary.
    use majit_translate::model::{CallTarget, OpKind};

    let llbc = load_corpus();
    let graph = lower_function(llbc, "tuple_roundtrip").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::tuple_roundtrip");

    let mut field_reads: Vec<(String, Option<String>)> = Vec::new();
    let mut field_writes: Vec<(String, Option<String>)> = Vec::new();
    let mut ctor_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } => {
                    field_reads.push((field.name.clone(), field.owner_root.clone()));
                }
                OpKind::FieldWrite { field, .. } => {
                    field_writes.push((field.name.clone(), field.owner_root.clone()));
                }
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                } => ctor_count += 1,
                _ => {}
            }
        }
    }

    // Exactly one synthetic ctor (the genuine tuple) and its two-field
    // `__pos_0` / `__pos_1` construction chain.  The per-shape tuple classdef
    // (default-ON) keys the owner on the tuple's element types, so the owner is
    // the suffixed `Tuple<i64,i64>` — the construction and projection sides must
    // agree on that exact spelling (the symmetry asserted below).
    assert_eq!(ctor_count, 1, "expected one tuple SyntheticTransparentCtor");
    field_writes.sort();
    assert_eq!(
        field_writes,
        vec![
            ("__pos_0".to_string(), Some("Tuple<i64,i64>".to_string())),
            ("__pos_1".to_string(), Some("Tuple<i64,i64>".to_string())),
        ],
        "tuple construction must emit a __pos_0 / __pos_1 FieldWrite chain"
    );

    // Exactly the two genuine tuple reads become FieldReads. The three
    // `*Checked` `.0` reads collapse, so a count of 2 (not 5) proves the
    // boundary holds.
    field_reads.sort();
    assert_eq!(
        field_reads,
        vec![
            ("__pos_0".to_string(), Some("Tuple<i64,i64>".to_string())),
            ("__pos_1".to_string(), Some("Tuple<i64,i64>".to_string())),
        ],
        "tuple reads must emit __pos_0 / __pos_1 FieldReads (owner_root \
         matching the FieldWrite chain) and *Checked .0 reads must collapse"
    );

    // Symmetry: every FieldRead pairs with an identically-keyed
    // FieldWrite (same name AND owner_root), so the read resolves the
    // value the construction stored.
    assert_eq!(
        field_reads, field_writes,
        "FieldRead keys must match the FieldWrite chain exactly"
    );
}

#[test]
fn unknown_function_name_errors() {
    let llbc = load_corpus();
    let err = lower_function(llbc, "no_such_function_anywhere").unwrap_err();
    assert!(matches!(err, LowerError::FunctionNotFound(_)));
}

#[test]
fn semantic_program_builder_lowers_every_corpus_function() {
    // Building a SemanticProgram from the corpus.ullbc should succeed
    // and surface every local function as a SemanticFunction with a
    // populated FunctionGraph.
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("builder");
    assert!(
        program.functions.len() >= 4,
        "expected at least the 4 corpus shapes, got {}",
        program.functions.len()
    );
    let names: std::collections::HashSet<_> =
        program.functions.iter().map(|f| f.name.as_str()).collect();
    // Names are crate-prefix-stripped (lib.rs
    // register_function_graph_alias walks bare leaf + crate aliases
    // off this shape).
    for required in [
        "straight_line_add",
        "branch_loop_sum",
        "strategy_len",
        "desugar_mix",
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    // The corpus declares one struct-shaped enum (Strategy + Token),
    // one type alias (PyResult), so we expect Strategy/Token and their
    // variant paths plus the leaf names.
    assert!(
        program.known_struct_names.contains("Strategy"),
        "expected Strategy in known_struct_names, got {:?}",
        program.known_struct_names
    );
    assert!(
        program
            .known_struct_names
            .contains("charon_corpus::Strategy::IntKeyed")
    );
    assert!(program.known_struct_names.contains("Token"));
}

#[test]
fn enum_variant_by_discriminant_round_trips_against_variant_paths() {
    // The discriminant→variant-name map must parse Charon's
    // `{"Scalar":{"Signed"|"Unsigned":[w,"K"]}}` discriminants and key
    // each enum under both its qualified path and bare leaf. Validate
    // against the corpus' Strategy enum without hard-coding variant
    // counts: every name the map produced must have a matching
    // `Strategy::<name>` variant path in known_struct_names, and the
    // leaf key must mirror the qualified key.
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("builder");

    let by_leaf = program
        .enum_variant_by_discriminant
        .get("Strategy")
        .expect("Strategy discriminant map present under bare leaf");
    assert!(!by_leaf.is_empty(), "Strategy must carry discriminants");

    // Discriminant 0 .. N-1 are distinct (HashMap keys) and every value
    // names a real Strategy variant.
    for name in by_leaf.values() {
        let path = format!("charon_corpus::Strategy::{name}");
        assert!(
            program.known_struct_names.contains(&path),
            "discriminant map produced {name:?} with no matching variant path {path:?}"
        );
    }
    // At least the variant the sibling test pins must round-trip.
    assert!(
        by_leaf.values().any(|n| n == "IntKeyed"),
        "expected IntKeyed among Strategy discriminants, got {by_leaf:?}"
    );

    // Qualified-path key mirrors the bare-leaf key.
    let by_qualified = program
        .enum_variant_by_discriminant
        .get("charon_corpus::Strategy")
        .expect("Strategy discriminant map present under qualified path");
    assert_eq!(by_leaf, by_qualified, "leaf and qualified maps must match");
}

#[test]
fn front_graph_carries_no_synthesized_exception_edges() {
    // The MIR driver drops every Call / Assert / Drop `on_unwind`
    // successor (a Rust panic-cleanup path) and routes only to the
    // success continuation, because Python exceptions ride the
    // `Result<_, PyError>` Switch/Return edges as ordinary control flow —
    // never a Rust unwind. Lock that structurally on the FRONT flow graph
    // (NOT the jitcode, where can-raise is re-derived op-locally as
    // guard_no_exception and is orthogonally correct):
    //
    //   A. No lowered block carries a `LastException` exitswitch — the
    //      driver never synthesizes a typed try/except handler dispatch.
    //   B. Every edge into the canonical exceptblock is a bare
    //      panic-propagation raise (`UnwindResume` / `Abort` -> set_raise),
    //      so the count of blocks linking to the exceptblock equals the
    //      count of `UnwindResume` / `Abort` MIR terminators. A Call /
    //      Assert / Drop success block contributes zero such edges.
    use majit_charon_reader::ullbc::{TermKind, Unstructured};
    use majit_translate::front::mir::{LowerContext, lower_fun_decl};
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};

    let llbc = load_corpus();
    let mut checked = 0usize;
    let context = LowerContext::new(llbc);
    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let graph = lower_fun_decl(&context, fd)
            .unwrap_or_else(|e| panic!("{} failed to lower: {e}", fd.item_meta.name_path()));

        // Invariant A.  The iterator `next`-diamond rewrite
        // (`front::iter_next`) is the one sanctioned synthesized
        // exception edge: a `for x in it` loop's `StopIteration` catch
        // (the RPython `next` op raising at exhaustion).  Its block — the
        // one carrying the `[__iter_next]` op — legitimately closes with
        // `LastException`; every other block must still drop on_unwind
        // rather than lower a try/except.
        for b in &graph.blocks {
            let is_next_handler = b.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.len() == 1 && segments[0] == "__iter_next"
                )
            });
            if is_next_handler {
                continue;
            }
            assert!(
                b.exitswitch != Some(ExitSwitch::LastException),
                "{}: block {:?} carries a LastException exitswitch — a typed \
                 exception-handler edge was synthesized; the MIR driver must \
                 drop on_unwind, not lower it as try/except",
                graph.name,
                b.id,
            );
        }

        // Invariant B: no Call/Assert/Drop on_unwind edge leaks into the
        // front graph, i.e. every live edge into the exceptblock is a bare
        // panic-propagation raise (`UnwindResume` / `Abort` -> set_raise),
        // so the live count never EXCEEDS the MIR's raise terminators.  It
        // may be fewer: a graph that runs `clear_unreachable_blocks` (the
        // iterator `next`-diamond and `?` rewrites do) prunes the dead
        // panic-cleanup blocks the driver leaves unreachable — those
        // pruned blocks were already dead exceptblock edges, never live
        // control flow.  A leak, by contrast, sits in a REACHABLE success
        // block and would push the live count above the raise count.
        let raises_in_mir = body
            .body
            .iter()
            .filter(|blk| {
                matches!(
                    blk.term(llbc),
                    Ok(TermKind::UnwindResume)
                        | Ok(TermKind::UnwindTerminate)
                        | Ok(TermKind::Abort(_))
                        | Ok(TermKind::UndefinedBehavior)
                )
            })
            .count();
        let edges_into_exceptblock = graph
            .blocks
            .iter()
            .filter(|b| b.exits.iter().any(|l| l.target == graph.exceptblock))
            .count();
        assert!(
            edges_into_exceptblock <= raises_in_mir,
            "{}: {} block(s) link to the exceptblock but the MIR has only {} \
             UnwindResume/Abort terminator(s) — a Call/Assert/Drop on_unwind \
             edge leaked into the front graph",
            graph.name,
            edges_into_exceptblock,
            raises_in_mir,
        );
        checked += 1;
    }
    assert!(
        checked >= 4,
        "expected to lower at least the 4 corpus shapes, got {checked}",
    );
}

/// `branch_loop_sum` iterates `&[i64]`, so the element `[__iter_next]`
/// yields is an `i64` — the list's item repr, the way
/// `rlist.py ll_listnext` hands one back.
///
/// This is the corpus half of the element-type fix.  The fold used to
/// answer `Ref` for every container that was not `front::range_iter`'s
/// `range()` builtin, because the `iter` op carries the iterator and not
/// the container's item type — so this graph typed a raw `i64` into the
/// ref register bank.  `result_ty` is not a hint the rtyper can overrule:
/// `resolve_call_result_kind` consults `concretetype` only when
/// `result_ty` is `Unknown`, and `authoritative_result_types` stamps the
/// derived kind back over it.
#[test]
fn branch_loop_sum_next_yields_an_int_element() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");

    let element_types: Vec<ValueType> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } if segments.len() == 1 && segments[0] == "__iter_next" => Some(result_ty.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(
        element_types,
        vec![ValueType::Int],
        "the `&[i64]` element must keep its own kind, not be typed as a GC reference",
    );
}

/// `for &v in slice: &[i64]`: `ll_slice_getitem_fast_i` returns `usize`
/// (`rvec.rs` `VecItemKind::Int`). The item is Signed, so the front end
/// must `intmask` (`cast_uint_to_int`) the helper result the way indexed
/// `items[i]` already does (`assert_exchange_pair_item`). Without that
/// cast the annotator unions the signed `i64` argument of
/// `int_or_float_encode_int` with the unsigned helper result.
#[test]
fn pair_slice_iter_getitem_intmasks_a_signed_item() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};
    let graph = lower_function(load_corpus(), "branch_loop_sum").expect("lowering");
    let leaf = |op: &majit_translate::model::SpaceOperation| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty,
            ..
        } => segments
            .last()
            .map(|leaf| (leaf.clone(), args.clone(), result_ty.clone())),
        _ => None,
    };
    let ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
    let getitem_at = ops
        .iter()
        .position(|op| leaf(op).is_some_and(|(l, _, _)| l == "ll_slice_getitem_fast_i"))
        .expect("branch_loop_sum reads the slice item");
    let getitem_result = ops[getitem_at]
        .result
        .as_ref()
        .expect("getitem has a result");
    assert!(
        leaf(ops[getitem_at]).is_some_and(|(_, _, ty)| ty == ValueType::Unsigned),
        "ll_slice_getitem_fast_i returns usize"
    );
    let intmasked = ops.iter().any(|op| {
        leaf(op).is_some_and(|(l, args, result_ty)| {
            l == "intmask"
                && args.first().and_then(|a| a.as_variable()) == Some(getitem_result)
                && result_ty == ValueType::Int
        })
    });
    assert!(
        intmasked,
        "signed pair-slice iter item must intmask the usize helper result"
    );
}

/// The element `[__iter_next]` yields is a `&i64` for both of these, and the
/// two get there differently: `slice_of_refs_sum` iterates `&[&i64]`, whose
/// `core::slice::iter::Iter` yields `Option<&&i64>` — one reference the
/// iterator added over one the element owns — while `array_of_refs_sum`
/// iterates `[&i64; 3]` by value, whose `core::array::iter::IntoIter` yields
/// `Option<&i64>` with no reference of its own.
///
/// So neither a blanket peel nor a blanket keep answers both: peeling every
/// reference types the first element `Int`, and peeling one unconditionally
/// types the second `Int`.  Either way a pointer lands in the integer
/// register bank, which is why the decision reads the receiver's iterator
/// ADT rather than the payload's shape alone.
#[test]
fn a_reference_element_stays_a_reference_through_either_iterator() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};
    let llbc = load_corpus();

    for name in ["slice_of_refs_sum", "array_of_refs_sum"] {
        let graph = lower_function(llbc, name).expect("lowering");
        let element_types: Vec<ValueType> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty,
                    ..
                } if segments.len() == 1 && segments[0] == "__iter_next" => Some(result_ty.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            element_types,
            vec![ValueType::Ref(None)],
            "{name}: a `&i64` element is a pointer and must keep the ref bank",
        );
    }
}

/// `branch_loop_sum`'s `for &v in slice` lifts to the native `iter` +
/// `[__iter_next]` ops: Layer 3 of the iterator vertical replaces the
/// residual `Iterator::next()` call (an unregistered callee that would
/// make the rtyper census Skip) with the `next` op + a `LastException`
/// block, the way `front::iter_next` rewrites the `Option` match diamond.
#[test]
fn branch_loop_sum_lifts_next_to_iter_next_op() {
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");

    let mut iter_next_blocks = Vec::new();
    let mut residual_next = 0usize;
    for (i, b) in graph.blocks.iter().enumerate() {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.len() == 1 && segments[0] == "__iter_next" => {
                    iter_next_blocks.push(i);
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "next" => residual_next += 1,
                _ => {}
            }
        }
    }

    assert_eq!(
        iter_next_blocks.len(),
        1,
        "expected exactly one `[__iter_next]` op after the rewrite",
    );
    assert_eq!(
        residual_next, 0,
        "the residual `Iterator::next()` call must be replaced",
    );

    // The `[__iter_next]` block is a `canraise` block: `LastException`
    // exitswitch with a normal (Some) exit and a StopIteration (break)
    // exit.  No catch-all propagation edge — list `next` raises only
    // StopIteration.
    let a = iter_next_blocks[0];
    assert!(
        matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
        "the next block must close with LastException exits",
    );
    assert_eq!(
        graph.blocks[a].exits.len(),
        2,
        "normal -> Some, StopIteration -> break",
    );
    // Exactly one exit (the normal/Some arm) carries no exitcase.
    let normal = graph.blocks[a]
        .exits
        .iter()
        .filter(|l| l.exitcase.is_none())
        .count();
    assert_eq!(normal, 1, "exactly one non-exception (Some) exit");
    // The exceptblock gains no edge from the rewrite.
    assert!(
        !graph.blocks[a]
            .exits
            .iter()
            .any(|l| l.target == graph.exceptblock),
        "the next rewrite must not link to the exceptblock",
    );
}

/// `bool::then` and `bool::then_some` both lift to the short-circuit
/// `Option` diamond. `then` calls the closure's `call_once`; `then_some`
/// wraps an already-evaluated value and emits no `call_once`.
#[test]
fn bool_then_lifts_to_short_circuit_diamond() {
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
    let cases = [
        (
            "closure",
            "bool_then_closure",
            "then",
            1usize,
            "the residual `core::bool::then` call must be replaced by the diamond",
            "the then arm calls the closure's `call_once`",
            "the then arm builds `Some(payload)`",
            "the else arm builds `None`",
        ),
        (
            "some",
            "bool_then_some",
            "then_some",
            0usize,
            "the residual `core::bool::then_some` call must be replaced by the diamond",
            "the then_some arm wraps the eager value directly — no `call_once`",
            "the then arm builds `Some(value)`",
            "the else arm builds `None`",
        ),
    ];
    for (
        name,
        fn_name,
        residual_name,
        want_call_once,
        residual_msg,
        call_once_msg,
        some_msg,
        none_msg,
    ) in cases
    {
        let llbc = load_corpus();
        let graph = lower_function(llbc, fn_name).expect("lowering");
        let reachable = reachable_blocks(&graph);
        let mut residual = 0usize;
        let mut call_once = 0usize;
        let mut bool_branches = 0usize;
        let mut some_ctor = 0usize;
        let mut none_ctor = 0usize;
        for b in &graph.blocks {
            if !reachable.contains(&b.id) {
                continue;
            }
            if matches!(b.exitswitch, Some(ExitSwitch::Value(_))) {
                bool_branches += 1;
            }
            let mut last_disc: Option<i64> = None;
            for op in &b.operations {
                match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().map(String::as_str) == Some(residual_name)
                        && segments.iter().any(|s| s == "bool") =>
                    {
                        residual += 1
                    }
                    OpKind::Call {
                        target: CallTarget::Method { name: method, .. },
                        ..
                    } if method == "call_once" => call_once += 1,
                    OpKind::ConstInt(d) => last_disc = Some(*d),
                    OpKind::FieldWrite { field, .. } if field.name == "__discriminant" => {
                        match last_disc {
                            Some(1) => some_ctor += 1,
                            Some(0) => none_ctor += 1,
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(residual, 0, "case {name}: {residual_msg}");
        assert_eq!(call_once, want_call_once, "case {name}: {call_once_msg}");
        assert_eq!(
            bool_branches, 1,
            "case {name}: the call block closes with a single `bool(cond)` branch"
        );
        assert_eq!(some_ctor, 1, "case {name}: {some_msg}");
        assert_eq!(none_ctor, 1, "case {name}: {none_msg}");
    }
}

/// `option_question_mark`'s `let v = opt?` lifts the residual
/// `Try::branch(opt)` / `ControlFlow` diamond into a direct switch on
/// `opt.__discriminant`: `Some` extracts `opt.__pos_0` and continues,
/// `None` builds a normal `None` return value.
///
/// The owners are the per-instantiation `Option<i64>` root, not the bare
/// template: the fixture's own `Some(v + addend)` writes `__pos_0` under the
/// suffixed root, so a bare read would take the payload off a different
/// classdef than the one the producer wrote.
#[test]
fn option_question_mark_lifts_to_direct_option_switch() {
    use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, OpKind};
    const OPTION_ROOT: &str = "core::option::Option<i64>";
    const SOME_ROOT: &str = "core::option::Option<i64>::Some";
    let llbc = load_corpus();
    let graph = lower_function(llbc, "option_question_mark").expect("lowering");

    let reachable = reachable_blocks(&graph);

    let mut residual_branch = 0usize;
    let mut direct_option_switch = 0usize;
    let mut some_payload_reads = 0usize;
    let mut none_ctor = 0usize;
    for b in &graph.blocks {
        if !reachable.contains(&b.id) {
            continue;
        }
        let mut last_disc: Option<i64> = None;
        let mut option_disc_read = None;
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "branch" => residual_branch += 1,
                OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref() == Some(OPTION_ROOT) =>
                {
                    option_disc_read = op.result.clone();
                }
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some(SOME_ROOT) =>
                {
                    some_payload_reads += 1;
                }
                OpKind::ConstInt(d) => last_disc = Some(*d),
                OpKind::FieldWrite { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref() == Some(OPTION_ROOT)
                        && last_disc == Some(0) =>
                {
                    none_ctor += 1;
                }
                _ => {}
            }
        }
        if let (Some(ExitSwitch::Value(sw)), Some(disc)) = (&b.exitswitch, option_disc_read)
            && *sw == disc
        {
            let mut cases: Vec<i64> = b
                .exits
                .iter()
                .filter_map(|l| match &l.exitcase {
                    Some(ExitCase::Const(majit_translate::flowspace::model::ConstValue::Int(
                        i,
                    ))) => Some(*i),
                    _ => None,
                })
                .collect();
            cases.sort_unstable();
            if cases == [0, 1] {
                direct_option_switch += 1;
            }
        }
    }

    assert_eq!(
        residual_branch, 0,
        "the residual `Try::branch` call must be replaced",
    );
    assert_eq!(
        direct_option_switch, 1,
        "the rewrite must leave one direct Option discriminant switch",
    );
    assert!(
        some_payload_reads >= 1,
        "the Some arm extracts the Option payload",
    );
    assert_eq!(none_ctor, 1, "the None arm builds the normal None return");
}

/// A field read through a raw object pointer retains the pointee's declared
/// type. The frontend represents the narrowing explicitly so descriptor
/// lookup does not receive a classless instance.
#[test]
fn header_read_narrows_to_a_typed_field_read() {
    use majit_translate::model::{CallTarget, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "w_object_type").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::w_object_type");

    let mut narrows = 0usize;
    let mut typed_header_reads = 0usize;
    let mut untyped_field_reads = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if majit_translate::model::cast_instance_root(&op.kind)
                    == Some("ObjectHeader") =>
                {
                    narrows += 1
                }
                OpKind::FieldRead { field, .. } if field.name == "ob_type" => {
                    if field.owner_root.as_deref() == Some("ObjectHeader")
                        && field.owner_id.is_some()
                    {
                        typed_header_reads += 1;
                    } else {
                        untyped_field_reads += 1;
                    }
                }
                _ => {}
            }
        }
    }
    assert_eq!(narrows, 1, "the deref base is narrowed exactly once");
    assert_eq!(typed_header_reads, 1, "ob_type reads as a typed FieldRead");
    assert_eq!(
        untyped_field_reads, 0,
        "no classdef-less header read survives the narrow",
    );
}

/// A boxing allocation becomes `NewWithVtable` only after its class-static
/// address is known. Exercise the fusion directly so the unresolved and
/// resolved cases differ by that input alone.
#[test]
fn boxing_cluster_fuses_once_the_class_address_resolves() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let attrs = std::collections::HashMap::from([
        (
            "ObjectHeader".to_string(),
            vec![
                ("ob_type".to_string(), ValueType::Ref(None)),
                ("w_class".to_string(), ValueType::Ref(None)),
            ],
        ),
        (
            "W_IntObject".to_string(),
            vec![
                ("ob_header".to_string(), ValueType::Ref(None)),
                ("intval".to_string(), ValueType::Int),
            ],
        ),
    ]);

    let llbc = load_corpus();

    // Without a resolvable class address the cluster is left alone.
    let mut graph = lower_function(llbc, "w_new_int").expect("lowering");
    assert_eq!(
        majit_translate::model::fuse_boxing_alloc(&mut graph, &attrs),
        0,
        "an unresolvable class-static address declines the fuse, silently",
    );

    // Stand in for the driver-supplied static address.
    let mut graph = lower_function(llbc, "w_new_int").expect("lowering");
    let mut substituted = 0usize;
    for b in &mut graph.blocks {
        for op in &mut b.operations {
            let is_class_static = matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("INT_CLASS")
            );
            if is_class_static {
                op.kind = OpKind::ConstRefAddr(CLASS_ADDR);
                substituted += 1;
            }
        }
    }
    assert_eq!(
        substituted, 2,
        "ob_type and w_class each read the class static",
    );

    assert_eq!(
        majit_translate::model::fuse_boxing_alloc(&mut graph, &attrs),
        1,
        "the cluster fuses once the class address is a constant",
    );

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut residual_mallocs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } if field.name == "intval" => payload_stores += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("malloc_typed") => {
                    residual_mallocs += 1
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        fused,
        vec![("W_IntObject".to_string(), CLASS_ADDR)],
        "one NewWithVtable carrying the real class pointer",
    );
    // Two: the re-emitted store after the `NewWithVtable`, plus the original
    // aggregate store, which is dead but not yet swept — `fuse_boxing_alloc`
    // leaves that to the `remove_dead_aggregates` pass in
    // `simplify_lowered_graph`.
    assert_eq!(payload_stores, 2, "the intval payload store is re-emitted");
    assert_eq!(
        residual_mallocs, 0,
        "the malloc_typed call is consumed, not left residual",
    );
}

/// The production lowering receives class-static addresses through
/// `HostStaticAddrs`. It must both fuse the allocation and preserve the
/// static's declared `ClassObject` root for pointer-identity comparisons.
#[test]
fn boxing_cluster_fuses_from_the_host_supplied_class_address() {
    use majit_translate::front::mir::{LowerContext, lower_fun_decl_with_static_addrs};
    use majit_translate::model::{CallTarget, OpKind, ValueType};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let llbc = load_corpus();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &[("INT_CLASS", CLASS_ADDR)],
        ..Default::default()
    };

    let fd = llbc.local_fn("w_new_int").expect("w_new_int in corpus");
    let context = LowerContext::new(llbc);
    let graph = lower_fun_decl_with_static_addrs(&context, fd, static_addrs).expect("lowering");

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut residual_mallocs = 0usize;
    let mut residual_class_reads = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } if field.name == "intval" => payload_stores += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => match segments.last().map(String::as_str) {
                    Some("malloc_typed") => residual_mallocs += 1,
                    Some("INT_CLASS") => residual_class_reads += 1,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    assert_eq!(
        fused,
        vec![("W_IntObject".to_string(), CLASS_ADDR)],
        "the real lowering fuses to one NewWithVtable carrying the class pointer",
    );
    assert_eq!(
        residual_mallocs, 0,
        "no residual lltype::malloc_typed survives",
    );
    assert_eq!(
        residual_class_reads, 0,
        "the class-static read resolved to the host address, not a residual call",
    );
    // One, not the two the hand-substituted sibling sees: the whole-graph
    // lowering runs `remove_dead_aggregates` after the fuse, so the
    // orphaned aggregate store is already swept here.
    assert_eq!(payload_stores, 1, "the intval payload store is re-emitted");

    // `w_number_add` keeps the class-static narrowing live after lowering.
    // Collect every narrowing so an incorrectly stamped root is reported as
    // its own entry rather than disappearing from an expected-root search.
    let add = llbc
        .local_fn("w_number_add")
        .expect("w_number_add in corpus");
    let add_graph =
        lower_fun_decl_with_static_addrs(&context, add, static_addrs).expect("lowering");
    let mut narrow_roots = std::collections::BTreeMap::new();
    let mut class_addr_narrowed = 0usize;
    for b in &add_graph.blocks {
        for op in &b.operations {
            let OpKind::Call {
                args, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            let Some(root) = majit_translate::model::cast_instance_root(&op.kind) else {
                continue;
            };
            let root = root.to_string();
            *narrow_roots.entry(root.clone()).or_insert(0usize) += 1;
            assert_eq!(
                result_ty,
                &ValueType::Ref(Some(root.clone())),
                "a narrow's result type is its own root",
            );
            // The narrow whose operand is the host-supplied class address
            // is the one this test is about.
            let narrows_class_addr = add_graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|p| {
                    p.result.as_ref()
                        == args
                            .first()
                            .and_then(majit_translate::model::LinkArg::as_variable)
                        && matches!(p.kind, OpKind::ConstRefAddr(a) if a == CLASS_ADDR)
                });
            if narrows_class_addr {
                class_addr_narrowed += 1;
                assert_eq!(
                    root, "ClassObject",
                    "the class-static address narrows to the corpus's own class root",
                );
            }
        }
    }
    assert_eq!(
        class_addr_narrowed, 1,
        "the host-supplied class address is narrowed exactly once",
    );
    assert_eq!(
        narrow_roots,
        std::collections::BTreeMap::from([
            ("ClassObject".to_string(), 1usize),
            ("ObjectHeader".to_string(), 2),
            ("W_IntObject".to_string(), 2),
        ]),
        "every narrow in the corpus carries a root the corpus itself declares",
    );
}

/// A header matching RPython's root object layout has a type pointer but no
/// per-instance class word. Lower it through the production metadata path so
/// both layout registration and allocation fusion are covered.
#[test]
fn boxing_cluster_fuses_where_the_header_declares_no_class_word() {
    use majit_translate::front::mir::{LowerContext, lower_fun_decl_with_static_addrs};
    use majit_translate::model::{CallTarget, OpKind};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let llbc = load_corpus();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &[("INT_CLASS", CLASS_ADDR)],
        ..Default::default()
    };

    let fd = llbc
        .local_fn("w_new_type_only_int")
        .expect("w_new_type_only_int in corpus");
    let context = LowerContext::new(llbc);
    let graph = lower_fun_decl_with_static_addrs(&context, fd, static_addrs).expect("lowering");

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut class_word_stores = 0usize;
    let mut residual_mallocs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } => match field.name.as_str() {
                    "intval" => payload_stores += 1,
                    "w_class" => class_word_stores += 1,
                    _ => {}
                },
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("malloc_typed") => {
                    residual_mallocs += 1
                }
                _ => {}
            }
        }
    }
    // The premise the arm rests on, asserted rather than assumed: nothing in
    // this cluster writes a class word, so a fuse here can only have come
    // through the no-class-word arm.
    assert_eq!(
        class_word_stores, 0,
        "the one-word header's cluster stores no class word",
    );
    assert_eq!(
        fused,
        vec![("W_TypeOnlyIntObject".to_string(), CLASS_ADDR)],
        "the one-word header's cluster fuses to one NewWithVtable",
    );
    assert_eq!(
        residual_mallocs, 0,
        "no residual lltype::malloc_typed survives",
    );
    assert_eq!(payload_stores, 1, "the intval payload store is re-emitted");
}

/// A pointer-identity type dispatch keeps its concrete arm as a direct call,
/// allowing graph discovery and inlining to continue through it.
#[test]
fn narrowing_chain_arm_lowers_to_a_direct_call() {
    use majit_translate::model::{CallTarget, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "w_number_add").expect("lowering");

    let mut direct_arm_calls = 0usize;
    let mut dyn_calls = 0usize;
    let mut header_reads = 0usize;
    let mut identity_tests = 0usize;
    let mut value_eqs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => match segments.last().map(String::as_str) {
                    Some("w_int_add") => direct_arm_calls += 1,
                    Some("__dyn_call") => dyn_calls += 1,
                    _ => {}
                },
                OpKind::FieldRead { field, .. } if field.name == "ob_type" => header_reads += 1,
                OpKind::BinOp { op, .. } if op == "is_" => identity_tests += 1,
                OpKind::BinOp { op, .. } if op == "eq" => value_eqs += 1,
                _ => {}
            }
        }
    }
    assert_eq!(direct_arm_calls, 1, "the taken arm is a direct call");
    assert_eq!(
        dyn_calls, 0,
        "no arm degrades to the __dyn_call placeholder"
    );
    assert_eq!(header_reads, 2, "both receivers' class words are read");
    assert_eq!(
        identity_tests, 2,
        "type(a) is type(b), then the per-class shortcut",
    );
    assert_eq!(
        value_eqs, 0,
        "a class-word compare is `is_`, not a value `eq`"
    );
}

/// Counts of the three lowerings a scalar `v[i]` can end in — the eager
/// `ArrayRead` `front::mir`'s `is_vec_index_call` emits, a residual
/// `Index::index` where that arm's width proof declined, or a residual
/// `<[T]>::get` — plus the element-bank and classdef facts that say whether
/// the projections off the element resolved.
///
/// `array_descr_keys` is parallel to `array_reads`: the `(array_type_id,
/// nolength)` pair the same `ArrayRead` carries. Together with the item bank
/// that pair is the whole descr key `codewriter::assembler`'s `ArrayRead` arm
/// hands to `arraydescrof`, so recording it lets a test mint the very descr
/// the bytecode emit would.
///
/// `residual_indexes` separates "declined" from "never reached the arm":
/// both leave no `ArrayRead`, and only the surviving call says which.
struct SlotReadShape {
    array_reads: Vec<majit_translate::model::ValueType>,
    array_descr_keys: Vec<(Option<String>, bool)>,
    residual_gets: usize,
    residual_indexes: usize,
    typed_discriminant_reads: usize,
    classdefless_discriminant_reads: usize,
    vec_helper_getitems: usize,
    residual_derefs: usize,
    slice_get_addrs: usize,
}

fn slot_read_shape(name: &str) -> SlotReadShape {
    use majit_translate::model::{CallTarget, OpKind};
    let graph = lower_function(load_corpus(), name).expect("lowering");
    let mut shape = SlotReadShape {
        array_reads: Vec::new(),
        array_descr_keys: Vec::new(),
        residual_gets: 0,
        residual_indexes: 0,
        typed_discriminant_reads: 0,
        classdefless_discriminant_reads: 0,
        vec_helper_getitems: 0,
        residual_derefs: 0,
        slice_get_addrs: 0,
    };
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::ArrayRead {
                    item_ty,
                    array_type_id,
                    nolength,
                    ..
                } => {
                    shape.array_reads.push(item_ty.clone());
                    shape
                        .array_descr_keys
                        .push((array_type_id.clone(), *nolength));
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("get") => shape.residual_gets += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    .last()
                    .is_some_and(|leaf| leaf.starts_with("ll_vec_getitem_fast_")) =>
                {
                    shape.vec_helper_getitems += 1
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("deref") => {
                    shape.residual_derefs += 1
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    .last()
                    .is_some_and(|leaf| leaf.starts_with("ll_slice_get_addr_")) =>
                {
                    shape.slice_get_addrs += 1
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "get" => shape.residual_gets += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if matches!(
                    segments.last().map(String::as_str),
                    Some("index" | "index_mut")
                ) =>
                {
                    shape.residual_indexes += 1
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "index" || name == "index_mut" => shape.residual_indexes += 1,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    if field.owner_root.is_some() && field.owner_id.is_some() {
                        shape.typed_discriminant_reads += 1;
                    } else {
                        shape.classdefless_discriminant_reads += 1;
                    }
                }
                _ => {}
            }
        }
    }
    shape
}

/// `&v[i]` on a `Vec<T>` whose `T` is a multi-word by-value ADT stored inline
/// declines rather than lowering to an `ArrayRead`. An `ArrayRead` here would
/// carry a descr that strides by **one word**: every index > 0 would address
/// the wrong element, and even index 0 would load the element's first word —
/// whichever of the tag or a payload field `repr(Rust)` puts there — and bank
/// it as a GC reference.
///
/// `front::mir`'s index arm admits only an element whose true width reaches the
/// descr, by one of two proofs: an ARRAY identity minted from the receiver
/// spelling (`narrow_item_array_type_id`, the narrow-int case), or an
/// addressable element — a scalar naming its own spelling, or a thin pointer,
/// which IS the one target word the identity-less descr assumes. A multi-word
/// ADT answers neither, so the call stays residual and real Rust computes the
/// element address at the real `size_of::<SlotValue>()` stride.
///
/// The descr behind the declined shape is minted below, from the key the leg
/// would have carried, and asserted to be one word wide. That width is the
/// whole reason the decline is required, so asserting it keeps the two facts
/// tied together.
///
/// `aggregate_slot_get` is the control. Both spellings are residual, so the
/// pair separates *which* call survives rather than a lowered read from a
/// residual one; the sibling
/// `a_one_word_vec_element_reads_through_the_vec_helper` supplies the positive
/// case where the index arm does lower the item read.
#[test]
fn an_aggregate_element_index_declines_instead_of_striding_by_one_word() {
    use majit_translate::model::ValueType;

    // `charon_corpus` declares its own `[workspace]` table and is not a
    // dependency of this crate, so `size_of::<SlotValue>()` is not callable
    // here. This mirrors the corpus declaration field for field — an `i64`
    // variant, a thin raw-pointer variant and a two-`i64` variant, none of
    // which leaves a niche free — so the `Pair` payload alone is two words
    // before any tag. The load-bearing claim is only "wider than one word",
    // which holds under any `repr(Rust)` layout choice.
    #[allow(dead_code)]
    enum SlotValueLayout {
        Int(i64),
        Object(*const u8),
        Pair { lhs: i64, rhs: i64 },
    }
    let elem_size = std::mem::size_of::<SlotValueLayout>();
    let word = majit_translate::layout::target_word_size();
    assert!(
        elem_size > word,
        "the fixture element must be wider than one word for the decline to \
         be the load-bearing outcome, got {elem_size} vs {word}",
    );

    let indexed = slot_read_shape("aggregate_slot_index");
    // The width proof declines, so no `ArrayRead` is emitted at all — and the
    // surviving `Vec::index` call says the arm was *reached* and refused,
    // rather than never matched.
    assert!(
        indexed.array_reads.is_empty(),
        "the aggregate element reaches no ArrayRead, got {:?}",
        indexed.array_reads,
    );
    assert_eq!(
        indexed.residual_indexes, 1,
        "the declined element leaves its `Index::index` call residual",
    );
    assert_eq!(
        indexed.residual_gets, 0,
        "the index spelling reaches no `get`",
    );
    // The discriminant read downstream of the element still resolves against
    // `SlotValue`'s own classdef rather than arriving as a bare pointer: the
    // decline costs the eager read, not the typing.
    assert_eq!(
        indexed.typed_discriminant_reads, 1,
        "the match reads __discriminant once, against a resolved owner",
    );
    assert_eq!(
        indexed.classdefless_discriminant_reads, 0,
        "no classdef-less discriminant read survives the residual call",
    );

    // The descr the leg would have carried, minted directly. `codewriter::
    // assembler`'s `ArrayRead` arm calls the module-level
    // `arraydescrof(item_ty, array_type_id, len_offset, callcontrol)` with
    // `len_offset = None` when `nolength` and `Some(0)` otherwise, and that
    // routes straight through `CallControl::arraydescrof_for_type`. The
    // `ir_type` it passes comes from the private
    // `value_type_to_ir_type_for_descr`, whose wildcard arm answers
    // `Type::Ref` for `ValueType::Ref(_)`.
    //
    // With `array_type_id: None`, `arraydescrof_concrete` never consults
    // `is_known_struct`, so no registered struct layout can reach this descr's
    // item size and the else arm sets `item_size = target_word_size()`. Both
    // assertions below still hold; they are why the arm above has to decline
    // rather than emit.
    let callcontrol = majit_translate::codewriter::call::CallControl::new();
    let descr = callcontrol.arraydescrof_for_type(
        &ValueType::Ref(None),
        &None,
        majit_ir::value::Type::Ref,
        Some(0),
    );
    let array_descr = descr
        .as_array_descr()
        .expect("arraydescrof_for_type must answer an ArrayDescr");
    assert_eq!(
        array_descr.item_size(),
        word,
        "an identity-less descr still strides by one word ({word}) while the \
         element is {elem_size} bytes wide — which is why no ArrayRead may \
         carry it over this element",
    );
    assert_eq!(
        array_descr.item_type(),
        majit_ir::value::Type::Ref,
        "and the single word it would load is banked as a GC reference, so \
         even at index 0 the backend would hand a non-pointer word to the \
         ref bank",
    );

    // The control: an aggregate-reference payload has no scalar bank or
    // element-layout descriptor, so the `get` spelling must stay residual too.
    let got = slot_read_shape("aggregate_slot_get");
    assert_eq!(
        got.residual_gets, 1,
        "the `get` spelling leaves its call residual, so real Rust computes \
         the element address at the real stride",
    );
    assert!(
        got.array_reads.is_empty(),
        "the `get` spelling emits no ArrayRead, got {:?}",
        got.array_reads,
    );
}

/// `v[1]` over a `Vec<char>` reaches the index arm and lowers to an
/// int-banked `ArrayRead` whose descr strides by the 4-byte `char`. Left
/// residual, the call returned a `&char` reference and the following `*`
/// collapsed onto it, so the `match` switched on a Ref and `flatten`
/// rejected the switch.
#[test]
fn a_char_element_indexes_to_an_int_banked_array_read() {
    use majit_translate::model::ValueType;

    let indexed = slot_read_shape("char_slot_index");
    assert_eq!(
        indexed.residual_indexes, 0,
        "the char element leaves no residual `Index::index` call",
    );
    assert_eq!(
        indexed.array_reads,
        vec![ValueType::Int],
        "the char element reads as one ArrayRead in the int bank",
    );
    let (array_type_id, _) = indexed.array_descr_keys[0].clone();
    let callcontrol = majit_translate::codewriter::call::CallControl::new();
    let descr = callcontrol.arraydescrof_for_type(
        &ValueType::Int,
        &array_type_id,
        majit_ir::value::Type::Int,
        None,
    );
    let array_descr = descr
        .as_array_descr()
        .expect("arraydescrof_for_type must answer an ArrayDescr");
    assert_eq!(
        array_descr.item_size(),
        4,
        "the char descr ({array_type_id:?}) strides by 4 bytes",
    );
}

/// `align.unwrap_or('>')` over an `Option<char>` joins the `Some` payload with
/// the literal default. A `char` is an int-kind scalar, so both links into the
/// join carry an Int: the payload read and the literal's `ConstInt` code point.
/// A literal lowered as a `__str_const` string would put a Ref on one link and
/// an Int on the other, which `flatten` cannot rename into one register.
#[test]
fn a_char_literal_default_joins_an_option_char_payload_in_the_int_bank() {
    use majit_translate::flowspace::model::Variable;
    use majit_translate::model::{LinkArg, OpKind, ValueType};

    let graph = lower_function(load_corpus(), "char_unwrap_or_join").expect("lowering");
    let producer = |var: &Variable| -> Option<&OpKind> {
        graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .find(|op| op.result.as_ref() == Some(var))
            .map(|op| &op.kind)
    };
    assert!(
        !graph.blocks.iter().flat_map(|b| b.operations.iter()).any(|op| matches!(
            &op.kind,
            OpKind::Call { target: majit_translate::model::CallTarget::FunctionPath { segments, .. }, .. }
                if segments.first().map(String::as_str) == Some("__str_const")
        )),
        "a char literal lowers to no __str_const",
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .any(|op| matches!(op.kind, OpKind::ConstInt(0x3e))),
        "the '>' default is the Int code point 0x3e",
    );
    // Every link argument into a block input produced by the literal or by
    // the `Some.__pos_0` payload read is int-kind, and at least one input
    // receives both — the `unwrap_or` join.
    let mut joins = 0;
    for target in &graph.blocks {
        for (slot, _) in target.inputargs.iter().enumerate() {
            let mut kinds = Vec::new();
            for block in &graph.blocks {
                for link in block.exits.iter().filter(|l| l.target == target.id) {
                    let Some(LinkArg::Value(v)) = link.args.get(slot) else {
                        continue;
                    };
                    match producer(v) {
                        Some(OpKind::ConstInt(0x3e)) => kinds.push(("literal", ValueType::Int)),
                        Some(OpKind::FieldRead { field, ty, .. }) if field.name == "__pos_0" => {
                            kinds.push(("payload", ty.clone()))
                        }
                        _ => {}
                    }
                }
            }
            if kinds.iter().any(|(k, _)| *k == "literal")
                && kinds.iter().any(|(k, _)| *k == "payload")
            {
                joins += 1;
                assert!(
                    kinds.iter().all(|(_, ty)| *ty == ValueType::Int),
                    "both links into the unwrap_or join are int-kind, got {kinds:?}",
                );
            }
        }
    }
    assert_eq!(
        joins, 1,
        "one block input joins the payload and the literal default"
    );
}

/// The same pair over `Vec<i64>`, whose one-word items make the receiver the
/// address of a raw `{ptr, len, cap}` header. The index spelling reads the
/// item through `ll_vec_getitem_fast_i`, not an `ArrayRead` on the header.
/// The `get` spelling goes through the `Vec` deref to the `(items, length)`
/// slice pair and reads the item address with `ll_slice_get_addr_i`; neither
/// the deref nor the `get` stays a residual call.
#[test]
fn a_one_word_vec_element_reads_through_the_vec_helper() {
    let indexed = slot_read_shape("scalar_slot_index");
    assert_eq!(
        indexed.vec_helper_getitems, 1,
        "an i64 element reads through one `ll_vec_getitem_fast_i` call",
    );
    assert!(
        indexed.array_reads.is_empty(),
        "no ArrayRead addresses the Vec header, got {:?}",
        indexed.array_reads,
    );
    assert_eq!(indexed.residual_gets, 0, "the index spelling has no `get`");

    let got = slot_read_shape("scalar_slot_get");
    assert_eq!(
        got.residual_derefs, 0,
        "the `Vec` deref behind the scalar `get` spelling is the slice pair",
    );
    assert_eq!(got.residual_gets, 0, "`get` over the pair is lowered");
    assert_eq!(got.slice_get_addrs, 1);
    assert_eq!(got.vec_helper_getitems, 0);
}

/// A borrowed primitive banks by its container, not by its own type.
///
/// `charon-corpus` §10's three shapes each put a shared borrow of a primitive
/// in a payload position, and all three serialize that borrow identically, so
/// no predicate over the payload's own type separates them:
///
/// | shape                        | payload         | reached through            |
/// |------------------------------|-----------------|----------------------------|
/// | `slice_get_tag_dispatch`     | `Option<&u8>`   | `<[T]>::get` then `?`      |
/// | `range_start_index`          | `Bound<&usize>` | `RangeBounds::start_bound` |
/// | `borrowed_byte_fields_alias` | `&u8`           | a struct field             |
///
/// The first two are enum-variant payloads, reached by matching on them, so
/// the borrow belongs to the match rather than to the program — and a sibling
/// arm supplying the merged value by value (`Bound::Unbounded => 0`) forces
/// one bank across the merge.  The third is a reference the program declared
/// and stores, which `ptr::eq` compares by address, so it keeps the ref bank.
///
/// Each shape falls to a different wrong answer, which is why all three are
/// asserted together: never peeling types the first `Ref`; peeling only the
/// `?`-desugaring shells types the second `Ref`, because `Bound` is not one;
/// peeling every borrowed primitive types the third `Unsigned`.
/// `jtransform.py rewrite_op_cast_bool_to_int` deletes every cast between integer primitives —
/// `rewrite_op_cast_char_to_int`, `cast_int_to_uint`, `cast_uint_to_int` and
/// the rest are each `pass` — because the two share one register kind. A
/// numeric `From` reaches the same conversion through a call, and `core`
/// carries no graph body, so left alone it stays residual.
///
/// The float sibling is asserted in the same test because it is the only thing
/// stopping the rule from being read as "numeric `From` is always identity":
/// `f64::from(i32)` moves banks and has to keep its call.
#[test]
fn an_integer_widening_from_aliases_but_a_float_one_does_not() {
    use majit_translate::{CallTarget, OpKind};
    let llbc = load_corpus();

    let numeric_from_calls = |name: &str| -> usize {
        let graph = lower_function(llbc, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => {
                    segments.iter().any(|s| s == "num")
                        && segments.last().map(String::as_str) == Some("from")
                }
                _ => false,
            })
            .count()
    };

    assert_eq!(
        numeric_from_calls("widening_int_from"),
        0,
        "`u32::from(u16)` is a no-op in the Int bank and keeps no call",
    );
    assert_eq!(
        numeric_from_calls("widening_float_from"),
        1,
        "`f64::from(i32)` crosses banks, so it is a conversion and keeps its call",
    );

    // The alias has to bind the destination to the argument, not drop it: a
    // lowering that discarded the operand would also report zero calls.
    let graph = lower_function(llbc, "widening_int_from").expect("lowering");
    let adds = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op.contains("add")))
        .count();
    assert_eq!(adds, 1, "the widened value still reaches the `+ 1`");
}

#[test]
fn a_borrowed_primitive_banks_by_its_container() {
    use majit_translate::model::{OpKind, ValueType};
    let llbc = load_corpus();

    // `__discriminant` is the tag read the match itself needs, not a payload.
    let payloads = |name: &str| -> Vec<(String, ValueType)> {
        let graph = lower_function(llbc, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name != "__discriminant" => {
                    Some((field.owner_root.clone().unwrap_or_default(), ty.clone()))
                }
                _ => None,
            })
            .collect()
    };

    assert_eq!(
        payloads("slice_get_tag_dispatch"),
        vec![(
            "core::option::Option<u8>::Some".to_string(),
            ValueType::Unsigned
        )],
        "the `?` payload of an `Option<&u8>` is the byte, not a pointer to it",
    );
    assert_eq!(
        payloads("range_start_index"),
        vec![
            ("Bound<usize>::Included".to_string(), ValueType::Unsigned),
            ("Bound<usize>::Excluded".to_string(), ValueType::Unsigned),
        ],
        "a `Bound` payload is an enum variant's too, though no `?` produces it",
    );
    assert_eq!(
        payloads("borrowed_byte_fields_alias"),
        vec![
            ("BorrowedByte".to_string(), ValueType::Ref(None)),
            ("BorrowedByte".to_string(), ValueType::Ref(None)),
        ],
        "a struct's `&u8` field is a pointer the program stores and compares",
    );
}

/// `stored` is `new_arg`, or a block input that a link feeds with `new_arg`
/// (one hop is what the slice replace splits into).
fn operand_is_new_argument(
    graph: &majit_translate::model::FunctionGraph,
    new_arg: &majit_translate::flowspace::model::Variable,
    stored: &majit_translate::flowspace::model::Variable,
) -> bool {
    if stored == new_arg {
        return true;
    }
    graph.blocks.iter().any(|block| {
        block.exits.iter().any(|link| {
            let target = graph.block(link.target);
            link.args
                .iter()
                .zip(target.inputargs.iter())
                .any(|(arg, input)| arg.as_variable() == Some(new_arg) && input == stored)
        })
    })
}

/// `mem::replace(&mut place, new)` is a read of `place` then a store of `new`.
/// The read's variable is what the function returns; the store's value is the
/// new argument.
#[test]
fn mem_replace_field_and_slice_element_read_then_store() {
    use majit_translate::model::OpKind;
    let llbc = load_corpus();

    let assert_exchange = |name: &str, read_is_field: bool| {
        let graph = lower_function(llbc, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut read_at = None;
        let mut write_at = None;
        let mut read_result = None;
        let mut write_value = None;
        let mut replace_calls = 0usize;
        let mut step = 0usize;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::FieldRead { .. } if read_is_field => {
                        read_at = Some(step);
                        read_result = op.result.clone();
                    }
                    OpKind::ArrayRead { .. } if !read_is_field => {
                        read_at = Some(step);
                        read_result = op.result.clone();
                    }
                    OpKind::FieldWrite { value, .. } if read_is_field => {
                        write_at = Some(step);
                        write_value = Some(value.clone());
                    }
                    OpKind::ArrayWrite { value, .. } if !read_is_field => {
                        write_at = Some(step);
                        write_value = Some(value.clone());
                    }
                    OpKind::Call { target, .. } => {
                        if format!("{target:?}").contains("replace") {
                            replace_calls += 1;
                        }
                    }
                    _ => {}
                }
                step += 1;
            }
        }
        assert_eq!(replace_calls, 0, "{name} still calls mem::replace");
        let kinds: Vec<String> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .map(|op| format!("{:?}", op.kind).chars().take(80).collect())
            .collect();
        let read_at = read_at.unwrap_or_else(|| panic!("{name} no read: {kinds:?}"));
        let write_at = write_at.unwrap_or_else(|| panic!("{name} no store: {kinds:?}"));
        assert!(
            read_at < write_at,
            "{name} store at {write_at} precedes read at {read_at}: {kinds:?}"
        );
        let read_result = read_result.unwrap_or_else(|| panic!("{name} no read: {kinds:?}"));
        let write_value = write_value.unwrap_or_else(|| panic!("{name} no store: {kinds:?}"));
        // `new` is the last parameter. A later block may take it as its own
        // input; the store must be that parameter, not some other operand.
        let new_arg = graph
            .block(graph.startblock)
            .inputargs
            .last()
            .cloned()
            .unwrap_or_else(|| panic!("{name} has no new argument"));
        let stored = write_value
            .as_variable()
            .unwrap_or_else(|| panic!("{name} store is not a variable: {kinds:?}"));
        assert!(
            operand_is_new_argument(&graph, &new_arg, stored),
            "{name} store operand is not the new argument: {kinds:?}"
        );
        assert_ne!(
            write_value.as_variable(),
            Some(&read_result),
            "{name} stores the old value back"
        );
        let returned = graph.blocks.iter().flat_map(|b| &b.exits).any(|link| {
            link.target == graph.returnblock
                && link
                    .args
                    .iter()
                    .any(|arg| arg.as_variable() == Some(&read_result))
        });
        assert!(returned, "{name} does not return the old value");
    };

    assert_exchange("replace_field", true);
    assert_exchange_pair_item("replace_elem");
}

/// `mem::replace(&mut items[i], new)` over a `&mut [i64]` pair: the old item
/// is read with `ll_slice_getitem_fast_i` before `new` is stored with
/// `ll_slice_setitem_fast_i`, and the old item is returned.  Both words pass
/// through the `usize` the helpers take: `intmask` on the way out, `r_uint` on
/// the way in.
fn assert_exchange_pair_item(name: &str) {
    use majit_translate::model::{CallTarget, OpKind};
    let graph = lower_function(load_corpus(), name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let leaf = |op: &majit_translate::model::SpaceOperation| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => segments.last().map(|leaf| (leaf.clone(), args.clone())),
        _ => None,
    };
    let ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
    let write_at = ops
        .iter()
        .position(|op| leaf(op).is_some_and(|(l, _)| l == "ll_slice_setitem_fast_i"))
        .unwrap_or_else(|| panic!("{name} no item store"));
    // The exchange's read is the last item read before the store; the
    // borrow `&mut items[i]` may read the item once more before it.
    let read_at = ops[..write_at]
        .iter()
        .rposition(|op| leaf(op).is_some_and(|(l, _)| l == "ll_slice_getitem_fast_i"))
        .unwrap_or_else(|| panic!("{name} no item read before the store"));
    assert!(read_at < write_at, "{name} stores before it reads");
    let producer = |v: &majit_translate::flowspace::model::Variable| {
        ops.iter()
            .find(|op| op.result.as_ref() == Some(v))
            .and_then(|op| leaf(op))
    };
    let new_arg = graph
        .block(graph.startblock)
        .inputargs
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("{name} has no new argument"));
    let (_, write_args) = leaf(ops[write_at]).expect("store is a call");
    let stored = write_args[2]
        .as_variable()
        .expect("stored word is a variable");
    let (retype, retype_args) = producer(stored).expect("stored word is retyped");
    assert_eq!(retype, "r_uint");
    assert!(
        operand_is_new_argument(&graph, &new_arg, retype_args[0].as_variable().unwrap()),
        "{name} store operand is not the new argument"
    );
    let read_result = ops[read_at].result.clone().expect("read has a result");
    let returned = graph.blocks.iter().flat_map(|b| &b.exits).any(|link| {
        link.target == graph.returnblock
            && link.args.iter().any(|arg| {
                arg.as_variable()
                    .and_then(|v| producer(v))
                    .is_some_and(|(l, a)| {
                        l == "intmask" && a[0].as_variable() == Some(&read_result)
                    })
            })
    });
    assert!(returned, "{name} does not return the old item");
}

/// `let r = &mut slot; mem::replace(&mut *r, new); *r` returns `new`.
/// The reborrow's read follows the local the replace wrote.
#[test]
fn mem_replace_reborrow_then_read_returns_new() {
    let llbc = load_corpus();
    let graph =
        lower_function(llbc, "replace_reborrow_then_read").unwrap_or_else(|e| panic!("{e}"));
    let new_arg = graph
        .block(graph.startblock)
        .inputargs
        .last()
        .cloned()
        .expect("new argument");
    let returned = graph.blocks.iter().flat_map(|b| &b.exits).any(|link| {
        link.target == graph.returnblock
            && link
                .args
                .iter()
                .any(|arg| arg.as_variable() == Some(&new_arg))
    });
    let kinds: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .map(|op| format!("{:?}", op.kind).chars().take(80).collect())
        .collect();
    assert!(returned, "reborrow read did not return new: {kinds:?}");
    let replace_calls = kinds.iter().filter(|k| k.contains("replace")).count();
    assert_eq!(replace_calls, 0, "still calls mem::replace: {kinds:?}");
}

/// A multi-word value moves one field at a time. `TwoWords` is two `i64`
/// fields (`getfield` / `setfield`). `WordUnion` is a 16-byte enum: the tag
/// and each non-overlapping payload field are `getfield` / `setfield`.
#[test]
fn mem_replace_of_a_multi_word_value_is_field_wise() {
    use majit_translate::model::OpKind;
    let llbc = load_corpus();

    let field_names = |name: &str, want_read: bool| -> Vec<String> {
        let graph = lower_function(llbc, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut names = Vec::new();
        let mut replace_calls = 0usize;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::FieldRead { field, .. } if want_read => names.push(field.name.clone()),
                    OpKind::FieldWrite { field, .. } if !want_read => {
                        names.push(field.name.clone())
                    }
                    OpKind::Call { target, .. } => {
                        if format!("{target:?}").contains("replace")
                            || format!("{target:?}").contains("swap")
                            || format!("{target:?}").contains("take")
                        {
                            replace_calls += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(
            replace_calls, 0,
            "{name} still calls mem::replace/swap/take"
        );
        names.sort();
        names
    };

    assert_eq!(
        field_names("replace_two_words", true),
        vec![
            "hi".to_string(),
            "hi".to_string(),
            "lo".to_string(),
            "lo".to_string()
        ],
        "replace reads each field of the slot and of the new value"
    );
    assert_eq!(
        field_names("replace_two_words", false),
        vec![
            "hi".to_string(),
            "hi".to_string(),
            "lo".to_string(),
            "lo".to_string()
        ],
        "replace writes each field of the slot and of the saved value"
    );
    assert_eq!(
        field_names("swap_two_words", true),
        vec![
            "hi".to_string(),
            "hi".to_string(),
            "lo".to_string(),
            "lo".to_string()
        ],
    );
    assert_eq!(
        field_names("take_two_words", true),
        vec![
            "hi".to_string(),
            "hi".to_string(),
            "lo".to_string(),
            "lo".to_string()
        ],
        "take reads the slot and the Default value"
    );

    let odd = lower_function(llbc, "take_odd_default").unwrap_or_else(|e| panic!("{e}"));
    let mut called_default = false;
    let mut wrote = Vec::new();
    for block in &odd.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::Call { target, .. } => {
                    let rendered = format!("{target:?}");
                    assert!(
                        !rendered.contains("take"),
                        "take_odd_default still calls take: {rendered}"
                    );
                    if rendered.contains("default") {
                        called_default = true;
                    }
                }
                OpKind::FieldWrite { field, .. } => wrote.push(field.name.clone()),
                _ => {}
            }
        }
    }
    assert!(
        called_default,
        "take_odd_default did not call Default::default"
    );
    wrote.sort();
    assert!(
        wrote.iter().filter(|name| name.as_str() == "lo").count() >= 1
            && wrote.iter().filter(|name| name.as_str() == "hi").count() >= 1,
        "take_odd_default did not store the default fields: {wrote:?}"
    );

    let names = field_names("replace_word_union", true);
    assert!(
        names.iter().any(|name| name == "__discriminant"),
        "enum exchange reads the tag, got {names:?}"
    );
    assert!(
        names.iter().any(|name| name == "__pos_0"),
        "enum exchange reads a payload field, got {names:?}"
    );

    let wide = lower_function(llbc, "replace_wide_payload").unwrap_or_else(|e| panic!("{e}"));
    let wide_calls: Vec<String> = wide
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call { target, .. } => Some(format!("{target:?}")),
            _ => None,
        })
        .collect();
    assert!(
        wide_calls.iter().any(|call| call.contains("replace")),
        "a u128 variant field makes move_plan None, so replace stays: {wide_calls:?}"
    );
}

/// `mem::replace` of a `Dynamic`-like enum reached through `Box::as_mut`.
/// The stores are the variant fields (`store_enum_variant`). The caller's
/// write set names `__discriminant` and the payload. No `mem::replace` call
/// remains.
#[test]
fn mem_replace_through_box_deref_names_enum_fields() {
    use majit_ir::descr::OopSpecIndex;
    use majit_ir::effectinfo::DescrSetMember;
    use majit_ir::value::Type;
    use majit_translate::CallPath;
    use majit_translate::call::{AnalysisCache, CallControl};
    use majit_translate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("builder");

    let assert_lowered = |name: &str| {
        let func = program
            .functions
            .iter()
            .find(|f| f.name == name || f.name.ends_with(&format!("::{name}")))
            .unwrap_or_else(|| panic!("missing {name}"));
        let mut replace_calls = 0usize;
        let mut writes = Vec::new();
        for block in &func.graph().blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::FieldWrite { field, .. } => writes.push(field.name.clone()),
                    OpKind::Call { target, .. } => {
                        if format!("{target:?}").contains("replace") {
                            replace_calls += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(replace_calls, 0, "{name} still calls mem::replace");
        assert!(
            writes.iter().any(|n| n == "__discriminant"),
            "{name} writes no tag: {writes:?}"
        );
        assert!(
            writes.iter().any(|n| n.starts_with("__pos_")),
            "{name} writes no payload: {writes:?}"
        );

        let mut cc = CallControl::new();
        cc.set_struct_fields(program.struct_fields.clone());
        let path = CallPath::from_segments([name]);
        cc.register_function_graph(path.clone(), func.graph().clone());
        cc.add_candidate_graph(path);
        let mut cache = AnalysisCache::default();
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::function_path([name]),
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
        };
        // `getcalldescr` compares these actual kinds with `FUNC.ARGS`.
        // `index: usize` is `int` (`getkind`); the pointer args stay `ref`.
        let start = func.graph().block(func.graph().startblock);
        let arg_types: Vec<Type> = start
            .inputargs
            .iter()
            .map(|arg| {
                let ty = start.operations.iter().find_map(|op| match &op.kind {
                    OpKind::Input { ty, .. } if op.result.as_ref() == Some(arg) => Some(ty),
                    _ => None,
                });
                match ty {
                    Some(
                        ValueType::Int
                        | ValueType::Unsigned
                        | ValueType::Bool
                        | ValueType::SingleFloat,
                    ) => Type::Int,
                    Some(ValueType::Float) => Type::Float,
                    _ => Type::Ref,
                }
            })
            .collect();
        let descriptor = cc.getcalldescr(
            &op,
            arg_types,
            Type::Ref,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        let named: Vec<String> = descriptor
            .extra_info
            .descr_set_keys
            .iter()
            .flat_map(|keys| keys.write_fields.iter())
            .filter_map(|member| match member {
                DescrSetMember::Field { field_name, .. } => Some(field_name.clone()),
                _ => None,
            })
            .collect();
        assert!(
            named.iter().any(|n| n.contains("__discriminant")),
            "{name} write set {named:?}"
        );
        assert!(
            named.iter().any(|n| n.contains("__pos_")),
            "{name} write set {named:?}"
        );
    };

    assert_lowered("replace_boxed_held");
    assert_lowered("replace_boxed_held_call");
    assert_lowered("replace_indexed_box");
    assert_lowered("replace_indexed_box_call");
    assert_lowered("replace_boxed_dynlike");
}

/// `*held = i64` through `&mut HeldUnion` is `setfield` of
/// `HeldUnion::Int.__pos_0`. The SSA dump of that graph carries no
/// `__deref_write` symbol for the assembler to resolve.
#[test]
fn store_through_union_int_is_setfield_not_deref_write() {
    use majit_translate::model::OpKind;
    let llbc = load_corpus();
    let graph = lower_function(llbc, "store_held_int").unwrap_or_else(|e| panic!("{e}"));
    let mut ssa_dump = String::new();
    let mut writes = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            ssa_dump.push_str(&format!("{:?}\n", op.kind));
            if let OpKind::FieldWrite { field, .. } = &op.kind {
                writes.push((field.name.clone(), field.owner_root.clone()));
            }
        }
    }
    assert!(
        !ssa_dump.contains("__deref_write"),
        "deref-write symbol reached the SSA dump:\n{ssa_dump}"
    );
    assert!(
        writes.iter().any(|(name, owner)| {
            name == "__pos_0"
                && owner
                    .as_deref()
                    .is_some_and(|owner| owner.contains("HeldUnion::Int"))
        }),
        "expected HeldUnion::Int.__pos_0 setfield, writes={writes:?}\n{ssa_dump}"
    );
}

/// `slot.0 = HeldUnion::Int(value)` moves the enum into the inline field.
/// The field is not a pointer, so the graph has no `HeldCell.__pos_0` store
/// of the temporary's address.
#[test]
fn store_inline_enum_field_moves_the_variant() {
    use majit_translate::model::{ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "store_held_cell").unwrap_or_else(|e| panic!("{e}"));
    let mut writes = Vec::new();
    let mut switches = 0usize;
    for block in &graph.blocks {
        if matches!(block.exitswitch, Some(ExitSwitch::Value(_))) {
            switches += 1;
        }
        for op in &block.operations {
            if let OpKind::FieldWrite { field, .. } = &op.kind {
                writes.push((
                    field.name.clone(),
                    field.owner_root.clone().unwrap_or_default(),
                ));
            }
        }
    }
    assert!(
        switches >= 1,
        "inline enum move has no discriminant switch, writes={writes:?}"
    );
    assert!(
        writes.iter().all(|(_, owner)| !owner.contains("HeldCell")),
        "inline enum field stored as one HeldCell word: {writes:?}"
    );
    assert!(
        writes
            .iter()
            .any(|(name, owner)| name == "__discriminant" && owner.contains("HeldUnion")),
        "missing discriminant move, writes={writes:?}"
    );
    assert!(
        writes
            .iter()
            .any(|(name, owner)| name == "__pos_0" && owner.contains("HeldUnion::Int")),
        "missing Int payload move, writes={writes:?}"
    );
}

/// A whole `HeldUnion` move switches on `__discriminant`. The `Ref`
/// payload is read only in its own arm, so an `Int` value is never
/// loaded as a reference.
#[test]
fn whole_enum_move_switches_per_variant() {
    use majit_translate::model::{ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "replace_held_union").unwrap_or_else(|e| panic!("{e}"));
    let mut switch_blocks = Vec::new();
    let mut ref_blocks = Vec::new();
    let mut int_blocks = Vec::new();
    for block in &graph.blocks {
        if matches!(block.exitswitch, Some(ExitSwitch::Value(_))) {
            switch_blocks.push(block.id);
        }
        for op in &block.operations {
            if let OpKind::FieldRead { field, ty, .. } = &op.kind {
                if field.name != "__pos_0" {
                    continue;
                }
                let owner = field.owner_root.clone().unwrap_or_default();
                if owner.contains("HeldUnion::Ref") {
                    assert!(
                        matches!(ty, majit_translate::model::ValueType::Ref(_)),
                        "Ref payload read as {ty:?}"
                    );
                    ref_blocks.push(block.id);
                }
                if owner.contains("HeldUnion::Int") {
                    assert!(
                        matches!(
                            ty,
                            majit_translate::model::ValueType::Int
                                | majit_translate::model::ValueType::Unsigned
                        ),
                        "Int payload read as {ty:?}"
                    );
                    int_blocks.push(block.id);
                }
            }
        }
    }
    assert!(
        !switch_blocks.is_empty(),
        "whole-enum move has no discriminant switch"
    );
    assert!(!ref_blocks.is_empty(), "no HeldUnion::Ref.__pos_0 read");
    assert!(!int_blocks.is_empty(), "no HeldUnion::Int.__pos_0 read");
    assert!(
        ref_blocks
            .iter()
            .all(|block| !switch_blocks.contains(block)),
        "Ref payload read sits on the switch block: ref={ref_blocks:?} switch={switch_blocks:?}"
    );
    assert!(
        ref_blocks.iter().all(|block| !int_blocks.contains(block)),
        "Ref and Int payloads are read in one block: ref={ref_blocks:?} int={int_blocks:?}"
    );
}

/// `clear_inline_tag` borrows an inline `Vec<u8>` after a word field.
/// The index operand is that field, retargeted to the buffer word
/// (`vec_part = Buf`) or marked as the field's address. It is not a
/// load of the Vec's first word used as a pointer.
#[test]
fn clear_inline_tag_indexes_the_buffer_not_the_capacity_word() {
    use majit_translate::model::{OpKind, VecFieldPart};

    let llbc = load_corpus();
    let graph = lower_function(llbc, "clear_inline_tag").expect("lowering");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let tags_reads: Vec<_> = ops
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "tags"
            )
        })
        .collect();
    assert!(
        !tags_reads.is_empty(),
        "expected a tags field read; ops={ops:?}"
    );
    assert!(
        tags_reads.iter().all(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => {
                field.vec_part == Some(VecFieldPart::Buf) || field.taken_by_address
            }
            _ => false,
        }),
        "tags must be the buffer word or its address, not word 0; reads={tags_reads:?}"
    );
    for op in &ops {
        let OpKind::FieldRead { base, field, .. } = &op.kind else {
            continue;
        };
        if field.name != "buf" {
            continue;
        }
        let producer = ops.iter().find(|src| src.result.as_ref() == Some(base));
        if let Some(src) = producer
            && let OpKind::FieldRead { field: tags, .. } = &src.kind
            && tags.name == "tags"
        {
            assert!(
                tags.vec_part == Some(VecFieldPart::Buf) || tags.taken_by_address,
                "buf read off a capacity-word copy of tags"
            );
        }
    }
}

/// A flag const built the way `bitflags!` builds one — an associated const
/// initialised through a `const fn` constructor of a `repr(transparent)`
/// wrapper around a `repr(transparent)` wrapper around a `u16` — reads as the
/// prebuilt integer, not as a nullary call to the const's path that no host
/// symbol backs.
#[test]
fn a_transparent_flag_const_folds_to_its_integer() {
    use majit_translate::model::{CallTarget, OpKind};

    let graph = lower_function(load_corpus(), "code_flags_bits_or").expect("lowering");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .collect();
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments.last().map(String::as_str) == Some("FLAT")
        )),
        "the flag const lowers to no accessor call",
    );
    assert!(
        ops.iter()
            .any(|op| matches!(op.kind, OpKind::ConstUInt(0x100) | OpKind::ConstInt(0x100))),
        "the flag const is the integer 0x100",
    );
}

/// An array literal whose borrow reaches a slice argument through a copied
/// reference is a raw buffer: allocated, filled item by item, passed as
/// `(buffer, 2)` and freed at the return.
#[test]
fn an_array_borrowed_through_a_copied_reference_is_a_raw_buffer() {
    let calls = corpus_call_leaves("sum_of_array_literal");
    let count = |leaf: &str| calls.iter().filter(|(l, _)| l == leaf).count();
    assert_eq!(count("ll_slice_buffer_new_i"), 1, "{calls:?}");
    assert_eq!(count("ll_slice_setitem_fast_i"), 2, "{calls:?}");
    assert_eq!(count("ll_slice_buffer_free"), 1, "{calls:?}");
    assert!(
        calls.iter().any(|(l, n)| l == "sum_two_items" && *n == 2),
        "the slice argument is the (buffer, length) pair: {calls:?}"
    );
}

/// The leaf names and argument counts of `name`'s `FunctionPath` calls.
fn corpus_call_leaves(name: &str) -> Vec<(String, usize)> {
    use majit_translate::model::{CallTarget, OpKind};
    let graph = lower_function(load_corpus(), name).unwrap_or_else(|e| panic!("{name}: {e}"));
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => segments.last().map(|leaf| (leaf.clone(), args.len())),
            _ => None,
        })
        .collect()
}

/// A slice of pointer items is a pair of the reference kind: `reverse` and
/// the item read go through the `_r` helpers.
#[test]
fn a_pointer_item_slice_uses_the_reference_helpers() {
    let calls = corpus_call_leaves("reverse_then_first_ref");
    let has = |leaf: &str| calls.iter().any(|(l, _)| l == leaf);
    assert!(has("ll_slice_reverse_r"), "{calls:?}");
    assert!(has("ll_slice_getitem_fast_r"), "{calls:?}");
}

/// `v.extend_from_slice(s)` over a pair is `ll_extend` from `(items, length)`.
#[test]
fn a_vec_extends_from_a_pair_slice_through_the_vec_helper() {
    let calls = corpus_call_leaves("extend_vec_from_slice");
    assert!(
        calls
            .iter()
            .any(|(l, n)| l == "ll_vec_extend_from_slice_i" && *n == 3),
        "{calls:?}"
    );
    assert!(
        !calls.iter().any(|(l, _)| l == "extend_from_slice"),
        "{calls:?}"
    );
}

/// `s.get(1..).unwrap_or(&[]).len()` lowers the tail as a pair. Neither
/// `get` nor `unwrap_or` remains as a call.
#[test]
fn a_range_get_unwrap_or_lowers_to_the_pair() {
    let calls = corpus_call_leaves("tail_len");
    assert!(
        !calls
            .iter()
            .any(|(leaf, _)| leaf == "get" || leaf == "unwrap_or"),
        "{calls:?}"
    );
}

/// `&buf[..n]` over a pointer array local is a subslice of its raw buffer:
/// `(buffer, n)` reaches the callee.
#[test]
fn a_range_index_of_an_array_local_is_a_subslice_of_its_buffer() {
    let calls = corpus_call_leaves("first_of_array_prefix");
    let count = |leaf: &str| calls.iter().filter(|(l, _)| l == leaf).count();
    assert_eq!(count("ll_slice_buffer_new_r"), 1, "{calls:?}");
    assert_eq!(count("ll_slice_setitem_fast_r"), 2, "{calls:?}");
    assert_eq!(count("ll_slice_buffer_free"), 1, "{calls:?}");
    assert!(
        calls.iter().any(|(l, n)| l == "first_ref" && *n == 2),
        "{calls:?}"
    );
}

/// The attribute names `name`'s `FieldWrite`s and `FieldRead`s name.
fn corpus_field_names(name: &str) -> (Vec<String>, Vec<String>) {
    use majit_translate::model::OpKind;
    let graph = lower_function(load_corpus(), name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        match &op.kind {
            OpKind::FieldWrite { field, .. } => writes.push(field.name.clone()),
            OpKind::FieldRead { field, .. } => reads.push(field.name.clone()),
            _ => {}
        }
    }
    (writes, reads)
}

/// A pair slice item of a tuple is stored as its two words, the length
/// beside the pointer.
#[test]
fn a_tuple_stores_a_pair_slice_item_as_two_words() {
    let (writes, _) = corpus_field_names("tuple_a_slice");
    for name in ["__pos_0", "__pos_0.len", "__pos_1"] {
        assert!(writes.iter().any(|w| w == name), "{name}: {writes:?}");
    }
}

/// Reading the pair slice item back out of a tuple reads both words.
#[test]
fn a_pair_slice_tuple_item_reads_back_both_words() {
    let (_, reads) = corpus_field_names("item_of_tupled_slice");
    for name in ["__pos_0", "__pos_0.len"] {
        assert!(reads.iter().any(|r| r == name), "{name}: {reads:?}");
    }
    let calls = corpus_call_leaves("item_of_tupled_slice");
    assert!(
        calls.iter().any(|(l, _)| l == "ll_slice_getitem_fast_r"),
        "{calls:?}"
    );
}

/// A closure capturing a slice by reference stores the slice's two words,
/// since a reference aliases its referent.
#[test]
fn a_closure_capturing_a_slice_stores_both_words() {
    let (writes, _) = corpus_field_names("slice_through_a_closure");
    assert!(
        writes.iter().any(|w| writes.contains(&format!("{w}.len"))),
        "{writes:?}"
    );
}

/// The shaped tuple of a pair slice item registers the length word as a
/// field of its own. Shaped tuples are not stored ahead of the first
/// lookup; `FieldRows::get` derives the rows from the spelling the
/// lowering actually wrote.
#[test]
fn a_pair_slice_tuple_shape_registers_the_length_word() {
    use majit_translate::model::OpKind;
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("builder");
    let graph = lower_function(llbc, "tuple_a_slice").expect("tuple_a_slice");
    let shape = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } | OpKind::FieldRead { field, .. }
                if field.name == "__pos_0.len" =>
            {
                field.owner_root.clone()
            }
            _ => None,
        })
        .expect("tuple_a_slice names the length word");
    let rows = program
        .struct_fields
        .fields
        .get(&shape)
        .unwrap_or_else(|| panic!("no rows for {shape}"))
        .clone();
    assert_eq!(rows[0], ("__pos_0".to_string(), "usize".to_string()));
    assert_eq!(rows[1], ("__pos_0.len".to_string(), "usize".to_string()));
}

/// `Option<&T>` of a pointer item passed as a value is the item, or null
/// for `None`: `ll_slice_load_or_r` of the item address.
#[test]
fn a_pointer_item_reference_passed_as_a_value_is_the_item_or_null() {
    let calls = corpus_call_leaves("get_item_or_null");
    let has = |leaf: &str| calls.iter().any(|(l, _)| l == leaf);
    assert!(has("ll_slice_get_addr_r"), "{calls:?}");
    assert!(has("ll_slice_load_or_r"), "{calls:?}");
}

/// A closure body indexing the slice it captured reads the item through
/// the captured pair's item pointer.
#[test]
fn a_closure_indexes_its_captured_slice_through_the_pair() {
    use majit_translate::model::{CallTarget, OpKind};
    let llbc = load_corpus();
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path()
                .ends_with("index_through_a_closure::<Impl>::call")
        })
        .expect("closure body");
    let context = majit_translate::front::mir::LowerContext::new(llbc);
    let graph = majit_translate::front::mir::lower_fun_decl(&context, fd).expect("lowering");
    let ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments.last().map(String::as_str) == Some("ll_slice_getitem_fast_r")
        )),
        "{ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(op.kind, OpKind::ArrayRead { .. })),
        "{ops:?}"
    );
}

/// `CodeFlags::{contains,intersects,bitor}` on the real interpreter bodies
/// lower to integer ops. The external `bitflags` methods stay opaque in the
/// LLBC, so a residual call would name an unresolvable `CodeFlags` path.
#[test]
fn code_flags_methods_lower_to_integer_ops() {
    use majit_translate::model::{CallTarget, OpKind};

    let llbc = load_llbc(INTERPRETER_LLBC);
    let tails: &[&[&str]] = &[
        &["CodeFlags", "contains"],
        &["CodeFlags", "intersects"],
        &["CodeFlags", "bitor"],
    ];
    for name in [
        "fill_user_function_args",
        "pyre_interpreter::pyframe::code_flags_make_generator",
    ] {
        let graph = lower_named(llbc, name).unwrap_or_else(|e| panic!("lower {name}: {e}"));
        let calls: Vec<Vec<String>> = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                OpKind::Call {
                    target:
                        CallTarget::Method {
                            name,
                            receiver_root,
                            resolved_path,
                            ..
                        },
                    ..
                } => {
                    let mut segments = receiver_root
                        .as_ref()
                        .map(|root| vec![root.clone()])
                        .unwrap_or_default();
                    if let Some(path) = resolved_path {
                        segments = path.segments.clone();
                    }
                    segments.push(name.clone());
                    Some(segments)
                }
                _ => None,
            })
            .collect();
        for tail in tails {
            assert!(
                !calls.iter().any(|segments| {
                    segments.len() >= tail.len()
                        && segments[segments.len() - tail.len()..] == tail[..]
                }),
                "{name}: residual {tail:?} remains in {calls:?}"
            );
        }
        // The `*self` word is an integer, not the address of the flags field.
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .collect();
        for op in &ops {
            let OpKind::BinOp {
                op: label,
                lhs,
                rhs,
                ..
            } = &op.kind
            else {
                continue;
            };
            if label != "bitand" {
                continue;
            }
            for operand in [lhs, rhs] {
                let producer = ops.iter().find(|p| p.result.as_ref() == Some(operand));
                if let Some(OpKind::FieldRead { ty, field, .. }) = producer.map(|p| &p.kind) {
                    assert!(
                        matches!(ty, majit_translate::model::ValueType::Unsigned)
                            && !field.taken_by_address,
                        "{name}: bitand operand {field:?} reads {ty:?}"
                    );
                }
            }
        }
    }
}

/// `FrameBox::new` copies a by-value `PyFrame` with `core::ptr::write`.
/// That write becomes one field store per registered field.
#[test]
fn frame_box_new_ptr_write_lowers_to_field_stores() {
    use majit_translate::front::mir::lower_fun_decl;
    use majit_translate::model::{CallTarget, OpKind};

    let llbc = load_llbc(INTERPRETER_LLBC);
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .source_text
                .as_deref()
                .is_some_and(|text| text.starts_with("pub fn new(mut frame: PyFrame)"))
        })
        .expect("FrameBox::new");
    let context = lower_context_for(llbc);
    let graph = lower_fun_decl(&context, fd).expect("lower FrameBox::new");
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.iter().map(String::as_str).eq(["core", "ptr", "write"])
                )
            }),
        "FrameBox::new still has core::ptr::write"
    );
    let cast_results: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| {
            (majit_translate::model::cast_instance_root(&op.kind) == Some("PyFrame"))
                .then(|| op.result.as_ref().map(|var| var.id()))
                .flatten()
        })
        .collect();
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { base, field, .. }
                        if cast_results.contains(&base.id()) && field.name == "pycode"
                )
            }),
        "the PyFrame copy must store pycode into the cast destination"
    );
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. } if field.name == "ob_header"
                )
            }),
        "FrameBox::new must not store the inlined ob_header as one word"
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. }
                        if field.name == "w_class"
                            && field
                                .owner_root
                                .as_deref()
                                .is_some_and(|owner| owner.ends_with("PyObject"))
                )
            }),
        "FrameBox::new must copy ob_header.w_class through the inner struct descriptor"
    );
}

/// `FrameBox::new` stores the operand of `OwnerRootGuard::new(r)` in
/// `owner_root`. `frame_ptr`'s `Some` arm returns that word, and
/// `deref_mut` passes it on as the frame.
#[test]
fn frame_box_owner_root_word_roundtrips() {
    use majit_translate::front::mir::{LowerContext, lower_fun_decl};
    use majit_translate::model::{FunctionGraph, LinkArg, OpKind, SpaceOperation};

    let llbc = load_llbc(INTERPRETER_LLBC);
    let context = lower_context_for(llbc);

    fn value_id(arg: &LinkArg) -> Option<u64> {
        match arg {
            LinkArg::Value(var) => Some(var.id()),
            LinkArg::Const(_) => None,
        }
    }

    fn operand_ids(op: &SpaceOperation) -> Vec<u64> {
        match &op.kind {
            OpKind::Call { args, .. } => args.iter().filter_map(value_id).collect(),
            OpKind::FieldRead { base, .. } => vec![base.id()],
            OpKind::UnaryOp { operand, .. } => vec![operand.id()],
            OpKind::BinOp { lhs, rhs, .. } => vec![lhs.id(), rhs.id()],
            _ => Vec::new(),
        }
    }

    fn op_result<'a>(graph: &'a FunctionGraph, id: u64) -> Option<&'a SpaceOperation> {
        graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find(|op| op.result.as_ref().is_some_and(|var| var.id() == id))
        })
    }

    fn phi_sources(graph: &FunctionGraph, id: u64) -> Vec<u64> {
        let Some(block) = graph
            .blocks
            .iter()
            .find(|block| block.inputargs.iter().any(|var| var.id() == id))
        else {
            return Vec::new();
        };
        let pos = block
            .inputargs
            .iter()
            .position(|var| var.id() == id)
            .expect("input position");
        graph
            .blocks
            .iter()
            .flat_map(|pred| pred.exits.iter())
            .filter(|link| link.target == block.id)
            .filter_map(|link| link.args.get(pos).and_then(value_id))
            .collect()
    }

    /// Field read reached only by forwarding block arguments.
    fn pure_field_read<'a>(
        graph: &'a FunctionGraph,
        id: u64,
        seen: &mut Vec<u64>,
    ) -> &'a majit_translate::model::FieldDescriptor {
        assert!(!seen.contains(&id), "cycle at v{id}");
        seen.push(id);
        if let Some(op) = op_result(graph, id) {
            match &op.kind {
                OpKind::FieldRead { field, .. } => field,
                other => panic!("v{id} is {other:?}, not the owner_root word"),
            }
        } else {
            let srcs = phi_sources(graph, id);
            assert_eq!(
                srcs.len(),
                1,
                "v{id} is not a forward of one field read: {srcs:?}"
            );
            pure_field_read(graph, srcs[0], seen)
        }
    }

    fn ancestors(graph: &FunctionGraph, start: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if out.contains(&id) {
                continue;
            }
            out.push(id);
            if let Some(op) = op_result(graph, id) {
                stack.extend(operand_ids(op));
            } else {
                stack.extend(phi_sources(graph, id));
            }
        }
        out
    }

    fn defining_op<'a>(graph: &'a FunctionGraph, start: u64) -> &'a SpaceOperation {
        let mut id = start;
        let mut seen = Vec::new();
        loop {
            assert!(!seen.contains(&id), "cycle at v{id}");
            seen.push(id);
            if let Some(op) = op_result(graph, id) {
                return op;
            }
            let srcs = phi_sources(graph, id);
            assert_eq!(srcs.len(), 1, "v{id} has no single producer: {srcs:?}");
            id = srcs[0];
        }
    }

    fn lower_pyframe(llbc: &Llbc, context: &LowerContext<'_>, source: &str) -> FunctionGraph {
        let fd = llbc
            .iter_local_fns()
            .find(|fd| {
                fd.item_meta.name_path().contains("pyframe")
                    && fd
                        .item_meta
                        .source_text
                        .as_deref()
                        .is_some_and(|text| text.starts_with(source))
            })
            .unwrap_or_else(|| panic!("missing {source}"));
        lower_fun_decl(context, fd).unwrap_or_else(|err| panic!("lower {source}: {err}"))
    }

    fn calls_cast_int_to_ptr(graph: &FunctionGraph) -> bool {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| match &op.kind {
                OpKind::Call { target, .. } => {
                    call_target_text(target).ends_with("cast_int_to_ptr")
                }
                OpKind::UnaryOp { op, .. } => op == "cast_int_to_ptr",
                _ => false,
            })
    }

    let frame_ptr = lower_pyframe(&llbc, &context, "fn frame_ptr(&self)");
    let live = reachable_blocks(&frame_ptr);
    let mut arms = Vec::new();
    for block in &frame_ptr.blocks {
        if !live.contains(&block.id) {
            continue;
        }
        for link in &block.exits {
            if link.target != frame_ptr.returnblock {
                continue;
            }
            let arg = link
                .args
                .first()
                .and_then(value_id)
                .unwrap_or_else(|| panic!("frame_ptr return link has no value"));
            let mut seen = Vec::new();
            let field = pure_field_read(&frame_ptr, arg, &mut seen);
            arms.push((field.name.clone(), field.taken_by_address));
        }
    }
    assert!(
        arms.iter()
            .any(|(name, address)| name == "owner_root" && !address),
        "Some arm must return the owner_root word, not an address or another value: {arms:?}"
    );
    assert!(
        arms.iter().any(|(name, _)| name == "ptr"),
        "None arm must return the ptr field: {arms:?}"
    );
    assert!(
        !calls_cast_int_to_ptr(&frame_ptr),
        "frame_ptr still casts the guard word through the integer bank"
    );

    let deref_mut = lower_pyframe(&llbc, &context, "fn deref_mut(&mut self)");
    let live = reachable_blocks(&deref_mut);
    let mut deref_returns = Vec::new();
    for block in &deref_mut.blocks {
        if !live.contains(&block.id) {
            continue;
        }
        for link in &block.exits {
            if link.target == deref_mut.returnblock {
                deref_returns.push(
                    link.args
                        .first()
                        .and_then(value_id)
                        .unwrap_or_else(|| panic!("deref_mut return link has no value")),
                );
            }
        }
    }
    assert_eq!(
        deref_returns.len(),
        1,
        "deref_mut returns: {deref_returns:?}"
    );
    let ret_op = defining_op(&deref_mut, deref_returns[0]);
    assert_eq!(
        majit_translate::model::cast_instance_root(&ret_op.kind),
        Some("PyFrame"),
        "deref_mut must retarget frame_ptr's word to PyFrame, got {:?}",
        ret_op.kind
    );
    let frame_ptr_result = operand_ids(ret_op).first().copied().expect("cast operand");
    let call = defining_op(&deref_mut, frame_ptr_result);
    let call_name = match &call.kind {
        OpKind::Call { target, .. } => call_target_text(target),
        other => panic!("deref_mut cast operand is {other:?}"),
    };
    assert!(
        call_name.ends_with("frame_ptr"),
        "deref_mut must return frame_ptr's word, got {call_name}"
    );
    assert!(
        !calls_cast_int_to_ptr(&deref_mut),
        "deref_mut casts the frame pointer through the integer bank"
    );

    let new_graph = lower_pyframe(&llbc, &context, "pub fn new(mut frame: PyFrame)");
    let mut owner_value = None;
    let mut ptr_value = None;
    for block in &new_graph.blocks {
        for op in &block.operations {
            if let OpKind::FieldWrite { field, value, .. } = &op.kind {
                let LinkArg::Value(var) = value else {
                    panic!("{} is a constant, not new(r)", field.name);
                };
                if field.name == "owner_root" {
                    assert!(owner_value.is_none(), "two owner_root writes");
                    owner_value = Some(var.id());
                } else if field.name == "ptr" {
                    assert!(ptr_value.is_none(), "two ptr writes");
                    ptr_value = Some(var.id());
                }
            }
        }
    }
    let owner_value = owner_value.expect("FrameBox::new writes owner_root");
    let ptr_value = ptr_value.expect("FrameBox::new writes ptr");
    let stored = defining_op(&new_graph, owner_value);
    assert_eq!(
        majit_translate::model::cast_instance_root(&stored.kind),
        Some("GCREF"),
        "owner_root must receive the GcRef word new(r) lowers to, got {:?}",
        stored.kind
    );
    let new_operand = operand_ids(stored)
        .first()
        .copied()
        .expect("GCREF cast operand");
    let ptr_ancestors = ancestors(&new_graph, ptr_value);
    assert!(
        ptr_ancestors.contains(&new_operand),
        "owner_root's new(r) operand is not the frame pointer stored in ptr"
    );
}

/// `FrameBox` on the fib path roots the frame with `OwnerRootGuard`. The
/// lowered graph keeps the reference and does not call the guard.
#[test]
fn frame_box_owner_root_guard_lowers_without_guard_calls() {
    use majit_translate::front::mir::lower_fun_decl;
    use majit_translate::model::OpKind;

    let llbc = load_llbc(INTERPRETER_LLBC);
    let sources = [
        "pub fn new(mut frame: PyFrame)",
        "fn frame_ptr(&self)",
        "pub fn is_gc_owned(&self)",
        "pub fn into_raw(mut self)",
        "pub unsafe fn from_raw(ptr: *mut PyFrame)",
    ];
    let context = lower_context_for(llbc);
    for source in sources {
        let fd = llbc
            .iter_local_fns()
            .find(|fd| {
                fd.item_meta.name_path().contains("pyframe")
                    && fd
                        .item_meta
                        .source_text
                        .as_deref()
                        .is_some_and(|text| text.starts_with(source))
            })
            .unwrap_or_else(|| panic!("missing {source}"));
        let graph = lower_fun_decl(&context, fd)
            .unwrap_or_else(|err| panic!("lower {}: {err}", fd.item_meta.name_path()));
        let calls: Vec<String> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(call_target_text(target)),
                _ => None,
            })
            .collect();
        assert!(
            !calls.iter().any(|call| call_names_owner_root_guard(call)),
            "{source} still calls the guard: {calls:?}"
        );
        assert!(
            !calls.iter().any(|call| {
                call.ends_with("::is_some")
                    || call.ends_with("::is_none")
                    || call.ends_with("::take")
                    || call.ends_with("::drop")
                    || call.contains("DropGlue")
            }),
            "{source} still calls an option/drop of the guard: {calls:?}"
        );
    }
}

/// Graphs that pass a guard-containing pointer to a residual callee are
/// declined. The interpreter corpus has none. `pyre-jit` resumes a native
/// `FailArgSource` (it holds the guard) through `llmodel` methods that stay
/// residual, so those graphs are the decline set.
#[test]
fn owner_root_guard_pointer_does_not_reach_a_residual_callee() {
    use majit_translate::front::mir::residual_owner_root_guard_escapes;

    let mut escapes_by_artefact = Vec::new();
    for artefact in [
        "pyre-interpreter.ullbc",
        "pyre-object.ullbc",
        "majit-rlib.ullbc",
        "pyre-jit.ullbc",
    ] {
        let path = format!("{}/../../build/llbc/{artefact}", env!("CARGO_MANIFEST_DIR"));
        let llbc = Llbc::load(&path).unwrap_or_else(|err| panic!("load {artefact}: {err}"));
        let mut escapes = residual_owner_root_guard_escapes(&llbc);
        escapes.sort();
        escapes.dedup();
        escapes_by_artefact.push((artefact, escapes));
    }
    for (artefact, escapes) in &escapes_by_artefact[..3] {
        assert!(
            escapes.is_empty(),
            "{artefact} residual OwnerRootGuard pointer escapes: {escapes:?}"
        );
    }
    let jit = &escapes_by_artefact[3].1;
    let expected = [
        (
            "pyre_jit::call_jit::blackhole_resume_via_rd_numb",
            "majit_backend::llmodel::<Impl>::get",
        ),
        (
            "pyre_jit::call_jit::blackhole_resume_via_rd_numb",
            "majit_backend::llmodel::<Impl>::len",
        ),
        (
            "pyre_jit::call_jit::blackhole_resume_via_rd_numb::<Impl>::call_once",
            "core::fmt::rt::<Impl>::new_debug",
        ),
        (
            "pyre_jit::call_jit::blackhole_resume_via_rd_numb::<Impl>::call_once",
            "majit_backend::llmodel::<Impl>::clone",
        ),
        (
            "pyre_jit::call_jit::jit_blackhole_resume_from_guard",
            "majit_backend::llmodel::<Impl>::get",
        ),
        (
            "pyre_jit::call_jit::jit_blackhole_resume_from_guard",
            "majit_backend::llmodel::<Impl>::len",
        ),
    ];
    let got: Vec<(&str, &str)> = jit
        .iter()
        .map(|(graph, callee)| (graph.as_str(), callee.as_str()))
        .collect();
    assert_eq!(got, expected);
}

fn call_target_text(target: &majit_translate::model::CallTarget) -> String {
    use majit_translate::model::CallTarget;
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.join("::"),
        CallTarget::Method {
            name,
            receiver_root,
            resolved_path,
            ..
        } => {
            let mut text = receiver_root.clone().unwrap_or_default();
            if let Some(path) = resolved_path {
                text = path.segments.join("::");
            }
            if text.is_empty() {
                name.clone()
            } else {
                format!("{text}::{name}")
            }
        }
        other => format!("{other:?}"),
    }
}

fn call_names_owner_root_guard(call: &str) -> bool {
    call.contains("OwnerRootGuard")
        || call.contains("shadow_stack::<Impl>::new")
        || call.contains("shadow_stack::<Impl>::get")
        || call.contains("shadow_stack::<Impl>::set")
        || call.contains("shadow_stack::<Impl>::drop")
}
