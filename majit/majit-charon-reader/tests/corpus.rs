//! End-to-end test: load the Charon fixture corpus, walk every function,
//! confirm every terminator/statement decodes into the typed enums.
//!
//! Run with: `cargo test -p majit-charon-reader --features dynasm`.

use majit_charon_reader::{
    GlobalDecl, Llbc,
    ullbc::{CallClass, NameSeg, Operand, Place, PlaceKind, Rvalue, StmtKind, TermKind},
};
use serde_json::Value;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc",);

#[test]
fn loads_fixture_corpus() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    assert_eq!(llbc.crate_name(), "charon_corpus");
    assert!(!llbc.file.has_errors);
    let local_count = llbc
        .iter_local_fns()
        .filter(|f| f.item_meta.name_path().starts_with("charon_corpus::"))
        .count();
    // 6 base fns (straight_line_add, branch_loop_sum, strategy_len,
    // parse_one, desugar_mix, tuple_roundtrip) + `bool_then_closure` and
    // the two local fns Charon emits for its `|| x + 1` closure (the
    // closure body and its transparent `<Impl>::call_once` inherent method)
    // + `option_source` and `option_question_mark` (the Option `?` fixture)
    // + `bool_then_some` (the eager `then_some` sibling, no closure).
    //
    // + 10 for the header-first object model: `w_object_type`, `w_new_int`,
    // `w_new_type_only_int`, `w_number_add`, `w_int_add`,
    // `lltype::malloc_typed`, the fixture's `object_model::get_instantiate`, and
    // the initializer bodies for `INT_CLASS`, `DOUBLE_CLASS`, and
    // `_immutable_fields_W_IntObject` — a `static`/`const` carries its
    // initializer as a function body, so it lands in `iter_local_fns` too.
    //
    // + 2 for the host-registered callback table: `host_registry_dispatch`
    // and `host_registry_dispatch_optional`. `HostCallback` is a type alias,
    // not an item, so it contributes no body.
    //
    // + 2 for the iterator element-kind pair, `slice_of_refs_sum` and
    // `array_of_refs_sum`.
    //
    // + 4 for the aggregate-element array read and its controls:
    // `aggregate_slot_index`, `aggregate_slot_get`, `scalar_slot_index` and
    // `scalar_slot_get`. `SlotValue` is a type, so it contributes no body.
    //
    // + 3 for the borrowed-primitive banking trio, `slice_get_tag_dispatch`,
    // `range_start_index` and `borrowed_byte_fields_alias`.  `BorrowedByte`
    // is a struct, so it contributes no body.
    //
    // + 2 for the numeric-`From` pair, `widening_int_from` and
    // `widening_float_from` — the integer one aliases, the float one must not.
    //
    // + 3 for the `mem::replace` trio, `replace_field`, `replace_elem`, and
    // `replace_reborrow_then_read`. Each is one local body.
    //
    // + 3 for the two-word cell trio, `replace_two_words`, `swap_two_words`
    // and `take_two_words`.
    //
    // + 2 for `take_odd_default` and its hand-written `Default::default`.
    // + 1 for `replace_wide_payload` (a `u128` variant field).
    //
    // + 13 for the union, enum, and box-deref fixtures:
    // `clear_inline_tag`, `store_held_int`, `replace_held_union`,
    // `replace_boxed_held`, `held_as_mut`, `replace_boxed_held_call`,
    // `replace_indexed_box`, `replace_indexed_box_call`,
    // `replace_boxed_dynlike`, `store_held_cell`, and the extra local
    // bodies Charon emits beside those items.
    //
    // Charon since nightly-2026.09.26 no longer emits
    // `bool_then_closure::closure::<Impl>::drop_in_place` as its own local
    // item. The closure env is not a separate `closure` path; its body is
    // `bool_then_closure::<Impl>::call_once`. That drop glue was a local fn
    // on nightly-2026.05.29, and it is absent from the artefact rather than
    // dropped by the reader.
    //
    // + 1 for `char_slot_index`, the `char` element array read.
    //
    // + 1 for `char_unwrap_or_join`, the `Option<char>` literal-default join.
    //
    // + 6 for the `bitflags!`-shaped flag type: `code_flags_bits_or`, the
    // `FLAT` initializer, and the constructor and accessor on both wrappers.
    //
    // + 2 for the raw-buffer array pair, `sum_two_items` and
    // `sum_of_array_literal`.
    //
    // + 3 for the pointer-item slices, `reverse_then_first_ref`,
    // `first_of_array_prefix` and `first_ref`.
    //
    // + 1 for `extend_vec_from_slice`.
    //
    // + 8 for pair slices inside aggregates: `item_of_tupled_slice`,
    // `tuple_a_slice`, `slice_through_a_closure`, its closure's `call`,
    // `call_mut` and `call_once`, `item_or_null` and `get_item_or_null`.
    // Charon 0.1.281 does not emit that closure's `drop_in_place`.
    //
    // + 4 for `index_through_a_closure` and its closure's `call`,
    // `call_mut` and `call_once`. Same missing `drop_in_place`.
    //
    // + 3 for the pair-slice copy and pointer projections:
    // `copy_slice_from_slice`, `ptr_of_slice` and `mut_ptr_of_slice`.
    //
    // + 2 for `vec![item; count]`: `repeat_vec_i64` and `repeat_vec_ptr`.
    //
    // + 6 for an explicit `drop` of a root bracket and of two integers:
    // `mem_drop_root_scope`, `mem_drop_i64`, `mem_drop_usize`,
    // `gc_roots::push_roots`, `RootScope`'s initializer, and
    // `RootScope`'s `Drop::drop`. Charon 0.1.281 does not emit
    // `drop_in_place` as its own local item.
    //
    // + 1 for `tail_len`, `get(1..).unwrap_or(&[])` of a pair slice.
    //
    // + 6 for the three `majit_gc::GcType` impls (`ClassObject`,
    // `ObjectHeader`, `TypeOnlyHeader`): each `type_id` and each `SIZE`
    // associated const is a function body.
    //
    // + 2 for the `Option<*mut T>` pair, `option_raw_c_void_from_int` and
    // `option_raw_struct_from_int`. `SomeRawStruct` is a type, so it
    // contributes no body.
    //
    // Measured on Charon 0.1.281 (nightly-2026.10.04) with
    // `--reconstruct-panic-calls --inline-anon-consts`: drop glue is no
    // longer its own local item, and this artefact holds 105.
    assert_eq!(local_count, 105, "105 local fns expected");
}

#[test]
fn every_corpus_function_decodes() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");

    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        let Some(u) = fd.unstructured() else { continue };

        for (bb_idx, bb) in u.body.iter().enumerate() {
            for (s_idx, st) in bb.statements.iter().enumerate() {
                let stmt = st.stmt_kind().unwrap_or_else(|e| {
                    panic!("stmt decode failed in {name} bb{bb_idx} stmt{s_idx}: {e}")
                });
                assert!(
                    !matches!(stmt, StmtKind::Unknown),
                    "Unknown StmtKind in {name} bb{bb_idx} stmt{s_idx}",
                );
            }
            let term = bb
                .term(&llbc)
                .unwrap_or_else(|e| panic!("terminator decode failed in {name} bb{bb_idx}: {e}"));
            assert!(
                !matches!(term, TermKind::Unknown),
                "Unknown TermKind in {name} bb{bb_idx}",
            );
        }
    }
}

#[test]
fn straight_line_add_shape() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc
        .local_fn("straight_line_add")
        .expect("function present");
    let u = fd.unstructured().expect("Unstructured body");
    assert_eq!(u.locals.arg_count, 3);
    assert_eq!(u.body.len(), 7);
    assert!(
        u.body.iter().any(|bb| bb.is_cleanup),
        "cleanup blocks stay in the body the reader returns"
    );

    // bb0 should end in an overflow Assert (AddChecked + Assert).
    let bb0 = &u.body[0];
    assert!(
        matches!(bb0.term(&llbc).unwrap(), TermKind::Assert { .. }),
        "bb0 terminator was not Assert",
    );

    // bb3 should be the return block.
    let bb3 = &u.body[3];
    assert!(matches!(bb3.term(&llbc).unwrap(), TermKind::Return));
}

#[test]
fn branch_loop_sum_has_switch_int_and_switch_if() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc.local_fn("branch_loop_sum").expect("function present");
    let u = fd.unstructured().expect("Unstructured body");

    let mut saw_switch_int = false;
    let mut saw_switch_if = false;
    for bb in &u.body {
        if let Ok(TermKind::Switch { targets, .. }) = bb.term(&llbc) {
            match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(..) => saw_switch_if = true,
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(..) => saw_switch_int = true,
            }
        }
    }
    assert!(saw_switch_int, "expected SwitchInt in branch_loop_sum");
    assert!(saw_switch_if, "expected If switch in branch_loop_sum");
}

#[test]
fn call_classify_covers_corpus() {
    use std::collections::BTreeMap;
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for fd in llbc.iter_local_fns() {
        let Some(u) = fd.unstructured() else { continue };
        for bb in &u.body {
            if let Ok(TermKind::Call { call, .. }) = bb.term(&llbc) {
                let label = match call.func.classify() {
                    CallClass::Direct => "direct",
                    CallClass::Trait => "trait",
                    CallClass::Dynamic => "dynamic",
                    CallClass::Ptr => "ptr",
                    CallClass::Unknown => "unknown",
                };
                *counts.entry(label).or_default() += 1;
            }
        }
    }
    // The corpus is straightforward Rust — every call should classify
    // as Direct (or Trait for the iterator `?` desugaring helpers).
    let unknown = counts.get("unknown").copied().unwrap_or(0);
    assert_eq!(
        unknown, 0,
        "corpus should not produce unknown call classifications: {counts:?}"
    );
    assert!(
        counts.get("direct").copied().unwrap_or(0) > 0,
        "expected at least one direct call: {counts:?}",
    );
}

#[test]
fn dedup_body_resolves_inline_shape() {
    // Every `Value: [id, body]` occurrence must surface
    // through `Llbc::dedup_body(id)` so MIR's TyRef projection can
    // resolve `Deduplicated` references.
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");

    // Collect a (dedup_id, body_kind) sample by walking every
    // FunDecl's `inputs` / `output` TyRefs.  At least one
    // `Deduplicated` id should appear in the corpus and round-trip
    // through `dedup_body`.
    let mut sampled = 0usize;
    for fd in llbc.iter_local_fns() {
        for ty in &fd.signature.inputs {
            if let majit_charon_reader::ullbc::TyRef::Dedup { id } = ty {
                let body = llbc.dedup_body(*id).unwrap_or_else(|| {
                    panic!(
                        "dedup_body({id}) returned None for an input TyRef in {}",
                        fd.item_meta.name_path()
                    )
                });
                assert!(
                    body.is_object() || body.is_string(),
                    "dedup_body({id}) body was unexpectedly typed: {body}"
                );
                sampled += 1;
            }
        }
    }
    assert!(
        sampled > 0,
        "expected at least one Deduplicated input TyRef in the corpus"
    );
}

fn is_promoted_anon_const(gd: &GlobalDecl) -> bool {
    if gd.rest.get("global_kind").and_then(Value::as_str) != Some("AnonConst") {
        return false;
    }
    match gd.item_meta.name.last() {
        Some(NameSeg::Other(v)) => {
            v.get("Builtin")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(Value::as_str)
                == Some("PromotedConst")
        }
        _ => false,
    }
}

fn json_global_ids(v: &Value, out: &mut Vec<u64>) {
    if let Some(id) = v
        .get("kind")
        .and_then(|k| k.get("Global"))
        .and_then(|g| g.get("id"))
        .and_then(Value::as_u64)
    {
        out.push(id);
    }
    match v {
        Value::Array(arr) => {
            for item in arr {
                json_global_ids(item, out);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                json_global_ids(item, out);
            }
        }
        _ => {}
    }
}

fn unstructured_global_ids(u: &majit_charon_reader::ullbc::Unstructured) -> Vec<u64> {
    let mut ids = Vec::new();
    for bb in &u.body {
        for st in &bb.statements {
            json_global_ids(st.kind_value(), &mut ids);
        }
        json_global_ids(bb.terminator.kind_value(), &mut ids);
    }
    ids
}

fn assert_no_promoted_anon_const_read(llbc: &Llbc, u: &majit_charon_reader::ullbc::Unstructured) {
    for id in unstructured_global_ids(u) {
        if let Some(gd) = llbc.global_by_id(id) {
            assert!(
                !is_promoted_anon_const(gd),
                "body still reads AnonConst promoted global {id} ({})",
                gd.item_meta.name_path()
            );
        }
    }
}

fn dest_local(place: &Place) -> Option<u64> {
    match place.kind {
        PlaceKind::Local(n) => Some(n),
        _ => None,
    }
}

fn copy_global_id(rv: &Rvalue) -> Option<u64> {
    match rv {
        Rvalue::Use(Operand::Copy(place), _) => match place.kind {
            PlaceKind::Global { id, .. } => Some(id),
            _ => None,
        },
        _ => None,
    }
}

fn ref_of_local(rv: &Rvalue, local: u64) -> bool {
    matches!(rv, Rvalue::Ref { place, .. } if matches!(place.kind, PlaceKind::Local(n) if n == local))
}

fn is_array_aggregate(rv: &Rvalue) -> bool {
    matches!(rv, Rvalue::Aggregate(kind, _) if kind.get("Array").is_some())
}

fn consecutive_assigns(
    u: &majit_charon_reader::ullbc::Unstructured,
) -> Vec<(Place, Rvalue, Place, Rvalue)> {
    let mut out = Vec::new();
    for bb in &u.body {
        for pair in bb.statements.windows(2) {
            let Ok(StmtKind::Assign(p0, rv0)) = pair[0].stmt_kind() else {
                continue;
            };
            let Ok(StmtKind::Assign(p1, rv1)) = pair[1].stmt_kind() else {
                continue;
            };
            out.push((p0, rv0, p1, rv1));
        }
    }
    out
}

/// Charon 0.1.281 emits the `CodeFlags::FLAT` borrow as a promoted
/// constant item. The reader splices the initializer so the body copies
/// `FLAT` into a local and takes a reference, the inline shape.
#[test]
fn code_flags_bits_or_inlines_promoted_flat_borrow() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc
        .local_fn("code_flags_bits_or")
        .expect("function present");
    let u = fd.unstructured().expect("Unstructured body");
    assert_no_promoted_anon_const_read(&llbc, &u);

    let flat_id = llbc
        .iter_global_decls()
        .find(|gd| gd.item_meta.name_path().ends_with("::FLAT"))
        .map(|gd| gd.def_id)
        .expect("CodeFlags::FLAT global");
    let found = consecutive_assigns(&u).iter().any(|(p0, rv0, _, rv1)| {
        copy_global_id(rv0) == Some(flat_id)
            && dest_local(p0).is_some_and(|loc| ref_of_local(rv1, loc))
    });
    assert!(
        found,
        "expected copy Global(FLAT) into a local followed by a Ref of that local"
    );
}

/// Charon 0.1.281 emits the `&[]` fallback in `tail_len` as a promoted
/// constant item. The reader splices the empty-array aggregate and a
/// reference to it.
#[test]
fn tail_len_inlines_promoted_empty_array() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc.local_fn("tail_len").expect("function present");
    let u = fd.unstructured().expect("Unstructured body");
    assert_no_promoted_anon_const_read(&llbc, &u);

    let found = consecutive_assigns(&u).iter().any(|(p0, rv0, _, rv1)| {
        is_array_aggregate(rv0) && dest_local(p0).is_some_and(|loc| ref_of_local(rv1, loc))
    });
    assert!(
        found,
        "expected Aggregate Array assign followed by a Ref of that local"
    );
}

/// A body that never reads a promoted constant keeps its block and
/// statement counts after the splice pass. The counts are the body
/// Charon wrote, cleanup blocks included.
#[test]
fn straight_line_add_unchanged_without_promoted_read() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc
        .local_fn("straight_line_add")
        .expect("function present");
    let u = fd.unstructured().expect("Unstructured body");
    assert_eq!(u.body.len(), 7);
    let stmt_counts: Vec<usize> = u.body.iter().map(|bb| bb.statements.len()).collect();
    assert_eq!(stmt_counts, vec![10, 8, 8, 5, 0, 0, 0]);
    assert!(
        u.body[4].is_cleanup && u.body[5].is_cleanup && u.body[6].is_cleanup,
        "cleanup blocks stay in the body the reader returns"
    );
}
