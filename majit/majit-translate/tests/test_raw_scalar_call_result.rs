//! A call that returns a non-byte raw scalar pointer uses the integer
//! bank. `history.py` `getkind` banks a raw `Ptr` as `int`, and
//! `pygraph_initial_block` already records that parameter as `Int`.
//! `CallControl::getcalldescr` compares the two.
//!
//! `IntArray::base` and `FloatArray::base` return the GC block header,
//! so those calls stay `Ref`. An integer cast to a non-byte raw scalar
//! stays in the int bank.
//!
//! A byte pointer stays `Ref`: pyre erases a GC reference to `*mut u8`.
//!
//! A call that passes `&` / `&mut` of a primitive into a raw
//! scalar-pointer parameter materializes an address first
//! (`RawMalloc` / `RawStore` / `RawFree`). `history.py` `getkind` banks
//! that parameter as `int`, and `Rvalue::Ref` aliases the pointee word.
//! Two borrows of one place share that address. A raw pointer of that
//! place uses the same address. A raw pointer with no referent is not
//! lowered. A raw pointer taken while its local is still clean reloads
//! a later store of the address, including through a callee that
//! returns that load and through a field that holds the pointer. A
//! borrow of a field or a constant index names that path
//! (`place_referent`), so a later store into the leaf is visible
//! through the pointer. Drop glue sees a field referent lifted from
//! the dropped place (`dropped_address`). A
//! cast keeps that name only when the destination pointee still
//! covers the source (`cast_covers_referent_pointee`). A nested field
//! store refolds its parents. A constant index is a
//! `const_expr_literal`. A mutable raw parameter
//! copies the written word back into the borrowed place, including a
//! field projection. An address-sized integer whose provenance is the
//! spilled place is passed that same pointer
//! (`attach_aliasing_raw_arguments`). A pointer packed into a tuple,
//! struct, or array stays on the caller's place
//! (`local_hides_spill_alias`), so that call stays unlowered. An
//! address-sized integer whose place this walk cannot name stays
//! unlowered (`spill_address_source`): `pair.0`, a parameter, and a
//! call result. An integer literal still lowers. `[q; N]` keeps `q`'s
//! referent on each element (`repeat_slots`). A union folds that
//! referent onto the local (`rvalue_referent`); its fields overlap, so
//! they are not split (`aggregate_slots`). A store through an index
//! that is not a constant stays unlowered when the value or a cleared
//! element names a place (`untracked_projection_carries_referent`).
//! Drop glue refolds a lifted child into its parent
//! (`lift_referent_field`). A call result keeps the callee's fields
//! and a direct return (`AddressEscape`). Several null checks stay
//! the same branch (`callee_returns_spill_address`). The return
//! summary is taken after the join settles
//! (`unstructured_address_escape`), so a return block above the
//! assignment still sees that assignment. A block that becomes
//! reachable after its own turn still runs (`join_reached`), so a
//! later null return keeps the pointer from staying direct.
//! `Len` of a fixed array is the
//! const generic. `Len` of a slice is the length recorded for that
//! slice (`rvalue_length_metadata`). A call that returns the spill
//! address is not lowered: the free would run before the caller
//! dereferences it.

mod common;

use common::{
    INTERPRETER_LLBC, MODULE_LLBC, OBJECT_LLBC, load_llbc, lower_context_for, lower_named,
};
use majit_charon_reader::ullbc::NameSeg;
use majit_charon_reader::{FunDecl, Llbc};
use majit_translate::{
    ErrorCarrierSpec, HostStaticAddrs,
    call::CallControl,
    codewriter::jtransform::{GraphTransformConfig, Transformer},
    flowspace::model::ConstValue,
    front::llbc_hints::harvest_immutable_fields_from_llbcs,
    front::mir::{lower_fun_decl, lower_function, lower_function_with_static_addrs},
    model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType},
    translator::rtyper::lltypesystem::lltype::{GCREF, GcKind, LowLevelType, PtrTarget},
};
use serde_json::{Value, json};

fn defining_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| (op.result.as_ref() == Some(var)).then_some(&op.kind))
    })
}

fn root_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    let mut current = var.clone();
    for _ in 0..32 {
        if let Some(op) = defining_op(graph, &current) {
            return Some(op);
        }
        let mut next = None;
        for block in &graph.blocks {
            let Some(slot) = block.inputargs.iter().position(|arg| arg == &current) else {
                continue;
            };
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == block.id
                        && let Some(src) = exit.args.get(slot).and_then(LinkArg::as_variable)
                        && src != &current
                    {
                        next = Some(src.clone());
                    }
                }
            }
        }
        match next {
            Some(src) => current = src,
            None => return None,
        }
    }
    None
}

fn returned_var(graph: &FunctionGraph) -> majit_translate::flowspace::model::Variable {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.exits)
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .and_then(LinkArg::as_variable)
        .cloned()
        .unwrap_or_else(|| panic!("{} does not return a variable", graph.name))
}

fn call_leaf(target: &CallTarget) -> Option<&str> {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
        CallTarget::Method { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

fn ops(graph: &FunctionGraph) -> impl Iterator<Item = &SpaceOperation> {
    graph.blocks.iter().flat_map(|block| &block.operations)
}

fn call_result<'a>(kind: &'a OpKind) -> Option<(&'a ValueType, Option<&'a str>)> {
    match kind {
        OpKind::Call {
            result_ty, target, ..
        } => Some((result_ty, call_leaf(target))),
        OpKind::IndirectCall { result_ty, .. } => Some((result_ty, None)),
        _ => None,
    }
}

/// `name_path` renders every inherent impl as `<Impl>`, so
/// `ActionFlag::ticker_addr` and `SpaceActionFlag::ticker_addr` share one
/// suffix. The impl segment's `Self` type is the receiver.
fn receiver_path(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let id = fd.item_meta.name.iter().find_map(|seg| match seg {
        NameSeg::Other(value) => value
            .pointer("/Impl/Ty/skip_binder/Deduplicated")
            .and_then(|value| value.as_u64()),
        NameSeg::Ident { .. } => None,
    })?;
    let def_id = llbc.dedup_to_adt_def_id(id)?;
    Some(llbc.type_by_id(def_id)?.item_meta.name_path())
}

fn find_method<'a>(llbc: &'a Llbc, leaf: &str, owner: &str) -> &'a FunDecl {
    let suffix = format!("::{leaf}");
    let owner_suffix = format!("::{owner}");
    llbc.iter_local_fns()
        .find(|fd| {
            fd.item_meta.name_path().ends_with(&suffix)
                && receiver_path(llbc, fd)
                    .is_some_and(|path| path == owner || path.ends_with(&owner_suffix))
        })
        .unwrap_or_else(|| panic!("missing {owner}::{leaf}"))
}

fn find_exact<'a>(llbc: &'a Llbc, path: &str) -> &'a FunDecl {
    llbc.iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == path)
        .unwrap_or_else(|| panic!("missing {path}"))
}

fn call_results_named<'a>(graph: &'a FunctionGraph, leaf: &str) -> Vec<&'a ValueType> {
    ops(graph)
        .filter_map(|op| {
            let (ty, name) = call_result(&op.kind)?;
            (name == Some(leaf)).then_some(ty)
        })
        .collect()
}

fn assert_returned_call(graph: &FunctionGraph, leaf: &str) {
    match root_op(graph, &returned_var(graph)).and_then(call_result) {
        Some((ValueType::Int, Some(got))) if got == leaf => {}
        other => panic!(
            "{} must return an int `{leaf}` call, got {other:?}",
            graph.name
        ),
    }
}

#[test]
fn call_returned_raw_scalar_pointer_is_int() {
    let llbc = load_llbc(INTERPRETER_LLBC);
    let context = lower_context_for(llbc);

    let inner = lower_fun_decl(&context, find_method(&llbc, "ticker_addr", "ActionFlag"))
        .expect("lower ActionFlag::ticker_addr");
    assert_returned_call(&inner, "as_ptr");

    let producer = lower_fun_decl(
        &context,
        find_method(&llbc, "ticker_addr", "SpaceActionFlag"),
    )
    .expect("lower SpaceActionFlag::ticker_addr");
    assert_returned_call(&producer, "ticker_addr");
}

fn mapdict_fns<'a>(llbc: &'a Llbc, leaf: &str) -> Vec<&'a FunDecl> {
    let suffix = format!("::{leaf}");
    llbc.iter_local_fns()
        .filter(|fd| {
            let path = fd.item_meta.name_path();
            path.ends_with(&suffix) && path.contains("mapdict")
        })
        .collect()
}

#[test]
fn mapdict_get_reads_slots_map_pointer_as_int() {
    let llbc = load_llbc(INTERPRETER_LLBC);
    let context = lower_context_for(llbc);
    let gets = mapdict_fns(&llbc, "_get_mapdict_map");
    assert!(
        !gets.is_empty(),
        "mapdict _get_mapdict_map is in interpreter LLBC"
    );
    let mut saw_pos0 = false;
    for fd in gets {
        let graph = lower_fun_decl(&context, fd)
            .unwrap_or_else(|err| panic!("lower {}: {err}", fd.item_meta.name_path()));
        let pos0: Vec<_> = ops(&graph)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name == "__pos_0" => Some(ty),
                _ => None,
            })
            .collect();
        if pos0.is_empty() {
            continue;
        }
        saw_pos0 = true;
        assert!(
            pos0.iter().all(|ty| matches!(ty, ValueType::Int)),
            "{} slots().0 is *mut usize; getkind of a raw Ptr is int, got {pos0:?}",
            graph.name
        );
    }
    assert!(
        saw_pos0,
        "MapdictCarrier::_get_mapdict_map FieldReads slots().0"
    );

    let slots = mapdict_fns(&llbc, "slots");
    assert_eq!(slots.len(), 1, "one MapdictCarrier::slots");
    let slots = lower_fun_decl(&context, slots[0]).expect("lower MapdictCarrier::slots");
    let writes: Vec<_> = ops(&slots)
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, ty, .. } if field.name == "__pos_0" => Some(ty),
            _ => None,
        })
        .collect();
    assert!(
        !writes.is_empty() && writes.iter().all(|ty| matches!(ty, ValueType::Int)),
        "slots() stores *mut usize as int, got {writes:?}"
    );
}

#[test]
fn typed_array_base_call_stays_ref() {
    let llbc = load_llbc(OBJECT_LLBC);
    let context = lower_context_for(llbc);
    for path in [
        "pyre_object::int_array::<Impl>::index",
        "pyre_object::float_array::<Impl>::index",
    ] {
        let graph = lower_fun_decl(&context, find_exact(&llbc, path)).unwrap_or_else(|err| {
            panic!("lower {path}: {err}");
        });
        let bases = call_results_named(&graph, "base");
        assert!(
            !bases.is_empty(),
            "{path} must call base, ops in {}",
            graph.name
        );
        assert!(
            bases.iter().all(|ty| matches!(ty, ValueType::Ref(_))),
            "{path} base() stays the GC block Ref, got {bases:?}"
        );
    }
}

#[test]
fn integer_cast_to_raw_scalar_stays_int() {
    let llbc = load_llbc(MODULE_LLBC);
    let context = lower_context_for(llbc);
    let graph = lower_fun_decl(&context, find_method(&llbc, "as_ptr", "HashStateStorage"))
        .expect("lower HashStateStorage::as_ptr");
    let leaves: Vec<_> = ops(&graph)
        .filter_map(|op| call_result(&op.kind).and_then(|(_, leaf)| leaf))
        .collect();
    assert!(
        leaves.contains(&"cast_ptr_to_int"),
        "the field address still crosses into the integer, got {leaves:?}"
    );
    assert!(
        !leaves.contains(&"cast_int_to_ptr"),
        "usize as *const usize stays in the int bank, got {leaves:?} in {}",
        graph.name
    );
    match root_op(&graph, &returned_var(&graph)) {
        Some(OpKind::BinOp { result_ty, .. })
            if matches!(result_ty, ValueType::Int | ValueType::Unsigned) => {}
        // `rarithmetic.py` `intmask`: usize arithmetic is Unsigned, the
        // `*const usize` return is the Signed address word
        // (`history.py` `getkind` of a raw Ptr).
        Some(OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty,
            ..
        }) if segments.last().map(String::as_str) == Some("intmask")
            && matches!(result_ty, ValueType::Int | ValueType::Unsigned) => {}
        other => panic!(
            "{} must return the aligned integer word, got {other:?}",
            graph.name
        ),
    }

    let widened = lower_fun_decl(
        &context,
        find_method(&llbc, "as_mut_ptr", "HashStateStorage"),
    )
    .expect("lower HashStateStorage::as_mut_ptr");
    assert_returned_call(&widened, "as_ptr");
}

#[test]
fn sizehint_i64_cast_stays_gcarray_write() {
    let llbc = load_llbc(OBJECT_LLBC);
    let graph = lower_named(llbc, "pyre_object::listobject::set_sizehint_state_value")
        .expect("lower set_sizehint_state_value");
    let raw_stores = ops(&graph)
        .filter(|op| matches!(op.kind, OpKind::RawStore { .. }))
        .count();
    assert_eq!(
        raw_stores, 0,
        "set_sizehint_state_value must keep the GcArray write"
    );
    let ptr_to_int = call_results_named(&graph, "cast_ptr_to_int");
    assert!(
        ptr_to_int.is_empty(),
        "*mut u8 as *mut i64 must not become cast_ptr_to_int, got {ptr_to_int:?}"
    );
    assert!(
        ops(&graph).any(|op| {
            matches!(
                &op.kind,
                OpKind::ArrayWrite {
                    item_ty: ValueType::Int,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == "[i64]"
            )
        }),
        "set_sizehint_state_value must write GcArray(Signed)"
    );
}

#[test]
fn call_returned_byte_pointer_stays_ref() {
    let llbc = load_llbc(OBJECT_LLBC);
    let graph = lower_named(llbc, "gc_hook::try_gc_alloc_young_nonmoving_raw")
        .expect("lower try_gc_alloc_young_nonmoving_raw");
    let byte_calls: Vec<_> = ops(&graph)
        .filter_map(|op| call_result(&op.kind))
        .filter(|(_, leaf)| *leaf == Some("try_gc_alloc_stable_raw"))
        .map(|(ty, _)| ty.clone())
        .collect();
    assert!(
        !byte_calls.is_empty(),
        "try_gc_alloc_young_nonmoving_raw must call try_gc_alloc_stable_raw, ops in {}",
        graph.name
    );
    assert!(
        byte_calls.iter().all(|ty| matches!(ty, ValueType::Ref(_))),
        "*mut u8 call result stays Ref, got {byte_calls:?} in {}",
        graph.name
    );
}

/// Front-end: `p as *const RawStruct` + field read lowers through
/// `ann_cast_int_to_ptr` as 2-arg `cast_int_to_ptr(PTRTYPE, addr)` with
/// `SomePtr(Ptr(Struct ...))`, then `SomePtr.getattr` of `terminator` /
/// `w_cls`. The unary-op test that constructs `SomePtr` directly does not
/// count. A dest whose GC pointer field is not in `_immutable_fields_`
/// stays off `SomePtr` (`dest_overlay_rejects_raw_struct_with_mutable_gc_pointer_field`).
#[test]
fn map_terminator_w_cls_overlay_cast_is_dest_typed_someptr() {
    let llbc = load_llbc(OBJECT_LLBC);
    let context = lower_context_for(llbc);
    let ranks = harvest_immutable_fields_from_llbcs(std::slice::from_ref(llbc));
    assert!(
        ranks.get("MapNodeTerminatorView").is_some_and(|fields| {
            fields.iter().any(|(name, _)| name == "terminator")
                && fields.iter().any(|(name, _)| name == "w_cls")
        }),
        "harvested _immutable_fields_ for MapNodeTerminatorView: {:?}",
        ranks.get("MapNodeTerminatorView")
    );
    assert!(
        ranks
            .get("MapNodeAbstractAttribute")
            .is_some_and(|fields| fields.iter().any(|(name, _)| name == "terminator")),
        "harvested _immutable_fields_ for MapNodeAbstractAttribute: {:?}",
        ranks.get("MapNodeAbstractAttribute")
    );
    let mut saw_dest_typed_struct = false;
    let mut saw_terminator = false;
    let mut saw_w_cls = false;
    let mut saw_raw_overlay = false;
    let mut saw_w_cls_gcref = false;
    let mut saw_terminator_ptr = false;
    let mut saw_overlay_hints = false;
    let mut saw_pure_w_cls = false;
    for leaf in ["map_terminator_w_cls", "instance_getclass"] {
        let fns = mapdict_fns(&llbc, leaf);
        assert!(!fns.is_empty(), "{leaf} is in pyre-object LLBC");
        for fd in fns {
            let graph = lower_fun_decl(&context, fd)
                .unwrap_or_else(|err| panic!("lower {}: {err}", fd.item_meta.name_path()));
            for op in ops(&graph) {
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                    && segments.last().map(String::as_str) == Some("cast_int_to_ptr")
                    && args.len() == 2
                    && let LinkArg::Const(c) = &args[0]
                    && let ConstValue::LowLevelType(ll) = &c.value
                    && let LowLevelType::Ptr(ptr) = ll.as_ref()
                    && let PtrTarget::Struct(st) = &ptr.TO
                {
                    saw_dest_typed_struct = true;
                    if st._gckind == GcKind::Raw {
                        saw_raw_overlay = true;
                    }
                    if let Some(w_cls_ty) = st._flds.get("w_cls")
                        && w_cls_ty == &*GCREF
                    {
                        saw_w_cls_gcref = true;
                    }
                    if let Some(term_ty) = st._flds.get("terminator")
                        && matches!(term_ty, LowLevelType::Ptr(_))
                        && term_ty != &*GCREF
                    {
                        saw_terminator_ptr = true;
                    }
                    if let Some(ConstValue::Dict(fields)) = st._hints.get("immutable_fields") {
                        let has_terminator =
                            fields.contains_key(&ConstValue::byte_str("terminator"));
                        let has_w_cls = fields.contains_key(&ConstValue::byte_str("w_cls"));
                        if has_terminator && (st._flds.get("w_cls").is_none() || has_w_cls) {
                            saw_overlay_hints = true;
                        }
                    }
                }
                if let OpKind::FieldRead { field, .. } = &op.kind {
                    if field.name == "terminator" {
                        saw_terminator = true;
                    }
                    if field.name == "w_cls" {
                        saw_w_cls = true;
                    }
                }
            }
            let mut cc = CallControl::new();
            cc.immutable_fields_by_struct = ranks.clone();
            let out = Transformer::new(&GraphTransformConfig::default())
                .with_callcontrol(&mut cc)
                .transform(&graph);
            for op in ops(&out.graph) {
                if let OpKind::FieldRead { field, pure, .. } = &op.kind
                    && field.name == "w_cls"
                    && *pure
                {
                    saw_pure_w_cls = true;
                }
            }
        }
    }
    assert!(
        saw_dest_typed_struct,
        "overlay dest is 2-arg cast_int_to_ptr(Ptr(Struct ...), addr)"
    );
    assert!(saw_raw_overlay, "overlay Struct is GcKind::Raw");
    assert!(saw_w_cls_gcref, "overlay w_cls field lltype is GCREF");
    assert!(
        saw_terminator_ptr,
        "overlay terminator field is a raw Ptr, not Address"
    );
    assert!(
        saw_overlay_hints,
        "overlay Struct carries harvested immutable_fields hints"
    );
    assert!(
        saw_terminator,
        "overlay FieldRead terminator via SomePtr.getattr"
    );
    assert!(saw_w_cls, "overlay FieldRead w_cls via SomePtr.getattr");
    assert!(
        saw_pure_w_cls,
        "jtransform rewrite_op_getfield marks w_cls as pure getfield_raw_r"
    );
}

fn i64_ty() -> Value {
    json!({"Scalar": {"Integer": {"Signed": "I64"}}})
}

fn u64_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U64"}}})
}

fn u8_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
}

fn u32_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U32"}}})
}

fn usize_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

fn raw_ptr(pointee: &Value, kind: &str) -> Value {
    json!({"RawPtr": [pointee, kind]})
}

fn option_of(inner: &Value) -> Value {
    json!({"Adt": {
        "id": 0,
        "generics": {
            "regions": [],
            "types": [inner],
            "const_generics": [],
            "trait_refs": []
        }
    }})
}

fn nest_option(inner: Value, depth: usize) -> Value {
    let mut ty = inner;
    for _ in 0..depth {
        ty = option_of(&ty);
    }
    ty
}

fn borrow_ty(pointee: &Value, kind: &str) -> Value {
    json!({"Ref": ["Erased", pointee, kind]})
}

fn place(id: u64, ty: &Value) -> Value {
    json!({"kind": {"Local": id}, "ty": ty})
}

/// How `write_hash` passes local 2 into `sink_pair`.
enum Pass {
    /// `_b = &kind word`, then pass `_b`. `ret_local` is copied into the
    /// return slot after the call (`2` is the word, `1` is `flag`).
    Borrow { kind: &'static str, ret_local: u64 },
    /// Pass local 2 itself.
    Direct { ret_local: u64 },
    /// Copy the word, then pass `&mut` of the original, then return the copy.
    CopyThenMutBorrow,
}

fn lower_probe(word_name: &str, word_ty: &Value, sink_param: &Value, pass: Pass) -> FunctionGraph {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let ret = i64_ty();
    // Narrower than a pointer. An address-sized integer parameter beside
    // the spill stays unlowered (`spill_address_source`).
    let flag = u32_ty();
    let local = |index: u64, name: Option<&str>, ty: &Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let mut locals = vec![
        local(0, None, &ret),
        local(1, Some("flag"), &flag),
        local(2, Some(word_name), word_ty),
    ];
    let (statements, call_args, dest, ret_local) = match pass {
        Pass::Borrow { kind, ret_local } => {
            let borrowed = borrow_ty(word_ty, kind);
            locals.push(local(3, None, &borrowed));
            locals.push(local(4, None, &ret));
            let word = place(2, word_ty);
            let borrow = place(3, &borrowed);
            (
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word, "kind": kind, "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Copy": place(1, &flag)}), json!({"Move": borrow})],
                place(4, &ret),
                ret_local,
            )
        }
        Pass::Direct { ret_local } => {
            locals.push(local(3, None, &ret));
            (
                vec![],
                vec![
                    json!({"Copy": place(1, &flag)}),
                    json!({"Move": place(2, word_ty)}),
                ],
                place(3, &ret),
                ret_local,
            )
        }
        Pass::CopyThenMutBorrow => {
            let borrowed = borrow_ty(word_ty, "Mut");
            locals.push(local(3, None, word_ty));
            locals.push(local(4, None, &borrowed));
            locals.push(local(5, None, &ret));
            let word = place(2, word_ty);
            let borrow = place(4, &borrowed);
            (
                vec![
                    json!({"span": span, "kind": {"Assign": [
                        place(3, word_ty),
                        {"Use": [{"Copy": word.clone()}, "Yes"]}
                    ]}}),
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Copy": place(1, &flag)}), json!({"Move": borrow})],
                place(5, &ret),
                3,
            )
        }
    };
    let ret_place_ty = if ret_local == 1 {
        flag.clone()
    } else {
        word_ty.clone()
    };
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        vec![flag.clone(), word_ty.clone()],
        json!({"Unstructured": {"span": span, "locals": {"arg_count": 2, "locals": locals}, "body": [
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args, "dest": dest},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret_place_ty)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let params = vec![flag.clone(), sink_param.clone()];
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        params.clone(),
        idle_body(&ret, &params),
    );
    let file = json!({"charon_version": "0.1.201", "has_errors": false, "translated": {
        "crate_name": "probe", "type_decls": [], "fun_decls": [caller, sink],
        "global_decls": [], "trait_decls": [], "trait_impls": []
    }});
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, "write_hash").unwrap_or_else(|err| panic!("lower write_hash: {err}"))
}

fn op_lines(graph: &FunctionGraph) -> String {
    ops(graph)
        .map(|op| format!("{:?}", op.kind))
        .collect::<Vec<_>>()
        .join("\n")
}

fn input_var<'a>(
    graph: &'a FunctionGraph,
    name: &str,
) -> &'a majit_translate::flowspace::model::Variable {
    ops(graph)
        .find_map(|op| match &op.kind {
            OpKind::Input { name: got, .. } if got == name => op.result.as_ref(),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "missing input {name} in {}\n{}",
                graph.name,
                op_lines(graph)
            )
        })
}

fn sink_call(graph: &FunctionGraph) -> &OpKind {
    let mut found = None;
    for op in ops(graph) {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        if call_leaf(target) == Some("sink_pair") {
            assert!(found.is_none(), "two sink_pair calls\n{}", op_lines(graph));
            found = Some(&op.kind);
        }
    }
    found.unwrap_or_else(|| panic!("missing sink_pair\n{}", op_lines(graph)))
}

fn call_arg<'a>(kind: &'a OpKind, index: usize) -> &'a majit_translate::flowspace::model::Variable {
    let OpKind::Call { args, .. } = kind else {
        panic!("not a call");
    };
    args.get(index)
        .and_then(LinkArg::as_variable)
        .unwrap_or_else(|| panic!("call arg {index} is not a variable: {kind:?}"))
}

fn raw_ops<'a>(
    graph: &'a FunctionGraph,
    pred: impl Fn(&OpKind) -> bool,
) -> Vec<&'a SpaceOperation> {
    ops(graph).filter(|op| pred(&op.kind)).collect()
}

fn assert_no_raw_address(graph: &FunctionGraph, label: &str) {
    let spilled = raw_ops(graph, |kind| {
        matches!(
            kind,
            OpKind::RawMalloc { .. }
                | OpKind::RawStore { .. }
                | OpKind::RawLoad { .. }
                | OpKind::RawFree { .. }
        )
    });
    assert!(
        spilled.is_empty(),
        "{label} must pass the word, got\n{}",
        op_lines(graph)
    );
}

fn assert_address_spill(graph: &FunctionGraph, copy_out: bool, return_is_reload: bool) {
    let flag = input_var(graph, "flag");
    let keyhash = input_var(graph, "keyhash");
    let mallocs = raw_ops(graph, |kind| matches!(kind, OpKind::RawMalloc { .. }));
    assert_eq!(
        mallocs.len(),
        1,
        "one raw spill allocation\n{}",
        op_lines(graph)
    );
    let ptr = mallocs[0].result.as_ref().expect("RawMalloc result");
    match &mallocs[0].kind {
        OpKind::RawMalloc { owner, zero: false } if owner == "Tuple<i64>" => {}
        other => panic!("spill owner is Tuple<i64>, got {other:?}"),
    }
    let stores = raw_ops(graph, |kind| matches!(kind, OpKind::RawStore { .. }));
    assert_eq!(
        stores.len(),
        1,
        "one store of the word\n{}",
        op_lines(graph)
    );
    match &stores[0].kind {
        OpKind::RawStore {
            base,
            offset,
            value,
            item_ty: ValueType::Int,
            itemsize: 8,
            is_item_signed: true,
        } if base == ptr && value == keyhash => {
            assert!(
                matches!(defining_op(graph, offset), Some(OpKind::ConstInt(0))),
                "store offset is the zero constant"
            );
        }
        other => panic!("store must write keyhash through the spill, got {other:?}"),
    }
    let call = sink_call(graph);
    assert_eq!(call_arg(call, 0), flag, "flag stays the first argument");
    assert_eq!(
        call_arg(call, 1),
        ptr,
        "the raw parameter receives the address\n{}",
        op_lines(graph)
    );
    let loads = raw_ops(graph, |kind| matches!(kind, OpKind::RawLoad { .. }));
    let frees = raw_ops(graph, |kind| matches!(kind, OpKind::RawFree { .. }));
    assert_eq!(frees.len(), 1, "the spill is freed\n{}", op_lines(graph));
    assert!(
        matches!(&frees[0].kind, OpKind::RawFree { ptr: freed } if freed == ptr),
        "RawFree releases the spill"
    );
    let call_block = graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| std::ptr::eq(&op.kind, call))
        })
        .expect("call block");
    let pos = |pred: &dyn Fn(&OpKind) -> bool| {
        call_block
            .operations
            .iter()
            .position(|op| pred(&op.kind))
            .expect("op in the call block")
    };
    let malloc_at = pos(&|kind| matches!(kind, OpKind::RawMalloc { .. }));
    let store_at = pos(&|kind| matches!(kind, OpKind::RawStore { .. }));
    let call_at = pos(&|kind| std::ptr::eq(kind, call));
    let free_at = pos(&|kind| matches!(kind, OpKind::RawFree { .. }));
    assert!(malloc_at < store_at && store_at < call_at && call_at < free_at);
    if copy_out {
        assert_eq!(
            loads.len(),
            1,
            "mutable out-parameter reloads\n{}",
            op_lines(graph)
        );
        match &loads[0].kind {
            OpKind::RawLoad {
                base,
                offset,
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
            } if base == ptr => {
                assert!(matches!(
                    defining_op(graph, offset),
                    Some(OpKind::ConstInt(0))
                ));
            }
            other => panic!("reload must read the spill, got {other:?}"),
        }
        let load_at = pos(&|kind| matches!(kind, OpKind::RawLoad { .. }));
        assert!(call_at < load_at && load_at < free_at);
    } else {
        assert!(
            loads.is_empty(),
            "a shared or const parameter is not written back\n{}",
            op_lines(graph)
        );
    }
    let returned = root_op(graph, &returned_var(graph));
    if return_is_reload {
        match returned {
            Some(OpKind::RawLoad {
                base,
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
                ..
            }) if base == ptr => {}
            other => panic!("the caller must observe the reloaded word, got {other:?}"),
        }
    } else {
        match returned {
            Some(OpKind::Input {
                name,
                ty: ValueType::Int,
                ..
            }) if name == "keyhash" => {}
            other => panic!("the returned word must stay the pre-call input, got {other:?}"),
        }
    }
}

#[test]
fn mut_borrow_of_i64_passed_to_raw_pointer_reloads_the_word() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Mut"),
        Pass::Borrow {
            kind: "Mut",
            ret_local: 2,
        },
    );
    assert_address_spill(&graph, true, true);
}

#[test]
fn pre_call_copy_keeps_the_word_the_borrow_reloads() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Mut"),
        Pass::CopyThenMutBorrow,
    );
    assert_address_spill(&graph, true, false);
}

#[test]
fn shared_borrow_of_i64_into_const_pointer_is_not_copied_out() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Const"),
        Pass::Borrow {
            kind: "Shared",
            ret_local: 2,
        },
    );
    assert_address_spill(&graph, false, false);
}

#[test]
fn raw_scalar_address_spill_leaves_other_arguments_as_the_word() {
    let i64_ty = i64_ty();
    let cases: [(&str, &str, Value, Value, Pass); 5] = [
        (
            "&mut i64 parameter",
            "keyhash",
            i64_ty.clone(),
            borrow_ty(&i64_ty, "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 2,
            },
        ),
        (
            "already raw *mut i64",
            "keyhash",
            raw_ptr(&i64_ty, "Mut"),
            raw_ptr(&i64_ty, "Mut"),
            Pass::Direct { ret_local: 2 },
        ),
        (
            "&mut u8 into *mut u8",
            "byte",
            u8_ty(),
            raw_ptr(&u8_ty(), "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 1,
            },
        ),
        (
            "i64 passed by value",
            "keyhash",
            i64_ty.clone(),
            i64_ty.clone(),
            Pass::Direct { ret_local: 2 },
        ),
        (
            "&mut i64 into *mut u64",
            "keyhash",
            i64_ty.clone(),
            raw_ptr(&u64_ty(), "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 2,
            },
        ),
    ];
    for (label, word_name, word_ty, sink_param, pass) in cases {
        let graph = lower_probe(word_name, &word_ty, &sink_param, pass);
        assert_no_raw_address(&graph, label);
        let call = sink_call(&graph);
        assert_eq!(
            call_arg(call, 0),
            input_var(&graph, "flag"),
            "{label}: flag"
        );
        assert_eq!(
            call_arg(call, 1),
            input_var(&graph, word_name),
            "{label}: the argument stays the word\n{}",
            op_lines(&graph)
        );
    }
}

enum BorrowAlias {
    /// `&word` passed twice.
    SamePlace,
    /// `&left` and `&right`.
    TwoInputs,
    /// `let copy = word; f(&word, &copy)` — one word, two places.
    CopiedPlace,
}

fn probe_graph(type_decls: Value, fun_decls: Value, name: &str) -> FunctionGraph {
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": type_decls,
            "fun_decls": fun_decls,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err}"))
}

fn probe_parts() -> (
    Value,
    Value,
    impl Fn(&[&str]) -> Value,
    impl Fn(u64, Option<&str>, &Value) -> Value,
) {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let span_meta = span.clone();
    let meta = move |path: &[&str]| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span_meta, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let span_local = span.clone();
    let local = move |index: u64, name: Option<&str>, ty: &Value| json!({"index": index, "name": name, "span": span_local, "ty": ty});
    (
        span,
        json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []}),
        meta,
        local,
    )
}

fn lower_two_borrows(alias: BorrowAlias) -> FunctionGraph {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let ret = i64_ty();
    let borrowed = borrow_ty(&word, "Shared");
    let ptr = raw_ptr(&word, "Const");
    let (arg_count, inputs, locals, statements, call_args, ret_local, ret_ty) = match alias {
        BorrowAlias::SamePlace => {
            let borrow_a = place(2, &borrowed);
            let borrow_b = place(3, &borrowed);
            let word_place = place(1, &word);
            (
                1u64,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, None, &borrowed),
                    local(3, None, &borrowed),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow_a.clone(), {"Ref": {
                        "place": word_place.clone(), "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [borrow_b.clone(), {"Ref": {
                        "place": word_place, "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow_a}), json!({"Move": borrow_b})],
                1u64,
                word.clone(),
            )
        }
        BorrowAlias::TwoInputs => {
            let left = place(1, &word);
            let right = place(2, &word);
            let borrow_left = place(3, &borrowed);
            let borrow_right = place(4, &borrowed);
            (
                2,
                vec![word.clone(), word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("left"), &word),
                    local(2, Some("right"), &word),
                    local(3, None, &borrowed),
                    local(4, None, &borrowed),
                    local(5, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow_left.clone(), {"Ref": {
                        "place": left, "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [borrow_right.clone(), {"Ref": {
                        "place": right, "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow_left}), json!({"Move": borrow_right})],
                1,
                word.clone(),
            )
        }
        BorrowAlias::CopiedPlace => {
            let word_place = place(1, &word);
            let copy = place(2, &word);
            let borrow_word = place(3, &borrowed);
            let borrow_copy = place(4, &borrowed);
            (
                1,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, Some("copy"), &word),
                    local(3, None, &borrowed),
                    local(4, None, &borrowed),
                    local(5, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [
                        copy.clone(),
                        {"Use": [{"Copy": word_place.clone()}, "Yes"]}
                    ]}}),
                    json!({"span": span, "kind": {"Assign": [borrow_word.clone(), {"Ref": {
                        "place": word_place, "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [borrow_copy.clone(), {"Ref": {
                        "place": copy, "kind": "Shared", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow_word}), json!({"Move": borrow_copy})],
                1,
                word.clone(),
            )
        }
    };
    let dest = place(locals.len() as u64 - 1, &ret);
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        inputs,
        json!({"Unstructured": {"span": span, "locals": {"arg_count": arg_count, "locals": locals}, "body": [
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args, "dest": dest},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret_ty)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let params = vec![ptr.clone(), ptr.clone()];
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        params.clone(),
        idle_body(&ret, &params),
    );
    probe_graph(json!([]), json!([caller, sink]), "write_hash")
}

fn malloc_ptrs(graph: &FunctionGraph) -> Vec<majit_translate::flowspace::model::Variable> {
    raw_ops(graph, |kind| matches!(kind, OpKind::RawMalloc { .. }))
        .into_iter()
        .map(|op| {
            match &op.kind {
                OpKind::RawMalloc { owner, zero: false } if owner == "Tuple<i64>" => {}
                other => panic!("spill owner is Tuple<i64>, got {other:?}"),
            }
            op.result.clone().expect("RawMalloc result")
        })
        .collect()
}

fn assert_const_spills(
    graph: &FunctionGraph,
    ptrs: &[majit_translate::flowspace::model::Variable],
) {
    let stores = raw_ops(graph, |kind| matches!(kind, OpKind::RawStore { .. }));
    assert_eq!(
        stores.len(),
        ptrs.len(),
        "one store per address\n{}",
        op_lines(graph)
    );
    let frees = raw_ops(graph, |kind| matches!(kind, OpKind::RawFree { .. }));
    assert_eq!(
        frees.len(),
        ptrs.len(),
        "one free per address\n{}",
        op_lines(graph)
    );
    assert!(
        raw_ops(graph, |kind| matches!(kind, OpKind::RawLoad { .. })).is_empty(),
        "a const parameter is not written back\n{}",
        op_lines(graph)
    );
    let call = sink_call(graph);
    let call_block = graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| std::ptr::eq(&op.kind, call))
        })
        .expect("call block");
    let pos = |pred: &dyn Fn(&OpKind) -> bool| {
        call_block
            .operations
            .iter()
            .position(|op| pred(&op.kind))
            .expect("op")
    };
    let call_at = pos(&|kind| std::ptr::eq(kind, call));
    let first_free = call_block
        .operations
        .iter()
        .position(|op| matches!(op.kind, OpKind::RawFree { .. }))
        .expect("free");
    let last_store = call_block
        .operations
        .iter()
        .rposition(|op| matches!(op.kind, OpKind::RawStore { .. }))
        .expect("store");
    assert!(last_store < call_at && call_at < first_free);
}

#[test]
fn two_borrows_of_one_local_share_the_spill_address() {
    let graph = lower_two_borrows(BorrowAlias::SamePlace);
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        1,
        "one address for one place\n{}",
        op_lines(&graph)
    );
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(call_arg(call, 1), &ptrs[0]);
    match &raw_ops(&graph, |kind| matches!(kind, OpKind::RawStore { .. }))[0].kind {
        OpKind::RawStore { value, base, .. }
            if base == &ptrs[0] && value == input_var(&graph, "word") => {}
        other => panic!("the store writes the borrowed word, got {other:?}"),
    }
    assert_const_spills(&graph, &ptrs);
}

#[test]
fn borrows_of_two_locals_keep_distinct_spill_addresses() {
    let graph = lower_two_borrows(BorrowAlias::TwoInputs);
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        2,
        "two places, two addresses\n{}",
        op_lines(&graph)
    );
    assert_ne!(&ptrs[0], &ptrs[1]);
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(call_arg(call, 1), &ptrs[1]);
    let stores = raw_ops(&graph, |kind| matches!(kind, OpKind::RawStore { .. }));
    match (&stores[0].kind, &stores[1].kind) {
        (
            OpKind::RawStore {
                base: base0,
                value: value0,
                ..
            },
            OpKind::RawStore {
                base: base1,
                value: value1,
                ..
            },
        ) if base0 == &ptrs[0]
            && base1 == &ptrs[1]
            && value0 == input_var(&graph, "left")
            && value1 == input_var(&graph, "right") => {}
        other => panic!("each store writes its own local, got {other:?}"),
    }
    assert_const_spills(&graph, &ptrs);
}

#[test]
fn copied_local_keeps_a_distinct_spill_address() {
    let graph = lower_two_borrows(BorrowAlias::CopiedPlace);
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        2,
        "a copy is a different place even when the word aliases\n{}",
        op_lines(&graph)
    );
    assert_ne!(&ptrs[0], &ptrs[1]);
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(call_arg(call, 1), &ptrs[1]);
    let word = input_var(&graph, "word");
    for store in raw_ops(&graph, |kind| matches!(kind, OpKind::RawStore { .. })) {
        match &store.kind {
            OpKind::RawStore { value, .. } if value == word => {}
            other => panic!("both stores write the aliased word, got {other:?}"),
        }
    }
    assert_const_spills(&graph, &ptrs);
}

#[derive(Clone, Copy)]
enum RawAlias {
    /// `q = &mut word as *mut i64`, passed with `&mut word`.
    CastSamePlace,
    /// `q = &raw mut other`.
    OtherPlace,
    /// `q` is an argument. This body never names its referent.
    Unknown,
    /// `&mut pair.word` with `q = &raw mut pair`.
    CoveringPlace,
    /// `q = &mut word as *mut i64 as usize`, passed with `&mut word`.
    UsizeSamePlace,
    /// The callee returns that `usize`.
    UsizeReturned,
    /// `q` is a constant `usize` with no pointer provenance.
    UsizeConst,
    /// `q = &mut other as usize`.
    UsizeOtherPlace,
    /// `q = &mut word as u32`. The integer is narrower than a pointer.
    UsizeNarrow,
    /// `&mut pair.word` with `q = &pair as usize`.
    UsizeCovering,
}

fn lower_raw_alias(
    alias: RawAlias,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let ret = i64_ty();
    let borrowed = borrow_ty(&word, "Mut");
    let ptr = raw_ptr(&word, "Mut");
    let pair_ty = json!({"Adt": {"id": 0, "generics": generics.clone()}});
    let pair_ptr = raw_ptr(&pair_ty, "Mut");
    let (arg_count, inputs, locals, statements, call_args, decls, q_ty) = match alias {
        RawAlias::CastSamePlace => {
            let word_place = place(1, &word);
            let borrow = place(2, &borrowed);
            let q = place(3, &ptr);
            (
                1u64,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, None, &borrowed),
                    local(3, Some("q"), &ptr),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_to(q.clone(), ptr_cast(borrow.clone(), &borrowed, &ptr)),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                ptr.clone(),
            )
        }
        RawAlias::OtherPlace => {
            let word_place = place(1, &word);
            let other = place(2, &word);
            let borrow = place(3, &borrowed);
            let q = place(4, &ptr);
            (
                2,
                vec![word.clone(), word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, Some("other"), &word),
                    local(3, None, &borrowed),
                    local(4, Some("q"), &ptr),
                    local(5, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [q.clone(), {"RawPtr": {
                        "place": other, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                ptr.clone(),
            )
        }
        RawAlias::Unknown => {
            let word_place = place(1, &word);
            let borrow = place(3, &borrowed);
            let q = place(2, &ptr);
            (
                2,
                vec![word.clone(), ptr.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, Some("q"), &ptr),
                    local(3, None, &borrowed),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow}), json!({"Copy": q})],
                json!([]),
                ptr.clone(),
            )
        }
        RawAlias::CoveringPlace => {
            let field = json!({
                "kind": {"Projection": [place(1, &pair_ty), {"Field": [null, 0]}]},
                "ty": word
            });
            let borrow = place(2, &borrowed);
            let q = place(3, &pair_ptr);
            (
                1,
                vec![pair_ty.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("pair"), &pair_ty),
                    local(2, None, &borrowed),
                    local(3, Some("q"), &pair_ptr),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": field, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [q.clone(), {"RawPtr": {
                        "place": place(1, &pair_ty), "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([{
                    "def_id": 0,
                    "item_meta": meta(&["probe", "Pair"]),
                    "kind": {"Struct": [{"name": "word", "ty": word, "attr_info": null}]}
                }]),
                pair_ptr.clone(),
            )
        }
        RawAlias::UsizeSamePlace | RawAlias::UsizeReturned => {
            let word_place = place(1, &word);
            let borrow = place(2, &borrowed);
            let raw = place(3, &ptr);
            let q = place(4, &usize_ty());
            (
                1,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, None, &borrowed),
                    local(3, None, &ptr),
                    local(4, Some("q"), &usize_ty()),
                    local(5, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_to(raw.clone(), ptr_cast(borrow.clone(), &borrowed, &ptr)),
                    assign_scalar_cast(4, 3, &ptr, &usize_ty()),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                usize_ty(),
            )
        }
        RawAlias::UsizeConst => {
            let word_place = place(1, &word);
            let borrow = place(2, &borrowed);
            let q = place(3, &usize_ty());
            (
                1,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, None, &borrowed),
                    local(3, Some("q"), &usize_ty()),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_to(
                        q.clone(),
                        json!({"Use": [{"Const": [
                        {"Integer": {"Unsigned": ["Usize", "0"]}},
                        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
                    ]}, "Yes"]}),
                    ),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                usize_ty(),
            )
        }
        RawAlias::UsizeOtherPlace => {
            let word_place = place(1, &word);
            let other = place(2, &word);
            let borrow = place(3, &borrowed);
            let other_borrow = place(4, &borrowed);
            let q = place(5, &usize_ty());
            (
                2,
                vec![word.clone(), word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, Some("other"), &word),
                    local(3, None, &borrowed),
                    local(4, None, &borrowed),
                    local(5, Some("q"), &usize_ty()),
                    local(6, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [other_borrow.clone(), {"Ref": {
                        "place": other, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_scalar_cast(5, 4, &borrowed, &usize_ty()),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                usize_ty(),
            )
        }
        RawAlias::UsizeNarrow => {
            let word_place = place(1, &word);
            let borrow = place(2, &borrowed);
            let narrow = u32_ty();
            let q = place(3, &narrow);
            (
                1,
                vec![word.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("word"), &word),
                    local(2, None, &borrowed),
                    local(3, Some("q"), &narrow),
                    local(4, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word_place, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_scalar_cast(3, 2, &borrowed, &narrow),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([]),
                narrow,
            )
        }
        RawAlias::UsizeCovering => {
            let field = json!({
                "kind": {"Projection": [place(1, &pair_ty), {"Field": [null, 0]}]},
                "ty": word
            });
            let borrow = place(2, &borrowed);
            let pair_ref = place(3, &borrow_ty(&pair_ty, "Mut"));
            let q = place(4, &usize_ty());
            (
                1,
                vec![pair_ty.clone()],
                vec![
                    local(0, None, &ret),
                    local(1, Some("pair"), &pair_ty),
                    local(2, None, &borrowed),
                    local(3, None, &borrow_ty(&pair_ty, "Mut")),
                    local(4, Some("q"), &usize_ty()),
                    local(5, None, &ret),
                ],
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": field, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    json!({"span": span, "kind": {"Assign": [pair_ref.clone(), {"Ref": {
                        "place": place(1, &pair_ty), "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                    assign_scalar_cast(4, 3, &borrow_ty(&pair_ty, "Mut"), &usize_ty()),
                ],
                vec![json!({"Move": borrow}), json!({"Move": q})],
                json!([{
                    "def_id": 0,
                    "item_meta": meta(&["probe", "Pair"]),
                    "kind": {"Struct": [{"name": "word", "ty": word, "attr_info": null}]}
                }]),
                usize_ty(),
            )
        }
    };
    let ret_local = locals.len() as u64 - 1;
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        inputs,
        json!({"Unstructured": {"span": span, "locals": {"arg_count": arg_count, "locals": locals}, "body": [
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args, "dest": place(ret_local, &ret)},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let (sink_extra, sink_statements): (Vec<Value>, Vec<Value>) = match alias {
        RawAlias::UsizeSamePlace => (
            vec![local(3, Some("r"), &ptr)],
            vec![
                assign_deref(1, &ptr, const_use()),
                assign_scalar_cast(3, 2, &q_ty, &ptr),
                assign_to(place(0, &ret), copy_use(deref_place(place(3, &ptr), &ret))),
            ],
        ),
        RawAlias::UsizeReturned => (vec![], vec![assign_scalar_cast(0, 2, &q_ty, &ret)]),
        RawAlias::UsizeConst
        | RawAlias::UsizeOtherPlace
        | RawAlias::UsizeNarrow
        | RawAlias::UsizeCovering => (
            vec![],
            vec![
                assign_deref(1, &ptr, const_use()),
                assign_to(place(0, &ret), const_use()),
            ],
        ),
        RawAlias::CastSamePlace
        | RawAlias::OtherPlace
        | RawAlias::Unknown
        | RawAlias::CoveringPlace => (
            vec![],
            vec![
                assign_deref(1, &ptr, const_use()),
                assign_to(place(0, &ret), copy_use(deref_place(place(2, &q_ty), &ret))),
            ],
        ),
    };
    let mut sink_locals = vec![
        local(0, None, &ret),
        local(1, Some("p"), &ptr),
        local(2, Some("q"), &q_ty),
    ];
    sink_locals.extend(sink_extra);
    let sink_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 2, "locals": sink_locals},
        "body": [{"statements": sink_statements, "terminator": {"span": span, "kind": "Return"}}]
    }});
    let sink = fun(1, &["probe", "sink_pair"], vec![ptr, q_ty], sink_body);
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": decls,
            "fun_decls": [caller, sink],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, "write_hash")
}

#[test]
fn raw_pointer_to_the_spilled_place_uses_that_address() {
    let graph = lower_raw_alias(RawAlias::CastSamePlace).unwrap_or_else(|err| {
        panic!("a raw pointer of the spilled place must use that address: {err}")
    });
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        1,
        "one address for the place\n{}",
        op_lines(&graph)
    );
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(call_arg(call, 1), &ptrs[0]);
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn raw_pointer_to_another_place_stays_put() {
    let graph = lower_raw_alias(RawAlias::OtherPlace).unwrap_or_else(|err| {
        panic!("a raw pointer of another place must still free the spill: {err}")
    });
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(call_arg(call, 1), &ptrs[0]);
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn raw_pointer_with_no_referent_is_not_lowered() {
    let err = lower_raw_alias(RawAlias::Unknown)
        .expect_err("a raw pointer with no referent must not lower");
    let msg = err.to_string();
    assert!(msg.contains("unspilled raw argument"), "{msg}");
}

#[test]
fn raw_pointer_covering_the_spilled_field_is_not_lowered() {
    let err = lower_raw_alias(RawAlias::CoveringPlace)
        .expect_err("a pointer that covers the spilled field must not lower");
    let msg = err.to_string();
    assert!(msg.contains("unspilled raw argument"), "{msg}");
}

#[test]
fn usize_alias_of_the_spilled_place_uses_that_address() {
    let graph = lower_raw_alias(RawAlias::UsizeSamePlace).unwrap_or_else(|err| {
        panic!("a usize cast of the spilled place must use that address: {err}")
    });
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        1,
        "one address for the place\n{}",
        op_lines(&graph)
    );
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(call_arg(call, 1), &ptrs[0]);
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn usize_alias_returned_from_the_callee_is_not_lowered() {
    let err = lower_raw_alias(RawAlias::UsizeReturned)
        .expect_err("returning the usize form of the spill address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("call result"), "{msg}");
}

#[test]
fn usize_constant_stays_beside_the_spill() {
    let graph = lower_raw_alias(RawAlias::UsizeConst)
        .unwrap_or_else(|err| panic!("a constant usize must still free the spill: {err}"));
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(call_arg(call, 1), &ptrs[0]);
}

#[test]
fn usize_of_another_place_stays_put() {
    let graph = lower_raw_alias(RawAlias::UsizeOtherPlace)
        .unwrap_or_else(|err| panic!("a usize of another place must still free the spill: {err}"));
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(call_arg(call, 1), &ptrs[0]);
}

#[test]
fn narrow_integer_of_the_spilled_place_stays_put() {
    let graph = lower_raw_alias(RawAlias::UsizeNarrow).unwrap_or_else(|err| {
        panic!("a u32 cast of the spilled place must still free the spill: {err}")
    });
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(call_arg(call, 1), &ptrs[0]);
}

#[test]
fn usize_covering_the_spilled_field_is_not_lowered() {
    let err = lower_raw_alias(RawAlias::UsizeCovering)
        .expect_err("a usize that covers the spilled field must not lower");
    let msg = err.to_string();
    assert!(msg.contains("unspilled raw argument"), "{msg}");
}

#[derive(Clone, Copy)]
enum PackedArg {
    /// `f(&mut word, (q,))` with `q = &raw mut word`.
    TupleRawSame,
    /// `q = &raw mut other`.
    TupleRawOther,
    /// `f(&mut word, ((q,),))`.
    NestedTupleRawSame,
    /// `f(&mut word, Hold { p: q })`.
    StructRawSame,
    /// `let pair2 = pair; f(&mut word, pair2)`.
    CopiedTupleRawSame,
    /// `f(&mut word, (1i64, 2i64))`.
    TupleI64,
    /// `q = &mut word as *mut i64 as usize`, packed in a tuple.
    TupleUsizeSame,
    /// `q = &mut other as usize`.
    TupleUsizeOther,
    /// `q` is the constant `0usize`.
    TupleUsizeConst,
    /// `q` is an incoming `*mut i64`.
    TupleUnknownRaw,
    /// `q = &mut word as u32`.
    TupleNarrow,
    /// `f(&mut word, &pair)` and `pair` holds `q`.
    RefOfTupleRawSame,
    /// `pair = (q_other,); pair.0 = q` after `q` names `word`.
    FieldStoreRawSame,
    /// `f(&mut word, [q])`.
    ArrayRawSame,
}

fn i64_const_operand(text: &str) -> Value {
    json!({"Const": [
        {"Integer": {"Signed": ["I64", text]}},
        {"Scalar": {"Integer": {"Signed": "I64"}}}
    ]})
}

fn usize_const_operand(text: &str) -> Value {
    json!({"Const": [
        {"Integer": {"Unsigned": ["Usize", text]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]})
}

fn raw_mut_assign(dest: u64, dest_ty: &Value, src: Value) -> Value {
    assign_to(
        place(dest, dest_ty),
        json!({"RawPtr": {"place": src, "kind": "Mut", "ptr_metadata": null}}),
    )
}

fn lower_packed_alias(
    alias: PackedArg,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let ret = i64_ty();
    let borrowed = borrow_ty(&word, "Mut");
    let ptr = raw_ptr(&word, "Mut");
    let narrow = u32_ty();
    let bits = usize_ty();
    let struct_ty = json!({"Adt": {"id": 0, "generics": generics.clone()}});
    let ptr_tuple = tuple_ty(vec![ptr.clone()]);
    let nested_tuple = tuple_ty(vec![ptr_tuple.clone()]);
    let i64_tuple = tuple_ty(vec![word.clone(), word.clone()]);
    let usize_tuple = tuple_ty(vec![bits.clone()]);
    let narrow_tuple = tuple_ty(vec![narrow.clone()]);
    let array = array_ty(&ptr);
    let pair_ref = borrow_ty(&ptr_tuple, "Shared");
    let (arg_count, inputs, locals, statements, second_ty, decls) = match alias {
        PackedArg::TupleRawSame => (
            1u64,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("pair"), &ptr_tuple),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(3, &ptr)})]),
                ),
            ],
            ptr_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleRawOther => (
            2,
            vec![word.clone(), word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("other"), &word),
                local(3, None, &borrowed),
                local(4, Some("q"), &ptr),
                local(5, Some("pair"), &ptr_tuple),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                raw_mut_assign(4, &ptr, place(2, &word)),
                assign_to(
                    place(5, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &ptr)})]),
                ),
            ],
            ptr_tuple.clone(),
            json!([]),
        ),
        PackedArg::NestedTupleRawSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("inner"), &ptr_tuple),
                local(5, Some("outer"), &nested_tuple),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(3, &ptr)})]),
                ),
                assign_to(
                    place(5, &nested_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &ptr_tuple)})]),
                ),
            ],
            nested_tuple.clone(),
            json!([]),
        ),
        PackedArg::StructRawSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("hold"), &struct_ty),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &struct_ty),
                    json!({"Aggregate": [{"Adt": [0, null]}, [json!({"Copy": place(3, &ptr)})]]}),
                ),
            ],
            struct_ty.clone(),
            json!([{
                "def_id": 0,
                "item_meta": meta(&["probe", "Hold"]),
                "kind": {"Struct": [field_decl("p", &ptr)]},
                "src": "Normal"
            }]),
        ),
        PackedArg::CopiedTupleRawSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("pair"), &ptr_tuple),
                local(5, Some("pair2"), &ptr_tuple),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(3, &ptr)})]),
                ),
                assign_to(place(5, &ptr_tuple), copy_use(place(4, &ptr_tuple))),
            ],
            ptr_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleI64 => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("pair"), &i64_tuple),
                local(4, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &i64_tuple),
                    tuple_of(vec![i64_const_operand("1"), i64_const_operand("2")]),
                ),
            ],
            i64_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleUsizeSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ptr),
                local(4, Some("q"), &bits),
                local(5, Some("pair"), &usize_tuple),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &ptr),
                    ptr_cast(place(2, &borrowed), &borrowed, &ptr),
                ),
                assign_scalar_cast(4, 3, &ptr, &bits),
                assign_to(
                    place(5, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &bits)})]),
                ),
            ],
            usize_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleUsizeOther => (
            2,
            vec![word.clone(), word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("other"), &word),
                local(3, None, &borrowed),
                local(4, None, &borrowed),
                local(5, Some("q"), &bits),
                local(6, Some("pair"), &usize_tuple),
                local(7, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                ref_assign_mut(4, &borrowed, place(2, &word)),
                assign_scalar_cast(5, 4, &borrowed, &bits),
                assign_to(
                    place(6, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(5, &bits)})]),
                ),
            ],
            usize_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleUsizeConst => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("pair"), &usize_tuple),
                local(4, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &usize_tuple),
                    tuple_of(vec![usize_const_operand("0")]),
                ),
            ],
            usize_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleUnknownRaw => (
            2,
            vec![word.clone(), ptr.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("q"), &ptr),
                local(3, None, &borrowed),
                local(4, Some("pair"), &ptr_tuple),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                assign_to(
                    place(4, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(2, &ptr)})]),
                ),
            ],
            ptr_tuple.clone(),
            json!([]),
        ),
        PackedArg::TupleNarrow => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &narrow),
                local(4, Some("pair"), &narrow_tuple),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_scalar_cast(3, 2, &borrowed, &narrow),
                assign_to(
                    place(4, &narrow_tuple),
                    tuple_of(vec![json!({"Copy": place(3, &narrow)})]),
                ),
            ],
            narrow_tuple.clone(),
            json!([]),
        ),
        PackedArg::RefOfTupleRawSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("pair"), &ptr_tuple),
                local(5, Some("r"), &pair_ref),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(3, &ptr)})]),
                ),
                ref_assign(5, &pair_ref, place(4, &ptr_tuple)),
            ],
            pair_ref.clone(),
            json!([]),
        ),
        PackedArg::FieldStoreRawSame => (
            2,
            vec![word.clone(), word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("other"), &word),
                local(3, None, &borrowed),
                local(4, Some("q_other"), &ptr),
                local(5, Some("q"), &ptr),
                local(6, Some("pair"), &ptr_tuple),
                local(7, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                raw_mut_assign(4, &ptr, place(2, &word)),
                raw_mut_assign(5, &ptr, place(1, &word)),
                assign_to(
                    place(6, &ptr_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &ptr)})]),
                ),
                assign_to(
                    field_place(6, &ptr_tuple, 0, &ptr),
                    copy_use(place(5, &ptr)),
                ),
            ],
            ptr_tuple.clone(),
            json!([]),
        ),
        PackedArg::ArrayRawSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &ptr),
                local(4, Some("arr"), &array),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                raw_mut_assign(3, &ptr, place(1, &word)),
                assign_to(
                    place(4, &array),
                    array_of(&ptr, vec![json!({"Copy": place(3, &ptr)})]),
                ),
            ],
            array.clone(),
            json!([]),
        ),
    };
    let ret_local = locals.len() as u64 - 1;
    let pair_local = ret_local - 1;
    let fun = |id: u64, name: &[&str], fun_inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": fun_inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        inputs,
        json!({"Unstructured": {"span": span, "locals": {"arg_count": arg_count, "locals": locals}, "body": [
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": [{"Move": place(if arg_count == 1 { 2 } else { 3 }, &borrowed)}, {"Move": place(pair_local, &second_ty)}],
                    "dest": place(ret_local, &ret)},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let sink_inputs = vec![ptr.clone(), second_ty.clone()];
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        sink_inputs.clone(),
        idle_body(&ret, &sink_inputs),
    );
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": decls,
            "fun_decls": [caller, sink],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, "write_hash")
}

fn ref_assign_mut(dest: u64, dest_ty: &Value, src: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": {"Assign": [
        place(dest, dest_ty),
        {"Ref": {"place": src, "kind": "Mut", "ptr_metadata": null}}
    ]}})
}

fn assert_packed_escapes(alias: PackedArg) {
    let err = lower_packed_alias(alias).expect_err("a packed spill alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("unspilled raw argument"), "{msg}");
}

fn assert_packed_frees(alias: PackedArg) {
    let graph = lower_packed_alias(alias).unwrap_or_else(|err| {
        panic!("a packed value that misses the spill must still lower: {err}")
    });
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(
        call_arg(call, 1),
        &ptrs[0],
        "the packed argument stays the caller's value\n{}",
        op_lines(&graph)
    );
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn tuple_raw_alias_of_the_spilled_place_is_not_lowered() {
    assert_packed_escapes(PackedArg::TupleRawSame);
}

#[test]
fn tuple_raw_alias_of_another_place_still_frees() {
    assert_packed_frees(PackedArg::TupleRawOther);
}

#[test]
fn nested_tuple_raw_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::NestedTupleRawSame);
}

#[test]
fn struct_raw_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::StructRawSame);
}

#[test]
fn copied_tuple_raw_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::CopiedTupleRawSame);
}

#[test]
fn tuple_of_i64_beside_the_spill_still_frees() {
    assert_packed_frees(PackedArg::TupleI64);
}

#[test]
fn tuple_usize_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::TupleUsizeSame);
}

#[test]
fn tuple_usize_of_another_place_still_frees() {
    assert_packed_frees(PackedArg::TupleUsizeOther);
}

#[test]
fn tuple_usize_constant_beside_the_spill_still_frees() {
    assert_packed_frees(PackedArg::TupleUsizeConst);
}

#[test]
fn tuple_unknown_raw_is_not_lowered() {
    assert_packed_escapes(PackedArg::TupleUnknownRaw);
}

#[test]
fn tuple_narrow_integer_beside_the_spill_still_frees() {
    assert_packed_frees(PackedArg::TupleNarrow);
}

#[test]
fn reference_to_a_tuple_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::RefOfTupleRawSame);
}

#[test]
fn field_store_of_a_raw_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::FieldStoreRawSame);
}

#[test]
fn array_raw_alias_is_not_lowered() {
    assert_packed_escapes(PackedArg::ArrayRawSame);
}

#[derive(Clone, Copy)]
enum TracedInt {
    /// `f(&mut word, pair.0)` with `pair = (q,)` and `q` naming `word`.
    FieldSame,
    /// `pair.0` names `other`.
    FieldOther,
    /// `pair.0` is the literal `0usize`.
    FieldConst,
    /// The second parameter is an incoming `usize`.
    Param,
    /// `q = usize_source(); f(&mut word, q)`.
    CallResult,
    /// The second argument is the literal `0usize`.
    DirectConst,
    /// `f(&mut word, (q,))` and `q` is an incoming `usize`.
    TupleParam,
    /// `f(&mut word, (pair.0,))` and `pair.0` names `word`.
    TupleFieldSame,
    /// `q = 1usize + 2usize`.
    Sum,
    /// `let bits = pair.0` and `pair.0` names `word`.
    CopiedField,
    /// `pair = (q,); pair.0 = q` with `q` naming `word`. Pass `pair.0`.
    SameStore,
    /// `pair = (0usize,); pair.0 = q` with `q` naming `word`. Pass `pair.0`.
    ConflictStore,
    /// `q = &mut word as u8 as usize`.
    RoundTripU8,
}

fn lower_traced_int(
    kind: TracedInt,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let ret = i64_ty();
    let borrowed = borrow_ty(&word, "Mut");
    let ptr = raw_ptr(&word, "Mut");
    let bits = usize_ty();
    let narrow = u8_ty();
    let usize_tuple = tuple_ty(vec![bits.clone()]);
    let (arg_count, inputs, locals, statements, call_args, second_ty) = match kind {
        TracedInt::FieldSame | TracedInt::SameStore => (
            1u64,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ptr),
                local(4, Some("q"), &bits),
                local(5, Some("pair"), &usize_tuple),
                local(6, None, &ret),
            ],
            {
                let mut stmts = vec![
                    ref_assign_mut(2, &borrowed, place(1, &word)),
                    assign_to(
                        place(3, &ptr),
                        ptr_cast(place(2, &borrowed), &borrowed, &ptr),
                    ),
                    assign_scalar_cast(4, 3, &ptr, &bits),
                    assign_to(
                        place(5, &usize_tuple),
                        tuple_of(vec![json!({"Copy": place(4, &bits)})]),
                    ),
                ];
                if matches!(kind, TracedInt::SameStore) {
                    stmts.push(assign_to(
                        field_place(5, &usize_tuple, 0, &bits),
                        copy_use(place(4, &bits)),
                    ));
                }
                stmts
            },
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": field_place(5, &usize_tuple, 0, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::FieldOther => (
            2,
            vec![word.clone(), word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("other"), &word),
                local(3, None, &borrowed),
                local(4, None, &borrowed),
                local(5, Some("q"), &bits),
                local(6, Some("pair"), &usize_tuple),
                local(7, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                ref_assign_mut(4, &borrowed, place(2, &word)),
                assign_scalar_cast(5, 4, &borrowed, &bits),
                assign_to(
                    place(6, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(5, &bits)})]),
                ),
            ],
            vec![
                json!({"Move": place(3, &borrowed)}),
                json!({"Move": field_place(6, &usize_tuple, 0, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::FieldConst => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("pair"), &usize_tuple),
                local(4, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &usize_tuple),
                    tuple_of(vec![usize_const_operand("0")]),
                ),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": field_place(3, &usize_tuple, 0, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::Param => (
            2,
            vec![word.clone(), bits.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("q"), &bits),
                local(3, None, &borrowed),
                local(4, None, &ret),
            ],
            vec![ref_assign_mut(3, &borrowed, place(1, &word))],
            vec![
                json!({"Move": place(3, &borrowed)}),
                json!({"Copy": place(2, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::CallResult => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &bits),
                local(4, None, &ret),
            ],
            vec![ref_assign_mut(2, &borrowed, place(1, &word))],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": place(3, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::DirectConst => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ret),
            ],
            vec![ref_assign_mut(2, &borrowed, place(1, &word))],
            vec![
                json!({"Move": place(2, &borrowed)}),
                usize_const_operand("0"),
            ],
            bits.clone(),
        ),
        TracedInt::TupleParam => (
            2,
            vec![word.clone(), bits.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, Some("q"), &bits),
                local(3, None, &borrowed),
                local(4, Some("pair"), &usize_tuple),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(3, &borrowed, place(1, &word)),
                assign_to(
                    place(4, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(2, &bits)})]),
                ),
            ],
            vec![
                json!({"Move": place(3, &borrowed)}),
                json!({"Move": place(4, &usize_tuple)}),
            ],
            usize_tuple.clone(),
        ),
        TracedInt::TupleFieldSame => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ptr),
                local(4, Some("q"), &bits),
                local(5, Some("pair"), &usize_tuple),
                local(6, Some("outer"), &usize_tuple),
                local(7, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &ptr),
                    ptr_cast(place(2, &borrowed), &borrowed, &ptr),
                ),
                assign_scalar_cast(4, 3, &ptr, &bits),
                assign_to(
                    place(5, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &bits)})]),
                ),
                assign_to(
                    place(6, &usize_tuple),
                    tuple_of(vec![json!({
                        "Copy": field_place(5, &usize_tuple, 0, &bits)
                    })]),
                ),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": place(6, &usize_tuple)}),
            ],
            usize_tuple.clone(),
        ),
        TracedInt::Sum => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("q"), &bits),
                local(4, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &bits),
                    json!({"BinaryOp": [
                        "Add",
                        usize_const_operand("1"),
                        usize_const_operand("2")
                    ]}),
                ),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": place(3, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::CopiedField => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ptr),
                local(4, Some("q"), &bits),
                local(5, Some("pair"), &usize_tuple),
                local(6, Some("bits"), &bits),
                local(7, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &ptr),
                    ptr_cast(place(2, &borrowed), &borrowed, &ptr),
                ),
                assign_scalar_cast(4, 3, &ptr, &bits),
                assign_to(
                    place(5, &usize_tuple),
                    tuple_of(vec![json!({"Copy": place(4, &bits)})]),
                ),
                assign_to(
                    place(6, &bits),
                    copy_use(field_place(5, &usize_tuple, 0, &bits)),
                ),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": place(6, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::ConflictStore => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, None, &ptr),
                local(4, Some("q"), &bits),
                local(5, Some("pair"), &usize_tuple),
                local(6, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_to(
                    place(3, &ptr),
                    ptr_cast(place(2, &borrowed), &borrowed, &ptr),
                ),
                assign_scalar_cast(4, 3, &ptr, &bits),
                assign_to(
                    place(5, &usize_tuple),
                    tuple_of(vec![usize_const_operand("0")]),
                ),
                assign_to(
                    field_place(5, &usize_tuple, 0, &bits),
                    copy_use(place(4, &bits)),
                ),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": field_place(5, &usize_tuple, 0, &bits)}),
            ],
            bits.clone(),
        ),
        TracedInt::RoundTripU8 => (
            1,
            vec![word.clone()],
            vec![
                local(0, None, &ret),
                local(1, Some("word"), &word),
                local(2, None, &borrowed),
                local(3, Some("narrow"), &narrow),
                local(4, Some("q"), &bits),
                local(5, None, &ret),
            ],
            vec![
                ref_assign_mut(2, &borrowed, place(1, &word)),
                assign_scalar_cast(3, 2, &borrowed, &narrow),
                assign_scalar_cast(4, 3, &narrow, &bits),
            ],
            vec![
                json!({"Move": place(2, &borrowed)}),
                json!({"Move": place(4, &bits)}),
            ],
            bits.clone(),
        ),
    };
    let ret_local = locals.len() as u64 - 1;
    let fun = |id: u64, name: &[&str], fun_inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": fun_inputs, "output": ret.clone()},
            "body": body
        })
    };
    let sink_call = json!({"statements": statements, "terminator": {"span": span, "kind": {"Call": {
        "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics.clone()}},
            "args": call_args, "dest": place(ret_local, &ret)},
        "target": if matches!(kind, TracedInt::CallResult) { 2 } else { 1 },
        "on_unwind": if matches!(kind, TracedInt::CallResult) { 3 } else { 2 }
    }}}});
    let body = if matches!(kind, TracedInt::CallResult) {
        json!([
            {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics.clone()}},
                    "args": [], "dest": place(3, &bits)},
                "target": 1, "on_unwind": 3
            }}}},
            sink_call,
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    } else {
        json!([
            sink_call,
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        inputs,
        json!({"Unstructured": {"span": span, "locals": {"arg_count": arg_count, "locals": locals}, "body": body}}),
    );
    let sink_inputs = vec![ptr.clone(), second_ty];
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        sink_inputs.clone(),
        idle_body(&ret, &sink_inputs),
    );
    let mut funs = vec![caller, sink];
    if matches!(kind, TracedInt::CallResult) {
        funs.push(json!({
            "def_id": 2,
            "item_meta": meta(&["probe", "usize_source"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": bits.clone()},
            "body": idle_body(&bits, &[])
        }));
    }
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": [],
            "fun_decls": funs,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, "write_hash")
}

fn assert_traced_escapes(kind: TracedInt) {
    let err = lower_traced_int(kind).expect_err("an untraced address-sized integer must not lower");
    let msg = err.to_string();
    assert!(msg.contains("unspilled raw argument"), "{msg}");
}

fn assert_traced_frees(kind: TracedInt) {
    let graph = lower_traced_int(kind)
        .unwrap_or_else(|err| panic!("an integer that misses the spill must still lower: {err}"));
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_ne!(
        call_arg(call, 1),
        &ptrs[0],
        "the integer stays the caller's value\n{}",
        op_lines(&graph)
    );
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn assert_traced_relocates(kind: TracedInt) {
    let graph = lower_traced_int(kind)
        .unwrap_or_else(|err| panic!("a field that names the spill must use that address: {err}"));
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(ptrs.len(), 1, "one spilled place\n{}", op_lines(&graph));
    let call = sink_call(&graph);
    assert_eq!(call_arg(call, 0), &ptrs[0]);
    assert_eq!(
        call_arg(call, 1),
        &ptrs[0],
        "the integer is the spill pointer\n{}",
        op_lines(&graph)
    );
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn field_usize_alias_uses_the_spill_address() {
    assert_traced_relocates(TracedInt::FieldSame);
}

#[test]
fn field_usize_of_another_place_still_frees() {
    assert_traced_frees(TracedInt::FieldOther);
}

#[test]
fn field_usize_constant_still_frees() {
    assert_traced_frees(TracedInt::FieldConst);
}

#[test]
fn usize_parameter_beside_the_spill_is_not_lowered() {
    assert_traced_escapes(TracedInt::Param);
}

#[test]
fn usize_call_result_beside_the_spill_is_not_lowered() {
    assert_traced_escapes(TracedInt::CallResult);
}

#[test]
fn direct_usize_constant_beside_the_spill_still_frees() {
    assert_traced_frees(TracedInt::DirectConst);
}

#[test]
fn tuple_of_a_usize_parameter_is_not_lowered() {
    assert_traced_escapes(TracedInt::TupleParam);
}

#[test]
fn tuple_of_a_field_usize_alias_is_not_lowered() {
    assert_traced_escapes(TracedInt::TupleFieldSame);
}

#[test]
fn usize_sum_beside_the_spill_is_not_lowered() {
    assert_traced_escapes(TracedInt::Sum);
}

#[test]
fn copied_field_usize_alias_uses_the_spill_address() {
    assert_traced_relocates(TracedInt::CopiedField);
}

#[test]
fn field_store_of_the_same_usize_alias_uses_the_spill_address() {
    assert_traced_relocates(TracedInt::SameStore);
}

#[test]
fn field_store_over_a_usize_constant_is_not_lowered() {
    assert_traced_escapes(TracedInt::ConflictStore);
}

#[test]
fn usize_round_trip_through_u8_is_not_lowered() {
    assert_traced_escapes(TracedInt::RoundTripU8);
}

fn lower_field_out() -> FunctionGraph {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let ret = i64_ty();
    let empty_g = generics.clone();
    let pair_ty = json!({"Adt": {"id": 0, "generics": empty_g}});
    let borrowed = borrow_ty(&word, "Mut");
    let field = json!({
        "kind": {"Projection": [place(1, &pair_ty), {"Field": [null, 0]}]},
        "ty": word
    });
    let borrow = place(2, &borrowed);
    let pair_decl = json!({
        "def_id": 0,
        "item_meta": meta(&["probe", "Pair"]),
        "kind": {"Struct": [{"name": "word", "ty": word, "attr_info": null}]}
    });
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        vec![pair_ty.clone()],
        json!({"Unstructured": {"span": span, "locals": {"arg_count": 1, "locals": [
            local(0, None, &ret),
            local(1, Some("pair"), &pair_ty),
            local(2, None, &borrowed),
            local(3, None, &ret)
        ]}, "body": [
            {"statements": [
                {"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                    "place": field, "kind": "Mut", "ptr_metadata": null
                }}]}}
            ], "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": [{"Move": borrow}], "dest": place(3, &ret)},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(3, &ret)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let params = vec![raw_ptr(&word, "Mut")];
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        params.clone(),
        idle_body(&ret, &params),
    );
    probe_graph(json!([pair_decl]), json!([caller, sink]), "write_hash")
}

#[test]
fn mut_borrow_of_struct_field_writes_the_word_back() {
    let graph = lower_field_out();
    let pair = input_var(&graph, "pair");
    let ptrs = malloc_ptrs(&graph);
    assert_eq!(
        ptrs.len(),
        1,
        "one spill for the field\n{}",
        op_lines(&graph)
    );
    let ptr = &ptrs[0];
    let reads = raw_ops(&graph, |kind| matches!(kind, OpKind::FieldRead { .. }));
    assert_eq!(reads.len(), 1, "one field read\n{}", op_lines(&graph));
    let read_var = reads[0].result.as_ref().expect("FieldRead result");
    match &reads[0].kind {
        OpKind::FieldRead {
            base,
            field,
            ty: ValueType::Int,
            ..
        } if base == pair
            && field.name == "word"
            && field.owner_root.as_deref() == Some("Pair") => {}
        other => panic!("the spill reads pair.word, got {other:?}"),
    }
    match &raw_ops(&graph, |kind| matches!(kind, OpKind::RawStore { .. }))[0].kind {
        OpKind::RawStore {
            base,
            value,
            item_ty: ValueType::Int,
            itemsize: 8,
            is_item_signed: true,
            ..
        } if base == ptr && value == read_var => {}
        other => panic!("the store writes the field read, got {other:?}"),
    }
    let call = sink_call(&graph);
    assert_eq!(
        call_arg(call, 0),
        ptr,
        "the raw parameter receives the address"
    );
    let loads = raw_ops(&graph, |kind| matches!(kind, OpKind::RawLoad { .. }));
    assert_eq!(
        loads.len(),
        1,
        "the mutable parameter reloads\n{}",
        op_lines(&graph)
    );
    let loaded = loads[0].result.as_ref().expect("RawLoad result");
    match &loads[0].kind {
        OpKind::RawLoad {
            base,
            item_ty: ValueType::Int,
            itemsize: 8,
            is_item_signed: true,
            ..
        } if base == ptr => {}
        other => panic!("reload reads the spill, got {other:?}"),
    }
    let writes = raw_ops(&graph, |kind| matches!(kind, OpKind::FieldWrite { .. }));
    assert_eq!(
        writes.len(),
        1,
        "the word is written back\n{}",
        op_lines(&graph)
    );
    match &writes[0].kind {
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty: ValueType::Int,
        } if base == pair
            && field.name == "word"
            && field.owner_root.as_deref() == Some("Pair")
            && value.as_variable() == Some(loaded) => {}
        other => panic!("FieldWrite must store the reload into pair.word, got {other:?}"),
    }
    let call_block = graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| std::ptr::eq(&op.kind, call))
        })
        .expect("call block");
    let pos = |pred: &dyn Fn(&OpKind) -> bool| {
        call_block
            .operations
            .iter()
            .position(|op| pred(&op.kind))
            .expect("op in the call block")
    };
    let read_at = pos(&|kind| matches!(kind, OpKind::FieldRead { .. }));
    let malloc_at = pos(&|kind| matches!(kind, OpKind::RawMalloc { .. }));
    let store_at = pos(&|kind| matches!(kind, OpKind::RawStore { .. }));
    let call_at = pos(&|kind| std::ptr::eq(kind, call));
    let load_at = pos(&|kind| matches!(kind, OpKind::RawLoad { .. }));
    let write_at = pos(&|kind| matches!(kind, OpKind::FieldWrite { .. }));
    let free_at = pos(&|kind| matches!(kind, OpKind::RawFree { .. }));
    assert!(
        read_at < malloc_at
            && malloc_at < store_at
            && store_at < call_at
            && call_at < load_at
            && load_at < write_at
            && write_at < free_at,
        "field read, spill, call, reload, field write, free\n{}",
        op_lines(&graph)
    );
}

fn lower_returned_address_with(
    result_ty: &Value,
    extra_decls: &[Value],
    carrier_path: Option<&str>,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    lower_returned_address_sink(result_ty, extra_decls, carrier_path, None, &[])
}

fn lower_returned_address_sink(
    result_ty: &Value,
    extra_decls: &[Value],
    carrier_path: Option<&str>,
    sink_body: Option<&Value>,
    extra_funs: &[Value],
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, generics, meta, local) = probe_parts();
    let word = i64_ty();
    let borrowed = borrow_ty(&word, "Shared");
    let ptr = raw_ptr(&word, "Const");
    let borrow = place(2, &borrowed);
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, output: &Value, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        vec![word.clone()],
        result_ty,
        json!({"Unstructured": {"span": span, "locals": {"arg_count": 1, "locals": [
            local(0, None, result_ty),
            local(1, Some("word"), &word),
            local(2, None, &borrowed),
            local(3, None, result_ty)
        ]}, "body": [
            {"statements": [
                {"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                    "place": place(1, &word), "kind": "Shared", "ptr_metadata": null
                }}]}}
            ], "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": [{"Move": borrow}], "dest": place(3, result_ty)},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, result_ty),
                {"Use": [{"Copy": place(3, result_ty)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        vec![ptr.clone()],
        result_ty,
        sink_body
            .cloned()
            .unwrap_or_else(|| idle_body(result_ty, &[ptr.clone()])),
    );
    let mut fun_decls = vec![caller, sink];
    fun_decls.extend(extra_funs.iter().cloned());
    let mut type_decls = vec![json!({
        "def_id": 0,
        "item_meta": meta(&["core", "option", "Option"]),
        "kind": "Opaque",
        "src": "Normal"
    })];
    type_decls.extend(extra_decls.iter().cloned());
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": type_decls,
            "fun_decls": fun_decls,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    match carrier_path {
        None => lower_function(&llbc, "write_hash"),
        Some(path) => lower_function_with_static_addrs(
            &llbc,
            "write_hash",
            HostStaticAddrs {
                error_carrier: ErrorCarrierSpec {
                    carrier_path: path,
                    carrier_class: "",
                    carrier_wrappers: &[],
                    to_exc_object: None,
                    from_exc_object: None,
                },
                ..HostStaticAddrs::default()
            },
        ),
    }
}

fn adt_ty(id: u64, types: Vec<Value>) -> Value {
    json!({"Adt": {
        "id": id,
        "generics": {
            "regions": [],
            "types": types,
            "const_generics": [],
            "trait_refs": []
        }
    }})
}

fn tuple_ty(types: Vec<Value>) -> Value {
    json!({"Adt": {
        "id": 99,
        "builtin": "Tuple",
        "generics": {
            "regions": [],
            "types": types,
            "const_generics": [],
            "trait_refs": []
        }
    }})
}

fn named_decl(id: u64, path: &[&str], kind: Value) -> Value {
    let (_, _, meta, _) = probe_parts();
    json!({
        "def_id": id,
        "item_meta": meta(path),
        "kind": kind,
        "src": "Normal"
    })
}

fn decl_with_layout(id: u64, path: &[&str], kind: Value, layout: Value) -> Value {
    let mut decl = named_decl(id, path, kind);
    decl["layout"] = layout;
    decl
}

fn zero_sized_layout() -> Value {
    json!([{"key": "host", "value": {"size": 0, "align": 1}}])
}

fn type_var_field(name: &str, index: u64) -> Value {
    field_decl(name, &json!({"TypeVar": {"Bound": [0, index]}}))
}

fn field_decl(name: &str, ty: &Value) -> Value {
    json!({"name": name, "ty": ty, "attr_info": null})
}

fn result_decl() -> Value {
    named_decl(1, &["core", "result", "Result"], json!("Opaque"))
}

fn result_of(ok: &Value, err: &Value) -> Value {
    adt_ty(1, vec![ok.clone(), err.clone()])
}

fn assert_spill_freed(result_ty: &Value, extra: &[Value], carrier: Option<&str>) {
    let graph = lower_returned_address_with(result_ty, extra, carrier)
        .unwrap_or_else(|err| panic!("a status result must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawMalloc { .. })),
        "the spill is allocated\n{}",
        op_lines(&graph)
    );
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn assert_spill_escapes(result_ty: &Value, extra: &[Value], carrier: Option<&str>) {
    let err = lower_returned_address_with(result_ty, extra, carrier)
        .expect_err("a returned spill address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn returned_spill_address_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&ptr, &[], None);
}

#[test]
fn returned_pointer_under_four_options_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&nest_option(ptr, 4), &[], None);
}

#[test]
fn returned_i64_under_four_options_still_frees_the_spill() {
    assert_spill_freed(&nest_option(i64_ty(), 4), &[], None);
}

#[test]
fn returned_i64_under_nine_options_still_frees_the_spill() {
    assert_spill_freed(&nest_option(i64_ty(), 9), &[], None);
}

#[test]
fn returned_pointer_under_nine_options_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&nest_option(ptr, 9), &[], None);
}

#[test]
fn returned_err_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&result_of(&i64_ty(), &ptr), &[result_decl()], None);
}

#[test]
fn returned_ok_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&result_of(&ptr, &i64_ty()), &[result_decl()], None);
}

#[test]
fn returned_tuple_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&tuple_ty(vec![i64_ty(), ptr]), &[], None);
}

#[test]
fn returned_tuple_of_i64_still_frees_the_spill() {
    assert_spill_freed(&tuple_ty(vec![i64_ty(), i64_ty()]), &[], None);
}

#[test]
fn returned_struct_pointer_field_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [field_decl("ptr", &ptr)]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![]), &[decl], None);
}

#[test]
fn returned_struct_i64_field_still_frees_the_spill() {
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [field_decl("word", &i64_ty())]}),
    );
    assert_spill_freed(&adt_ty(1, vec![]), &[decl], None);
}

#[test]
fn returned_enum_pointer_field_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Enum": [
            {"name": "Word", "fields": [field_decl("value", &i64_ty())]},
            {"name": "Ptr", "fields": [field_decl("value", &ptr)]}
        ]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![]), &[decl], None);
}

#[test]
fn returned_array_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&json!({"Array": [ptr, null, null]}), &[], None);
}

#[test]
fn returned_carrier_err_still_frees_the_spill() {
    let carrier = "pyre_interpreter::error::PyError";
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        2,
        &["pyre_interpreter", "error", "PyError"],
        json!({"Struct": [field_decl("obj", &ptr)]}),
    );
    let result = result_of(&i64_ty(), &adt_ty(2, vec![]));
    assert_spill_freed(&result, &[result_decl(), decl], Some(carrier));
}

#[test]
fn returned_opaque_newtype_is_not_lowered() {
    let decl = named_decl(1, &["probe", "Handle"], json!("Opaque"));
    assert_spill_escapes(&adt_ty(1, vec![]), &[decl], None);
}

#[test]
fn returned_opaque_i64_generic_is_not_lowered() {
    let decl = named_decl(1, &["probe", "Handle"], json!("Opaque"));
    assert_spill_escapes(&adt_ty(1, vec![i64_ty()]), &[decl], None);
}

#[test]
fn returned_phantom_pointer_still_frees_the_spill() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(1, &["core", "marker", "PhantomData"], json!({"Struct": []}));
    assert_spill_freed(&adt_ty(1, vec![ptr]), &[decl], None);
}

#[test]
fn returned_zero_sized_opaque_still_frees_the_spill() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = decl_with_layout(
        1,
        &["probe", "Marker"],
        json!("Opaque"),
        zero_sized_layout(),
    );
    assert_spill_freed(&adt_ty(1, vec![ptr]), &[decl], None);
}

#[test]
fn returned_typevar_pointer_field_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [type_var_field("value", 0)]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![ptr]), &[decl], None);
}

#[test]
fn returned_typevar_i64_field_still_frees_the_spill() {
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [type_var_field("value", 0)]}),
    );
    assert_spill_freed(&adt_ty(1, vec![i64_ty()]), &[decl], None);
}

#[test]
fn returned_non_carrier_struct_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        2,
        &["pyre_interpreter", "error", "PyError"],
        json!({"Struct": [field_decl("obj", &ptr)]}),
    );
    let result = result_of(&i64_ty(), &adt_ty(2, vec![]));
    assert_spill_escapes(&result, &[result_decl(), decl], None);
}

fn option_typevar_field() -> Value {
    field_decl("value", &option_of(&json!({"TypeVar": {"Bound": [0, 0]}})))
}

#[test]
fn returned_option_typevar_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [option_typevar_field()]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![ptr]), &[decl], None);
}

#[test]
fn returned_option_typevar_i64_still_frees_the_spill() {
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [option_typevar_field()]}),
    );
    assert_spill_freed(&adt_ty(1, vec![i64_ty()]), &[decl], None);
}

#[test]
fn returned_nested_struct_typevar_pointer_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let inner_ty = adt_ty(2, vec![json!({"TypeVar": {"Bound": [0, 0]}})]);
    let hold = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [field_decl("value", &inner_ty)]}),
    );
    let inner = named_decl(
        2,
        &["probe", "Inner"],
        json!({"Struct": [type_var_field("word", 0)]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![ptr]), &[hold, inner], None);
}

#[test]
fn returned_nested_struct_typevar_i64_still_frees_the_spill() {
    let inner_ty = adt_ty(2, vec![json!({"TypeVar": {"Bound": [0, 0]}})]);
    let hold = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [field_decl("value", &inner_ty)]}),
    );
    let inner = named_decl(
        2,
        &["probe", "Inner"],
        json!({"Struct": [type_var_field("word", 0)]}),
    );
    assert_spill_freed(&adt_ty(1, vec![i64_ty()]), &[hold, inner], None);
}

#[test]
fn returned_deeper_binder_typevar_is_not_lowered() {
    let decl = named_decl(
        1,
        &["probe", "Hold"],
        json!({"Struct": [field_decl("value", &json!({"TypeVar": {"Bound": [1, 0]}}))]}),
    );
    assert_spill_escapes(&adt_ty(1, vec![i64_ty()]), &[decl], None);
}

fn pair_decl() -> Value {
    named_decl(
        1,
        &["probe", "Pair"],
        json!({"Struct": [type_var_field("a", 0), type_var_field("b", 1)]}),
    )
}

fn dedup_holder(body: Value) -> Value {
    named_decl(
        2,
        &["probe", "Dedup"],
        json!({"Struct": [field_decl("body", &json!({"Value": [8, body]}))]}),
    )
}

#[test]
fn returned_caller_typevar_in_a_generic_argument_is_not_lowered() {
    let caller = json!({"TypeVar": {"Bound": [0, 0]}});
    assert_spill_escapes(&adt_ty(1, vec![i64_ty(), caller]), &[pair_decl()], None);
}

#[test]
fn returned_pair_of_i64_still_frees_the_spill() {
    assert_spill_freed(&adt_ty(1, vec![i64_ty(), i64_ty()]), &[pair_decl()], None);
}

#[test]
fn returned_pair_pointer_argument_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(&adt_ty(1, vec![i64_ty(), ptr]), &[pair_decl()], None);
}

#[test]
fn returned_deduped_caller_typevar_is_not_lowered() {
    let caller = json!({"TypeVar": {"Bound": [0, 0]}});
    assert_spill_escapes(
        &adt_ty(1, vec![i64_ty(), json!({"Deduplicated": 8})]),
        &[pair_decl(), dedup_holder(caller)],
        None,
    );
}

#[test]
fn returned_deduped_i64_argument_still_frees_the_spill() {
    assert_spill_freed(
        &adt_ty(1, vec![i64_ty(), json!({"Deduplicated": 8})]),
        &[pair_decl(), dedup_holder(i64_ty())],
        None,
    );
}

#[test]
fn returned_deduped_pointer_argument_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    assert_spill_escapes(
        &adt_ty(1, vec![i64_ty(), json!({"Deduplicated": 8})]),
        &[pair_decl(), dedup_holder(ptr)],
        None,
    );
}

fn sink_unstructured(result_ty: &Value, ptr_ty: &Value, statements: Vec<Value>) -> Value {
    let (span, _, _, local) = probe_parts();
    json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, result_ty),
            local(1, Some("p"), ptr_ty)
        ]},
        "body": [{"statements": statements, "terminator": {"span": span, "kind": "Return"}}]
    }})
}

fn assign_scalar_cast(dest: u64, src: u64, src_ty: &Value, dest_ty: &Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": {"Assign": [
        place(dest, dest_ty),
        {"UnaryOp": [
            {"Cast": {"Scalar": [src_ty, dest_ty]}},
            {"Copy": place(src, src_ty)}
        ]}
    ]}})
}

fn assert_sink_escapes(result_ty: &Value, sink_body: &Value) {
    let err = lower_returned_address_sink(result_ty, &[], None, Some(sink_body), &[])
        .expect_err("a returned spill address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn assert_sink_frees(result_ty: &Value, sink_body: &Value) {
    let graph = lower_returned_address_sink(result_ty, &[], None, Some(sink_body), &[])
        .unwrap_or_else(|err| panic!("a status result must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawMalloc { .. })),
        "the spill is allocated\n{}",
        op_lines(&graph)
    );
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn returned_pointer_cast_to_u64_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let body = sink_unstructured(&result, &ptr, vec![assign_scalar_cast(0, 1, &ptr, &result)]);
    assert_sink_escapes(&result, &body);
}

#[test]
fn returned_pointer_cast_through_a_local_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let mut body = sink_unstructured(
        &result,
        &ptr,
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"Use": [{"Copy": place(2, &result)}, "Yes"]}
            ]}}),
        ],
    );
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("bits"), &result));
    assert_sink_escapes(&result, &body);
}

#[test]
fn loaded_pointee_still_frees_the_spill() {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let loaded = json!({"kind": {"Projection": [place(1, &ptr), "Deref"]}, "ty": word});
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![json!({"span": span, "kind": {"Assign": [
            place(0, &word),
            {"Use": [{"Copy": loaded}, "Yes"]}
        ]}})],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn load_through_a_copied_spill_pointer_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("q"), &ptr)],
        vec![
            assign_to(place(2, &ptr), copy_use(place(1, &ptr))),
            json!({"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Use": [{"Copy": deref_place(place(2, &ptr), &word)}, "Yes"]}
            ]}}),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn load_through_a_derived_pointer_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits), local(3, Some("q"), &ptr)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            json!({"span": span, "kind": {"Assign": [
                place(2, &bits),
                {"BinaryOp": ["Add", {"Copy": place(2, &bits)}, {"Const": zero_const()}]}
            ]}}),
            assign_scalar_cast(3, 2, &bits, &ptr),
            json!({"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Use": [{"Copy": deref_place(place(3, &ptr), &word)}, "Yes"]}
            ]}}),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn callee_body_that_does_not_return_the_address_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(&word, &ptr, vec![]);
    assert_sink_frees(&word, &body);
}

fn idle_body(output: &Value, inputs: &[Value]) -> Value {
    let (span, _, _, local) = probe_parts();
    let mut locals = vec![local(0, None, output)];
    for (index, ty) in inputs.iter().enumerate() {
        locals.push(local((index + 1) as u64, None, ty));
    }
    json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": inputs.len() as u64, "locals": locals},
        "body": [{"statements": [], "terminator": {"span": span, "kind": "Return"}}]
    }})
}

fn probe_fun(id: u64, name: &[&str], inputs: Vec<Value>, output: &Value, body: Value) -> Value {
    let (_, _, meta, _) = probe_parts();
    json!({
        "def_id": id,
        "item_meta": meta(name),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": body
    })
}

fn call_into_return(result: &Value, ptr: &Value, callee: u64) -> Value {
    let (span, generics, _, _) = probe_parts();
    let mut body = sink_unstructured(result, ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": callee}, "generics": generics}},
                "args": [{"Move": place(1, ptr)}], "dest": place(0, result)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body
}

fn zero_const() -> Value {
    json!([
        {"Integer": {"Unsigned": ["U64", "0"]}},
        {"Scalar": {"Integer": {"Unsigned": "U64"}}}
    ])
}

fn int_const(text: &str) -> Value {
    json!([
        {"Integer": {"Unsigned": ["U64", text]}},
        {"Scalar": {"Integer": {"Unsigned": "U64"}}}
    ])
}

fn assign_binop(op: &str, src_ty: &Value) -> Value {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    json!({"span": span, "kind": {"Assign": [
        place(0, &word),
        {"BinaryOp": [op, {"Copy": place(1, src_ty)}, {"Const": null}]}
    ]}})
}

fn assign_compare(op: &str, src_ty: &Value, rhs: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    json!({"span": span, "kind": {"Assign": [
        place(0, &word),
        {"BinaryOp": [op, {"Copy": place(1, src_ty)}, {"Const": rhs}]}
    ]}})
}

#[test]
fn status_from_a_call_still_frees_the_spill() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = call_into_return(&word, &ptr, 2);
    let helper = probe_fun(
        2,
        &["probe", "write_narrowed"],
        vec![ptr.clone()],
        &word,
        idle_body(&word, &[ptr]),
    );
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a status result must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn opaque_scalar_result_is_not_lowered() {
    let body = json!("Opaque");
    assert_sink_escapes(&u64_ty(), &body);
}

#[test]
fn call_of_opaque_into_the_return_slot_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let body = call_into_return(&result, &ptr, 2);
    let expose = probe_fun(2, &["probe", "expose"], vec![ptr], &result, json!("Opaque"));
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[expose])
        .expect_err("a call that may return the address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn pointer_comparison_still_frees_the_spill() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let body = sink_unstructured(
        &i64_ty(),
        &ptr,
        vec![assign_compare("Eq", &ptr, zero_const())],
    );
    assert_sink_frees(&i64_ty(), &body);
}

#[test]
fn returned_sign_of_the_address_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &word)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &word),
            compare_with_zero("Lt", 0, &word, 2, &word),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn pointer_compared_with_another_pointer_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("q"), &ptr)],
        vec![json!({"span": span, "kind": {"Assign": [
            place(0, &word),
            {"BinaryOp": ["Eq", {"Copy": place(1, &ptr)}, {"Copy": place(2, &ptr)}]}
        ]}})],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn pointer_compared_with_a_nonzero_constant_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let body = sink_unstructured(
        &i64_ty(),
        &ptr,
        vec![assign_compare("Eq", &ptr, int_const("1"))],
    );
    assert_sink_escapes(&i64_ty(), &body);
}

#[test]
fn pointer_compared_with_an_unknown_constant_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let body = sink_unstructured(
        &i64_ty(),
        &ptr,
        vec![assign_compare("Eq", &ptr, Value::Null)],
    );
    assert_sink_escapes(&i64_ty(), &body);
}

#[test]
fn deduped_null_comparison_still_frees_the_spill() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let anchor = named_decl(
        1,
        &["probe", "Anchor"],
        json!({"Struct": [field_decl("n", &json!({"Value": [11, zero_const()]}))]}),
    );
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![assign_compare("Eq", &ptr, json!({"Deduplicated": 11}))],
    );
    let graph = lower_returned_address_sink(&word, &[anchor], None, Some(&body), &[])
        .unwrap_or_else(|err| panic!("a deduped null comparison must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn cast_pointer_compared_with_zero_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(0, &word),
                json!({"BinaryOp": ["Eq", {"Copy": place(2, &bits)}, {"Const": zero_const()}]}),
            ),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn usize_pointer_compared_with_zero_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = usize_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(0, &word),
                json!({"BinaryOp": ["Eq", {"Copy": place(2, &bits)}, {"Const": zero_const()}]}),
            ),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn narrowed_pointer_byte_compared_with_zero_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = usize_ty();
    let low = u8_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits), local(3, Some("low"), &low)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_scalar_cast(3, 2, &bits, &low),
            assign_to(
                place(0, &word),
                json!({"BinaryOp": ["Eq", {"Copy": place(3, &low)}, {"Const": zero_const()}]}),
            ),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn masked_pointer_compared_with_zero_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("masked"), &bits),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(3, &bits),
                json!({"BinaryOp": [
                    "BitAnd",
                    {"Copy": place(2, &bits)},
                    {"Const": int_const("7")}
                ]}),
            ),
            assign_to(
                place(0, &word),
                json!({"BinaryOp": ["Eq", {"Copy": place(3, &bits)}, {"Const": zero_const()}]}),
            ),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn reborrow_compared_with_zero_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let r_ty = borrow_ty(&word, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("r"), &r_ty)],
        vec![
            ref_assign(2, &r_ty, deref_place(place(1, &ptr), &word)),
            assign_to(
                place(0, &word),
                json!({"BinaryOp": ["Eq", {"Copy": place(2, &r_ty)}, {"Const": zero_const()}]}),
            ),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn pointer_bitand_is_not_lowered() {
    let ptr = raw_ptr(&i64_ty(), "Const");
    let body = sink_unstructured(&i64_ty(), &ptr, vec![assign_binop("BitAnd", &ptr)]);
    assert_sink_escapes(&i64_ty(), &body);
}

fn assign_deref(ptr_local: u64, ptr_ty: &Value, rvalue: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    let dest = json!({"kind": {"Projection": [place(ptr_local, ptr_ty), "Deref"]}, "ty": word});
    json!({"span": span, "kind": {"Assign": [dest, rvalue]}})
}

fn const_use() -> Value {
    json!({"Use": [{"Const": null}, "Yes"]})
}

#[test]
fn store_of_a_status_through_the_pointer_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(&word, &ptr, vec![assign_deref(1, &ptr, const_use())]);
    assert_sink_frees(&word, &body);
}

#[test]
fn store_through_a_copied_spill_pointer_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("q"), &ptr)],
        vec![
            assign_to(place(2, &ptr), copy_use(place(1, &ptr))),
            assign_deref(2, &ptr, const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn store_through_a_derived_pointer_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits), local(3, Some("q"), &ptr)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            json!({"span": span, "kind": {"Assign": [
                place(2, &bits),
                {"BinaryOp": ["Add", {"Copy": place(2, &bits)}, {"Const": zero_const()}]}
            ]}}),
            assign_scalar_cast(3, 2, &bits, &ptr),
            assign_deref(3, &ptr, const_use()),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn store_through_a_reference_to_the_pointer_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            assign_to(deref_place(place(2, &q_ty), &ptr), const_use()),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn reference_overwrite_then_the_pointer_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            assign_to(deref_place(place(2, &q_ty), &ptr), const_use()),
            assign_scalar_cast(0, 1, &ptr, &result),
        ],
    );
    assert_sink_frees(&result, &body);
}

fn joined_reference_overwrite(rename: bool) -> Value {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty), local(3, Some("other"), &word)],
        vec![],
    );
    let other = if rename {
        ref_assign(2, &q_ty, place(3, &word))
    } else {
        ref_assign(2, &q_ty, place(1, &ptr))
    };
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Const": null}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [ref_assign(2, &q_ty, place(1, &ptr))],
            "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
        {"statements": [other],
            "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
        {"statements": [
            assign_to(deref_place(place(2, &q_ty), &ptr), const_use()),
            assign_scalar_cast(0, 1, &ptr, &result)
        ], "terminator": {"span": span, "kind": "Return"}}
    ]);
    body
}

#[test]
fn joined_reference_overwrite_still_frees() {
    assert_sink_frees(&u64_ty(), &joined_reference_overwrite(false));
}

#[test]
fn joined_reference_to_another_local_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &joined_reference_overwrite(true));
}

#[test]
fn reference_overwrite_then_the_reload_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            assign_to(deref_place(place(2, &q_ty), &ptr), const_use()),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&ptr, &result]}},
                    {"Copy": deref_place(place(2, &q_ty), &ptr)}
                ]}
            ]}}),
        ],
    );
    assert_sink_frees(&result, &body);
}

#[test]
fn copied_reference_overwrite_then_the_pointer_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty), local(3, Some("r"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            assign_to(place(3, &q_ty), copy_use(place(2, &q_ty))),
            assign_to(deref_place(place(3, &q_ty), &ptr), const_use()),
            assign_scalar_cast(0, 1, &ptr, &result),
        ],
    );
    assert_sink_frees(&result, &body);
}

#[test]
fn narrowed_reference_overwrite_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let low = u8_ty();
    let back = raw_ptr(&ptr, "Mut");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("q"), &q_ty),
            local(3, Some("low"), &low),
            local(4, Some("back"), &back),
        ],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            assign_scalar_cast(3, 2, &q_ty, &low),
            assign_scalar_cast(4, 3, &low, &back),
            assign_to(deref_place(place(4, &back), &ptr), const_use()),
            assign_scalar_cast(0, 1, &ptr, &result),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn store_of_the_address_through_the_pointer_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![assign_deref(1, &ptr, {
            let (span, _, _, _) = probe_parts();
            let _ = span;
            json!({"UnaryOp": [
                {"Cast": {"Scalar": [&ptr, &word]}},
                {"Copy": place(1, &ptr)}
            ]})
        })],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn address_stored_through_an_out_param_is_not_lowered() {
    let graph = out_param_case(true);
    let err = graph.expect_err("an address stored through an out parameter must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn status_stored_through_an_out_param_still_frees() {
    let graph = out_param_case(false)
        .unwrap_or_else(|err| panic!("a status out parameter must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn out_param_case(
    store_address: bool,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, generics, _, local) = probe_parts();
    let word = u64_ty();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let bits_ref = borrow_ty(&word, "Mut");
    let stored = json!({"kind": {"Projection": [place(1, &bits_ref), "Deref"]}, "ty": word});
    let stored_value = if store_address {
        json!({"UnaryOp": [
            {"Cast": {"Scalar": [ptr, word]}},
            {"Copy": place(2, &ptr)}
        ]})
    } else {
        const_use()
    };
    let inner_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 2, "locals": [
            local(0, None, &word),
            local(1, Some("bits"), &bits_ref),
            local(2, Some("p"), &ptr)
        ]},
        "body": [{"statements": [
            {"span": span, "kind": {"Assign": [stored, stored_value]}},
            {"span": span, "kind": {"Assign": [place(0, &word), const_use()]}}
        ], "terminator": {"span": span, "kind": "Return"}}]
    }});
    let inner = probe_fun(
        2,
        &["probe", "inner"],
        vec![bits_ref.clone(), ptr.clone()],
        &word,
        inner_body,
    );
    let mut outer = sink_unstructured(&word, &ptr, vec![]);
    outer["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .extend([
            local(2, Some("bits"), &word),
            local(3, None, &bits_ref),
            local(4, None, &word),
        ]);
    outer["Unstructured"]["body"] = json!([
        {"statements": [
            {"span": span, "kind": {"Assign": [
                place(3, &bits_ref),
                {"Ref": {"place": place(2, &word), "kind": "Mut", "ptr_metadata": null}}
            ]}}
        ], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(3, &bits_ref)}, {"Copy": place(1, &ptr)}],
                "dest": place(4, &word)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [
            {"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Use": [{"Copy": place(2, &word)}, "Yes"]}
            ]}}
        ], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    lower_returned_address_sink(&word, &[], None, Some(&outer), &[inner])
}

#[test]
fn opaque_call_then_a_status_is_not_lowered() {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("status"), &word));
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(1, &ptr)}], "dest": place(2, &word)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [
            {"span": span, "kind": {"Assign": [place(0, &word), const_use()]}}
        ], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let expose = probe_fun(2, &["probe", "expose"], vec![ptr], &word, json!("Opaque"));
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[expose])
        .expect_err("an opaque call that receives the address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn deref_place(base: Value, ty: &Value) -> Value {
    json!({"kind": {"Projection": [base, "Deref"]}, "ty": ty})
}

fn ref_assign(dest: u64, dest_ty: &Value, src: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": {"Assign": [
        place(dest, dest_ty),
        {"Ref": {"place": src, "kind": "Shared", "ptr_metadata": null}}
    ]}})
}

fn raw_const_assign(dest: u64, dest_ty: &Value, src: Value) -> Value {
    assign_to(
        place(dest, dest_ty),
        json!({"RawPtr": {"place": src, "kind": "Const", "ptr_metadata": null}}),
    )
}

/// `let mut bits = 0; let q = &raw const bits; bits = p as usize`.
fn clean_raw_alias_body(reload: CleanAliasReload) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let bits = u64_ty();
    let q_ty = raw_ptr(&bits, "Const");
    let reload_stored = match reload {
        CleanAliasReload::AfterStore => vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(0, &result),
                copy_use(deref_place(place(3, &q_ty), &result)),
            ),
        ],
        CleanAliasReload::BeforeStore => vec![
            assign_to(
                place(0, &result),
                copy_use(deref_place(place(3, &q_ty), &result)),
            ),
            assign_scalar_cast(2, 1, &ptr, &bits),
        ],
        CleanAliasReload::Never => vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(place(0, &result), const_use()),
        ],
    };
    let mut statements = vec![
        assign_to(place(2, &bits), const_use()),
        raw_const_assign(3, &q_ty, place(2, &bits)),
    ];
    statements.extend(reload_stored);
    sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("bits"), &bits), local(3, Some("q"), &q_ty)],
        statements,
    )
}

enum CleanAliasReload {
    /// `return *q` after `bits` holds the address.
    AfterStore,
    /// `return *q` while `bits` is still zero, then store the address.
    BeforeStore,
    /// Store the address and return a constant.
    Never,
}

#[test]
fn clean_raw_alias_reloaded_after_the_store_is_not_lowered() {
    assert_sink_escapes(
        &u64_ty(),
        &clean_raw_alias_body(CleanAliasReload::AfterStore),
    );
}

#[test]
fn reload_before_the_store_through_a_clean_alias_still_frees() {
    assert_sink_frees(
        &u64_ty(),
        &clean_raw_alias_body(CleanAliasReload::BeforeStore),
    );
}

#[test]
fn clean_raw_alias_that_is_not_reloaded_still_frees() {
    assert_sink_frees(&u64_ty(), &clean_raw_alias_body(CleanAliasReload::Never));
}

enum AliasCallee {
    /// `read` returns `*q` after `bits` holds the address.
    Reload,
    /// `read` returns `*q` before that store.
    ReloadFirst,
    /// `read` returns a constant after the store.
    Constant,
    /// `read` has no body.
    Opaque,
}

fn read_through_clean_alias(kind: AliasCallee) -> (Value, Value) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let store = assign_scalar_cast(2, 1, &ptr, &result);
    let mut early = vec![
        assign_to(place(2, &result), const_use()),
        raw_const_assign(3, &q_ty, place(2, &result)),
    ];
    let mut late = Vec::new();
    match kind {
        AliasCallee::ReloadFirst => late.push(store),
        _ => early.push(store),
    }
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("bits"), &result), local(3, Some("q"), &q_ty)],
        vec![],
    );
    body["Unstructured"]["body"] = json!([
        {"statements": early, "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Copy": place(3, &q_ty)}], "dest": place(0, &result)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": late, "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let helper_body = match kind {
        AliasCallee::Constant => sink_unstructured(
            &result,
            &q_ty,
            vec![assign_to(place(0, &result), const_use())],
        ),
        AliasCallee::Opaque => json!("Opaque"),
        AliasCallee::Reload | AliasCallee::ReloadFirst => sink_unstructured(
            &result,
            &q_ty,
            vec![assign_to(
                place(0, &result),
                copy_use(deref_place(place(1, &q_ty), &result)),
            )],
        ),
    };
    let helper = probe_fun(2, &["probe", "read"], vec![q_ty], &result, helper_body);
    (body, helper)
}

#[test]
fn read_of_a_clean_alias_after_the_store_is_not_lowered() {
    let result = u64_ty();
    let (body, helper) = read_through_clean_alias(AliasCallee::Reload);
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .expect_err("read of the alias after the store must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn read_of_a_clean_alias_before_the_store_still_frees() {
    let result = u64_ty();
    let (body, helper) = read_through_clean_alias(AliasCallee::ReloadFirst);
    let graph = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("read before the store must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn read_of_a_clean_alias_that_returns_a_constant_still_frees() {
    let result = u64_ty();
    let (body, helper) = read_through_clean_alias(AliasCallee::Constant);
    let graph = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a constant read must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn opaque_read_of_a_clean_alias_is_not_lowered() {
    let result = u64_ty();
    let (body, helper) = read_through_clean_alias(AliasCallee::Opaque);
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .expect_err("an opaque read of the alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn compare_through_clean_alias(op: &str) -> (Value, Value) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &word), local(3, Some("q"), &ptr)],
        vec![],
    );
    body["Unstructured"]["body"] = json!([
        {"statements": [
            assign_to(place(2, &word), const_use()),
            raw_const_assign(3, &ptr, place(2, &word)),
            compare_with_zero(op, 2, &word, 1, &ptr)
        ], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Copy": place(3, &ptr)}], "dest": place(0, &word)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let helper = probe_fun(
        2,
        &["probe", "read"],
        vec![ptr.clone()],
        &word,
        sink_unstructured(
            &word,
            &ptr,
            vec![assign_to(
                place(0, &word),
                copy_use(deref_place(place(1, &ptr), &word)),
            )],
        ),
    );
    (body, helper)
}

#[test]
fn read_of_an_ordering_through_a_clean_alias_is_not_lowered() {
    let word = i64_ty();
    let (body, helper) = compare_through_clean_alias("Lt");
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .expect_err("read of p < 0 through the alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn read_of_a_null_check_through_a_clean_alias_still_frees() {
    let word = i64_ty();
    let (body, helper) = compare_through_clean_alias("Eq");
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("read of p == null through the alias must still free: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

enum PackedAlias {
    /// `return *pair.0` after `bits` holds the address.
    Reload,
    /// `return *pair.0` before that store.
    ReloadFirst,
    /// `return pair.1`, the clean sibling.
    Sibling,
    /// `read` returns `*pair.0`.
    Helper,
}

fn pair_holding_clean_alias(kind: PackedAlias) -> (Value, Option<Value>) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let pair_ty = tuple_ty(vec![q_ty.clone(), result.clone()]);
    let store = assign_scalar_cast(2, 1, &ptr, &result);
    let load_alias = assign_to(
        place(0, &result),
        copy_use(deref_place(field_place(4, &pair_ty, 0, &q_ty), &result)),
    );
    let load_sibling = assign_to(
        place(0, &result),
        copy_use(field_place(4, &pair_ty, 1, &result)),
    );
    let mut statements = vec![
        assign_to(place(2, &result), const_use()),
        raw_const_assign(3, &q_ty, place(2, &result)),
        assign_to(
            place(4, &pair_ty),
            tuple_of(vec![
                json!({"Copy": place(3, &q_ty)}),
                json!({"Const": null}),
            ]),
        ),
    ];
    let helper = match kind {
        PackedAlias::Reload => {
            statements.push(store);
            statements.push(load_alias);
            None
        }
        PackedAlias::ReloadFirst => {
            statements.push(load_alias);
            statements.push(store);
            None
        }
        PackedAlias::Sibling => {
            statements.push(store);
            statements.push(load_sibling);
            None
        }
        PackedAlias::Helper => {
            statements.push(store);
            Some(probe_fun(
                2,
                &["probe", "read"],
                vec![pair_ty.clone()],
                &result,
                sink_unstructured(
                    &result,
                    &pair_ty,
                    vec![assign_to(
                        place(0, &result),
                        copy_use(deref_place(field_place(1, &pair_ty, 0, &q_ty), &result)),
                    )],
                ),
            ))
        }
    };
    let call_statements = statements.clone();
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("q"), &q_ty),
            local(4, Some("pair"), &pair_ty),
        ],
        statements,
    );
    if matches!(kind, PackedAlias::Helper) {
        body["Unstructured"]["body"] = json!([
            {"statements": call_statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [{"Copy": place(4, &pair_ty)}], "dest": place(0, &result)},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]);
    }
    (body, helper)
}

#[test]
fn alias_in_a_field_reloaded_after_the_store_is_not_lowered() {
    let result = u64_ty();
    let (body, _) = pair_holding_clean_alias(PackedAlias::Reload);
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[])
        .expect_err("a reload through the packed alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn alias_in_a_field_reloaded_before_the_store_still_frees() {
    let result = u64_ty();
    let (body, _) = pair_holding_clean_alias(PackedAlias::ReloadFirst);
    let graph = lower_returned_address_sink(&result, &[], None, Some(&body), &[])
        .unwrap_or_else(|err| panic!("a reload before the store must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn clean_field_beside_a_clean_alias_still_frees() {
    let result = u64_ty();
    let (body, _) = pair_holding_clean_alias(PackedAlias::Sibling);
    let graph = lower_returned_address_sink(&result, &[], None, Some(&body), &[])
        .unwrap_or_else(|err| panic!("the clean sibling field must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn read_of_a_packed_alias_is_not_lowered() {
    let result = u64_ty();
    let (body, helper) = pair_holding_clean_alias(PackedAlias::Helper);
    let helper = helper.expect("helper");
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .expect_err("a callee reload of the packed alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn sink_with_extra(
    result: &Value,
    ptr: &Value,
    extras: Vec<Value>,
    statements: Vec<Value>,
) -> Value {
    let mut body = sink_unstructured(result, ptr, statements);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .extend(extras);
    body
}

#[test]
fn address_reloaded_through_a_reference_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&ptr, &result]}},
                    {"Copy": deref_place(place(2, &q_ty), &ptr)}
                ]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn address_reloaded_through_a_copied_reference_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty), local(3, Some("r"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            json!({"span": span, "kind": {"Assign": [
                place(3, &q_ty),
                {"Use": [{"Copy": place(2, &q_ty)}, "Yes"]}
            ]}}),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&ptr, &result]}},
                    {"Copy": deref_place(place(3, &q_ty), &ptr)}
                ]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn address_reloaded_through_two_references_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let s_ty = borrow_ty(&q_ty, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty), local(3, Some("s"), &s_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            ref_assign(3, &s_ty, place(2, &q_ty)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&ptr, &result]}},
                    {"Copy": deref_place(deref_place(place(3, &s_ty), &q_ty), &ptr)}
                ]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn integer_reloaded_through_a_reference_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let bits_ref = borrow_ty(&result, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pb"), &bits_ref),
        ],
        vec![
            json!({"span": span, "kind": {"Assign": [
                place(2, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&ptr, &result]}},
                    {"Copy": place(1, &ptr)}
                ]}
            ]}}),
            ref_assign(3, &bits_ref, place(2, &result)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"Use": [{"Copy": deref_place(place(3, &bits_ref), &result)}, "Yes"]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn reference_to_the_pointer_then_a_status_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![ref_assign(2, &q_ty, place(1, &ptr)), {
            let (span, _, _, _) = probe_parts();
            json!({"span": span, "kind": {"Assign": [place(0, &word), const_use()]}})
        }],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn address_of_the_pointer_local_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            ref_assign(2, &q_ty, place(1, &ptr)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"UnaryOp": [
                    {"Cast": {"Scalar": [&q_ty, &result]}},
                    {"Copy": place(2, &q_ty)}
                ]}
            ]}}),
        ],
    );
    assert_sink_frees(&result, &body);
}

#[test]
fn reborrow_then_loaded_pointee_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let r_ty = borrow_ty(&word, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("r"), &r_ty)],
        vec![
            ref_assign(2, &r_ty, deref_place(place(1, &ptr), &word)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Use": [{"Copy": deref_place(place(2, &r_ty), &word)}, "Yes"]}
            ]}}),
        ],
    );
    assert_sink_frees(&word, &body);
}

fn global_place(ty: &Value) -> Value {
    json!({
        "kind": {"Global": {
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []},
            "id": 0
        }},
        "ty": ty
    })
}

fn assign_to(dest: Value, rvalue: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": {"Assign": [dest, rvalue]}})
}

fn ptr_cast(src: Value, src_ty: &Value, dest_ty: &Value) -> Value {
    json!({"UnaryOp": [
        {"Cast": {"Scalar": [src_ty, dest_ty]}},
        {"Copy": src}
    ]})
}

fn status_return_after_drop(result: &Value, ptr: &Value, dropped: Value, glue: u64) -> Value {
    let (span, generics, _, _) = probe_parts();
    let mut body = sink_unstructured(result, ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Drop": {
            "place": dropped,
            "fn_ptr": {"kind": {"Fun": glue}, "generics": generics},
            "target": 1,
            "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, result), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body
}

#[test]
fn store_of_the_address_into_a_global_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let saved = u64_ty();
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            assign_to(global_place(&saved), ptr_cast(place(1, &ptr), &ptr, &saved)),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn store_of_a_status_into_a_global_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            assign_to(global_place(&word), const_use()),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn drop_of_an_untainted_local_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = status_return_after_drop(&word, &ptr, place(2, &word), 2);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    let glue = probe_fun(
        2,
        &["probe", "drop_flag"],
        vec![word.clone()],
        &word,
        json!("Opaque"),
    );
    let graph =
        lower_returned_address_sink(&word, &[], None, Some(&body), &[glue]).unwrap_or_else(|err| {
            panic!("dropping an untainted local must still free the spill: {err}")
        });
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn opaque_drop_of_the_pointer_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = status_return_after_drop(&word, &ptr, place(1, &ptr), 2);
    let glue = probe_fun(
        2,
        &["probe", "drop_ptr"],
        vec![ptr.clone()],
        &word,
        json!("Opaque"),
    );
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .expect_err("an opaque drop of the pointer must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn drop_glue_that_returns_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let slot = raw_ptr(&ptr, "Mut");
    let body = status_return_after_drop(&word, &ptr, place(1, &ptr), 2);
    let glue = probe_fun(
        2,
        &["probe", "drop_ptr"],
        vec![slot.clone()],
        &word,
        idle_body(&word, &[slot]),
    );
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| panic!("a drop glue that returns must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn drop_glue_that_saves_the_address_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let saved = u64_ty();
    let slot = raw_ptr(&ptr, "Mut");
    let body = status_return_after_drop(&word, &ptr, place(1, &ptr), 2);
    let glue_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, &word),
            local(1, Some("slot"), &slot)
        ]},
        "body": [{"statements": [
            assign_to(
                global_place(&saved),
                ptr_cast(deref_place(place(1, &slot), &ptr), &ptr, &saved),
            )
        ], "terminator": {"span": span, "kind": "Return"}}]
    }});
    let glue = probe_fun(2, &["probe", "drop_ptr"], vec![slot], &word, glue_body);
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .expect_err("drop glue that stores the address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn deref_of_a_cast_pointer_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&ptr, "Const");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            assign_to(place(2, &q_ty), ptr_cast(place(1, &ptr), &ptr, &q_ty)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                ptr_cast(deref_place(place(2, &q_ty), &ptr), &ptr, &result)
            ]}}),
        ],
    );
    assert_sink_frees(&result, &body);
}

#[test]
fn deref_after_both_pointer_depths_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&ptr, "Const");
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("q"), &q_ty)],
        vec![
            assign_to(place(2, &q_ty), ptr_cast(place(1, &ptr), &ptr, &q_ty)),
            ref_assign(2, &q_ty, place(1, &ptr)),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                ptr_cast(deref_place(place(2, &q_ty), &ptr), &ptr, &result)
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

fn compare_with_zero(op: &str, dest: u64, dest_ty: &Value, src: u64, src_ty: &Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": {"Assign": [
        place(dest, dest_ty),
        {"BinaryOp": [op, {"Copy": place(src, src_ty)}, {"Const": zero_const()}]}
    ]}})
}

fn comparison_assign(dest: u64, dest_ty: &Value, src: u64, src_ty: &Value) -> Value {
    compare_with_zero("Lt", dest, dest_ty, src, src_ty)
}

#[test]
fn pointer_comparison_used_as_a_switch_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    body["Unstructured"]["body"] = json!([
        {"statements": [comparison_assign(2, &word, 1, &ptr)], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Copy": place(2, &word)}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_escapes(&word, &body);
}

#[test]
fn null_check_used_as_a_switch_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    body["Unstructured"]["body"] = json!([
        {"statements": [compare_with_zero("Eq", 2, &word, 1, &ptr)], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Copy": place(2, &word)}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_frees(&word, &body);
}

fn assert_term(cond: Value, target: u64, on_unwind: u64) -> Value {
    json!({"Assert": {
        "assert": {"cond": cond, "expected": true, "check_kind": null},
        "target": target,
        "on_unwind": on_unwind
    }})
}

#[test]
fn assertion_of_the_address_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    body["Unstructured"]["body"] = json!([
        {"statements": [comparison_assign(2, &word, 1, &ptr)], "terminator": {"span": span, "kind":
            assert_term(json!({"Copy": place(2, &word)}), 1, 2)
        }},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_escapes(&word, &body);
}

#[test]
fn statement_assertion_of_the_address_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("flag"), &word)],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            json!({"span": span, "kind": {"Assert": {
                "cond": {"Copy": place(2, &word)},
                "expected": true,
                "check_kind": null
            }}}),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn null_check_used_as_an_assertion_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    body["Unstructured"]["body"] = json!([
        {"statements": [compare_with_zero("Ne", 2, &word, 1, &ptr)], "terminator": {"span": span, "kind":
            assert_term(json!({"Copy": place(2, &word)}), 1, 2)
        }},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_frees(&word, &body);
}

#[test]
fn statement_assertion_of_a_null_check_still_frees() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("flag"), &word)],
        vec![
            compare_with_zero("Eq", 2, &word, 1, &ptr),
            json!({"span": span, "kind": {"Assert": {
                "cond": {"Copy": place(2, &word)},
                "expected": true,
                "check_kind": null
            }}}),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn assertion_of_a_status_still_frees() {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind":
            assert_term(json!({"Const": null}), 1, 2)
        }},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_frees(&word, &body);
}

fn zero_arg_call(func: Value, dest: Value) -> Value {
    json!({"Call": {
        "call": {"func": func, "args": [], "dest": dest},
        "target": 1,
        "on_unwind": 2
    }})
}

fn indirect_call_body(tainted: bool, func: Value) -> Value {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("table"), &word),
            local(4, Some("fp"), &word),
            local(5, Some("status"), &word),
        ],
        vec![],
    );
    let select = if tainted {
        vec![
            comparison_assign(2, &word, 1, &ptr),
            assign_to(
                place(4, &word),
                copy_use(index_place(3, &word, 2, &word, &word)),
            ),
        ]
    } else {
        vec![assign_to(place(4, &word), const_use())]
    };
    body["Unstructured"]["body"] = json!([
        {"statements": select, "terminator": {"span": span, "kind":
            zero_arg_call(func, place(5, &word))
        }},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body
}

#[test]
fn dynamic_call_through_the_address_is_not_lowered() {
    let word = i64_ty();
    let body = indirect_call_body(true, json!({"Dynamic": {"Copy": place(4, &word)}}));
    assert_sink_escapes(&word, &body);
}

#[test]
fn dynamic_call_through_a_clean_pointer_still_frees() {
    let word = i64_ty();
    let body = indirect_call_body(false, json!({"Dynamic": {"Copy": place(4, &word)}}));
    assert_sink_frees(&word, &body);
}

#[test]
fn pointer_call_through_the_address_is_not_lowered() {
    let (_, generics, _, _) = probe_parts();
    let word = i64_ty();
    let body = indirect_call_body(
        true,
        json!({"Regular": {"kind": {"Ptr": {"Copy": place(4, &word)}}, "generics": generics}}),
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn pointer_call_through_a_clean_pointer_still_frees() {
    let (_, generics, _, _) = probe_parts();
    let word = i64_ty();
    let body = indirect_call_body(
        false,
        json!({"Regular": {"kind": {"Ptr": {"Copy": place(4, &word)}}, "generics": generics}}),
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn arithmetic_on_a_pointer_comparison_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("flag"), &word)],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"BinaryOp": ["Add", {"Copy": place(2, &word)}, {"Const": null}]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

fn null_check_plus_one(op: &str) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("flag"), &word), local(3, Some("n"), &result)],
        vec![
            compare_with_zero(op, 2, &word, 1, &ptr),
            assign_scalar_cast(3, 2, &word, &result),
            assign_to(
                place(0, &result),
                json!({"BinaryOp": [
                    "Add",
                    {"Copy": place(3, &result)},
                    {"Const": int_const("1")}
                ]}),
            ),
        ],
    )
}

#[test]
fn null_check_plus_one_still_frees() {
    assert_sink_frees(&u64_ty(), &null_check_plus_one("Eq"));
}

#[test]
fn ordering_plus_one_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &null_check_plus_one("Lt"));
}

fn field_of_pair_passed_to_helper(field: u64) -> (Value, Value) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pair"), &result),
        ],
        vec![],
    );
    body["Unstructured"]["body"] = json!([
        {"statements": [
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(place(3, &result), tuple_of(vec![
                json!({"Copy": place(2, &result)}),
                json!({"Const": null})
            ]))
        ], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(3, &result)}], "dest": place(0, &result)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let helper = probe_fun(
        2,
        &["probe", "field"],
        vec![result.clone()],
        &result,
        sink_unstructured(
            &result,
            &result,
            vec![assign_to(
                place(0, &result),
                copy_use(field_place(1, &result, field, &result)),
            )],
        ),
    );
    (body, helper)
}

#[test]
fn clean_field_passed_to_a_helper_still_frees() {
    let result = u64_ty();
    let (body, helper) = field_of_pair_passed_to_helper(1);
    let graph = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a clean aggregate field must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn address_field_passed_to_a_helper_is_not_lowered() {
    let result = u64_ty();
    let (body, helper) = field_of_pair_passed_to_helper(0);
    let err = lower_returned_address_sink(&result, &[], None, Some(&body), &[helper])
        .expect_err("the address field of a passed aggregate must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn comparison_helper_switch(op: &str) -> (Value, Value) {
    let (span, generics, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(1, &ptr)}], "dest": place(2, &word)},
            "target": 1, "on_unwind": 3
        }}}},
        {"statements": [], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Copy": place(2, &word)}, "targets": {"If": [2, 2]}}
        }}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push({
            let (_, _, _, local) = probe_parts();
            local(2, Some("flag"), &word)
        });
    let helper = probe_fun(
        2,
        &["probe", "less"],
        vec![ptr.clone()],
        &word,
        sink_unstructured(&word, &ptr, vec![compare_with_zero(op, 0, &word, 1, &ptr)]),
    );
    (body, helper)
}

#[test]
fn comparison_helper_then_a_switch_is_not_lowered() {
    let word = i64_ty();
    let (body, helper) = comparison_helper_switch("Lt");
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .expect_err("a switch on a comparison helper must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn null_check_helper_then_a_switch_still_frees() {
    let word = i64_ty();
    let (body, helper) = comparison_helper_switch("Eq");
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a switch on p == null must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn comparison_aggregate_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("lo"), &word), local(3, Some("hi"), &word)],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            comparison_assign(3, &word, 1, &ptr),
            json!({"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Aggregate": ["Tuple", [
                    {"Copy": place(2, &word)},
                    {"Copy": place(3, &word)}
                ]]}
            ]}}),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn store_of_a_comparison_into_a_global_is_not_lowered() {
    let (span, _, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            json!({"span": span, "kind": {"Assign": [
                global_place(&word),
                {"BinaryOp": ["Lt", {"Copy": place(1, &ptr)}, {"Const": null}]}
            ]}}),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn store_of_a_comparison_through_the_pointer_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            assign_deref(
                1,
                &ptr,
                json!({"BinaryOp": ["Lt", {"Copy": place(1, &ptr)}, {"Const": null}]}),
            ),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn indexed_store_of_a_comparison_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("table"), &word),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            assign_to(index_place(3, &word, 2, &word, &word), const_use()),
            assign_to(place(0, &word), copy_use(place(3, &word))),
        ],
    );
    assert_sink_escapes(&word, &body);
}

fn index_place(base: u64, base_ty: &Value, index: u64, index_ty: &Value, elem_ty: &Value) -> Value {
    json!({
        "kind": {"Projection": [
            place(base, base_ty),
            {"Index": {"offset": {"Copy": place(index, index_ty)}, "from_end": false}}
        ]},
        "ty": elem_ty
    })
}

fn copy_use(src: Value) -> Value {
    json!({"Use": [{"Copy": src}, "Yes"]})
}

#[test]
fn indexed_comparison_still_frees_the_spill() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("table"), &word),
        ],
        vec![
            compare_with_zero("Eq", 2, &word, 1, &ptr),
            assign_to(place(3, &word), const_use()),
            assign_to(
                place(0, &word),
                copy_use(index_place(3, &word, 2, &word, &word)),
            ),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn indexed_ordering_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("table"), &word),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            assign_to(place(3, &word), const_use()),
            assign_to(
                place(0, &word),
                copy_use(index_place(3, &word, 2, &word, &word)),
            ),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn const_index_still_frees_the_spill() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let indexed = json!({
        "kind": {"Projection": [
            place(2, &word),
            {"Index": {"offset": {"Const": null}, "from_end": false}}
        ]},
        "ty": word
    });
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("table"), &word)],
        vec![
            assign_to(place(2, &word), const_use()),
            assign_to(place(0, &word), copy_use(indexed)),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn indexed_comparison_or_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("table"), &result),
            local(4, Some("lo"), &result),
            local(5, Some("hi"), &result),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            assign_to(place(3, &result), const_use()),
            assign_to(
                place(4, &result),
                copy_use(index_place(3, &result, 2, &word, &result)),
            ),
            assign_to(
                place(5, &result),
                copy_use(index_place(3, &result, 2, &word, &result)),
            ),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"BinaryOp": ["BitOr", {"Copy": place(4, &result)}, {"Copy": place(5, &result)}]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn index_by_the_address_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("table"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(place(3, &result), const_use()),
            assign_to(
                place(0, &result),
                copy_use(index_place(3, &result, 2, &result, &result)),
            ),
        ],
    );
    assert_sink_escapes(&result, &body);
}

fn field_place(base: u64, base_ty: &Value, field: u64, elem_ty: &Value) -> Value {
    project_field(place(base, base_ty), field, elem_ty)
}

fn project_field(base: Value, field: u64, elem_ty: &Value) -> Value {
    json!({
        "kind": {"Projection": [base, {"Field": field}]},
        "ty": elem_ty
    })
}

#[test]
fn field_stores_of_comparisons_are_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("lo"), &word),
            local(3, Some("hi"), &word),
            local(4, Some("parts"), &word),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            comparison_assign(3, &word, 1, &ptr),
            assign_to(field_place(4, &word, 0, &word), copy_use(place(2, &word))),
            assign_to(field_place(4, &word, 1, &word), copy_use(place(3, &word))),
            assign_to(place(0, &word), copy_use(place(4, &word))),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn const_index_stores_of_comparisons_are_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let slot = |index: i64| {
        json!({
            "kind": {"Projection": [
                place(4, &word),
                {"Index": {"offset": {"Const": index}, "from_end": false}}
            ]},
            "ty": word
        })
    };
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("lo"), &word),
            local(3, Some("hi"), &word),
            local(4, Some("parts"), &word),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            comparison_assign(3, &word, 1, &ptr),
            assign_to(slot(0), copy_use(place(2, &word))),
            assign_to(slot(1), copy_use(place(3, &word))),
            assign_to(place(0, &word), copy_use(place(4, &word))),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn drop_glue_that_publishes_a_comparison_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let slot = raw_ptr(&word, "Mut");
    let mut body = status_return_after_drop(&word, &ptr, place(2, &word), 2);
    body["Unstructured"]["body"][0]["statements"] = json!([comparison_assign(2, &word, 1, &ptr)]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    let glue_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, &word),
            local(1, Some("slot"), &slot)
        ]},
        "body": [{"statements": [
            assign_to(global_place(&word), copy_use(deref_place(place(1, &slot), &word)))
        ], "terminator": {"span": span, "kind": "Return"}}]
    }});
    let glue = probe_fun(2, &["probe", "drop_flag"], vec![slot], &word, glue_body);
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .expect_err("drop glue that publishes a comparison must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn drop_of_a_comparison_with_idle_glue_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let slot = raw_ptr(&word, "Mut");
    let mut body = status_return_after_drop(&word, &ptr, place(2, &word), 2);
    body["Unstructured"]["body"][0]["statements"] = json!([comparison_assign(2, &word, 1, &ptr)]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    let glue = probe_fun(
        2,
        &["probe", "drop_flag"],
        vec![slot.clone()],
        &word,
        idle_body(&word, &[slot]),
    );
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| panic!("idle drop glue must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn drop_glue_switching_on(
    op: &str,
) -> Result<FunctionGraph, majit_translate::front::mir::LowerError> {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let slot = raw_ptr(&word, "Mut");
    let mut body = status_return_after_drop(&word, &ptr, place(2, &word), 2);
    body["Unstructured"]["body"][0]["statements"] =
        json!([compare_with_zero(op, 2, &word, 1, &ptr)]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("flag"), &word));
    let glue_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, &word),
            local(1, Some("slot"), &slot)
        ]},
        "body": [
            {"statements": [], "terminator": {"span": span, "kind": {"Switch": {
                "discr": {"Copy": deref_place(place(1, &slot), &word)},
                "targets": {"If": [1, 2]}
            }}}},
            {"statements": [assign_to(place(0, &word), const_use())],
                "terminator": {"span": span, "kind": "Return"}},
            {"statements": [assign_to(place(0, &word), const_use())],
                "terminator": {"span": span, "kind": "Return"}}
        ]
    }});
    let glue = probe_fun(2, &["probe", "drop_flag"], vec![slot], &word, glue_body);
    lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
}

#[test]
fn drop_glue_switching_on_a_null_check_still_frees() {
    let graph = drop_glue_switching_on("Eq").unwrap_or_else(|err| {
        panic!("drop glue that switches on p == null must still free the spill: {err}")
    });
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn drop_glue_switching_on_an_ordering_is_not_lowered() {
    let err =
        drop_glue_switching_on("Lt").expect_err("drop glue that switches on p < 0 must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn dynamic_index_store_of_a_comparison_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("flag"), &word),
            local(3, Some("i"), &word),
            local(4, Some("table"), &word),
        ],
        vec![
            comparison_assign(2, &word, 1, &ptr),
            assign_to(place(3, &word), const_use()),
            assign_to(
                index_place(4, &word, 3, &word, &word),
                copy_use(place(2, &word)),
            ),
            assign_to(place(0, &word), copy_use(place(4, &word))),
        ],
    );
    assert_sink_escapes(&word, &body);
}

fn discriminant_assign(dest: u64, dest_ty: &Value, src: u64, src_ty: &Value) -> Value {
    assign_to(
        place(dest, dest_ty),
        json!({"Discriminant": place(src, src_ty)}),
    )
}

#[test]
fn combined_discriminants_of_the_address_are_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("lo"), &result),
            local(4, Some("hi"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            discriminant_assign(3, &result, 2, &result),
            discriminant_assign(4, &result, 2, &result),
            json!({"span": span, "kind": {"Assign": [
                place(0, &result),
                {"BinaryOp": ["BitOr", {"Copy": place(3, &result)}, {"Copy": place(4, &result)}]}
            ]}}),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn discriminant_of_the_address_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("bits"), &result)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            discriminant_assign(0, &result, 2, &result),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn discriminant_of_a_narrowed_address_tag_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = usize_ty();
    let low = u8_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("masked"), &bits),
            local(4, Some("low"), &low),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(3, &bits),
                json!({"BinaryOp": [
                    "BitAnd",
                    {"Copy": place(2, &bits)},
                    {"Const": int_const("3")}
                ]}),
            ),
            assign_scalar_cast(4, 3, &bits, &low),
            discriminant_assign(0, &word, 4, &low),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn discriminant_of_a_status_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("status"), &word)],
        vec![
            assign_to(place(2, &word), const_use()),
            discriminant_assign(0, &word, 2, &word),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn dynamic_index_store_of_a_status_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("i"), &word), local(3, Some("table"), &word)],
        vec![
            assign_to(place(2, &word), const_use()),
            assign_to(index_place(3, &word, 2, &word, &word), const_use()),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

fn call_through(dest: Value, inner_returns_address: bool) -> (Value, Value) {
    let (span, generics, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let inner_stmts = if inner_returns_address {
        vec![assign_scalar_cast(0, 1, &ptr, &result)]
    } else {
        vec![assign_to(place(0, &result), const_use())]
    };
    let inner = probe_fun(
        2,
        &["probe", "inner"],
        vec![ptr.clone()],
        &result,
        sink_unstructured(&result, &ptr, inner_stmts),
    );
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(1, &ptr)}], "dest": dest},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    (body, inner)
}

#[test]
fn call_result_stored_through_the_pointer_is_not_lowered() {
    let word = i64_ty();
    let result = u64_ty();
    let out = raw_ptr(&result, "Mut");
    let (mut body, inner) = call_through(deref_place(place(2, &out), &result), true);
    let (_, _, _, local) = probe_parts();
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("out"), &out));
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[inner])
        .expect_err("a call result stored through a pointer must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn call_result_stored_into_a_global_is_not_lowered() {
    let word = i64_ty();
    let result = u64_ty();
    let (body, inner) = call_through(global_place(&result), true);
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[inner])
        .expect_err("a call result stored into a global must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn call_result_status_stored_through_the_pointer_still_frees() {
    let word = i64_ty();
    let result = u64_ty();
    let out = raw_ptr(&result, "Mut");
    let (mut body, inner) = call_through(deref_place(place(2, &out), &result), false);
    let (_, _, _, local) = probe_parts();
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("out"), &out));
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[inner])
        .unwrap_or_else(|err| panic!("a status stored through a pointer must still free: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn overwritten_address_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("result"), &result)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(place(2, &result), const_use()),
            assign_to(place(0, &result), copy_use(place(2, &result))),
        ],
    );
    assert_sink_frees(&result, &body);
}

#[test]
fn address_kept_on_another_path_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("result"), &result)],
        vec![],
    );
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Const": null}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(place(0, &result), copy_use(place(2, &result)))
        ], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [
            assign_to(place(2, &result), const_use()),
            assign_to(place(0, &result), copy_use(place(2, &result)))
        ], "terminator": {"span": span, "kind": "Return"}}
    ]);
    assert_sink_escapes(&result, &body);
}

fn ref_of(src: Value) -> Value {
    json!({"Ref": {"place": src, "kind": "Shared", "ptr_metadata": null}})
}

fn tuple_of(fields: Vec<Value>) -> Value {
    json!({"Aggregate": ["Tuple", fields]})
}

#[test]
fn reference_to_the_address_stored_in_a_global_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let reference = borrow_ty(&ptr, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("cell"), &ptr)],
        vec![
            assign_to(place(2, &ptr), copy_use(place(1, &ptr))),
            assign_to(global_place(&reference), ref_of(place(2, &ptr))),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn reference_to_a_clean_local_stored_in_a_global_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let reference = borrow_ty(&word, "Shared");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("cell"), &word)],
        vec![
            assign_to(place(2, &word), const_use()),
            assign_to(global_place(&reference), ref_of(place(2, &word))),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn clean_aggregate_field_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pair"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(
                place(0, &result),
                copy_use(field_place(3, &result, 1, &result)),
            ),
        ],
    );
    assert_sink_frees(&result, &body);
}

fn nested_pair_body(leaf: u64) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("inner"), &result),
            local(4, Some("outer"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(
                place(4, &result),
                tuple_of(vec![json!({"Copy": place(3, &result)})]),
            ),
            assign_to(
                place(0, &result),
                copy_use(project_field(
                    field_place(4, &result, 0, &result),
                    leaf,
                    &result,
                )),
            ),
        ],
    )
}

#[test]
fn clean_nested_aggregate_field_still_frees() {
    assert_sink_frees(&u64_ty(), &nested_pair_body(1));
}

#[test]
fn nested_address_field_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &nested_pair_body(0));
}

fn len_of(src: Value) -> Value {
    json!({"Len": src})
}

fn array_ty(elem: &Value) -> Value {
    json!({"Array": [elem, null, null]})
}

fn slice_ty(elem: &Value) -> Value {
    json!({"Slice": [elem, null]})
}

fn array_of(elem: &Value, operands: Vec<Value>) -> Value {
    json!({"Aggregate": [{"Array": [elem, null]}, operands]})
}

fn unsize_to_slice(src: Value, src_ty: &Value, dest_ty: &Value) -> Value {
    json!({"UnaryOp": [
        {"Cast": {"Unsize": [src_ty, dest_ty, {"Length": null}]}},
        {"Copy": src}
    ]})
}

#[test]
fn len_of_a_pointer_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_unstructured(
        &result,
        &ptr,
        vec![assign_to(place(0, &result), len_of(place(1, &ptr)))],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn len_of_a_clean_array_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let array = array_ty(&word);
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("values"), &array)],
        vec![
            assign_to(
                place(2, &array),
                array_of(&word, vec![json!({"Const": null})]),
            ),
            assign_to(place(0, &word), len_of(place(2, &array))),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn len_of_an_address_array_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let array = array_ty(&bits);
    let body = sink_with_extra(
        &bits,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("values"), &array),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(3, &array),
                array_of(&bits, vec![json!({"Copy": place(2, &bits)})]),
            ),
            assign_to(place(0, &bits), len_of(place(3, &array))),
        ],
    );
    assert_sink_frees(&bits, &body);
}

#[test]
fn len_of_a_reference_to_an_address_array_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let array = array_ty(&bits);
    let array_ref = borrow_ty(&array, "Shared");
    let body = sink_with_extra(
        &bits,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("values"), &array),
            local(4, Some("r"), &array_ref),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(3, &array),
                array_of(&bits, vec![json!({"Copy": place(2, &bits)})]),
            ),
            ref_assign(4, &array_ref, place(3, &array)),
            assign_to(place(0, &bits), len_of(place(4, &array_ref))),
        ],
    );
    assert_sink_frees(&bits, &body);
}

#[test]
fn len_of_an_unsized_address_array_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let array = array_ty(&bits);
    let array_ref = borrow_ty(&array, "Shared");
    let slice = slice_ty(&bits);
    let slice_ref = borrow_ty(&slice, "Shared");
    let body = sink_with_extra(
        &bits,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("values"), &array),
            local(4, Some("r"), &array_ref),
            local(5, Some("s"), &slice_ref),
            local(6, Some("t"), &slice_ref),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(3, &array),
                array_of(&bits, vec![json!({"Copy": place(2, &bits)})]),
            ),
            ref_assign(4, &array_ref, place(3, &array)),
            assign_to(
                place(5, &slice_ref),
                unsize_to_slice(place(4, &array_ref), &array_ref, &slice_ref),
            ),
            assign_to(place(6, &slice_ref), copy_use(place(5, &slice_ref))),
            assign_to(
                place(0, &bits),
                len_of(deref_place(place(6, &slice_ref), &slice)),
            ),
        ],
    );
    assert_sink_frees(&bits, &body);
}

#[test]
fn len_of_an_untracked_slice_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let slice = slice_ty(&bits);
    let slice_ref = borrow_ty(&slice, "Shared");
    let body = sink_with_extra(
        &bits,
        &ptr,
        vec![local(2, Some("s"), &slice_ref)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &slice_ref),
            assign_to(
                place(0, &bits),
                len_of(deref_place(place(2, &slice_ref), &slice)),
            ),
        ],
    );
    assert_sink_escapes(&bits, &body);
}

fn mixed_address_join(return_pair: bool) -> Value {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let mut body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pair"), &result),
        ],
        vec![],
    );
    let ret = if return_pair {
        assign_to(place(0, &result), copy_use(place(3, &result)))
    } else {
        assign_to(place(0, &result), const_use())
    };
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Const": null}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(place(3, &result), tuple_of(vec![
                json!({"Copy": place(2, &result)}),
                json!({"Const": null})
            ]))
        ], "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
        {"statements": [
            assign_scalar_cast(3, 1, &ptr, &result)
        ], "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
        {"statements": [ret], "terminator": {"span": span, "kind": "Return"}}
    ]);
    body
}

#[test]
fn split_and_whole_local_join_of_the_address_is_not_lowered() {
    let result = u64_ty();
    assert_sink_escapes(&result, &mixed_address_join(true));
}

#[test]
fn split_and_whole_local_join_then_a_status_still_frees() {
    let result = u64_ty();
    assert_sink_frees(&result, &mixed_address_join(false));
}

#[test]
fn address_aggregate_field_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pair"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(
                place(0, &result),
                copy_use(field_place(3, &result, 0, &result)),
            ),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn deref_of_a_direct_field_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("pair"), &word)],
        vec![
            assign_to(
                place(2, &word),
                tuple_of(vec![json!({"Copy": place(1, &ptr)})]),
            ),
            assign_to(
                place(0, &word),
                copy_use(deref_place(field_place(2, &word, 0, &ptr), &word)),
            ),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn deref_of_an_offset_field_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("q"), &ptr),
            local(4, Some("pair"), &word),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(2, &bits),
                json!({"BinaryOp": [
                    "Add",
                    {"Copy": place(2, &bits)},
                    {"Const": zero_const()}
                ]}),
            ),
            assign_scalar_cast(3, 2, &bits, &ptr),
            assign_to(
                place(4, &word),
                tuple_of(vec![json!({"Copy": place(3, &ptr)})]),
            ),
            assign_to(
                place(0, &word),
                copy_use(deref_place(field_place(4, &word, 0, &ptr), &word)),
            ),
        ],
    );
    assert_sink_escapes(&word, &body);
}

fn constant_index_of_pair(index: i64) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let indexed = json!({
        "kind": {"Projection": [
            place(3, &result),
            {"Index": {"offset": {"Const": index}, "from_end": false}}
        ]},
        "ty": result
    });
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("arr"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(place(0, &result), copy_use(indexed)),
        ],
    )
}

#[test]
fn clean_constant_index_still_frees() {
    assert_sink_frees(&u64_ty(), &constant_index_of_pair(1));
}

#[test]
fn constant_index_of_the_address_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &constant_index_of_pair(0));
}

#[test]
fn dynamic_index_of_an_address_aggregate_is_not_lowered() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("arr"), &result),
            local(4, Some("i"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(place(4, &result), json!({"Use": [{"Const": 1}, "Yes"]})),
            assign_to(
                place(0, &result),
                copy_use(index_place(3, &result, 4, &result, &result)),
            ),
        ],
    );
    assert_sink_escapes(&result, &body);
}

fn copied_aggregate_body(field: u64, mov: bool) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let source = place(3, &result);
    let carried = if mov {
        json!({"Use": [{"Move": source}, "Yes"]})
    } else {
        copy_use(source)
    };
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("pair"), &result),
            local(4, Some("copy"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(place(4, &result), carried),
            assign_to(
                place(0, &result),
                copy_use(field_place(4, &result, field, &result)),
            ),
        ],
    )
}

#[test]
fn copied_aggregate_clean_field_still_frees() {
    let result = u64_ty();
    assert_sink_frees(&result, &copied_aggregate_body(1, false));
}

#[test]
fn moved_aggregate_clean_field_still_frees() {
    let result = u64_ty();
    assert_sink_frees(&result, &copied_aggregate_body(1, true));
}

#[test]
fn copied_aggregate_address_field_is_not_lowered() {
    let result = u64_ty();
    assert_sink_escapes(&result, &copied_aggregate_body(0, false));
}

fn projected_aggregate_body(field: u64) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("inner"), &result),
            local(4, Some("outer"), &result),
            local(5, Some("inner2"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(
                place(4, &result),
                tuple_of(vec![
                    json!({"Copy": place(3, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(
                place(5, &result),
                copy_use(field_place(4, &result, 0, &result)),
            ),
            assign_to(
                place(0, &result),
                copy_use(field_place(5, &result, field, &result)),
            ),
        ],
    )
}

#[test]
fn copied_projection_clean_field_still_frees() {
    assert_sink_frees(&u64_ty(), &projected_aggregate_body(1));
}

#[test]
fn copied_projection_address_field_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &projected_aggregate_body(0));
}

fn union_field_body(address: bool) -> (Value, Value) {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let union_ty = adt_ty(1, vec![]);
    let decl = named_decl(
        1,
        &["probe", "U"],
        json!({"Union": [
            field_decl("p", &ptr),
            field_decl("bits", &result)
        ]}),
    );
    let operand = if address {
        json!({"Copy": place(1, &ptr)})
    } else {
        json!({"Const": null})
    };
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("word"), &union_ty)],
        vec![
            assign_to(
                place(2, &union_ty),
                json!({"Aggregate": [{"Adt": [1, null]}, [operand]]}),
            ),
            assign_to(
                place(0, &result),
                copy_use(field_place(2, &union_ty, 1, &result)),
            ),
        ],
    );
    (decl, body)
}

#[test]
fn union_field_of_the_address_is_not_lowered() {
    let result = u64_ty();
    let (decl, body) = union_field_body(true);
    let err = lower_returned_address_sink(&result, &[decl], None, Some(&body), &[])
        .expect_err("another field of a union that holds the spill address must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn union_field_of_a_status_still_frees() {
    let result = u64_ty();
    let (decl, body) = union_field_body(false);
    let graph = lower_returned_address_sink(&result, &[decl], None, Some(&body), &[])
        .unwrap_or_else(|err| panic!("a clean union field must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

/// `let q = &raw const bits; let aliases = [q; N]; bits = p as usize`.
fn repeat_clean_alias(count: Value, index: u64, reload_after: bool) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let aliases_ty = array_ty(&q_ty);
    let store = assign_scalar_cast(2, 1, &ptr, &result);
    let load = assign_to(
        place(0, &result),
        copy_use(deref_place(
            const_index_place(4, &aliases_ty, index, &q_ty),
            &result,
        )),
    );
    let mut statements = vec![
        assign_to(place(2, &result), const_use()),
        raw_const_assign(3, &q_ty, place(2, &result)),
        assign_to(
            place(4, &aliases_ty),
            json!({"Repeat": [
                json!({"Copy": place(3, &q_ty)}),
                q_ty,
                count,
                null
            ]}),
        ),
    ];
    if reload_after {
        statements.push(store);
        statements.push(load);
    } else {
        statements.push(load);
        statements.push(store);
    }
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("q"), &q_ty),
            local(4, Some("aliases"), &aliases_ty),
        ],
        statements,
    )
}

fn usize_repeat_count(text: &str) -> Value {
    json!([
        {"Integer": {"Unsigned": ["Usize", text]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ])
}

#[test]
fn repeat_of_a_clean_alias_reloaded_after_the_store_is_not_lowered() {
    assert_sink_escapes(
        &u64_ty(),
        &repeat_clean_alias(usize_repeat_count("1"), 0, true),
    );
}

#[test]
fn repeat_of_a_clean_alias_reloaded_before_the_store_still_frees() {
    assert_sink_frees(
        &u64_ty(),
        &repeat_clean_alias(usize_repeat_count("1"), 0, false),
    );
}

#[test]
fn repeat_element_one_of_a_clean_alias_is_not_lowered() {
    assert_sink_escapes(
        &u64_ty(),
        &repeat_clean_alias(usize_repeat_count("2"), 1, true),
    );
}

#[test]
fn repeat_of_a_clean_alias_with_an_unknown_count_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &repeat_clean_alias(Value::Null, 0, true));
}

#[test]
fn repeat_of_a_constant_still_frees() {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let aliases_ty = array_ty(&result);
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("aliases"), &aliases_ty)],
        vec![
            assign_to(
                place(2, &aliases_ty),
                json!({"Repeat": [
                    json!({"Const": null}),
                    result,
                    usize_repeat_count("1"),
                    null
                ]}),
            ),
            assign_to(place(0, &result), const_use()),
        ],
    );
    assert_sink_frees(&result, &body);
}

/// `let q = &raw const bits; let u = U { q }; bits = p as usize; return *u.q`.
fn union_clean_alias(reload_after: bool) -> (Value, Value) {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let union_ty = adt_ty(1, vec![]);
    let decl = named_decl(
        1,
        &["probe", "U"],
        json!({"Union": [field_decl("q", &q_ty), field_decl("bits", &result)]}),
    );
    let store = assign_scalar_cast(2, 1, &ptr, &result);
    let load = assign_to(
        place(0, &result),
        copy_use(deref_place(field_place(4, &union_ty, 0, &q_ty), &result)),
    );
    let mut statements = vec![
        assign_to(place(2, &result), const_use()),
        raw_const_assign(3, &q_ty, place(2, &result)),
        assign_to(
            place(4, &union_ty),
            json!({"Aggregate": [
                {"Adt": [1, null]},
                [json!({"Copy": place(3, &q_ty)})]
            ]}),
        ),
    ];
    if reload_after {
        statements.push(store);
        statements.push(load);
    } else {
        statements.push(load);
        statements.push(store);
    }
    let body = sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("q"), &q_ty),
            local(4, Some("u"), &union_ty),
        ],
        statements,
    );
    (decl, body)
}

#[test]
fn union_of_a_clean_alias_reloaded_after_the_store_is_not_lowered() {
    let result = u64_ty();
    let (decl, body) = union_clean_alias(true);
    let err = lower_returned_address_sink(&result, &[decl], None, Some(&body), &[])
        .expect_err("a reload through a union alias must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn union_of_a_clean_alias_reloaded_before_the_store_still_frees() {
    let result = u64_ty();
    let (decl, body) = union_clean_alias(false);
    let graph = lower_returned_address_sink(&result, &[decl], None, Some(&body), &[])
        .unwrap_or_else(|err| panic!("a reload before the store must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn bare_stmt(kind: Value) -> Value {
    let (span, _, _, _) = probe_parts();
    json!({"span": span, "kind": kind})
}

fn status_after(statements: Vec<Value>) -> Value {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut statements = statements;
    statements.push(assign_to(place(0, &word), const_use()));
    sink_with_extra(&word, &ptr, vec![], statements)
}

fn unknown_place(ty: &Value) -> Value {
    json!({"kind": {"Mystery": null}, "ty": ty})
}

#[test]
fn storage_live_then_a_status_still_frees() {
    let word = i64_ty();
    assert_sink_frees(
        &word,
        &status_after(vec![bare_stmt(json!({"StorageLive": 1}))]),
    );
}

#[test]
fn set_discriminant_then_a_status_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    assert_sink_frees(
        &word,
        &status_after(vec![bare_stmt(
            json!({"SetDiscriminant": [place(1, &ptr), 0]}),
        )]),
    );
}

#[test]
fn set_discriminant_through_the_spill_pointer_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    assert_sink_frees(
        &word,
        &status_after(vec![bare_stmt(json!({
            "SetDiscriminant": [deref_place(place(1, &ptr), &word), 0]
        }))]),
    );
}

#[test]
fn set_discriminant_through_a_derived_pointer_is_not_lowered() {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let body = sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("bits"), &bits), local(3, Some("q"), &ptr)],
        vec![
            assign_scalar_cast(2, 1, &ptr, &bits),
            json!({"span": span, "kind": {"Assign": [
                place(2, &bits),
                {"BinaryOp": ["Add", {"Copy": place(2, &bits)}, {"Const": zero_const()}]}
            ]}}),
            assign_scalar_cast(3, 2, &bits, &ptr),
            bare_stmt(json!({
                "SetDiscriminant": [deref_place(place(3, &ptr), &word), 0]
            })),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn nop_then_a_status_still_frees() {
    let word = i64_ty();
    assert_sink_frees(&word, &status_after(vec![bare_stmt(json!({"Nop": null}))]));
}

#[test]
fn copy_nonoverlapping_is_not_lowered() {
    let word = i64_ty();
    assert_sink_escapes(
        &word,
        &status_after(vec![bare_stmt(json!({"CopyNonOverlapping": null}))]),
    );
}

#[test]
fn unparsed_statement_is_not_lowered() {
    let word = i64_ty();
    assert_sink_escapes(
        &word,
        &status_after(vec![bare_stmt(json!({"StorageLive": "nope"}))]),
    );
}

#[test]
fn unknown_rvalue_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            assign_to(place(0, &word), json!({"Mystery": null})),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn size_of_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![assign_to(
            place(0, &word),
            json!({"NullaryOp": ["SizeOf", word]}),
        )],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn read_of_an_unknown_place_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![assign_to(place(0, &word), copy_use(unknown_place(&word)))],
    );
    assert_sink_escapes(&word, &body);
}

#[test]
fn store_of_the_address_to_an_unknown_place_is_not_lowered() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let body = sink_unstructured(
        &result,
        &ptr,
        vec![
            assign_to(
                unknown_place(&result),
                ptr_cast(place(1, &ptr), &ptr, &result),
            ),
            assign_to(place(0, &result), const_use()),
        ],
    );
    assert_sink_escapes(&result, &body);
}

#[test]
fn store_of_a_status_to_an_unknown_place_still_frees() {
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let body = sink_unstructured(
        &word,
        &ptr,
        vec![
            assign_to(unknown_place(&word), const_use()),
            assign_to(place(0, &word), const_use()),
        ],
    );
    assert_sink_frees(&word, &body);
}

#[test]
fn call_with_an_unknown_place_is_not_lowered() {
    let (span, generics, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Copy": unknown_place(&word)}],
                "dest": place(0, &word)
            },
            "target": 1,
            "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_escapes(&word, &body);
}

#[test]
fn drop_of_an_unknown_place_is_not_lowered() {
    let (span, generics, _, _) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Drop": {
            "place": unknown_place(&word),
            "fn_ptr": {"kind": {"Fun": 2}, "generics": generics},
            "target": 1,
            "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    assert_sink_escapes(&word, &body);
}

/// `q = &raw mut bits; bits = p as usize; qcast = q as *mut T; *qcast = 0`.
/// `*mut u8` does not cover the `u64`. `*mut u64` does, so the store
/// clears `bits`.
fn cast_store_of_clean_alias(wide: bool) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let q_ty = raw_ptr(&bits, "Mut");
    let narrow = u8_ty();
    let cast_ty = if wide {
        q_ty.clone()
    } else {
        raw_ptr(&narrow, "Mut")
    };
    let stored = if wide { bits.clone() } else { narrow };
    sink_with_extra(
        &bits,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("q"), &q_ty),
            local(4, Some("qcast"), &cast_ty),
        ],
        vec![
            assign_to(place(2, &bits), const_use()),
            assign_to(
                place(3, &q_ty),
                json!({"RawPtr": {
                    "place": place(2, &bits),
                    "kind": "Mut",
                    "ptr_metadata": null
                }}),
            ),
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(
                place(4, &cast_ty),
                ptr_cast(place(3, &q_ty), &q_ty, &cast_ty),
            ),
            assign_to(deref_place(place(4, &cast_ty), &stored), const_use()),
            assign_to(place(0, &bits), copy_use(place(2, &bits))),
        ],
    )
}

#[test]
fn byte_store_through_a_narrow_cast_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &cast_store_of_clean_alias(false));
}

#[test]
fn word_store_through_a_wide_cast_still_frees() {
    assert_sink_frees(&u64_ty(), &cast_store_of_clean_alias(true));
}

/// `inner = (p as usize, 0); outer = (inner,); outer.0.0 = 0`.
fn nested_leaf_store_body(store: bool, field: u64) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let mut statements = vec![
        assign_scalar_cast(2, 1, &ptr, &result),
        assign_to(
            place(3, &result),
            tuple_of(vec![
                json!({"Copy": place(2, &result)}),
                json!({"Const": null}),
            ]),
        ),
        assign_to(
            place(4, &result),
            tuple_of(vec![json!({"Copy": place(3, &result)})]),
        ),
    ];
    if store {
        statements.push(assign_to(
            project_field(project_field(place(4, &result), 0, &result), 0, &result),
            const_use(),
        ));
    }
    statements.push(assign_to(
        place(0, &result),
        copy_use(project_field(
            project_field(place(4, &result), 0, &result),
            field,
            &result,
        )),
    ));
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("inner"), &result),
            local(4, Some("outer"), &result),
        ],
        statements,
    )
}

#[test]
fn nested_clean_store_of_the_address_still_frees() {
    assert_sink_frees(&u64_ty(), &nested_leaf_store_body(true, 0));
}

#[test]
fn nested_clean_store_leaves_the_sibling_free() {
    assert_sink_frees(&u64_ty(), &nested_leaf_store_body(true, 1));
}

#[test]
fn nested_address_without_the_store_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &nested_leaf_store_body(false, 0));
}

/// Same pair as `constant_index_of_pair`, with a `Usize` `ConstantExpr`
/// (`const_expr_literal`) instead of a bare JSON integer.
fn constant_expr_index_of_pair(text: &str) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let offset = json!([
        {"Integer": {"Unsigned": ["Usize", text]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let indexed = json!({
        "kind": {"Projection": [
            place(3, &result),
            {"Index": {"offset": {"Const": offset}, "from_end": false}}
        ]},
        "ty": result
    });
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("arr"), &result),
        ],
        vec![
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(3, &result),
                tuple_of(vec![
                    json!({"Copy": place(2, &result)}),
                    json!({"Const": null}),
                ]),
            ),
            assign_to(place(0, &result), copy_use(indexed)),
        ],
    )
}

#[test]
fn constant_expr_index_of_the_clean_element_still_frees() {
    assert_sink_frees(&u64_ty(), &constant_expr_index_of_pair("1"));
}

#[test]
fn constant_expr_index_of_the_address_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &constant_expr_index_of_pair("0"));
}

fn const_index_place(base: u64, base_ty: &Value, index: u64, elem_ty: &Value) -> Value {
    json!({
        "kind": {"Projection": [
            place(base, base_ty),
            {"Index": {"offset": {"Const": index}, "from_end": false}}
        ]},
        "ty": elem_ty
    })
}

fn const_expr_index_place(base: u64, base_ty: &Value, text: &str, elem_ty: &Value) -> Value {
    let offset = json!([
        {"Integer": {"Unsigned": ["Usize", text]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    json!({
        "kind": {"Projection": [
            place(base, base_ty),
            {"Index": {"offset": {"Const": offset}, "from_end": false}}
        ]},
        "ty": elem_ty
    })
}

fn cast_into(dest: Value, src: u64, src_ty: &Value, dest_ty: &Value) -> Value {
    assign_to(
        dest,
        json!({"UnaryOp": [
            {"Cast": {"Scalar": [src_ty, dest_ty]}},
            {"Copy": place(src, src_ty)}
        ]}),
    )
}

/// `pair = (0, 0); q = &raw const pair.field; pair.field = p as usize`.
fn field_alias_reload(store_before_return: bool) -> Value {
    let (_, _, _, local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let leaf = field_place(2, &result, 0, &result);
    let mut statements = vec![
        assign_to(
            place(2, &result),
            tuple_of(vec![json!({"Const": null}), json!({"Const": null})]),
        ),
        raw_const_assign(3, &q_ty, leaf),
    ];
    let reload = assign_to(
        place(0, &result),
        copy_use(deref_place(place(3, &q_ty), &result)),
    );
    let store = cast_into(field_place(2, &result, 0, &result), 1, &ptr, &result);
    if store_before_return {
        statements.push(store);
        statements.push(reload);
    } else {
        statements.push(reload);
        statements.push(store);
    }
    sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("pair"), &result), local(3, Some("q"), &q_ty)],
        statements,
    )
}

#[test]
fn field_alias_of_a_later_store_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &field_alias_reload(true));
}

#[test]
fn field_alias_read_before_the_store_still_frees() {
    assert_sink_frees(&u64_ty(), &field_alias_reload(false));
}

/// `arr = (0, 0); q = &arr[index]; arr[0] = p as usize; return *q`.
fn index_alias_reload(index_place: Value) -> Value {
    let (_, _, _, local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    sink_with_extra(
        &result,
        &ptr,
        vec![local(2, Some("arr"), &result), local(3, Some("q"), &q_ty)],
        vec![
            assign_to(
                place(2, &result),
                tuple_of(vec![json!({"Const": null}), json!({"Const": null})]),
            ),
            raw_const_assign(3, &q_ty, index_place),
            cast_into(const_index_place(2, &result, 0, &result), 1, &ptr, &result),
            assign_to(
                place(0, &result),
                copy_use(deref_place(place(3, &q_ty), &result)),
            ),
        ],
    )
}

#[test]
fn const_index_alias_of_a_later_store_is_not_lowered() {
    let result = u64_ty();
    assert_sink_escapes(
        &result,
        &index_alias_reload(const_index_place(2, &result, 0, &result)),
    );
}

#[test]
fn const_index_alias_of_the_sibling_still_frees() {
    let result = u64_ty();
    assert_sink_frees(
        &result,
        &index_alias_reload(const_index_place(2, &result, 1, &result)),
    );
}

#[test]
fn const_expr_index_alias_of_a_later_store_is_not_lowered() {
    let result = u64_ty();
    assert_sink_escapes(
        &result,
        &index_alias_reload(const_expr_index_place(2, &result, "0", &result)),
    );
}

/// `pair = (p as usize, p as usize); q = &pair.0; *q = 0; return pair.field`.
fn clear_named_field_through_alias(field: u64) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Mut");
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("pair"), &result),
            local(3, Some("q"), &q_ty),
            local(4, Some("bits"), &result),
        ],
        vec![
            assign_scalar_cast(4, 1, &ptr, &result),
            assign_to(
                place(2, &result),
                tuple_of(vec![
                    json!({"Copy": place(4, &result)}),
                    json!({"Copy": place(4, &result)}),
                ]),
            ),
            assign_to(
                place(3, &q_ty),
                json!({"RawPtr": {
                    "place": field_place(2, &result, 0, &result),
                    "kind": "Mut",
                    "ptr_metadata": null
                }}),
            ),
            assign_to(deref_place(place(3, &q_ty), &result), const_use()),
            assign_to(
                place(0, &result),
                copy_use(field_place(2, &result, field, &result)),
            ),
        ],
    )
}

#[test]
fn store_through_a_field_alias_still_frees() {
    assert_sink_frees(&u64_ty(), &clear_named_field_through_alias(0));
}

#[test]
fn store_through_a_field_alias_leaves_the_sibling() {
    assert_sink_escapes(&u64_ty(), &clear_named_field_through_alias(1));
}

/// `bits = 0; q = &raw const bits; wrapper = (q,); bits = p as usize; drop wrapper`.
fn alias_in_wrapper_drop(store_address: bool) -> Value {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let q_ty = raw_ptr(&bits, "Const");
    let mut statements = vec![
        assign_to(place(2, &bits), const_use()),
        raw_const_assign(3, &q_ty, place(2, &bits)),
        assign_to(
            place(4, &word),
            tuple_of(vec![json!({"Copy": place(3, &q_ty)})]),
        ),
    ];
    if store_address {
        statements.push(assign_scalar_cast(2, 1, &ptr, &bits));
    }
    let mut body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("q"), &q_ty),
            local(4, Some("wrapper"), &word),
        ],
        statements.clone(),
    );
    body["Unstructured"]["body"] = json!([
        {"statements": statements, "terminator": {"span": span, "kind": {"Drop": {
            "place": place(4, &word),
            "fn_ptr": {"kind": {"Fun": 2}, "generics": generics},
            "target": 1,
            "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body
}

fn glue_that_publishes_the_field() -> Value {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let bits = u64_ty();
    let saved = u64_ty();
    let field_ptr = raw_ptr(&bits, "Const");
    let slot = raw_ptr(&word, "Mut");
    let glue_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, &word),
            local(1, Some("slot"), &slot)
        ]},
        "body": [{"statements": [
            assign_to(
                global_place(&saved),
                ptr_cast(
                    deref_place(
                        project_field(deref_place(place(1, &slot), &word), 0, &field_ptr),
                        &bits,
                    ),
                    &bits,
                    &saved,
                ),
            )
        ], "terminator": {"span": span, "kind": "Return"}}]
    }});
    probe_fun(2, &["probe", "drop_wrapper"], vec![slot], &word, glue_body)
}

#[test]
fn drop_of_a_field_referent_is_not_lowered() {
    let body = alias_in_wrapper_drop(true);
    let glue = glue_that_publishes_the_field();
    let err = lower_returned_address_sink(&i64_ty(), &[], None, Some(&body), &[glue])
        .expect_err("drop glue that publishes a field referent must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn drop_of_a_clean_field_referent_still_frees() {
    let body = alias_in_wrapper_drop(false);
    let glue = glue_that_publishes_the_field();
    let graph = lower_returned_address_sink(&i64_ty(), &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| {
            panic!("dropping a clean field referent must still free the spill: {err}")
        });
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn idle_drop_of_a_field_referent_still_frees() {
    let word = i64_ty();
    let body = alias_in_wrapper_drop(true);
    let slot = raw_ptr(&word, "Mut");
    let glue = probe_fun(
        2,
        &["probe", "drop_wrapper"],
        vec![slot.clone()],
        &word,
        idle_body(&word, &[slot]),
    );
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| panic!("idle drop glue must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

/// `aliases = [q; 1]; aliases[i] = stored; bits = p as usize; return *aliases[0]`.
fn dynamic_store_into_repeat(stored_alias: bool, alias_elements: bool) -> Value {
    let (_, _, _, local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let result = u64_ty();
    let q_ty = raw_ptr(&result, "Const");
    let aliases_ty = array_ty(&q_ty);
    let element = if alias_elements {
        json!({"Copy": place(3, &q_ty)})
    } else {
        json!({"Const": null})
    };
    let stored = if stored_alias {
        copy_use(place(3, &q_ty))
    } else {
        const_use()
    };
    sink_with_extra(
        &result,
        &ptr,
        vec![
            local(2, Some("bits"), &result),
            local(3, Some("q"), &q_ty),
            local(4, Some("aliases"), &aliases_ty),
            local(5, Some("i"), &result),
        ],
        vec![
            assign_to(place(2, &result), const_use()),
            raw_const_assign(3, &q_ty, place(2, &result)),
            assign_to(
                place(4, &aliases_ty),
                json!({"Repeat": [element, q_ty, usize_repeat_count("1"), null]}),
            ),
            assign_to(place(5, &result), const_use()),
            assign_to(index_place(4, &aliases_ty, 5, &result, &q_ty), stored),
            assign_scalar_cast(2, 1, &ptr, &result),
            assign_to(
                place(0, &result),
                copy_use(deref_place(
                    const_index_place(4, &aliases_ty, 0, &q_ty),
                    &result,
                )),
            ),
        ],
    )
}

#[test]
fn dynamic_index_store_of_a_clean_alias_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &dynamic_store_into_repeat(true, true));
}

#[test]
fn dynamic_index_store_over_a_clean_alias_is_not_lowered() {
    assert_sink_escapes(&u64_ty(), &dynamic_store_into_repeat(false, true));
}

#[test]
fn dynamic_index_store_of_a_constant_repeat_still_frees() {
    assert_sink_frees(&u64_ty(), &dynamic_store_into_repeat(false, false));
}

/// `inner = (q,); outer = (inner,); bits = p as usize; drop(outer)`.
fn nested_alias_in_wrapper_drop(store_address: bool) -> Value {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let q_ty = raw_ptr(&bits, "Const");
    let mut statements = vec![
        assign_to(place(2, &bits), const_use()),
        raw_const_assign(3, &q_ty, place(2, &bits)),
        assign_to(
            place(4, &word),
            tuple_of(vec![json!({"Copy": place(3, &q_ty)})]),
        ),
        assign_to(
            place(5, &word),
            tuple_of(vec![json!({"Copy": place(4, &word)})]),
        ),
    ];
    if store_address {
        statements.push(assign_scalar_cast(2, 1, &ptr, &bits));
    }
    let mut body = sink_with_extra(
        &word,
        &ptr,
        vec![
            local(2, Some("bits"), &bits),
            local(3, Some("q"), &q_ty),
            local(4, Some("inner"), &word),
            local(5, Some("outer"), &word),
        ],
        statements.clone(),
    );
    body["Unstructured"]["body"] = json!([
        {"statements": statements, "terminator": {"span": span, "kind": {"Drop": {
            "place": place(5, &word),
            "fn_ptr": {"kind": {"Fun": 2}, "generics": generics},
            "target": 1,
            "on_unwind": 2
        }}}},
        {"statements": [assign_to(place(0, &word), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    body
}

fn glue_that_publishes_the_nested_field() -> Value {
    let (span, _, _, local) = probe_parts();
    let word = i64_ty();
    let bits = u64_ty();
    let saved = u64_ty();
    let field_ptr = raw_ptr(&bits, "Const");
    let slot = raw_ptr(&word, "Mut");
    let glue_body = json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            local(0, None, &word),
            local(1, Some("slot"), &slot)
        ]},
        "body": [{"statements": [
            assign_to(
                global_place(&saved),
                ptr_cast(
                    deref_place(
                        project_field(
                            project_field(deref_place(place(1, &slot), &word), 0, &word),
                            0,
                            &field_ptr,
                        ),
                        &bits,
                    ),
                    &bits,
                    &saved,
                ),
            )
        ], "terminator": {"span": span, "kind": "Return"}}]
    }});
    probe_fun(2, &["probe", "drop_nested"], vec![slot], &word, glue_body)
}

#[test]
fn drop_of_a_nested_field_referent_is_not_lowered() {
    let body = nested_alias_in_wrapper_drop(true);
    let glue = glue_that_publishes_the_nested_field();
    let err = lower_returned_address_sink(&i64_ty(), &[], None, Some(&body), &[glue])
        .expect_err("drop glue that publishes a nested field referent must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn drop_of_a_clean_nested_field_referent_still_frees() {
    let body = nested_alias_in_wrapper_drop(false);
    let glue = glue_that_publishes_the_nested_field();
    let graph = lower_returned_address_sink(&i64_ty(), &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| {
            panic!("dropping a clean nested field referent must still free the spill: {err}")
        });
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

fn pair_field_return(field: u64) -> (Value, Value) {
    let (helper_span, _, _, helper_local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let bits = u64_ty();
    let helper_body = json!({"Unstructured": {
        "span": helper_span,
        "locals": {"arg_count": 1, "locals": [
            helper_local(0, None, &bits),
            helper_local(1, Some("p"), &ptr),
            helper_local(2, Some("bits"), &bits)
        ]},
        "body": [{"statements": [
            assign_scalar_cast(2, 1, &ptr, &bits),
            assign_to(place(0, &bits), tuple_of(vec![
                json!({"Copy": place(2, &bits)}),
                json!({"Const": null})
            ]))
        ], "terminator": {"span": helper_span, "kind": "Return"}}]
    }});
    pair_field_against(field, helper_body)
}

/// Block 1 returns the pair. Block 2 writes it and jumps back, so the
/// first visit of the return has no slots yet.
fn pair_assigned_in_a_later_block() -> Value {
    let (span, _, _, helper_local) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let bits = u64_ty();
    json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 1, "locals": [
            helper_local(0, None, &bits),
            helper_local(1, Some("p"), &ptr),
            helper_local(2, Some("bits"), &bits)
        ]},
        "body": [
            {"statements": [], "terminator": {"span": span, "kind": {"Goto": {"target": 2}}}},
            {"statements": [], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [
                assign_scalar_cast(2, 1, &ptr, &bits),
                assign_to(place(0, &bits), tuple_of(vec![
                    json!({"Copy": place(2, &bits)}),
                    json!({"Const": null})
                ]))
            ], "terminator": {"span": span, "kind": {"Goto": {"target": 1}}}}
        ]
    }})
}

fn pair_field_against(field: u64, helper_body: Value) -> (Value, Value) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let bits = u64_ty();
    let mut body = sink_unstructured(&bits, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("pair"), &bits));
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(1, &ptr)}], "dest": place(2, &bits)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [assign_to(
            place(0, &bits),
            copy_use(field_place(2, &bits, field, &bits)),
        )], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let helper = probe_fun(2, &["probe", "pair"], vec![ptr], &bits, helper_body);
    (body, helper)
}

#[test]
fn clean_field_of_a_returned_pair_still_frees() {
    let bits = u64_ty();
    let (body, helper) = pair_field_return(1);
    let graph = lower_returned_address_sink(&bits, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("the clean field of a returned pair must still free: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn address_field_of_a_returned_pair_is_not_lowered() {
    let bits = u64_ty();
    let (body, helper) = pair_field_return(0);
    let err = lower_returned_address_sink(&bits, &[], None, Some(&body), &[helper])
        .expect_err("the address field of a returned pair must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn clean_field_after_a_later_pair_still_frees() {
    let bits = u64_ty();
    let (body, helper) = pair_field_against(1, pair_assigned_in_a_later_block());
    let graph = lower_returned_address_sink(&bits, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a clean field after a later pair must still free: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn address_field_after_a_later_pair_is_not_lowered() {
    let bits = u64_ty();
    let (body, helper) = pair_field_against(0, pair_assigned_in_a_later_block());
    let err = lower_returned_address_sink(&bits, &[], None, Some(&body), &[helper])
        .expect_err("the address field after a later pair must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn returned_pointer_check(op: &str) -> (Value, Value) {
    let ptr = raw_ptr(&i64_ty(), "Const");
    returned_pointer_against(
        op,
        sink_unstructured(
            &ptr,
            &ptr,
            vec![assign_to(place(0, &ptr), copy_use(place(1, &ptr)))],
        ),
    )
}

/// Block 1 returns. Block 2, which block 0 jumps to, writes `_0 = p`
/// and jumps back. The first visit of block 1 has no incoming state.
fn identity_assigned_in_a_later_block() -> Value {
    let (span, _, _, _) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let mut body = sink_unstructured(&ptr, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Goto": {"target": 2}}}},
        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
        {"statements": [assign_to(place(0, &ptr), copy_use(place(1, &ptr)))],
            "terminator": {"span": span, "kind": {"Goto": {"target": 1}}}}
    ]);
    body
}

fn returned_pointer_against(op: &str, helper_body: Value) -> (Value, Value) {
    let (span, generics, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    let mut body = sink_unstructured(&word, &ptr, vec![]);
    body["Unstructured"]["locals"]["locals"]
        .as_array_mut()
        .expect("locals")
        .push(local(2, Some("q"), &ptr));
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                "args": [{"Move": place(1, &ptr)}], "dest": place(2, &ptr)},
            "target": 1, "on_unwind": 2
        }}}},
        {"statements": [compare_with_zero(op, 0, &word, 2, &ptr)],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
    ]);
    let helper = probe_fun(
        2,
        &["probe", "identity"],
        vec![ptr.clone()],
        &ptr,
        helper_body,
    );
    (body, helper)
}

#[test]
fn null_check_of_a_returned_pointer_still_frees() {
    let word = i64_ty();
    let (body, helper) = returned_pointer_check("Eq");
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| panic!("a null check of a returned pointer must still free: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn ordering_of_a_returned_pointer_is_not_lowered() {
    let word = i64_ty();
    let (body, helper) = returned_pointer_check("Lt");
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .expect_err("an ordering of a returned pointer must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

#[test]
fn null_check_after_a_later_assignment_still_frees() {
    let word = i64_ty();
    let (body, helper) = returned_pointer_against("Eq", identity_assigned_in_a_later_block());
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .unwrap_or_else(|err| {
            panic!("a null check after a later assignment must still free: {err}")
        });
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}

#[test]
fn ordering_after_a_later_assignment_is_not_lowered() {
    let word = i64_ty();
    let (body, helper) = returned_pointer_against("Lt", identity_assigned_in_a_later_block());
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .expect_err("an ordering after a later assignment must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

/// Block 0 switches to block 1 (`return p`) and block 2. Block 2
/// clears `p` and jumps forward to block 6; blocks 6, 5, and 4 jump
/// back to block 3 (`return null`). Block 5 becomes reachable only
/// after its turn, and the cleared pointer adds no address facts, so
/// `join_reached` has to keep walking until block 3 is in the summary.
fn pointer_cleared_on_a_backward_chain() -> Value {
    let (span, _, _, _) = probe_parts();
    let ptr = raw_ptr(&i64_ty(), "Const");
    let mut body = sink_unstructured(&ptr, &ptr, vec![]);
    body["Unstructured"]["body"] = json!([
        {"statements": [], "terminator": {"span": span, "kind": {
            "Switch": {"discr": {"Const": null}, "targets": {"If": [1, 2]}}
        }}},
        {"statements": [assign_to(place(0, &ptr), copy_use(place(1, &ptr)))],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [assign_to(place(1, &ptr), const_use())],
            "terminator": {"span": span, "kind": {"Goto": {"target": 6}}}},
        {"statements": [assign_to(place(0, &ptr), const_use())],
            "terminator": {"span": span, "kind": "Return"}},
        {"statements": [], "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
        {"statements": [], "terminator": {"span": span, "kind": {"Goto": {"target": 4}}}},
        {"statements": [], "terminator": {"span": span, "kind": {"Goto": {"target": 5}}}}
    ]);
    body
}

#[test]
fn null_check_after_a_backward_null_return_is_not_lowered() {
    let word = i64_ty();
    let (body, helper) = returned_pointer_against("Eq", pointer_cleared_on_a_backward_chain());
    let err = lower_returned_address_sink(&word, &[], None, Some(&body), &[helper])
        .expect_err("a null check after a backward null return must not lower");
    let msg = err.to_string();
    assert!(msg.contains("spill address would escape"), "{msg}");
}

fn two_returned_comparisons(second: &str) -> Value {
    let (_, _, _, local) = probe_parts();
    let word = i64_ty();
    let ptr = raw_ptr(&word, "Const");
    sink_with_extra(
        &word,
        &ptr,
        vec![local(2, Some("eq"), &word), local(3, Some("other"), &word)],
        vec![
            compare_with_zero("Eq", 2, &word, 1, &ptr),
            compare_with_zero(second, 3, &word, 1, &ptr),
            assign_to(
                place(0, &word),
                tuple_of(vec![
                    json!({"Copy": place(2, &word)}),
                    json!({"Copy": place(3, &word)}),
                ]),
            ),
        ],
    )
}

#[test]
fn two_null_checks_still_free() {
    assert_sink_frees(&i64_ty(), &two_returned_comparisons("Ne"));
}

#[test]
fn null_check_beside_an_ordering_is_not_lowered() {
    assert_sink_escapes(&i64_ty(), &two_returned_comparisons("Lt"));
}

#[test]
fn idle_drop_of_a_nested_field_referent_still_frees() {
    let word = i64_ty();
    let body = nested_alias_in_wrapper_drop(true);
    let slot = raw_ptr(&word, "Mut");
    let glue = probe_fun(
        2,
        &["probe", "drop_nested"],
        vec![slot.clone()],
        &word,
        idle_body(&word, &[slot]),
    );
    let graph = lower_returned_address_sink(&word, &[], None, Some(&body), &[glue])
        .unwrap_or_else(|err| panic!("idle nested drop glue must still free the spill: {err}"));
    assert!(
        ops(&graph).any(|op| matches!(op.kind, OpKind::RawFree { .. })),
        "the spill is freed after the call\n{}",
        op_lines(&graph)
    );
}
