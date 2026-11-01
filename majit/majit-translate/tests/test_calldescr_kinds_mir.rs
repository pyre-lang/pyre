//! Front-end register kinds that `CallControl::getcalldescr` and
//! `func_result_kind` compare.
//!
//! `history.py` `getkind`: a primitive and a raw pointer are `int`; a GC
//! pointer is `ref`. These graphs used to lie about that — a `*const i64`
//! deref aliased the pointer (`ref`) out of an `i64` function, and `&i64` /
//! `*mut i64` parameters were `Ref(None)` while their callers passed the
//! scalar. An unspecialized `GcType::SIZE` is still a `Clause`, so it stays
//! the `__trait_const` ref sentinel. That template is `dont_look_inside`.

use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::mir::{
        build_semantic_program_from_llbcs_with_static_addrs_and_module_paths, lower_function,
    },
    model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType},
};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
);

fn load_object() -> Llbc {
    Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted")
}

fn int_family(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Int | ValueType::Unsigned | ValueType::Bool)
}

fn input_types(graph: &FunctionGraph) -> Vec<ValueType> {
    let start = graph.block(graph.startblock);
    start
        .inputargs
        .iter()
        .map(|arg| {
            start
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Input { ty, .. } if op.result.as_ref() == Some(arg) => Some(ty.clone()),
                    _ => None,
                })
                .unwrap_or(ValueType::Unknown)
        })
        .collect()
}

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

/// Follow block-argument links back to the operation that produced `var`.
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
                        && let Some(src) = exit.args.get(slot).and_then(|arg| arg.as_variable())
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

#[test]
fn sizehint_deref_loads_an_int_word() {
    let llbc = load_object();
    let graph = lower_function(&llbc, "listobject::sizehint_state_value")
        .expect("lower sizehint_state_value");
    let ret = graph
        .blocks
        .iter()
        .flat_map(|block| &block.exits)
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .and_then(majit_translate::model::LinkArg::as_variable)
        .cloned()
        .expect("sizehint_state_value returns the loaded word");
    match root_op(&graph, &ret) {
        Some(OpKind::ArrayRead {
            item_ty: ValueType::Int,
            array_type_id: Some(array_type_id),
            ..
        }) if array_type_id == "[i64]" => {}
        other => {
            let kinds: Vec<String> = ops(&graph)
                .map(|op| {
                    let rendered = format!("{:?}", op.kind);
                    rendered.chars().take(90).collect()
                })
                .collect();
            panic!("*const i64 deref must be a GcArray(Signed) read, got {other:?}; ops {kinds:?}")
        }
    }

    let store = lower_function(&llbc, "listobject::set_sizehint_state_value")
        .expect("lower set_sizehint_state_value");
    assert!(
        ops(&store).any(|op| matches!(
            &op.kind,
            OpKind::ArrayWrite {
                item_ty: ValueType::Int,
                array_type_id: Some(array_type_id),
                ..
            } if array_type_id == "[i64]"
        )),
        "set_sizehint_state_value must setarrayitem the Signed cell"
    );
    assert!(
        ops(&store).all(|op| !matches!(op.kind, OpKind::RawStore { .. })),
        "set_sizehint_state_value must not raw_store a GcArray cell"
    );
}

#[test]
fn unresolved_gc_type_size_stays_a_ref_sentinel() {
    let llbc = load_object();
    let graph =
        lower_function(&llbc, "lltype::malloc_typed_stable").expect("lower malloc_typed_stable");
    let call = ops(&graph)
        .find_map(|op| match &op.kind {
            OpKind::Call { target, args, .. }
                if call_leaf(target) == Some("try_gc_alloc_stable_raw") =>
            {
                Some(args.clone())
            }
            _ => None,
        })
        .expect("malloc_typed_stable calls try_gc_alloc_stable_raw");
    assert_eq!(call.len(), 2, "try_gc_alloc_stable_raw(type_id, T::SIZE)");
    let size = call[1].as_variable().expect("T::SIZE is a value").clone();
    match defining_op(&graph, &size) {
        Some(OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty: ValueType::Ref(_),
            ..
        }) if segments.len() == 2
            && segments[0] == "__str_const"
            && segments[1] == "__trait_const" => {}
        other => panic!("unresolved T::SIZE stays the ref sentinel, got {other:?}"),
    }
}

#[test]
fn scalar_pointer_params_match_int_callers() {
    let llbc = load_object();
    let callback = lower_function(
        &llbc,
        "dictmultiobject::w_dict_object_setitem_callback_free",
    )
    .expect("lower setitem callback");
    let tys = input_types(&callback);
    assert_eq!(tys.len(), 6, "callback params: {tys:?}");
    assert!(
        matches!(tys[0], ValueType::Ref(_)),
        "obj: *mut PyObject stays ref, got {:?}",
        tys[0]
    );
    assert!(
        matches!(tys[1], ValueType::Ref(_)) && matches!(tys[4], ValueType::Ref(_)),
        "key and value stay ref, got {tys:?}"
    );
    assert!(
        int_family(&tys[2]) && int_family(&tys[3]),
        "hash words stay int, got {tys:?}"
    );
    assert!(
        int_family(&tys[5]),
        "hash_out: *mut i64 is int, got {:?}",
        tys[5]
    );

    let barrier =
        lower_function(&llbc, "gc_hook::try_gc_write_barrier").expect("lower try_gc_write_barrier");
    let barrier_tys = input_types(&barrier);
    assert!(
        matches!(barrier_tys.first(), Some(ValueType::Ref(_))),
        "*mut u8 erased GC pointer stays ref, got {barrier_tys:?}"
    );

    let caller = lower_function(
        &llbc,
        "dictmultiobject::w_dict_store_object_strategy_checked_inner",
    )
    .expect("lower callback caller");
    let args = ops(&caller)
        .find_map(|op| match &op.kind {
            OpKind::Call { target, args, .. }
                if call_leaf(target) == Some("w_dict_object_setitem_callback_free") =>
            {
                Some(args.clone())
            }
            _ => None,
        })
        .expect("caller invokes the callback");
    let hash_out = args[5].as_variable().expect("hash_out value").clone();
    let hash_word = args[2].as_variable().expect("hash value").clone();
    assert_ne!(
        hash_out, hash_word,
        "hash_out is the address of keyhash, not the hash word"
    );
    let produced = root_op(&caller, &hash_out);
    // rustc lowers `&mut keyhash` into `*mut i64` as
    // `_r = &mut keyhash; _p = &raw mut (*_r)`. `_p` aliases the word, so
    // the call must spill it (`RawMalloc`) rather than pass `hash`.
    match produced {
        Some(OpKind::RawMalloc { .. }) => {}
        other => panic!("hash_out must be the spill address of &mut keyhash, producer {other:?}"),
    }

    let program = build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
        std::slice::from_ref(&llbc),
        HostStaticAddrs::default(),
        &["dictmultiobject", "rordereddict"],
    )
    .expect("lower dict modules");
    let hash_of: Vec<_> = program
        .functions
        .iter()
        .filter(|function| function.name.starts_with("hash_of"))
        .collect();
    assert!(
        hash_of.iter().any(|function| {
            let tys = input_types(function.graph());
            tys.len() >= 2 && int_family(&tys[1]) && function.name.contains("i64")
        }),
        "RDict<i64, IntKeyHasher>::hash_of key must be int, got {}",
        hash_of
            .iter()
            .map(|function| format!("{} {:?}", function.name, input_types(function.graph())))
            .collect::<Vec<_>>()
            .join("; ")
    );
    let lookup: Vec<_> = program
        .functions
        .iter()
        .filter(|function| function.name.starts_with("lookup_for_store"))
        .collect();
    assert!(
        lookup.iter().any(|function| {
            let tys = input_types(function.graph());
            tys.len() >= 3 && int_family(&tys[2]) && function.name.contains("i64")
        }),
        "RDict<i64>::lookup_for_store key must be int, got {}",
        lookup
            .iter()
            .map(|function| format!("{} {:?}", function.name, input_types(function.graph())))
            .collect::<Vec<_>>()
            .join("; ")
    );
}

fn call_segments(target: &CallTarget) -> Option<&[String]> {
    match target {
        CallTarget::FunctionPath { segments, .. } => Some(segments.as_slice()),
        _ => None,
    }
}

fn residual_named(graph: &FunctionGraph, owner: &str, leaf: &str) -> bool {
    ops(graph).any(|op| match &op.kind {
        OpKind::Call { target, .. } => call_segments(target).is_some_and(|segments| {
            segments.last().map(String::as_str) == Some(leaf)
                && segments
                    .get(segments.len().wrapping_sub(2))
                    .map(String::as_str)
                    == Some(owner)
        }),
        _ => false,
    })
}

/// `Layout::new::<T>()` in the grow `unwrap_or_else` abort closure used
/// to residualise as `["alloc", "layout", "Layout", "new"]`.
#[test]
fn grow_int_items_block_call_once_has_no_layout_new_residual() {
    let llbc = load_object();
    let graph = lower_function(&llbc, "object_array::grow_int_items_block")
        .expect("lower grow_int_items_block");
    assert!(
        !residual_named(&graph, "Layout", "new"),
        "Layout::new must fold to the virtualized {{size, align}} aggregate"
    );
}

/// `ll_list_int_length` is `l.int_items.len()` — `rlist.py` LIST
/// `("length", Signed)`, `jtransform.py` `_handle_list_call` `list.int_len`
/// → `getfield_gc_i(l, int_items.len)`. A residual `__len` was
/// `getattr(__len__)` on the nested `IntArray`.
#[test]
fn ll_list_int_length_reads_int_array_len_field() {
    let llbc = load_object();
    let graph =
        lower_function(&llbc, "listobject::ll_list_int_length").expect("lower ll_list_int_length");
    assert!(
        ops(&graph).any(|op| matches!(
            &op.kind,
            OpKind::FieldRead {
                field,
                ty: ValueType::Int,
                ..
            } if field.name == "len"
                && field.owner_root.as_deref() == Some("IntArray")
        )),
        "ll_list_int_length must getfield IntArray.len, ops {:?}",
        ops(&graph)
            .map(|op| {
                let rendered = format!("{:?}", op.kind);
                rendered.chars().take(90).collect::<String>()
            })
            .collect::<Vec<_>>()
    );
    assert!(
        ops(&graph).all(|op| !matches!(
            &op.kind,
            OpKind::Call { target, .. } if call_leaf(target) == Some("__len")
        )),
        "IntArray::len must not residualise as __len"
    );
}

/// `_orig_extend_from_tuple` is a cyclic `for` + `if let Some`. Pass 2
/// None-kills the header seed; `finish` must not reintroduce a Link.arg
/// that is not an inputarg or op result of its source block
/// (`flowspace_adapter` `undefined operand`).
#[test]
fn orig_extend_from_tuple_has_no_undefined_link_args() {
    const INTERPRETER_LLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("pyre-interpreter.ullbc is already extracted");
    let graph =
        lower_function(&llbc, "_orig_extend_from_tuple").expect("lower _orig_extend_from_tuple");
    let mut holes = Vec::new();
    for block in &graph.blocks {
        if block.dead {
            continue;
        }
        let mut defined: std::collections::HashSet<u64> =
            block.inputargs.iter().map(|v| v.id()).collect();
        for op in &block.operations {
            if let Some(result) = &op.result {
                defined.insert(result.id());
            }
        }
        for (ei, exit) in block.exits.iter().enumerate() {
            for (ai, arg) in exit.args.iter().enumerate() {
                let LinkArg::Value(var) = arg else {
                    continue;
                };
                if defined.contains(&var.id()) {
                    continue;
                }
                holes.push((block.id, ei, ai, var.id(), exit.target));
                eprintln!(
                    "HOLE BlockId({}) exit[{ei}] args[{ai}] var={} -> BlockId({})",
                    block.id.0,
                    var.id(),
                    exit.target.0
                );
                eprintln!(
                    "  src inputargs={:?}",
                    block.inputargs.iter().map(|v| v.id()).collect::<Vec<_>>()
                );
                for (i, op) in block.operations.iter().enumerate() {
                    eprintln!(
                        "    op[{i}] result={:?} kind={:?}",
                        op.result.as_ref().map(|v| v.id()),
                        op.kind
                    );
                }
                for (i, e) in block.exits.iter().enumerate() {
                    eprintln!("    exit[{i}] -> BlockId({}) args={:?}", e.target.0, e.args);
                }
            }
        }
    }
    assert!(
        holes.is_empty(),
        "_orig_extend_from_tuple has undefined Link.args: {holes:?}"
    );
}
