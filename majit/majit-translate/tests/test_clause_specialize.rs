//! Concrete generic instantiations get distinct graphs.
//!
//! `RDict::get` for a `String` key must reach `<str as PartialEq>::eq`.
//! The `ObjectKey` instantiation must reach `ObjectKey::eq`. The two
//! graphs do not share a key.

use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::llbc_hints::harvest_hints_from_llbcs,
    front::mir::{
        build_semantic_program_from_llbcs_with_static_addrs_and_function_names, lower_function,
    },
    model::{CallTarget, ConcreteType, FunctionGraph, OpKind, ValueType},
};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

#[test]
fn rdict_string_and_objectkey_instantiations_resolve_distinct_eq() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["celldict", "dictmultiobject", "rordereddict"],
        &[
            "_orig_module_dict_entries_get",
            "_orig_dict_entries_probe_object",
        ],
    )
    .expect("lower the two RDict callers");

    let specialized: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("__spec_"))
        .collect();
    assert!(
        specialized.len() >= 2,
        "expected a specialized graph per instantiation, got {}",
        specialized.len()
    );
    let names: Vec<_> = specialized.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len(),
        "instantiation keys collided: {names:?}"
    );

    let string_eq = specialized.iter().any(|f| graph_calls_str_eq(&f.graph()));
    let object_eq = specialized
        .iter()
        .any(|f| graph_calls_object_key_eq(&f.graph()));
    assert!(
        string_eq,
        "String-key instantiation did not reach <str as PartialEq>::eq"
    );
    assert!(
        object_eq,
        "ObjectKey instantiation did not reach ObjectKey::eq; graphs: {names:?}"
    );
    assert!(
        specialized
            .iter()
            .filter(|f| graph_calls_str_eq(&f.graph()))
            .all(|f| !graph_calls_object_key_eq(&f.graph()))
    );
}

fn graph_calls_str_eq(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph_calls_string_eq(graph, false)
}

/// `fn mk<T: Default>() -> T` at `T = i64` and `T = String` is two graphs.
/// The i64 copy's produced value is `Int`. The String copy's produced
/// value is the ref bank (`Str`, same register class as `Ref`).
#[test]
fn mk_i64_and_string_spec_graphs_have_int_and_ref_returns() {
    let llbc = Llbc::from_slice(mk_fixture_llbc().as_bytes()).expect("parse mk fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_i64", "use_string"],
    )
    .expect("lower mk callers");
    let specs: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("mk__s"))
        .collect();
    assert_eq!(
        specs.len(),
        2,
        "expected two mk instantiations, got {:?}",
        specs.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
    assert_ne!(specs[0].name, specs[1].name);
    let mut kinds = Vec::new();
    for spec in &specs {
        let kind = graph_return_kind(&spec.graph());
        assert!(
            kind == "Int" || kind == "Ref",
            "{} return kind {kind}, ops {:?}",
            spec.name,
            op_kinds(&spec.graph())
        );
        kinds.push(kind);
    }
    kinds.sort();
    assert_eq!(kinds, vec!["Int", "Ref"]);
}

/// `fn eqv<Q: Eq, K: Borrow<Q>>` at `Q = str`, `K = StrKey` (the module
/// dict key) calls `StrKey`'s own `borrow` impl, not the `Borrow::borrow`
/// trait method, and compares with the `str` eq.
#[test]
fn str_key_eqv_spec_calls_the_str_key_borrow_and_str_eq() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["celldict", "dictmultiobject", "rordereddict"],
        &[
            "_orig_module_dict_entries_get",
            "_orig_dict_entries_probe_object",
        ],
    )
    .expect("lower the two RDict callers");
    let eqv: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("equivalent__spec_") && graph_calls_str_eq(&f.graph()))
        .collect();
    assert!(
        !eqv.is_empty(),
        "no str equivalent copy; specs: {:?}",
        program
            .functions
            .iter()
            .filter(|f| f.name.contains("__spec_"))
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    for spec in &eqv {
        assert!(
            graph_calls_path(&spec.graph(), &["celldict", "StrKey", "borrow"]),
            "{} does not call StrKey::borrow: {:?}",
            spec.name,
            op_kinds(&spec.graph())
        );
        assert!(
            !graph_calls_trait_borrow(&spec.graph()),
            "{} still calls the Borrow::borrow trait method",
            spec.name
        );
        assert!(
            graph_calls_string_eq_impl(&spec.graph()),
            "{} does not call the str eq",
            spec.name
        );
    }
}

/// A spec copy at `V = ()` calls `put<V>(slot: &mut V, v: V) -> V`.
/// The specialized `put` graph keeps one non-void argument, the `Ref`
/// slot; the unit value is dropped.
#[test]
fn put_unit_spec_graph_drops_the_value_arg() {
    let llbc = Llbc::from_slice(put_fixture_llbc().as_bytes()).expect("parse put fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_unit"],
    )
    .expect("lower use_unit");
    let puts: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("put__s"))
        .collect();
    assert_eq!(
        puts.len(),
        1,
        "expected one specialized put, got {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    let kinds = non_void_input_kinds(&puts[0].graph());
    assert_eq!(
        kinds,
        vec!["Ref"],
        "{} non-void inputs {kinds:?}, ops {:?}",
        puts[0].name,
        op_kinds(&puts[0].graph())
    );
}

/// A spec copy calls `pyre_object::lltype::malloc_typed`. The call stays
/// on the bare path: `HOST_ENV` already owns that builtin, so no
/// `malloc_typed__s` graph is built.
#[test]
fn spec_copy_keeps_bare_lltype_malloc_typed() {
    let llbc =
        Llbc::from_slice(malloc_typed_fixture_llbc().as_bytes()).expect("parse malloc fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_wrap"],
    )
    .expect("lower use_wrap");
    let wraps: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("wrap__s"))
        .collect();
    assert_eq!(
        wraps.len(),
        1,
        "expected one specialized wrap, got {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        graph_calls_bare_malloc_typed(&wraps[0].graph()),
        "{} dropped the bare malloc_typed call, ops {:?}",
        wraps[0].name,
        op_kinds(&wraps[0].graph())
    );
    assert!(
        program
            .functions
            .iter()
            .all(|f| !f.name.contains("malloc_typed__s")),
        "host builtin was copied: {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .filter(|name| name.contains("malloc"))
            .collect::<Vec<_>>()
    );
}

/// `probe_i64_store` / `probe_i64_lookup` are `unroll_safe` and only reached
/// from a `BuildHasher` spec copy. `look_inside_graph` declines a loopy copy
/// that dropped `_jit_unroll_safe_`, and the call then residualizes at a
/// symbolic fnaddr.
#[test]
fn int_dict_lookup_orig_is_unroll_safe() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let harvested = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        HostStaticAddrs::default(),
        &["dictmultiobject", "rordereddict"],
        &[
            "ll_dict_lookup_orig",
            "ll_dict_lookup_trampoline",
            "w_dict_store_int_strategy",
            "w_dict_delitem_int_strategy",
        ],
    )
    .expect("lower int-strategy store and ll_dict_lookup_orig");
    let lookups: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("ll_dict_lookup_orig"))
        .collect();
    let harvested_keys: Vec<_> = harvested
        .iter()
        .filter(|(path, _)| {
            path.contains("ll_dict_lookup_orig") || path.contains("ll_dict_lookup_trampoline")
        })
        .map(|(path, hints)| format!("{path} -> {hints:?}"))
        .collect();
    let spec_rows: Vec<_> = lookups
        .iter()
        .map(|f| format!("{} hints {:?}", f.name, f.hints))
        .collect();
    assert!(
        !lookups.is_empty(),
        "no ll_dict_lookup_orig graph\nharvested {harvested_keys:?}\nspecs {spec_rows:?}"
    );
}

/// `fn f(s: &mut S) { replace(&mut s.u, ()) }` where `S.u` is `()`.
/// The borrow aliases a Void field read: no `getfield` of `u`.
#[test]
fn unit_field_borrow_is_void_and_emits_no_getfield() {
    let llbc = Llbc::from_slice(unit_field_fixture_llbc().as_bytes()).expect("parse S fixture");
    let graph = lower_function(&llbc, "f").expect("lower f");
    assert!(
        !graph_reads_field(&graph, "u"),
        "zero-sized field u must not be a getfield, ops {:?}",
        op_kinds(&graph)
    );
    let arg = replace_first_arg(&graph);
    assert_eq!(
        FunctionGraph::concretetype_of(arg),
        ConcreteType::Void,
        "&mut s.u must be Void, ops {:?}",
        op_kinds(&graph)
    );
}

fn graph_reads_field(graph: &FunctionGraph, name: &str) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == name))
}

fn replace_first_arg(graph: &FunctionGraph) -> &majit_translate::flowspace::model::Variable {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.as_slice() == ["fixture", "put"] && !args.is_empty() => {
                args[0].as_variable()
            }
            _ => None,
        })
        .expect("call fixture::put")
}

fn non_void_input_kinds(graph: &majit_translate::model::FunctionGraph) -> Vec<&'static str> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Input { ty, .. } if !matches!(ty, ValueType::Void) => match ty {
                ValueType::Ref(_) | ValueType::Str => Some("Ref"),
                ValueType::Int | ValueType::Unsigned => Some("Int"),
                _ => Some("other"),
            },
            _ => None,
        })
        .collect()
}

fn graph_return_kind(graph: &majit_translate::model::FunctionGraph) -> &'static str {
    let mut saw_int = false;
    let mut saw_ref = false;
    let mut saw = false;
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target != graph.returnblock {
                continue;
            }
            for arg in &link.args {
                saw = true;
                let kind = match arg.as_variable() {
                    Some(var) => returned_kind(graph, var),
                    None => "other",
                };
                match kind {
                    "Int" => saw_int = true,
                    "Ref" => saw_ref = true,
                    "Void" => {}
                    _ => return "mixed",
                }
            }
        }
    }
    if !saw {
        for var in &graph.block(graph.returnblock).inputargs {
            match returned_kind(graph, var) {
                "Int" => saw_int = true,
                "Ref" => saw_ref = true,
                "Void" => {}
                _ => return "mixed",
            }
        }
    }
    if saw_int && !saw_ref {
        "Int"
    } else if saw_ref && !saw_int {
        "Ref"
    } else {
        "mixed"
    }
}

/// Kind of a value that reaches the return block: its producer's type,
/// or the variable's concretetype when the producer carries none.
fn returned_kind(
    graph: &majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> &'static str {
    if let Some(ty) = producer_value_type(graph, var) {
        return value_type_kind(ty);
    }
    match FunctionGraph::concretetype_of(var) {
        majit_translate::model::ConcreteType::Signed => "Int",
        majit_translate::model::ConcreteType::GcRef => "Ref",
        majit_translate::model::ConcreteType::Void => "Void",
        _ => "other",
    }
}

fn value_type_kind(ty: &majit_translate::model::ValueType) -> &'static str {
    use majit_translate::model::ValueType;
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => "Int",
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => "Ref",
        ValueType::Void => "Void",
        _ => "other",
    }
}

fn producer_value_type<'a>(
    graph: &'a majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a majit_translate::model::ValueType> {
    graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| {
            if op.result.as_ref() != Some(var) {
                return None;
            }
            match &op.kind {
                OpKind::Input { ty, .. }
                | OpKind::Call { result_ty: ty, .. }
                | OpKind::UnaryOp { result_ty: ty, .. }
                | OpKind::BinOp { result_ty: ty, .. }
                | OpKind::FieldRead { ty, .. } => Some(ty),
                _ => None,
            }
        })
    })
}

fn op_kinds(graph: &majit_translate::model::FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| format!("{:?}", op.kind))
        .collect()
}

fn call_leaf_is_borrow(segments: &[String]) -> bool {
    segments.last().map(String::as_str) == Some("borrow")
}

fn graph_calls_path(graph: &majit_translate::model::FunctionGraph, path: &[&str]) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.iter().map(String::as_str).eq(path.iter().copied()),
            _ => false,
        })
}

/// A `borrow` call still dispatched through the trait: a `Method` call,
/// or a path whose owner is the `Borrow` trait.
fn graph_calls_trait_borrow(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                call_leaf_is_borrow(segments)
                    && segments.len() >= 2
                    && segments[segments.len() - 2] == "Borrow"
            }
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "borrow",
            _ => false,
        })
}

fn graph_calls_string_eq_impl(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph_calls_string_eq(graph, true)
}

/// `eq` on `str` / `String` / `&str`: a call whose target is that impl,
/// or a `BinOp("eq")` whose operands are typed that way.
fn graph_calls_string_eq(graph: &majit_translate::model::FunctionGraph, spec_leaf: bool) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if string_eq_segments(segments, spec_leaf)
                || (eq_leaf(segments, spec_leaf)
                    && args.len() >= 2
                    && args.iter().take(2).all(|arg| {
                        arg.as_variable()
                            .is_some_and(|var| operand_is_string(graph, var))
                    })) =>
            {
                true
            }
            OpKind::BinOp { op, lhs, rhs, .. } if op == "eq" => {
                operand_is_string(graph, lhs) && operand_is_string(graph, rhs)
            }
            _ => false,
        })
}

fn eq_leaf(segments: &[String], spec_leaf: bool) -> bool {
    let Some(leaf) = segments.last().map(String::as_str) else {
        return false;
    };
    leaf == "eq" || (spec_leaf && leaf.starts_with("eq__spec_"))
}

fn string_eq_segments(segments: &[String], spec_leaf: bool) -> bool {
    segments.len() >= 4
        && (segments[segments.len() - 4] == "str" || segments[segments.len() - 4] == "String")
        && segments[segments.len() - 3] == "traits"
        && segments[segments.len() - 2] == "<Impl>"
        && eq_leaf(segments, spec_leaf)
}

fn operand_is_string(
    graph: &majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> bool {
    use majit_translate::model::ValueType;
    match reaching_value_type(graph, var, 0) {
        Some(ValueType::Str) => true,
        Some(ValueType::Ref(Some(root))) => string_root(root),
        _ => false,
    }
}

/// Producer type of `var`, or of the link value that reaches it when `var`
/// is only a block input.
fn reaching_value_type<'a>(
    graph: &'a majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
    depth: usize,
) -> Option<&'a majit_translate::model::ValueType> {
    if depth > 8 {
        return None;
    }
    if let Some(ty) = producer_value_type(graph, var) {
        return Some(ty);
    }
    for block in &graph.blocks {
        let Some(slot) = block.inputargs.iter().position(|arg| arg == var) else {
            continue;
        };
        for pred in &graph.blocks {
            for link in &pred.exits {
                if link.target != block.id {
                    continue;
                }
                let Some(src) = link.args.get(slot).and_then(|arg| arg.as_variable()) else {
                    continue;
                };
                if src == var {
                    continue;
                }
                if let Some(ty) = reaching_value_type(graph, src, depth + 1) {
                    return Some(ty);
                }
            }
        }
    }
    None
}

fn string_root(root: &str) -> bool {
    matches!(
        root.rsplit("::").next().unwrap_or(root),
        "str" | "String" | "&str"
    )
}

fn mk_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let i64_ty = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let string_ty = json!({"Adt": {"id": 0, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let impl_ref = |id: u64, ty: &serde_json::Value| json!({"kind": {"TraitImpl": {"id": id, "generics": {"regions": [], "types": [ty], "const_generics": [], "trait_refs": []}}}});
    let opaque = |id: u64, path: &[&str], out: &serde_json::Value| {
        json!({
            "def_id": id,
            "item_meta": meta(path, false),
            "signature": {"is_unsafe": false, "inputs": [], "output": out},
            "body": null
        })
    };
    let body = |ret: &serde_json::Value, func: serde_json::Value| {
        json!({
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 0, "locals": [{"index": 0, "name": null, "span": span, "ty": ret}]},
                "body": [
                    {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                        "call": {"func": func, "args": [], "dest": {"kind": {"Local": 0}, "ty": ret}},
                        "target": 2,
                        "on_unwind": 1
                    }}}},
                    {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                    {"statements": [], "terminator": {"span": span, "kind": "Return"}}
                ]
            }
        })
    };
    let call_mk = |id: u64, name: &str, ty: &serde_json::Value, impl_id: u64| {
        json!({
            "def_id": id,
            "item_meta": meta(&["fixture", name], true),
            "signature": {"is_unsafe": false, "inputs": [], "output": ty},
            "body": body(ty, json!({"Regular": {
                "kind": {"Fun": 3},
                "generics": {"regions": [], "types": [ty], "const_generics": [], "trait_refs": [impl_ref(impl_id, ty)]}
            }}))
        })
    };
    let mk = json!({
        "def_id": 3,
        "item_meta": meta(&["fixture", "mk"], true),
        "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
        "generics": {
            "regions": [],
            "types": [{"index": 0, "name": "T"}],
            "const_generics": [],
            "trait_clauses": [{"clause_id": 0}],
            "regions_outlive": [],
            "types_outlive": [],
            "trait_type_constraints": []
        },
        "body": body(&tvar, json!({"Regular": {
            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0]},
            "generics": empty_g
        }}))
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                opaque(0, &["core", "default", "Default", "default"], &tvar),
                opaque(1, &["core", "default", "Default", "default"], &i64_ty),
                opaque(2, &["alloc", "string", "String", "default"], &string_ty),
                mk,
                call_mk(4, "use_i64", &i64_ty, 0),
                call_mk(5, "use_string", &string_ty, 1)
            ],
            "global_decls": [],
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta(&["alloc", "string", "String"], false),
                "kind": "Opaque",
                "src": "Normal"
            }],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "default", "Default"], false)
            }],
            "trait_impls": [
                {
                    "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                    "impl_trait": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}},
                    "implied_trait_refs": []
                },
                {
                    "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 2}}],
                    "impl_trait": {"id": 0, "generics": {"regions": [], "types": [string_ty], "const_generics": [], "trait_refs": []}},
                    "implied_trait_refs": []
                }
            ]
        }
    });
    file.to_string()
}

fn put_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": empty_g}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let slot_ty = json!({"Ref": {"region": "Erased", "ty": tvar, "kind": "Mut"}});
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let impl_ref = json!({
        "kind": {"TraitImpl": {"id": 0, "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": []}}}
    });
    let generics = |clauses: bool| {
        json!({
            "regions": [],
            "types": [{"index": 0, "name": "V"}],
            "const_generics": [],
            "trait_clauses": if clauses { json!([{"clause_id": 0}]) } else { json!([]) },
            "regions_outlive": [],
            "types_outlive": [],
            "trait_type_constraints": []
        })
    };
    let sig = json!({
        "is_unsafe": false,
        "inputs": [slot_ty, tvar],
        "output": tvar
    });
    let put_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 2,
                "locals": [local(0, None, &tvar), local(1, Some("slot"), &slot_ty), local(2, Some("v"), &tvar)]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let fill_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 2,
                "locals": [
                    local(0, None, &tvar),
                    local(1, Some("slot"), &slot_ty),
                    local(2, Some("v"), &tvar),
                    local(3, Some("tmp"), &tvar)
                ]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0]},
                            "generics": empty_g
                        }},
                        "args": [],
                        "dest": place(3, &tvar)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": {"regions": [], "types": [tvar], "const_generics": [], "trait_refs": []}
                        }},
                        "args": [{"Copy": place(1, &slot_ty)}, {"Copy": place(2, &tvar)}],
                        "dest": place(0, &tvar)
                    },
                    "target": 3,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let use_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0, None, &unit)]},
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 3},
                            "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": [impl_ref]}
                        }},
                        "args": [],
                        "dest": place(0, &unit)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["core", "default", "Default", "default"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
                    "body": null
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["core", "default", "Default", "default"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                    "body": null
                },
                {
                    "def_id": 2,
                    "item_meta": meta(&["fixture", "put"], true),
                    "signature": sig,
                    "generics": generics(false),
                    "body": put_body
                },
                {
                    "def_id": 3,
                    "item_meta": meta(&["fixture", "fill"], true),
                    "signature": sig,
                    "generics": generics(true),
                    "body": fill_body
                },
                {
                    "def_id": 4,
                    "item_meta": meta(&["fixture", "use_unit"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                    "body": use_body
                }
            ],
            "global_decls": [],
            "type_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "default", "Default"], false)
            }],
            "trait_impls": [{
                "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                "impl_trait": {"id": 0, "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": []}},
                "implied_trait_refs": []
            }]
        }
    });
    file.to_string()
}

fn unit_field_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let i64_ty = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": {"types": []}}});
    let s_ty = json!({"Adt": {"id": 0, "generics": empty_g}});
    let s_ref = json!({"Ref": ["Erased", s_ty, "Mut"]});
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let field_u = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &s_ref), "Deref"]}, "ty": s_ty},
            {"Field": [null, 1]}
        ]},
        "ty": unit
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "S"], true),
                "kind": {"Struct": [
                    {"name": "a", "ty": i64_ty, "attr_info": null},
                    {"name": "u", "ty": unit, "attr_info": null}
                ]}
            }],
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["fixture", "f"], true),
                    "signature": {"is_unsafe": false, "inputs": [s_ref], "output": unit},
                    "body": {"Unstructured": {
                        "span": span,
                        "locals": {"arg_count": 1, "locals": [
                            local(0, None, &unit),
                            local(1, Some("s"), &s_ref),
                            local(2, Some("slot"), &unit),
                            local(3, Some("val"), &unit)
                        ]},
                        "body": [
                            {"statements": [
                                {"span": span, "kind": {"Assign": [
                                    place(2, &unit),
                                    {"Ref": {"place": field_u, "kind": "Mut", "ptr_metadata": "None"}}
                                ]}},
                                {"span": span, "kind": {"Assign": [
                                    place(3, &unit),
                                    {"Aggregate": [{"Adt": ["Tuple", null, null, empty_g]}, []]}
                                ]}}
                            ], "terminator": {"span": span, "kind": {"Call": {
                                "call": {
                                    "func": {"Regular": {
                                        "kind": {"Fun": 1},
                                        "generics": empty_g
                                    }},
                                    "args": [{"Copy": place(2, &unit)}, {"Copy": place(3, &unit)}],
                                    "dest": place(0, &unit)
                                },
                                "target": 1,
                                "on_unwind": 2
                            }}}},
                            {"statements": [], "terminator": {"span": span, "kind": "Return"}},
                            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
                        ]
                    }}
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "put"], true),
                    "signature": {"is_unsafe": false, "inputs": [unit, unit], "output": unit},
                    "body": null
                }
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    file.to_string()
}

fn graph_calls_bare_malloc_typed(graph: &FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("malloc_typed")
                    && segments.iter().any(|segment| segment == "lltype")
            }
            _ => false,
        })
}

fn malloc_typed_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let i64_ty = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let impl_ref = json!({
        "kind": {"TraitImpl": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}}}
    });
    let generics = json!({
        "regions": [],
        "types": [{"index": 0, "name": "T"}],
        "const_generics": [],
        "trait_clauses": [{"clause_id": 0}],
        "regions_outlive": [],
        "types_outlive": [],
        "trait_type_constraints": []
    });
    let ret_body = |ty: &serde_json::Value| {
        json!({
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 0, "locals": [{"index": 0, "name": null, "span": span, "ty": ty}]},
                "body": [
                    {"statements": [], "terminator": {"span": span, "kind": "Return"}}
                ]
            }
        })
    };
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, ty: &serde_json::Value| json!({"index": index, "name": null, "span": span, "ty": ty});
    let wrap_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 0,
                "locals": [local(0, &i64_ty), local(1, &i64_ty)]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0]},
                            "generics": empty_g
                        }},
                        "args": [],
                        "dest": place(1, &i64_ty)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}
                        }},
                        "args": [],
                        "dest": place(0, &i64_ty)
                    },
                    "target": 3,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let use_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0, &i64_ty)]},
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 3},
                            "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": [impl_ref]}
                        }},
                        "args": [],
                        "dest": place(0, &i64_ty)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["core", "marker", "Marker", "mark"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
                    "body": null
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "mark_i64"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "body": null
                },
                {
                    "def_id": 2,
                    "item_meta": meta(&["pyre_object", "lltype", "malloc_typed"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "generics": {
                        "regions": [],
                        "types": [{"index": 0, "name": "T"}],
                        "const_generics": [],
                        "trait_clauses": [],
                        "regions_outlive": [],
                        "types_outlive": [],
                        "trait_type_constraints": []
                    },
                    "body": ret_body(&i64_ty)
                },
                {
                    "def_id": 3,
                    "item_meta": meta(&["fixture", "wrap"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "generics": generics,
                    "body": wrap_body
                },
                {
                    "def_id": 4,
                    "item_meta": meta(&["fixture", "use_wrap"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "body": use_body
                }
            ],
            "global_decls": [],
            "type_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "marker", "Marker"], false)
            }],
            "trait_impls": [{
                "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                "impl_trait": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}},
                "implied_trait_refs": []
            }]
        }
    });
    file.to_string()
}

fn path_is_object_key_eq(segments: &[String]) -> bool {
    segments.iter().any(|segment| segment == "ObjectKey")
        && segments.last().map(String::as_str) == Some("eq")
}

fn graph_calls_object_key_eq(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => path_is_object_key_eq(segments),
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
                resolved_path
                    .as_ref()
                    .is_some_and(|path| path_is_object_key_eq(&path.segments))
                    || (name == "eq"
                        && receiver_root
                            .as_deref()
                            .is_some_and(|root| root.contains("ObjectKey")))
            }
            _ => false,
        })
}

/// `ll_slice_getitem_fast_i` returns the unsigned helper word. A signed
/// i64 item retypes through `rarithmetic.intmask` before it is passed to
/// `int_or_float_encode_int` (`IntegerRepr.convert_from_to`).
#[test]
fn integer_to_int_or_float_passes_intmask_of_slice_getitem_to_encode_int() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let graph =
        lower_function(&llbc, "integer_to_int_or_float").expect("lower integer_to_int_or_float");
    let mut encode_arg = None;
    let mut by_result = std::collections::HashMap::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        if let Some(result) = op.result.as_ref() {
            by_result.insert(result.id(), op);
        }
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op.kind
            && segments.last().map(String::as_str) == Some("int_or_float_encode_int")
        {
            encode_arg = args[0].as_variable().cloned();
        }
    }
    let arg = encode_arg.expect("calls encode_int");
    let producer = *by_result
        .get(&arg.id())
        .expect("encode_int argument has a producer");
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = &producer.kind
    else {
        panic!("encode_int argument producer is not a call");
    };
    assert_eq!(
        segments.last().map(String::as_str),
        Some("intmask"),
        "signed i64 slice item must retype through intmask"
    );
    let getitem = args[0]
        .as_variable()
        .expect("intmask argument is a variable");
    let getitem_op = *by_result
        .get(&getitem.id())
        .expect("intmask argument has a producer");
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = &getitem_op.kind
    else {
        panic!("intmask argument producer is not a call");
    };
    assert_eq!(
        segments.last().map(String::as_str),
        Some("ll_slice_getitem_fast_i")
    );
}
