//! A marked enum value is the minted exception class at every use.
//! An unmarked enum of the same shape keeps the tagged `__discriminant`.

use std::panic::AssertUnwindSafe;

use crate::flowspace::model::Variable;
use crate::model::{BlockId, CallTarget, ExitCase, FunctionGraph, LinkArg, OpKind, ValueType};
use majit_charon_reader::ullbc::TyRef;
use serde_json::{Value, json};

fn span() -> Value {
    super::window_span()
}

fn usize_ty() -> Value {
    super::window_usize()
}

fn bool_ty() -> Value {
    json!({"Scalar": "Bool"})
}

fn isize_disc_ty() -> Value {
    json!({"Scalar": {"Integer": {"Signed": "Isize"}}})
}

fn adt(id: u64) -> Value {
    json!({
        "Adt": {"id": id, "generics": super::window_empty_generics()}
    })
}

fn shared_ref(inner: &Value) -> Value {
    json!({"Ref": ["Erased", inner, "Shared"]})
}

fn named_field(name: &str, ty: &Value) -> Value {
    json!({"name": name, "ty": ty, "attr_info": null})
}

fn tuple_field(ty: &Value) -> Value {
    json!({"name": "_0", "is_positional": true, "ty": ty, "attr_info": null})
}

fn variant(name: &str, fields: Vec<Value>, disc: i64) -> Value {
    json!({
        "name": name,
        "fields": fields,
        "discriminant": {"Scalar": {"Signed": ["Isize", disc.to_string()]}}
    })
}

fn enum_decl(id: u64, path: &[&str], variants: Vec<Value>) -> Value {
    json!({
        "def_id": id,
        "item_meta": super::window_meta(path, true),
        "kind": {"Enum": variants}
    })
}

fn marker(id: u64, path: &[&str], value: &str) -> Value {
    json!({
        "def_id": id,
        "item_meta": super::window_meta(path, true),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": value}, {"Deduplicated": 1}]]}
    })
}

fn named_local(index: u64, name: Option<&str>, ty: &Value) -> Value {
    json!({"index": index, "name": name, "span": span(), "ty": ty})
}

fn local_place(index: u64, ty: &Value) -> Value {
    super::window_place(json!({"Local": index}), ty)
}

fn copy_of(place: Value) -> Value {
    json!({"Read": place})
}

fn assign(dest: Value, rvalue: Value) -> Value {
    json!({"span": span(), "kind": {"Assign": [dest, rvalue]}})
}

fn regular(stmts: Vec<Value>, term: Value) -> Value {
    json!({
        "statements": stmts,
        "kind": "Regular",
        "terminator": {"span": span(), "kind": term}
    })
}

fn ret_block(stmts: Vec<Value>) -> Value {
    regular(stmts, json!("Return"))
}

fn aggregate(type_id: u64, variant_idx: u64, operands: Vec<Value>) -> Value {
    json!({"Aggregate": [{"Adt": [type_id, variant_idx, null]}, operands]})
}

fn const_usize_use(text: &str) -> Value {
    json!({"Use": [super::interior_const_usize(text, &usize_ty()), "No"]})
}

fn downcast_field(
    base: Value,
    base_ty: &Value,
    variant_idx: u64,
    field_idx: u64,
    field_ty: &Value,
) -> Value {
    let down = super::window_place(
        json!({"Projection": [base, {"Downcast": variant_idx}]}),
        base_ty,
    );
    super::window_place(
        json!({"Projection": [down, {"Field": [variant_idx, field_idx]}]}),
        field_ty,
    )
}

fn switch_int(discr: Value, arms: &[(i64, u64)], default_bb: u64) -> Value {
    let pairs: Vec<Value> = arms
        .iter()
        .map(|(disc, bb)| json!([{"Scalar": {"Signed": ["Isize", disc.to_string()]}}, bb]))
        .collect();
    json!({"Switch": {
        "discr": {"Read": discr},
        "targets": {"SwitchInt": [null, pairs, default_bb]}
    }})
}

fn unstructured(arg_count: u64, locals: Vec<Value>, body: Vec<Value>) -> Value {
    json!({"Unstructured": {
        "span": span(),
        "locals": {"arg_count": arg_count, "locals": locals},
        "body": body
    }})
}

fn fun(id: u64, path: &[&str], inputs: Vec<Value>, output: Value, body: Value) -> Value {
    json!({
        "def_id": id,
        "item_meta": super::window_meta(path, true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": body
    })
}

fn payload_enum(id: u64, leaf: &str) -> Value {
    let u = usize_ty();
    let b = bool_ty();
    enum_decl(
        id,
        &["fixture", "synth", leaf],
        vec![
            variant("A", vec![named_field("n", &u)], 0),
            variant("B", vec![named_field("n", &u), named_field("flag", &b)], 1),
            variant("C", vec![named_field("n", &u), named_field("flag", &b)], 2),
            variant("D", vec![], 3),
        ],
    )
}

fn build_aggregate(
    id: u64,
    name: &str,
    type_id: u64,
    variant_idx: u64,
    ret_ty: &Value,
    arg_tys: &[Value],
    operand_places: Vec<Value>,
) -> Value {
    let mut locals = vec![named_local(0, None, ret_ty)];
    let mut inputs = Vec::new();
    for (index, ty) in arg_tys.iter().enumerate() {
        inputs.push(ty.clone());
        locals.push(named_local((index + 1) as u64, None, ty));
    }
    let body = unstructured(
        arg_tys.len() as u64,
        locals,
        vec![ret_block(vec![assign(
            local_place(0, ret_ty),
            aggregate(type_id, variant_idx, operand_places),
        )])],
    );
    fun(
        id,
        &["fixture", "synth", name],
        inputs,
        ret_ty.clone(),
        body,
    )
}

fn const_match_body(
    scrutinee: Value,
    scrutinee_ty: &Value,
    arms: &[(i64, u64)],
    arm_consts: &[(u64, &str)],
    default_bb: u64,
    default_const: Option<&str>,
) -> Value {
    let disc_ty = isize_disc_ty();
    let u = usize_ty();
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &u);
    let last = arm_consts
        .iter()
        .map(|(bb, _)| *bb)
        .chain(std::iter::once(default_bb))
        .max()
        .unwrap_or(default_bb) as usize;
    let mut blocks = vec![None; last + 1];
    blocks[0] = Some(regular(
        vec![assign(disc.clone(), json!({"Discriminant": scrutinee}))],
        switch_int(disc, arms, default_bb),
    ));
    for (bb, text) in arm_consts {
        blocks[*bb as usize] = Some(ret_block(vec![assign(ret.clone(), const_usize_use(text))]));
    }
    blocks[default_bb as usize] = Some(match default_const {
        Some(text) => ret_block(vec![assign(ret.clone(), const_usize_use(text))]),
        None => regular(vec![], json!({"Abort": "UndefinedBehavior"})),
    });
    unstructured(
        1,
        vec![
            named_local(0, None, &u),
            named_local(1, Some("e"), scrutinee_ty),
            named_local(2, None, &disc_ty),
        ],
        blocks
            .into_iter()
            .map(|block| block.expect("const match block"))
            .collect(),
    )
}

/// Full match, same shape as [`match_body`].
///
/// Disc 0 and 1 return constants. Disc 2 copies `field_place`. The
/// panic-abort default is outside the lowered closure, so every variant
/// has an arm and that default stays unreachable.
fn read_variant_field_body(
    scrutinee: Value,
    scrutinee_ty: &Value,
    field_place: Value,
    field_ty: &Value,
) -> Value {
    let disc_ty = isize_disc_ty();
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, field_ty);
    unstructured(
        1,
        vec![
            named_local(0, None, field_ty),
            named_local(1, Some("e"), scrutinee_ty),
            named_local(2, None, &disc_ty),
        ],
        vec![
            regular(
                vec![assign(disc.clone(), json!({"Discriminant": scrutinee}))],
                switch_int(disc, &[(1, 1), (2, 2), (0, 3)], 4),
            ),
            ret_block(vec![assign(ret.clone(), const_usize_use("1"))]),
            ret_block(vec![assign(
                ret.clone(),
                json!({"Use": [{"Read": field_place}, "No"]}),
            )]),
            ret_block(vec![assign(ret, const_usize_use("0"))]),
            regular(vec![], json!({"Abort": "UndefinedBehavior"})),
        ],
    )
}

fn match_body(scrutinee: Value, scrutinee_ty: &Value, c_arm_place: Value) -> Value {
    let disc_ty = isize_disc_ty();
    let u = usize_ty();
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &u);
    unstructured(
        1,
        vec![
            named_local(0, None, &u),
            named_local(1, Some("e"), scrutinee_ty),
            named_local(2, None, &disc_ty),
        ],
        vec![
            regular(
                vec![assign(disc.clone(), json!({"Discriminant": scrutinee}))],
                switch_int(disc, &[(1, 1), (2, 2), (0, 3), (3, 4)], 5),
            ),
            ret_block(vec![assign(ret.clone(), const_usize_use("1"))]),
            ret_block(vec![assign(
                ret.clone(),
                json!({"Use": [{"Read": c_arm_place}, "No"]}),
            )]),
            ret_block(vec![assign(ret.clone(), const_usize_use("3"))]),
            ret_block(vec![assign(ret, const_usize_use("4"))]),
            regular(vec![], json!({"Abort": "UndefinedBehavior"})),
        ],
    )
}

fn classify_fun(e_adt: &Value, e_ref: &Value) -> Value {
    let u = usize_ty();
    let disc_ty = isize_disc_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, e_ref), "Deref"]}),
        e_adt,
    );
    let field = downcast_field(deref.clone(), e_adt, 2, 0, &u);
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &u);
    json!({
        "def_id": 7,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["synth", 0]},
                {"Ident": ["E", 0]},
                {"Impl": {"Ty": {
                    "skip_binder": {"Value": [0, e_adt]},
                    "kind": "InherentImplBlock"
                }}},
                {"Ident": ["classify", 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {"is_unsafe": false, "inputs": [e_ref], "output": u},
        "body": unstructured(
            1,
            vec![
                named_local(0, None, &u),
                named_local(1, Some("self"), e_ref),
                named_local(2, None, &disc_ty),
            ],
            vec![
                regular(
                    vec![assign(disc.clone(), json!({"Discriminant": deref}))],
                    switch_int(disc, &[(1, 1), (2, 2), (0, 3), (3, 4)], 5),
                ),
                ret_block(vec![assign(ret.clone(), const_usize_use("1"))]),
                ret_block(vec![assign(
                    ret.clone(),
                    json!({"Use": [{"Read": field}, "No"]}),
                )]),
                ret_block(vec![assign(ret.clone(), const_usize_use("3"))]),
                ret_block(vec![assign(ret, const_usize_use("4"))]),
                regular(vec![], json!({"Abort": "UndefinedBehavior"})),
            ],
        )
    })
}

fn call_classify_fun(e_ref: &Value) -> Value {
    let u = usize_ty();
    let (unwind, ret) = super::window_unwind_and_return_blocks();
    let arg = local_place(1, e_ref);
    let dest = local_place(0, &u);
    fun(
        8,
        &["fixture", "synth", "call_classify"],
        vec![e_ref.clone()],
        u.clone(),
        unstructured(
            1,
            vec![named_local(0, None, &u), named_local(1, Some("e"), e_ref)],
            vec![
                regular(
                    vec![],
                    json!({"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 7},
                            "generics": super::window_empty_generics()
                        }},
                        "args": [{"Read": arg}]}, "dest": dest, "target": 2, "on_unwind": 1}}),
                ),
                unwind,
                ret,
            ],
        ),
    )
}

fn copy_through(id: u64, name: &str, arg_ty: &Value) -> Value {
    fun(
        id,
        &["fixture", "synth", name],
        vec![arg_ty.clone()],
        arg_ty.clone(),
        unstructured(
            1,
            vec![
                named_local(0, None, arg_ty),
                named_local(1, Some("e"), arg_ty),
            ],
            vec![ret_block(vec![assign(
                local_place(0, arg_ty),
                json!({"Use": [{"Read": local_place(1, arg_ty)}, "No"]}),
            )])],
        ),
    )
}

fn fixture_llbc() -> majit_charon_reader::Llbc {
    let e = adt(0);
    let p = adt(1);
    let u = adt(2);
    let e_ref = shared_ref(&e);
    let u_size = usize_ty();
    let flag = bool_ty();
    let q_b = variant("B", vec![named_field("n", &u_size)], 0);
    let q_c = variant(
        "C",
        vec![named_field("n", &u_size), named_field("extra", &flag)],
        1,
    );
    let build_a = build_aggregate(
        0,
        "build_a",
        0,
        0,
        &e,
        &[u_size.clone()],
        vec![copy_of(local_place(1, &u_size))],
    );
    let build_b = build_aggregate(
        1,
        "build_b",
        0,
        1,
        &e,
        &[u_size.clone(), flag.clone()],
        vec![
            copy_of(local_place(1, &u_size)),
            copy_of(local_place(2, &flag)),
        ],
    );
    let build_c = build_aggregate(
        2,
        "build_c",
        0,
        2,
        &e,
        &[u_size.clone(), flag.clone()],
        vec![
            copy_of(local_place(1, &u_size)),
            copy_of(local_place(2, &flag)),
        ],
    );
    let build_d = build_aggregate(3, "build_d", 0, 3, &e, &[], vec![]);
    let build_p = build_aggregate(
        4,
        "build_p",
        1,
        0,
        &p,
        &[u_size.clone()],
        vec![copy_of(local_place(1, &u_size))],
    );
    let match_e = fun(
        5,
        &["fixture", "synth", "match_e"],
        vec![e.clone()],
        u_size.clone(),
        match_body(
            local_place(1, &e),
            &e,
            downcast_field(local_place(1, &e), &e, 2, 0, &u_size),
        ),
    );
    let match_u = fun(
        6,
        &["fixture", "synth", "match_u"],
        vec![u.clone()],
        u_size.clone(),
        match_body(local_place(1, &u), &u, local_place(1, &u)),
    );
    let match_b = fun(
        11,
        &["fixture", "synth", "match_b"],
        vec![e.clone()],
        u_size.clone(),
        const_match_body(local_place(1, &e), &e, &[(1, 1)], &[(1, "1")], 2, Some("2")),
    );
    let match_values = fun(
        12,
        &["fixture", "synth", "match_values"],
        vec![e.clone()],
        u_size.clone(),
        const_match_body(
            local_place(1, &e),
            &e,
            &[(1, 1), (2, 2), (0, 3), (3, 4)],
            &[(1, "1"), (2, "2"), (3, "3"), (4, "4")],
            5,
            None,
        ),
    );
    super::window_llbc_with_globals(
        vec![
            payload_enum(0, "E"),
            enum_decl(
                1,
                &["fixture", "synth", "P"],
                vec![variant("T", vec![tuple_field(&u_size)], 0)],
            ),
            payload_enum(2, "U"),
            enum_decl(3, &["fixture", "synth", "Q"], vec![q_b, q_c]),
        ],
        vec![
            build_a,
            build_b,
            build_c,
            build_d,
            build_p,
            match_e,
            match_u,
            classify_fun(&e, &e_ref),
            call_classify_fun(&e_ref),
            copy_through(9, "take_e", &e),
            copy_through(10, "take_ref", &e_ref),
            match_b,
            match_values,
        ],
        vec![
            marker(0, &["fixture", "synth", "_exception_class_E"], "C(B) B(A)"),
            marker(1, &["fixture", "synth", "_exception_class_P"], ""),
            marker(2, &["fixture", "synth", "_exception_class_Q"], "C(B)"),
        ],
    )
}

fn paint(ty: Value, llbc: &majit_charon_reader::Llbc) -> ValueType {
    let parsed: TyRef = serde_json::from_value(ty).expect("ty parses");
    super::super::lltype::tyref_to_value_type(&parsed, llbc)
}

fn program_metadata(
    llbc: &majit_charon_reader::Llbc,
) -> (
    std::collections::HashSet<String>,
    crate::front::semantic::StructFieldRegistry,
    std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
    std::collections::HashMap<String, String>,
    std::collections::HashMap<String, Vec<(String, ValueType)>>,
) {
    let hints = super::super::llbc_exception_class_hints(llbc);
    let raw =
        crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc));
    let gc = super::super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    let (known, _, fields, by_discr, origins, attrs, _, _) = super::super::derive_program_metadata(
        llbc,
        &gc,
        super::super::lltype::no_immutable_fields(),
        &raw,
        hints,
    );
    (known, fields, by_discr, origins, attrs)
}

fn lower(llbc: &majit_charon_reader::Llbc, name: &str) -> FunctionGraph {
    super::super::lower_function(llbc, name)
        .unwrap_or_else(|err| panic!("{name} failed to lower: {err}"))
}

fn row_names(
    fields: &crate::front::semantic::StructFieldRegistry,
    key: &str,
) -> Option<Vec<(String, String, bool)>> {
    fields.fields.get(key).map(|rows| {
        rows.iter()
            .map(|row| (row.name.clone(), row.ty.clone(), row.is_positional))
            .collect()
    })
}

fn assert_rows(
    fields: &crate::front::semantic::StructFieldRegistry,
    key: &str,
    expected: &[(&str, &str, bool)],
) {
    let rows = row_names(fields, key).unwrap_or_else(|| panic!("{key} has no rows"));
    let got: Vec<(&str, &str, bool)> = rows
        .iter()
        .map(|(name, ty, positional)| (name.as_str(), ty.as_str(), *positional))
        .collect();
    assert_eq!(got, expected, "{key}");
}

fn assert_absent(fields: &crate::front::semantic::StructFieldRegistry, key: &str) {
    assert!(
        row_names(fields, key).is_none(),
        "{key} must not register rows, got {:?}",
        row_names(fields, key)
    );
}

struct Ctor {
    name: String,
    owner_path: Vec<String>,
    is_struct: bool,
    variant_tag: Option<i64>,
    result_ty: ValueType,
    argc: usize,
}

fn ctors(graph: &FunctionGraph) -> Vec<Ctor> {
    super::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name,
                        owner_path,
                        is_struct,
                        variant_tag,
                        ..
                    },
                args,
                result_ty,
            } => Some(Ctor {
                name: name.clone(),
                owner_path: owner_path.clone(),
                is_struct: *is_struct,
                variant_tag: *variant_tag,
                result_ty: result_ty.clone(),
                argc: args.len(),
            }),
            _ => None,
        })
        .collect()
}

fn field_writes(graph: &FunctionGraph) -> Vec<(String, Option<String>)> {
    super::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } => {
                Some((field.name.clone(), field.owner_root.clone()))
            }
            _ => None,
        })
        .collect()
}

fn assert_ctor(graph: &FunctionGraph, name: &str, tag: i64, result: &str) {
    let found = ctors(graph);
    assert_eq!(found.len(), 1, "{name} graph={}", graph.dump());
    let ctor = &found[0];
    assert_eq!(ctor.name, name);
    assert_eq!(ctor.owner_path, vec!["synth::E".to_string()]);
    assert!(!ctor.is_struct);
    assert_eq!(ctor.variant_tag, Some(tag));
    assert_eq!(ctor.argc, 0);
    assert_eq!(ctor.result_ty, ValueType::Ref(Some(result.to_string())));
}

fn inputs(graph: &FunctionGraph) -> Vec<(ValueType, Option<String>)> {
    super::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { ty, class_root, .. } => Some((ty.clone(), class_root.clone())),
            _ => None,
        })
        .collect()
}

fn class_name_in(block: &crate::model::Block) -> Option<String> {
    let (carrier, result_ty) = block.operations.iter().find_map(|op| match &op.kind {
        OpKind::IsInstance {
            class_carrier,
            result_ty,
            ..
        } => Some((class_carrier, result_ty.clone())),
        _ => None,
    })?;
    assert_eq!(result_ty, ValueType::Bool);
    block.operations.iter().find_map(|op| {
        if op.result.as_ref() != Some(carrier) {
            return None;
        }
        match &op.kind {
            OpKind::ConstRef(host) => Some(host.qualname().to_string()),
            _ => None,
        }
    })
}

fn isinstance_chain(graph: &FunctionGraph) -> Vec<String> {
    let mut id = graph.startblock;
    let mut names = Vec::new();
    for _ in 0..8 {
        let block = graph.block(id);
        let Some(name) = class_name_in(block) else {
            break;
        };
        let next = block
            .exits
            .iter()
            .find_map(|link| (link.exitcase == Some(ExitCase::Bool(false))).then_some(link.target));
        names.push(name);
        let Some(next) = next else {
            break;
        };
        id = next;
    }
    names
}

fn assert_isinstance_cascade(graph: &FunctionGraph, label: &str) {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::IsInstance { result_ty, .. } = &op.kind {
                assert_eq!(
                    *result_ty,
                    ValueType::Bool,
                    "{label} arm is isinstance; graph={}",
                    graph.dump()
                );
            }
        }
        for link in &block.exits {
            assert!(
                !matches!(link.exitcase, Some(ExitCase::ExceptionClass(_))),
                "{label} switch arm is Bool, not a handler; graph={}",
                graph.dump()
            );
        }
    }
    assert_eq!(
        isinstance_chain(graph),
        vec![
            "synth::E::C".to_string(),
            "synth::E::B".to_string(),
            "synth::E::A".to_string(),
        ],
        "{label} children-first isinstance cascade; graph={}",
        graph.dump()
    );
}

fn cascade_arms(graph: &FunctionGraph) -> (Vec<(String, BlockId)>, Option<BlockId>) {
    let mut id = graph.startblock;
    let mut arms = Vec::new();
    let mut final_false = None;
    for _ in 0..8 {
        let block = graph.block(id);
        let Some(name) = class_name_in(block) else {
            break;
        };
        let true_target = block
            .exits
            .iter()
            .find_map(|link| (link.exitcase == Some(ExitCase::Bool(true))).then_some(link.target));
        let false_target = block
            .exits
            .iter()
            .find_map(|link| (link.exitcase == Some(ExitCase::Bool(false))).then_some(link.target));
        let Some(true_target) = true_target else {
            break;
        };
        arms.push((name, true_target));
        let Some(next) = false_target else {
            break;
        };
        final_false = Some(next);
        id = next;
    }
    (arms, final_false)
}

fn unsigned_reached(graph: &FunctionGraph, mut id: BlockId) -> Option<u64> {
    let mut seen = Vec::new();
    for _ in 0..8 {
        if seen.contains(&id) {
            return None;
        }
        seen.push(id);
        let block = graph.block(id);
        for op in &block.operations {
            if let OpKind::ConstUInt(value) = &op.kind {
                return Some(*value);
            }
        }
        if block.exits.len() == 1 {
            id = block.exits[0].target;
            continue;
        }
        return None;
    }
    None
}

fn true_edge_reaches_field(graph: &FunctionGraph, name: &str, owner: &str) -> bool {
    let start = graph.block(graph.startblock);
    let Some(mut id) = start
        .exits
        .iter()
        .find_map(|link| (link.exitcase == Some(ExitCase::Bool(true))).then_some(link.target))
    else {
        return false;
    };
    let mut seen = std::collections::HashSet::new();
    for _ in 0..4 {
        if !seen.insert(id) {
            return false;
        }
        let block = graph.block(id);
        let found = block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == name && field.owner_root.as_deref() == Some(owner)
            )
        });
        if found {
            return true;
        }
        let Some(next) = block.exits.first().map(|link| link.target) else {
            return false;
        };
        id = next;
    }
    false
}

fn assert_no_const_int(graph: &FunctionGraph) {
    let bad = super::window_graph_ops(graph).into_iter().any(|op| {
        matches!(
            op.kind,
            OpKind::ConstInt(_) | OpKind::ConstUInt(_) | OpKind::ConstBool(_)
        )
    });
    assert!(
        !bad,
        "empty variant is a class ctor; graph={}",
        graph.dump()
    );
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(text) => *text,
        Err(payload) => match payload.downcast::<&str>() {
            Ok(text) => (*text).to_string(),
            Err(_) => panic!("field-type panic payload was not a string"),
        },
    }
}

fn expect_one_word_panic(llbc: &majit_charon_reader::Llbc, class: &str, spelling: &str) {
    let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = program_metadata(llbc);
    }));
    let Err(payload) = caught else {
        panic!("{class}.{spelling} must fail at attr placement");
    };
    let message = panic_text(payload);
    let expected =
        format!("exception class {class}.missing: field type {spelling} is not a one-word attr");
    assert!(
        message.contains(&expected),
        "panic `{message}` must contain `{expected}`"
    );
}

fn bad_enum_llbc(
    enum_leaf: &str,
    field_ty: Value,
    extras: Vec<Value>,
    globals: Vec<Value>,
) -> majit_charon_reader::Llbc {
    bad_enum_with_funs(enum_leaf, field_ty, extras, globals, Vec::new())
}

fn bad_enum_with_funs(
    enum_leaf: &str,
    field_ty: Value,
    extras: Vec<Value>,
    globals: Vec<Value>,
    funs: Vec<Value>,
) -> majit_charon_reader::Llbc {
    let mut types = vec![enum_decl(
        0,
        &["fixture", enum_leaf],
        vec![variant("S", vec![named_field("missing", &field_ty)], 0)],
    )];
    types.extend(extras);
    super::window_llbc_with_globals(
        types,
        funs,
        std::iter::once(marker(
            0,
            &["fixture", &format!("_exception_class_{enum_leaf}")],
            "",
        ))
        .chain(globals)
        .collect(),
    )
}

fn opaque_string_decl(id: u64) -> Value {
    json!({
        "def_id": id,
        "item_meta": super::window_meta(&["alloc", "string", "String"], false),
        "kind": "Opaque"
    })
}

fn opaque_vec_decl(id: u64) -> Value {
    json!({
        "def_id": id,
        "item_meta": super::window_meta(&["alloc", "vec", "Vec"], false),
        "kind": "Opaque"
    })
}

fn vec_adt_of(vec_id: u64, item_id: u64) -> Value {
    json!({
        "Adt": {
            "id": vec_id,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [adt(item_id)],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

fn str_ref_ty() -> Value {
    json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]})
}

fn rust_string_copy_addr() -> i64 {
    crate::call::symbolic_fnaddr_for_target(&CallTarget::function_path([
        "jit_ll_str_from_rust_string",
    ]))
}

fn names_rust_string_copy(graph: &FunctionGraph) -> bool {
    let expected = rust_string_copy_addr();
    super::window_graph_ops(graph)
        .iter()
        .any(|op| matches!(&op.kind, OpKind::ConstFnAddr { value, .. } if *value == expected))
}

fn transform_graph(graph: &FunctionGraph) -> FunctionGraph {
    crate::codewriter::jtransform::Transformer::new(
        &crate::codewriter::jtransform::GraphTransformConfig::default(),
    )
    .transform(graph)
    .graph
}

fn sole_input(graph: &FunctionGraph) -> Variable {
    let found: Vec<Variable> = super::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { .. } => op.result.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "one argument; graph={}", graph.dump());
    found[0].clone()
}

fn missing_field_writes(graph: &FunctionGraph) -> Vec<(ValueType, LinkArg)> {
    super::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite {
                field, value, ty, ..
            } if field.name == "missing" => Some((ty.clone(), value.clone())),
            _ => None,
        })
        .collect()
}

fn assert_coerce_then_setfield(graph: &FunctionGraph, source: &Variable) {
    let expected = rust_string_copy_addr();
    let mut found = false;
    for block in &graph.blocks {
        let ops = &block.operations;
        for window in ops.windows(4) {
            let addr = matches!(
                &window[0].kind,
                OpKind::ConstFnAddr { value, .. } if *value == expected
            );
            let OpKind::CallResidual { args_r, .. } = &window[1].kind else {
                continue;
            };
            let live = matches!(window[2].kind, OpKind::Live { .. });
            let write = match &window[3].kind {
                OpKind::FieldWrite {
                    field, value, ty, ..
                } => {
                    // `rewrite_op_setfield` retypes from the coerced value.
                    // `Ptr(STR)` is GcRef, so the store's `ty` is `Ref`,
                    // the same retarget a plain `Ptr(STR)` store gets.
                    field.name == "missing"
                        && *ty == ValueType::Ref(None)
                        && matches!(value, LinkArg::Value(stored) if window[1].result.as_ref() == Some(stored))
                }
                _ => false,
            };
            if addr && args_r == &vec![source.clone()] && live && write {
                found = true;
            }
        }
    }
    assert!(
        found,
        "rust-heap String is copied into the rstr attr before setfield; graph={}",
        graph.dump()
    );
}

fn assert_by_value_string_attr_is_rstr(llbc: &majit_charon_reader::Llbc, class: &str) {
    let (_, fields, _, _, attrs) = program_metadata(llbc);
    assert_rows(&fields, class, &[("missing", "str", false)]);
    assert_eq!(
        attrs.get(class),
        Some(&vec![("missing".to_string(), ValueType::Str)])
    );
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    assert_eq!(
        bk.project_struct_field_type("str"),
        crate::annotator::model::s_str0()
    );
    let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
    let rtyper = std::rc::Rc::new(crate::translator::rtyper::rtyper::RPythonTyper::new(&ann));
    let repr = crate::translator::rtyper::rmodel::rtyper_makerepr(
        &crate::annotator::model::s_str0(),
        &rtyper,
    )
    .expect("SomeString rtyper_makerepr");
    assert_eq!(repr.class_name(), "StringRepr");
    assert_eq!(
        repr.lowleveltype(),
        &*crate::translator::rtyper::lltypesystem::rstr::STRPTR
    );
}

fn assert_missing_field_not_str(
    fields: &crate::front::semantic::StructFieldRegistry,
    attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    keys: &[&str],
) {
    let mut saw_row = false;
    for key in keys {
        if let Some(rows) = row_names(fields, key) {
            for (name, ty, _) in &rows {
                if name == "missing" {
                    saw_row = true;
                    assert_ne!(ty.as_str(), "str", "{key} row {rows:?}");
                }
            }
        }
        if let Some(rows) = attrs.get(*key) {
            for (name, ty) in rows {
                if name == "missing" {
                    assert_ne!(*ty, ValueType::Str, "{key} attr {rows:?}");
                }
            }
        }
    }
    assert!(saw_row, "missing row absent under {keys:?}");
}

fn call_regular(fun_id: u64, arg: Value, dest: Value, target: u64, unwind: u64) -> Value {
    json!({"Call": {"call": {
        "func": {"Regular": {
            "kind": {"Fun": fun_id},
            "generics": super::window_empty_generics()
        }},
        "args": [{"Read": arg}]}, "dest": dest, "target": target, "on_unwind": unwind}})
}

fn enum_string_field_llbc(leaf: &str, marked: bool, funs: Vec<Value>) -> majit_charon_reader::Llbc {
    let string_ty = adt(1);
    let types = vec![
        enum_decl(
            0,
            &["fixture", leaf],
            vec![variant("S", vec![named_field("missing", &string_ty)], 0)],
        ),
        opaque_string_decl(1),
    ];
    let globals = if marked {
        vec![marker(
            0,
            &["fixture", &format!("_exception_class_{leaf}")],
            "",
        )]
    } else {
        Vec::new()
    };
    super::window_llbc_with_globals(types, funs, globals)
}

fn build_struct_aggregate(
    id: u64,
    name: &str,
    type_id: u64,
    ret_ty: &Value,
    arg_tys: &[Value],
    operand_places: Vec<Value>,
) -> Value {
    let mut locals = vec![named_local(0, None, ret_ty)];
    let mut inputs = Vec::new();
    for (index, ty) in arg_tys.iter().enumerate() {
        inputs.push(ty.clone());
        locals.push(named_local((index + 1) as u64, None, ty));
    }
    let body = unstructured(
        arg_tys.len() as u64,
        locals,
        vec![ret_block(vec![assign(
            local_place(0, ret_ty),
            json!({"Aggregate": [{"Adt": [type_id, null, null]}, operand_places]}),
        )])],
    );
    fun(
        id,
        &["fixture", "synth", name],
        inputs,
        ret_ty.clone(),
        body,
    )
}

#[test]
fn marked_exception_class_enum_is_the_minted_hierarchy() {
    let llbc = fixture_llbc();
    let e = adt(0);
    let u = adt(2);
    assert_eq!(
        paint(e.clone(), &llbc),
        ValueType::Ref(Some("synth::E".into()))
    );
    assert_eq!(
        paint(shared_ref(&e), &llbc),
        ValueType::Ref(Some("synth::E".into()))
    );
    assert_eq!(paint(u, &llbc), ValueType::Ref(Some("synth::U".into())));

    let (known, fields, by_discr, origins, attrs) = program_metadata(&llbc);
    assert!(known.contains("fixture::synth::E"));
    assert!(known.contains("E"));
    assert!(!known.contains("synth::E"));
    assert!(!known.contains("fixture::synth::E::A"));
    assert!(known.contains("fixture::synth::U::A"));
    assert_eq!(origins.get("E").map(String::as_str), Some("synth"));
    assert_eq!(origins.get("U").map(String::as_str), Some("synth"));
    assert_eq!(origins.get("P").map(String::as_str), Some("synth"));
    assert_eq!(origins.get("Q").map(String::as_str), Some("synth"));

    assert_rows(&fields, "synth::E::A", &[("n", "usize", false)]);
    assert_rows(&fields, "synth::E::B", &[("flag", "bool", false)]);
    assert_absent(&fields, "synth::E::C");
    assert_absent(&fields, "synth::E::D");
    assert_absent(&fields, "EA");
    assert_absent(&fields, "EB");
    assert_absent(&fields, "E");
    assert_absent(&fields, "synth::E");
    assert_absent(&fields, "fixture::synth::E");
    assert_rows(&fields, "synth::P::T", &[("__pos_0", "usize", true)]);
    assert_rows(&fields, "synth::Q::B", &[("n", "usize", false)]);
    assert_rows(&fields, "synth::Q::C", &[("extra", "bool", false)]);
    assert_absent(&fields, "PT");
    assert_absent(&fields, "QB");
    assert_absent(&fields, "QC");
    assert_absent(&fields, "Q");

    for key in ["fixture::synth::U", "U", "synth::U"] {
        assert_rows(&fields, key, &[("__discriminant", "i64", false)]);
    }
    for key in ["fixture::synth::U::A", "U::A", "synth::U::A"] {
        assert_rows(
            &fields,
            key,
            &[("__discriminant", "i64", false), ("n", "usize", false)],
        );
    }
    for key in ["fixture::synth::U::D", "U::D", "synth::U::D"] {
        assert_eq!(row_names(&fields, key), Some(vec![]));
    }
    let u_by = by_discr
        .get("fixture::synth::U")
        .expect("U discriminant map");
    assert_eq!(u_by.get(&0).map(String::as_str), Some("A"));
    assert_eq!(u_by.get(&1).map(String::as_str), Some("B"));
    assert_eq!(u_by.get(&2).map(String::as_str), Some("C"));
    assert_eq!(u_by.get(&3).map(String::as_str), Some("D"));
    assert!(by_discr.contains_key("U"));
    assert!(!by_discr.contains_key("synth::U"));
    assert!(!by_discr.contains_key("fixture::synth::E"));
    assert!(!by_discr.contains_key("E"));
    assert!(!by_discr.contains_key("Q"));

    assert_eq!(
        attrs.get("synth::E::A"),
        Some(&vec![("n".to_string(), ValueType::Unsigned)])
    );
    assert_eq!(
        attrs.get("synth::E::B"),
        Some(&vec![("flag".to_string(), ValueType::Bool)])
    );
    assert!(attrs.get("synth::E::C").is_none());
    assert!(attrs.get("synth::E::D").is_none());
    assert_eq!(
        attrs.get("synth::P::T"),
        Some(&vec![("__pos_0".to_string(), ValueType::Unsigned)])
    );
    assert_eq!(
        attrs.get("synth::Q::C"),
        Some(&vec![("extra".to_string(), ValueType::Bool)])
    );
    let unmarked_a = attrs.get("synth::U::A").expect("unmarked variant attrs");
    assert!(unmarked_a.iter().all(|(name, _)| name != "__discriminant"));
    assert_eq!(unmarked_a, &vec![("n".to_string(), ValueType::Unsigned)]);
    assert_eq!(attrs.get("synth::U::D"), Some(&vec![]));

    for owner in [
        "E",
        "synth::E",
        "fixture::synth::E",
        "P",
        "Q",
        "synth::P",
        "synth::Q",
    ] {
        assert!(
            !fields.raw_word_owners.contains(owner),
            "{owner} is a minted class, not a raw word bag"
        );
    }
    for owner in ["synth::U", "U", "fixture::synth::U"] {
        assert!(fields.raw_word_owners.contains(owner), "{owner}");
    }

    let build_a = lower(&llbc, "build_a");
    assert_ctor(&build_a, "A", 0, "synth::E::A");
    assert_eq!(
        field_writes(&build_a),
        vec![("n".to_string(), Some("synth::E::A".to_string()))]
    );
    let build_b = lower(&llbc, "build_b");
    assert_ctor(&build_b, "B", 1, "synth::E::B");
    assert_eq!(
        field_writes(&build_b),
        vec![
            ("n".to_string(), Some("synth::E::A".to_string())),
            ("flag".to_string(), Some("synth::E::B".to_string())),
        ]
    );
    let build_c = lower(&llbc, "build_c");
    assert_ctor(&build_c, "C", 2, "synth::E::C");
    assert_eq!(
        field_writes(&build_c),
        vec![
            ("n".to_string(), Some("synth::E::A".to_string())),
            ("flag".to_string(), Some("synth::E::B".to_string())),
        ]
    );
    let build_d = lower(&llbc, "build_d");
    assert_ctor(&build_d, "D", 3, "synth::E::D");
    assert!(field_writes(&build_d).is_empty(), "{}", build_d.dump());
    assert_no_const_int(&build_d);

    let build_p = lower(&llbc, "build_p");
    let p_ctors = ctors(&build_p);
    assert_eq!(p_ctors.len(), 1, "{}", build_p.dump());
    assert_eq!(p_ctors[0].name, "T");
    assert_eq!(p_ctors[0].owner_path, vec!["synth::P".to_string()]);
    assert_eq!(p_ctors[0].variant_tag, Some(0));
    assert_eq!(
        field_writes(&build_p),
        vec![("__pos_0".to_string(), Some("synth::P::T".to_string()))]
    );

    for name in ["take_e", "take_ref", "classify"] {
        let graph = lower(&llbc, name);
        assert_eq!(
            inputs(&graph),
            vec![(
                ValueType::Ref(Some("synth::E".into())),
                Some("synth::E".into()),
            )],
            "{name} graph={}",
            graph.dump()
        );
    }

    let match_e = lower(&llbc, "match_e");
    assert_isinstance_cascade(&match_e, "match_e");
    assert!(
        true_edge_reaches_field(&match_e, "n", "synth::E::A"),
        "C arm getattr n on synth::E::A; graph={}",
        match_e.dump()
    );
    let classify = lower(&llbc, "classify");
    assert_isinstance_cascade(&classify, "classify");
    assert!(
        true_edge_reaches_field(&classify, "n", "synth::E::A"),
        "classify C arm getattr n on synth::E::A; graph={}",
        classify.dump()
    );

    let match_u = lower(&llbc, "match_u");
    let reads_tag = super::window_graph_ops(&match_u).into_iter().any(
        |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant"),
    );
    assert!(
        reads_tag,
        "unmarked match reads __discriminant; graph={}",
        match_u.dump()
    );
    let isinstance = super::window_graph_ops(&match_u)
        .into_iter()
        .any(|op| matches!(op.kind, OpKind::IsInstance { .. }));
    assert!(
        !isinstance,
        "unmarked match stays a tag switch; graph={}",
        match_u.dump()
    );

    let call = lower(&llbc, "call_classify");
    // `impl_method_owner` keys the method by the ADT name-path leaf.
    // The exception class of the value is the Input's `synth::E`.
    let method = super::window_graph_ops(&call).into_iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, receiver_root, .. },
                ..
            } if name == "classify" && receiver_root.as_deref() == Some("E")
        )
    });
    assert!(
        method,
        "classify stays getattr on struct leaf E; graph={}",
        call.dump()
    );
}

#[test]
fn exception_class_parent_arm_tests_the_child_first() {
    let llbc = fixture_llbc();
    let graph = lower(&llbc, "match_b");
    let (arms, final_false) = cascade_arms(&graph);
    let names: Vec<&str> = arms.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        ["synth::E::C", "synth::E::B"],
        "C ⊂ B is tested before B; graph={}",
        graph.dump()
    );
    let ec = arms
        .iter()
        .find(|(name, _)| name == "synth::E::C")
        .expect("synth::E::C");
    let eb = arms
        .iter()
        .find(|(name, _)| name == "synth::E::B")
        .expect("synth::E::B");
    assert_eq!(
        unsigned_reached(&graph, ec.1),
        Some(2),
        "synth::E::C takes the otherwise arm; graph={}",
        graph.dump()
    );
    assert_eq!(
        unsigned_reached(&graph, eb.1),
        Some(1),
        "synth::E::B takes the B arm; graph={}",
        graph.dump()
    );
    let final_false = final_false.expect("otherwise edge");
    assert_eq!(
        unsigned_reached(&graph, final_false),
        Some(2),
        "the residual value takes the otherwise arm; graph={}",
        graph.dump()
    );
}

#[test]
fn exception_class_listed_child_keeps_its_arm() {
    let llbc = fixture_llbc();
    let graph = lower(&llbc, "match_values");
    let (arms, final_false) = cascade_arms(&graph);
    let names: Vec<&str> = arms.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        ["synth::E::C", "synth::E::B", "synth::E::A"],
        "children-first cascade elides the covered last arm; graph={}",
        graph.dump()
    );
    let ec = arms
        .iter()
        .find(|(name, _)| name == "synth::E::C")
        .expect("synth::E::C");
    let eb = arms
        .iter()
        .find(|(name, _)| name == "synth::E::B")
        .expect("synth::E::B");
    assert_eq!(
        unsigned_reached(&graph, ec.1),
        Some(2),
        "synth::E::C takes the C arm; graph={}",
        graph.dump()
    );
    assert_eq!(
        unsigned_reached(&graph, eb.1),
        Some(1),
        "synth::E::B takes the B arm; graph={}",
        graph.dump()
    );
    let final_false = final_false.expect("D is the false edge");
    assert_eq!(
        unsigned_reached(&graph, final_false),
        Some(4),
        "the covered residual arm is D; graph={}",
        graph.dump()
    );
}

#[test]
fn exception_class_non_one_word_field_panics() {
    let string_adt = adt(1);
    let string_decl = json!({
        "def_id": 1,
        "item_meta": super::window_meta(&["alloc", "string", "String"], false),
        "kind": "Opaque"
    });
    let bad_str = bad_enum_llbc(
        "BadStr",
        string_adt.clone(),
        vec![string_decl.clone()],
        vec![],
    );
    assert_by_value_string_attr_is_rstr(&bad_str, "BadStr::S");

    let vec_adt = json!({
        "Adt": {
            "id": 1,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [adt(2)],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let vec_decl = json!({
        "def_id": 1,
        "item_meta": super::window_meta(&["alloc", "vec", "Vec"], false),
        "kind": "Opaque"
    });
    let string_decl = json!({
        "def_id": 2,
        "item_meta": super::window_meta(&["alloc", "string", "String"], false),
        "kind": "Opaque"
    });
    let bad_vec = bad_enum_llbc(
        "BadVec",
        vec_adt,
        vec![vec_decl.clone(), string_decl],
        vec![],
    );
    let (_, fields, _, _, attrs) = program_metadata(&bad_vec);
    assert_rows(
        &fields,
        "BadVec::S",
        &[("missing", crate::front::mir::RPY_LIST_OF_STR, false)],
    );
    assert_eq!(
        attrs.get("BadVec::S"),
        Some(&vec![(
            "missing".to_string(),
            ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string()))
        )])
    );
    let (flag, kind, size) = crate::codewriter::call::get_type_flag(
        crate::front::mir::RPY_LIST_OF_STR,
        &std::collections::HashSet::new(),
    );
    assert_eq!(flag, majit_ir::descr::ArrayFlag::Pointer);
    assert_eq!(kind, majit_ir::value::Type::Ref);
    assert_eq!(size, crate::layout::target_word_size());

    let signature_for_vec = super::interior_struct(
        2,
        &["fixture", "Signature"],
        &[("word", super::interior_usize())],
        8,
        &[0],
        false,
    );
    let vec_of_sig = vec_adt_of(1, 2);
    let bad_vec_sig = bad_enum_llbc(
        "BadVecSig",
        vec_of_sig,
        vec![vec_decl, signature_for_vec],
        vec![],
    );
    expect_one_word_panic(&bad_vec_sig, "BadVecSig::S", "Vec<Signature>");

    let bad_sig = bad_enum_llbc(
        "BadSig",
        adt(1),
        signature_types(),
        vec![signature_raw_marker()],
    );
    let ok_raw = bad_enum_llbc(
        "OkRawPtr",
        super::interior_raw_mut(adt(1)),
        signature_types(),
        vec![signature_raw_marker()],
    );
    let (_, bad_fields, _, _, bad_attrs) = program_metadata(&bad_sig);
    let (_, raw_fields, _, _, raw_attrs) = program_metadata(&ok_raw);
    let bad_row = missing_field_row(&bad_fields, "BadSig::S");
    let raw_row = missing_field_row(&raw_fields, "OkRawPtr::S");
    assert_eq!(bad_row.ty, "*mut Signature");
    assert_eq!(bad_row.ty, raw_row.ty);
    assert_eq!(bad_attrs.get("BadSig::S"), raw_attrs.get("OkRawPtr::S"));
    assert_eq!(
        bad_attrs.get("BadSig::S"),
        Some(&vec![(
            "missing".to_string(),
            ValueType::Ref(Some("Signature".to_string()))
        )])
    );
    let raw_set = crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(
        std::slice::from_ref(&bad_sig),
    );
    let bad_flag = crate::codewriter::call::get_type_flag(&bad_row.ty, &raw_set);
    let raw_flag = crate::codewriter::call::get_type_flag(&raw_row.ty, &raw_set);
    assert_eq!(bad_flag, raw_flag);
    assert_eq!(bad_flag.2, crate::layout::target_word_size());
    let owned = bad_row
        .raw_owned
        .as_ref()
        .expect("by-value raw-flavor struct is raw-owned");
    assert_eq!(owned.type_path, "fixture::Signature");
    assert!(owned.type_path.ends_with("Signature"));
    assert_eq!(owned.size, 8);
    assert_eq!(owned.align, 8);
    assert!(raw_row.raw_owned.is_none());

    let ok_sig = bad_enum_llbc("OkSig", adt(1), signature_types(), vec![]);
    let (_, fields, _, _, _) = program_metadata(&ok_sig);
    let ok_row = missing_field_row(&fields, "OkSig::S");
    assert!(
        !ok_row.ty.starts_with("*mut"),
        "one-word struct field stays inline, got {}",
        ok_row.ty
    );
    assert!(ok_row.raw_owned.is_none());

    let ok_ptr = bad_enum_llbc(
        "OkPtr",
        super::interior_raw_mut(string_adt),
        vec![json!({
            "def_id": 1,
            "item_meta": super::window_meta(&["alloc", "string", "String"], false),
            "kind": "Opaque"
        })],
        vec![],
    );
    let (_, fields, _, _, attrs) = program_metadata(&ok_ptr);
    let rows = row_names(&fields, "OkPtr::S").expect("OkPtr::S");
    assert!(
        rows.iter().any(|(name, _, _)| name == "missing"),
        "{rows:?}"
    );
    assert_missing_field_not_str(&fields, &attrs, &["OkPtr::S"]);
}

#[test]
fn exception_class_string_attr_coerces_rust_heap_on_write() {
    let string_ty = adt(1);
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[string_ty.clone()],
        vec![copy_of(local_place(1, &string_ty))],
    );
    let llbc = enum_string_field_llbc("BadStr", true, vec![build]);
    let graph = lower(&llbc, "build_s");
    let source = sole_input(&graph);
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert!(
        writes[0].0 == ValueType::Str,
        "front stores the attr as Ptr(STR); got {writes:?} graph={}",
        graph.dump()
    );
    assert!(
        matches!(&writes[0].1, LinkArg::Value(value) if value == &source),
        "the operand is still the rust-heap local; graph={}",
        graph.dump()
    );
    assert!(
        !names_rust_string_copy(&graph),
        "the copy is a jtransform rewrite; graph={}",
        graph.dump()
    );
    let inputs = inputs(&graph);
    assert!(
        inputs.iter().any(|(ty, root)| {
            matches!(ty, ValueType::Ref(Some(name)) if name == "String" || name == "string::String")
                || root
                    .as_deref()
                    .is_some_and(|name| name == "String" || name == "string::String")
        }),
        "the local stays a rust-heap header so the copy has a source; inputs={inputs:?}"
    );
    let transformed = transform_graph(&graph);
    assert_coerce_then_setfield(&transformed, &source);
}

#[test]
fn exception_class_string_attr_read_as_str_is_rstr() {
    let enum_ty = adt(0);
    let enum_ref = shared_ref(&enum_ty);
    let string_ty = adt(1);
    let out = str_ref_ty();
    let disc_ty = isize_disc_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, &enum_ref), "Deref"]}),
        &enum_ty,
    );
    let field = downcast_field(deref.clone(), &enum_ty, 0, 0, &string_ty);
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &out);
    let as_str = super::window_opaque_fun(
        0,
        &["alloc", "string", "String", "as_str"],
        vec![shared_ref(&string_ty)],
        out.clone(),
    );
    let reader = json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["BadStr", 0]},
                {"Impl": {"Ty": {
                    "skip_binder": {"Value": [0, enum_ty]},
                    "kind": "InherentImplBlock"
                }}},
                {"Ident": ["read_missing", 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {"is_unsafe": false, "inputs": [enum_ref.clone()], "output": out.clone()},
        "body": unstructured(
            1,
            vec![
                named_local(0, None, &out),
                named_local(1, Some("self"), &enum_ref),
                named_local(2, None, &disc_ty),
            ],
            vec![
                regular(
                    vec![assign(disc.clone(), json!({"Discriminant": deref}))],
                    switch_int(disc, &[(0, 1)], 2),
                ),
                regular(
                    vec![],
                    call_regular(0, field, ret, 3, 2),
                ),
                regular(vec![], json!({"Abort": "UndefinedBehavior"})),
                ret_block(vec![]),
            ],
        )
    });
    let llbc = enum_string_field_llbc("BadStr", true, vec![as_str, reader]);
    let graph = lower(&llbc, "read_missing");
    let reads: Vec<&crate::model::SpaceOperation> = super::window_graph_ops(&graph)
        .into_iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "missing" && *ty == ValueType::Str
            )
        })
        .collect();
    assert_eq!(reads.len(), 1, "{}", graph.dump());
    let produced = reads[0].result.as_ref().expect("field read result");
    assert_eq!(
        produced.concretetype().as_ref(),
        Some(&*crate::translator::rtyper::lltypesystem::rstr::STRPTR),
        "the getattr word is Ptr(STR); graph={}",
        graph.dump()
    );
    let returned = graph
        .blocks
        .iter()
        .flat_map(|block| &block.exits)
        .find_map(|link| {
            (link.target == graph.returnblock).then(|| match link.args.first() {
                Some(LinkArg::Value(value)) => Some(value.clone()),
                _ => None,
            })
        })
        .expect("return link")
        .expect("return value");
    let (block_id, index) = super::super::resolve_to_producer_op(&graph, &returned)
        .unwrap_or_else(|| panic!("as_str must alias the attr; graph={}", graph.dump()));
    let producer = &graph.block(block_id).operations[index];
    assert!(
        matches!(
            &producer.kind,
            OpKind::FieldRead { field, ty, .. }
                if field.name == "missing" && *ty == ValueType::Str
        ),
        "as_str is the rstr word, not a rust-heap helper; op={:?} graph={}",
        producer.kind,
        graph.dump()
    );
    assert!(
        producer.result.as_ref() == Some(produced),
        "the returned word is the stamped getattr"
    );
    let calls_as_str = super::window_graph_ops(&graph)
        .iter()
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("as_str"),
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "as_str",
            _ => false,
        });
    assert!(
        !calls_as_str,
        "as_str aliases the rstr attr; graph={}",
        graph.dump()
    );
    assert!(!names_rust_string_copy(&graph), "{}", graph.dump());
    let transformed = transform_graph(&graph);
    assert!(
        !names_rust_string_copy(&transformed),
        "jtransform must not copy a Ptr(STR) attr through the rust-heap helper; graph={}",
        transformed.dump()
    );
}

#[test]
fn exception_class_string_attr_rust_heap_use_is_unsupported() {
    let enum_ty = adt(0);
    let enum_ref = shared_ref(&enum_ty);
    let string_ty = adt(1);
    let out = usize_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, &enum_ref), "Deref"]}),
        &enum_ty,
    );
    let field = downcast_field(deref, &enum_ty, 0, 0, &string_ty);
    let eat = super::window_opaque_fun(0, &["fixture", "eat"], vec![string_ty], out.clone());
    let (unwind, ret) = super::window_unwind_and_return_blocks();
    let user = fun(
        1,
        &["fixture", "use_missing"],
        vec![enum_ref.clone()],
        out.clone(),
        unstructured(
            1,
            vec![
                named_local(0, None, &out),
                named_local(1, Some("self"), &enum_ref),
            ],
            vec![
                regular(vec![], call_regular(0, field, local_place(0, &out), 2, 1)),
                unwind,
                ret,
            ],
        ),
    );
    let llbc = enum_string_field_llbc("BadStr", true, vec![eat, user]);
    let err = match super::super::lower_function(&llbc, "use_missing") {
        Err(err) => err,
        Ok(graph) => panic!(
            "a rust-heap String ABI on the attr must fail; graph={}",
            graph.dump()
        ),
    };
    let text = err.to_string();
    assert!(
        text.contains("String attr read as rust-heap String"),
        "{text}"
    );
}

#[test]
fn exception_class_unmarked_string_field_stays_rust_heap() {
    let string_ty = adt(1);
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[string_ty.clone()],
        vec![copy_of(local_place(1, &string_ty))],
    );
    let unmarked = enum_string_field_llbc("Plain", false, vec![build]);
    let (_, fields, _, _, attrs) = program_metadata(&unmarked);
    assert_missing_field_not_str(
        &fields,
        &attrs,
        &["Plain::S", "fixture::Plain::S", "synth::Plain::S"],
    );
    let graph = lower(&unmarked, "build_s");
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert!(
        writes[0].0 != ValueType::Str,
        "an unmarked enum keeps the rust-heap store; got {writes:?} graph={}",
        graph.dump()
    );
    let transformed = transform_graph(&graph);
    assert!(
        !names_rust_string_copy(&transformed),
        "no coerce without the exception-class flag; graph={}",
        transformed.dump()
    );

    let holder_ty = adt(0);
    let holder = super::interior_struct(
        0,
        &["fixture", "Holder"],
        &[("missing", string_ty.clone())],
        24,
        &[0],
        false,
    );
    let build_holder = build_struct_aggregate(
        0,
        "build_holder",
        0,
        &holder_ty,
        &[string_ty.clone()],
        vec![copy_of(local_place(1, &string_ty))],
    );
    let plain = super::window_llbc_with_globals(
        vec![holder, opaque_string_decl(1)],
        vec![build_holder],
        Vec::new(),
    );
    let (_, fields, _, _, attrs) = program_metadata(&plain);
    assert_missing_field_not_str(&fields, &attrs, &["Holder", "fixture::Holder"]);
    let graph = lower(&plain, "build_holder");
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert!(
        writes[0].0 != ValueType::Str,
        "a plain struct keeps the rust-heap store; got {writes:?} graph={}",
        graph.dump()
    );
    let transformed = transform_graph(&graph);
    assert!(
        !names_rust_string_copy(&transformed),
        "a plain String field is not an rstr attr; graph={}",
        transformed.dump()
    );
}

#[test]
fn two_marked_enums_with_the_same_leaf_are_distinct_classes() {
    let n = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = super::window_llbc_with_globals(
        vec![
            enum_decl(
                0,
                &["fixture", "a", "E"],
                vec![variant("A", vec![named_field("n", &n)], 0)],
            ),
            enum_decl(
                1,
                &["fixture", "b", "E"],
                vec![variant("A", vec![named_field("n", &n)], 0)],
            ),
        ],
        Vec::new(),
        vec![
            marker(0, &["fixture", "a", "_exception_class_E"], ""),
            marker(1, &["fixture", "b", "_exception_class_E"], ""),
        ],
    );
    let hints = crate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(
        std::slice::from_ref(&llbc),
    );
    let paths: Vec<_> = hints.iter().map(|hint| hint.enum_path.as_str()).collect();
    assert_eq!(paths, ["a::E", "b::E"]);
    let (_, fields, _, _, attrs) = program_metadata(&llbc);
    assert_rows(&fields, "a::E::A", &[("n", "i64", false)]);
    assert_rows(&fields, "b::E::A", &[("n", "i64", false)]);
    assert_absent(&fields, "a::E");
    assert_absent(&fields, "b::E");
    assert_absent(&fields, "E");
    assert_absent(&fields, "EA");
    assert_eq!(
        attrs.get("a::E::A"),
        Some(&vec![("n".to_string(), ValueType::Int)])
    );
    assert_eq!(
        attrs.get("b::E::A"),
        Some(&vec![("n".to_string(), ValueType::Int)])
    );

    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.set_struct_fields(std::rc::Rc::new(fields));
    bk.mint_exception_class_hierarchies(&hints);
    let a_base = bk.exception_class_base("a::E").expect("a::E");
    let b_base = bk.exception_class_base("b::E").expect("b::E");
    assert!(!std::rc::Rc::ptr_eq(&a_base, &b_base));
    assert_eq!(a_base.borrow().name, "a::E");
    assert_eq!(b_base.borrow().name, "b::E");
    let a_var = bk.exception_class_variant("a::E", "A").expect("a::E::A");
    let b_var = bk.exception_class_variant("b::E", "A").expect("b::E::A");
    assert!(!std::rc::Rc::ptr_eq(&a_var, &b_var));
    assert_eq!(a_var.borrow().name, "a::E::A");
    assert_eq!(b_var.borrow().name, "b::E::A");
    let a_host = a_base.borrow().classdesc.borrow().pyobj.clone();
    let b_host = b_base.borrow().classdesc.borrow().pyobj.clone();
    assert_ne!(a_host, b_host);
    assert_eq!(bk.intern_class_by_qualname("a::E"), a_host);
    assert_eq!(bk.intern_class_by_qualname("b::E"), b_host);
    let exception = crate::flowspace::model::HOST_ENV
        .lookup_exception_class("Exception")
        .expect("Exception");
    assert_eq!(a_host.class_bases(), Some(std::slice::from_ref(&exception)));
    let again = bk.intern_class_by_qualname("a::E");
    assert_eq!(again, a_host);
    assert_eq!(again.class_bases(), Some(std::slice::from_ref(&exception)));
    let a_var_host = a_var.borrow().classdesc.borrow().pyobj.clone();
    let b_var_host = b_var.borrow().classdesc.borrow().pyobj.clone();
    assert_ne!(a_var_host, b_var_host);
    assert_eq!(
        a_var_host.class_bases(),
        Some(std::slice::from_ref(&a_host))
    );
    assert_eq!(
        b_var_host.class_bases(),
        Some(std::slice::from_ref(&b_host))
    );
    assert_eq!(bk.intern_class_by_qualname("a::E::A"), a_var_host);
    assert_eq!(bk.intern_class_by_qualname("b::E::A"), b_var_host);
}

#[test]
fn marked_variant_keys_do_not_collide_across_declarations() {
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.mint_exception_class_hierarchies(&[
        crate::front::llbc_hints::ExceptionClassHint::new("pkg::E", "", vec!["AB".into()]),
        crate::front::llbc_hints::ExceptionClassHint::new("pkg::EA", "", vec!["B".into()]),
    ]);
    let ab = bk
        .exception_class_variant("pkg::E", "AB")
        .expect("pkg::E::AB");
    let b = bk
        .exception_class_variant("pkg::EA", "B")
        .expect("pkg::EA::B");
    assert!(!std::rc::Rc::ptr_eq(&ab, &b));
    assert_eq!(ab.borrow().name, "pkg::E::AB");
    assert_eq!(b.borrow().name, "pkg::EA::B");
    let ab_host = ab.borrow().classdesc.borrow().pyobj.clone();
    let b_host = b.borrow().classdesc.borrow().pyobj.clone();
    assert_ne!(ab_host, b_host);
    assert_eq!(bk.intern_class_by_qualname("pkg::E::AB"), ab_host);
    assert_eq!(bk.intern_class_by_qualname("pkg::EA::B"), b_host);
}

#[test]
#[should_panic(expected = "already interned with bases")]
fn minting_refuses_a_plain_class_under_the_declaration_key() {
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    let _ = bk.intern_class_by_qualname("pkg::E");
    bk.mint_exception_class_hierarchies(&[crate::front::llbc_hints::ExceptionClassHint::new(
        "pkg::E",
        "",
        vec!["A".into()],
    )]);
}

#[test]
#[should_panic(expected = "already interned as a different class object")]
fn interning_a_different_arc_under_a_minted_key_panics() {
    let hint = crate::front::llbc_hints::ExceptionClassHint::new("pkg::E", "", vec!["A".into()]);
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    let exception = crate::flowspace::model::HOST_ENV
        .lookup_exception_class("Exception")
        .expect("Exception");
    let _ = bk.intern_class_by_qualname_with_bases("pkg::E", vec![exception]);
    bk.mint_exception_class_hierarchies(&[hint]);
}

#[test]
fn isinstance_constref_is_the_harvested_class_object() {
    let llbc = fixture_llbc();
    let graph = lower(&llbc, "match_e");
    let hints = super::super::llbc_exception_class_hints(&llbc);
    let hint = hints
        .iter()
        .find(|hint| hint.enum_path == "synth::E")
        .expect("harvested E");
    let mut id = graph.startblock;
    let mut seen = Vec::new();
    for _ in 0..8 {
        let block = graph.block(id);
        let Some(host) = block.operations.iter().find_map(|op| match &op.kind {
            OpKind::ConstRef(host) if host.is_class() => Some(host.clone()),
            _ => None,
        }) else {
            break;
        };
        let expected = hint
            .classes
            .variants
            .iter()
            .find(|(_, candidate)| candidate == &host)
            .map(|(_, candidate)| candidate.clone());
        let Some(expected) = expected else {
            panic!(
                "isinstance host {} is not a variant of the harvested hierarchy",
                host.qualname()
            );
        };
        assert_eq!(host, expected);
        seen.push(host.qualname().to_string());
        let Some(next) = block
            .exits
            .iter()
            .find_map(|link| (link.exitcase == Some(ExitCase::Bool(false))).then_some(link.target))
        else {
            break;
        };
        id = next;
    }
    assert_eq!(
        seen,
        vec![
            "synth::E::C".to_string(),
            "synth::E::B".to_string(),
            "synth::E::A".to_string(),
        ]
    );
}

fn lowering_state(llbc: &majit_charon_reader::Llbc) -> super::super::CrateLoweringState {
    let gc = std::collections::HashSet::<majit_ir::descr::StructId>::new();
    let raw = std::collections::HashSet::<String>::new();
    super::super::CrateLoweringState::new(
        llbc,
        &std::collections::HashSet::new(),
        std::collections::HashMap::new(),
        Default::default(),
        &gc,
        &raw,
        None,
        None,
    )
}

/// Artefact B carries a copied `E` and a function that takes `E`. Seeded
/// with A's marker plus B's own harvest, B paints the declaration key and
/// registers no tagged rows and no variant-field rows.
#[test]
fn copied_marked_enum_uses_the_program_wide_hint() {
    let defining = super::window_llbc_with_globals(
        vec![payload_enum(0, "E")],
        Vec::new(),
        vec![marker(0, &["fixture", "synth", "_exception_class_E"], "")],
    );
    let e_ty = adt(0);
    let mut copied_enum = payload_enum(0, "E");
    copied_enum["item_meta"] = super::window_meta(&["fixture", "synth", "E"], false);
    let take_e = super::window_opaque_fun(
        1,
        &["fixture", "synth", "take_e"],
        vec![e_ty.clone()],
        e_ty.clone(),
    );
    let copied = super::window_llbc_with_globals(vec![copied_enum], vec![take_e], Vec::new());

    let mut hints = Vec::new();
    crate::front::llbc_hints::extend_program_exception_class_hints(
        &mut hints,
        crate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(std::slice::from_ref(
            &defining,
        )),
    );
    crate::front::llbc_hints::extend_program_exception_class_hints(
        &mut hints,
        crate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(std::slice::from_ref(
            &copied,
        )),
    );
    super::super::seed_program_exception_class_hints(&copied, hints);

    let parsed: TyRef = serde_json::from_value(e_ty).expect("E ty");
    assert_eq!(
        super::super::lltype::tyref_exception_class_key(&parsed, &copied).as_deref(),
        Some("synth::E")
    );

    let state_a = lowering_state(&defining);
    let mut state_b = lowering_state(&copied);
    for key in ["E", "synth::E", "fixture::synth::E"] {
        if let Some(rows) = state_b.struct_fields().fields.get(key) {
            assert!(
                rows.iter().all(|row| row.name != "__discriminant"),
                "{key} registered __discriminant"
            );
        }
    }
    for key in [
        "synth::E::A",
        "fixture::synth::E::A",
        "E::A",
        "synth::E::B",
        "fixture::synth::E::B",
        "E::B",
    ] {
        assert!(
            state_b.struct_fields().fields.get(key).is_none(),
            "{key} registered rows on the copy"
        );
    }

    state_b.absorb_struct_fields(state_a.struct_fields());
    let program_a = state_a.finish(Vec::new());
    let program_b = state_b.finish(Vec::new());
    assert_eq!(program_a.exception_class_hints.len(), 1);
    assert_eq!(program_a.exception_class_hints[0].enum_path, "synth::E");
    assert_rows(
        &program_a.struct_fields,
        "synth::E::A",
        &[("n", "usize", false)],
    );
    assert!(program_b.exception_class_hints.is_empty());
    assert_rows(
        &program_b.struct_fields,
        "synth::E::A",
        &[("n", "usize", false)],
    );
    for key in ["E", "synth::E", "fixture::synth::E"] {
        assert!(
            !program_b.enum_variant_by_discriminant.contains_key(key),
            "{key} registered enum_variant_by_discriminant"
        );
        if let Some(rows) = program_b.struct_fields.fields.get(key) {
            assert!(
                rows.iter().all(|row| row.name != "__discriminant"),
                "{key} registered __discriminant"
            );
        }
    }
}

/// `C(B) B(A)`, `n` on A and C, none on B. The row lives on A, and a
/// read of C's `n` names that same owner.
#[test]
fn exception_class_field_read_skips_a_non_declaring_parent() {
    let e = adt(0);
    let u_size = usize_ty();
    let flag = bool_ty();
    let enum_ty = enum_decl(
        0,
        &["fixture", "synth", "E"],
        vec![
            variant("A", vec![named_field("n", &u_size)], 0),
            variant("B", vec![named_field("flag", &flag)], 1),
            variant("C", vec![named_field("n", &u_size)], 2),
        ],
    );
    let match_c = fun(
        0,
        &["fixture", "synth", "match_c"],
        vec![e.clone()],
        u_size.clone(),
        read_variant_field_body(
            local_place(1, &e),
            &e,
            downcast_field(local_place(1, &e), &e, 2, 0, &u_size),
            &u_size,
        ),
    );
    let llbc = super::window_llbc_with_globals(
        vec![enum_ty],
        vec![match_c],
        vec![marker(
            0,
            &["fixture", "synth", "_exception_class_E"],
            "C(B) B(A)",
        )],
    );
    let (_, fields, _, _, attrs) = program_metadata(&llbc);
    assert_rows(&fields, "synth::E::A", &[("n", "usize", false)]);
    assert_rows(&fields, "synth::E::B", &[("flag", "bool", false)]);
    assert_absent(&fields, "synth::E::C");
    assert_absent(&fields, "synth::E");
    let mut n_owners: Vec<&str> = fields
        .fields
        .iter()
        .filter(|(_, rows)| rows.iter().any(|row| row.name == "n"))
        .map(|(key, _)| key.as_str())
        .collect();
    n_owners.sort();
    assert_eq!(n_owners, vec!["synth::E::A"]);
    assert_eq!(
        attrs.get("synth::E::A"),
        Some(&vec![("n".to_string(), ValueType::Unsigned)])
    );
    assert_eq!(
        attrs.get("synth::E::B"),
        Some(&vec![("flag".to_string(), ValueType::Bool)])
    );
    assert!(attrs.get("synth::E::C").is_none());
    let graph = lower(&llbc, "match_c");
    assert!(
        true_edge_reaches_field(&graph, "n", "synth::E::A"),
        "C arm getattr n on synth::E::A; graph={}",
        graph.dump()
    );
}

/// Owner `x: i64`, subclass `x: isize`. `ClassDef._generalize_attr`
/// merges both signed integers onto the owner; the subclass has no row.
#[test]
fn exception_class_inherited_i64_isize_merges_on_owner() {
    let i64_ty = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let isize_ty = isize_disc_ty();
    let llbc = super::window_llbc_with_globals(
        vec![enum_decl(
            0,
            &["fixture", "synth", "E"],
            vec![
                variant("A", vec![named_field("x", &i64_ty)], 0),
                variant("B", vec![named_field("x", &isize_ty)], 1),
            ],
        )],
        Vec::new(),
        vec![marker(
            0,
            &["fixture", "synth", "_exception_class_E"],
            "B(A)",
        )],
    );
    let (_, fields, _, _, attrs) = program_metadata(&llbc);
    assert_rows(&fields, "synth::E::A", &[("x", "i64", false)]);
    assert_absent(&fields, "synth::E::B");
    assert_absent(&fields, "synth::E");
    let row = &fields.fields["synth::E::A"][0];
    let raw = std::collections::HashSet::new();
    assert_eq!(
        crate::codewriter::call::get_type_flag(&row.ty, &raw),
        crate::codewriter::call::get_type_flag("isize", &raw),
        "i64 and isize are one signed integer"
    );
    assert!(row.raw_owned.is_none());
    assert_eq!(
        attrs.get("synth::E::A"),
        Some(&vec![("x".to_string(), ValueType::Int)])
    );
    assert!(attrs.get("synth::E::B").is_none());
    let mut owners: Vec<&str> = fields
        .fields
        .iter()
        .filter(|(_, rows)| rows.iter().any(|candidate| candidate.name == "x"))
        .map(|(key, _)| key.as_str())
        .collect();
    owners.sort();
    assert_eq!(owners, vec!["synth::E::A"]);
}

fn ordinary_integer_decl(
    variant: &str,
    row_ty: &str,
    signed_size: usize,
) -> super::super::ExceptionClassAttrDecl {
    let s_value =
        crate::annotator::bookkeeper::integer_annotation_for_spelling(row_ty, signed_size)
            .unwrap_or_else(|| panic!("{row_ty} at signed size {signed_size}"));
    super::super::ExceptionClassAttrDecl {
        variant_name: variant.to_string(),
        field_index: 0,
        is_positional: false,
        is_atomic: false,
        declared: row_ty.to_string(),
        row_ty: row_ty.to_string(),
        attr_ty: ValueType::Int,
        raw_owned: None,
        s_value,
        one_word: true,
        kind: super::super::ExceptionClassAttrKind::Ordinary,
    }
}

/// Ancestor `x: i64`, subclass `x: isize`, with `Signed` 4 bytes.
/// `compute_restype` keeps `r_longlong`; the owner row stays `i64`
/// (`SignedLongLong`, 8 bytes). The owner declared as `isize` widens
/// to that same row.
#[test]
fn exception_class_inherited_i64_isize_keeps_i64_when_signed_is_four_bytes() {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let raw = std::collections::HashSet::new();
    let owner_i64 = vec![
        ordinary_integer_decl("A", "i64", 4),
        ordinary_integer_decl("B", "isize", 4),
    ];
    let (row, attr) = super::super::exception_class_generalized_owner_row(
        "synth::E::A",
        "x",
        "A",
        &owner_i64,
        &raw,
        4,
    );
    assert_eq!(row.name, "x");
    assert_eq!(row.ty, "i64");
    assert!(row.raw_owned.is_none());
    assert_eq!(attr, ("x".to_string(), ValueType::Int));
    assert_eq!(
        crate::annotator::bookkeeper::scalar_lltype_for("i64", 4),
        Some(LowLevelType::SignedLongLong)
    );

    let owner_isize = vec![
        ordinary_integer_decl("A", "isize", 4),
        ordinary_integer_decl("B", "i64", 4),
    ];
    let (row, attr) = super::super::exception_class_generalized_owner_row(
        "synth::E::A",
        "x",
        "A",
        &owner_isize,
        &raw,
        4,
    );
    assert_eq!(row.ty, "i64");
    assert_eq!(attr, ("x".to_string(), ValueType::Int));
    assert_eq!(
        crate::annotator::bookkeeper::scalar_lltype_for(&row.ty, 4),
        Some(LowLevelType::SignedLongLong)
    );
}

/// `int` → `isize`, `r_uint` → `usize`, `r_longlong` → `i64`,
/// `r_ulonglong` → `u64`. `i64` is `Signed` when `Signed` is 8 bytes
/// and `SignedLongLong` when it is 4.
#[test]
fn exception_class_annotation_repr_follows_integer_knowntype() {
    use crate::annotator::model::{KnownType, SomeInteger, SomeValue};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let repr = |knowntype, signed_size| {
        let ann = SomeValue::Integer(SomeInteger::new_with_knowntype(false, knowntype));
        super::super::exception_class_annotation_repr(&ann, signed_size).expect("integer repr")
    };
    assert_eq!(
        repr(KnownType::Int, 4),
        ("isize".to_string(), ValueType::Int)
    );
    assert_eq!(
        repr(KnownType::Ruint, 4),
        ("usize".to_string(), ValueType::Unsigned)
    );
    assert_eq!(
        repr(KnownType::LongLong, 4),
        ("i64".to_string(), ValueType::Int)
    );
    assert_eq!(
        repr(KnownType::LongLong, 8),
        ("i64".to_string(), ValueType::Int)
    );
    assert_eq!(
        repr(KnownType::ULongLong, 4),
        ("u64".to_string(), ValueType::Unsigned)
    );
    assert_eq!(
        repr(KnownType::ULongLong, 8),
        ("u64".to_string(), ValueType::Unsigned)
    );
    assert_eq!(
        crate::annotator::bookkeeper::scalar_lltype_for("i64", 8),
        Some(LowLevelType::Signed)
    );
    assert_eq!(
        crate::annotator::bookkeeper::scalar_lltype_for("i64", 4),
        Some(LowLevelType::SignedLongLong)
    );
}

#[test]
#[should_panic(expected = "exception class synth::E::A.x: field type i64 ∪ String")]
fn exception_class_inherited_field_type_mismatch_panics() {
    let i64_ty = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let string_ty = adt(1);
    let llbc = super::window_llbc_with_globals(
        vec![
            enum_decl(
                0,
                &["fixture", "synth", "E"],
                vec![
                    variant("A", vec![named_field("x", &i64_ty)], 0),
                    variant("B", vec![named_field("x", &string_ty)], 1),
                ],
            ),
            opaque_string_decl(1),
        ],
        Vec::new(),
        vec![marker(
            0,
            &["fixture", "synth", "_exception_class_E"],
            "B(A)",
        )],
    );
    let _ = program_metadata(&llbc);
}

fn list_of_str_helper_addr() -> i64 {
    crate::call::symbolic_fnaddr_for_target(&CallTarget::function_path([
        "lltypesystem",
        "rlist",
        "jit_ll_list_str_from_rust_vec",
    ]))
}

fn names_list_of_str_helper(graph: &FunctionGraph) -> bool {
    let expected = list_of_str_helper_addr();
    super::window_graph_ops(graph)
        .iter()
        .any(|op| matches!(&op.kind, OpKind::ConstFnAddr { value, .. } if *value == expected))
}

fn assert_list_coerce_then_setfield(graph: &FunctionGraph, source: &Variable) {
    let expected = list_of_str_helper_addr();
    let mut found = false;
    for block in &graph.blocks {
        let ops = &block.operations;
        for window in ops.windows(4) {
            let addr = matches!(
                &window[0].kind,
                OpKind::ConstFnAddr { value, .. } if *value == expected
            );
            let OpKind::CallResidual { args_r, .. } = &window[1].kind else {
                continue;
            };
            let live = matches!(window[2].kind, OpKind::Live { .. });
            let write = match &window[3].kind {
                OpKind::FieldWrite {
                    field, value, ty, ..
                } => {
                    field.name == "missing"
                        && *ty == ValueType::Ref(None)
                        && matches!(value, LinkArg::Value(stored) if window[1].result.as_ref() == Some(stored))
                }
                _ => false,
            };
            if addr && args_r == &vec![source.clone()] && live && write {
                found = true;
            }
        }
    }
    assert!(
        found,
        "rust-heap Vec<String> is converted to the list before setfield; graph={}",
        graph.dump()
    );
}

fn assert_rpy_list_of_str_is_fixed_list()
-> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::annotator::model::SomeValue;
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    let projected = bk.project_struct_field_type(crate::front::mir::RPY_LIST_OF_STR);
    let SomeValue::List(list) = &projected else {
        panic!("rpy_list_of_str projects to {projected:?}");
    };
    {
        let item = list.listdef.listitem_rc();
        let item = item.borrow();
        assert!(!item.resized, "the list is unresized");
        assert!(!item.mutated, "the list is not mutated");
        assert_eq!(item.s_value, crate::annotator::model::s_str0());
    }
    let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
    let rtyper = std::rc::Rc::new(crate::translator::rtyper::rtyper::RPythonTyper::new(&ann));
    rtyper.initialize_exceptiondata().expect("exceptiondata");
    let repr = crate::translator::rtyper::rmodel::rtyper_makerepr(&projected, &rtyper)
        .expect("SomeList rtyper_makerepr");
    assert_eq!(repr.class_name(), "FixedSizeListRepr");
    // `externalvsinternal(..., gcref=True)` wraps `Ptr(STR)` as `GCREF`,
    // so the list pointer is `Ptr(GcArray(GCREF))`, not `Ptr(GcArray(Ptr(STR)))`.
    let ll = crate::translator::rtyper::rlist::list_of_str_lltype();
    assert_eq!(repr.lowleveltype(), &ll);
    let helper = crate::translator::rtyper::rlist::build_ll_fixed_length_helper_graph(
        "ll_fixed_length",
        ll.clone(),
    )
    .expect("ll_fixed_length");
    let graph = helper.graph.borrow();
    let block = graph.startblock.borrow();
    assert!(
        block
            .operations
            .iter()
            .any(|op| op.opname == "getarraysize"),
        "ll_fixed_length is getarraysize"
    );
    ll
}

fn enum_vec_string_field_llbc(
    leaf: &str,
    marked: bool,
    funs: Vec<Value>,
) -> majit_charon_reader::Llbc {
    let vec_ty = vec_adt_of(1, 2);
    let types = vec![
        enum_decl(
            0,
            &["fixture", leaf],
            vec![variant("S", vec![named_field("missing", &vec_ty)], 0)],
        ),
        opaque_vec_decl(1),
        opaque_string_decl(2),
    ];
    let globals = if marked {
        vec![marker(
            0,
            &["fixture", &format!("_exception_class_{leaf}")],
            "",
        )]
    } else {
        Vec::new()
    };
    super::window_llbc_with_globals(types, funs, globals)
}

fn vec_string_len_reader(enum_leaf: &str, fun_id: u64, len_fun_id: u64) -> Value {
    let enum_ty = adt(0);
    let enum_ref = shared_ref(&enum_ty);
    let vec_ty = vec_adt_of(1, 2);
    let out = usize_ty();
    let disc_ty = isize_disc_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, &enum_ref), "Deref"]}),
        &enum_ty,
    );
    let field = downcast_field(deref.clone(), &enum_ty, 0, 0, &vec_ty);
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &out);
    fun(
        fun_id,
        &["fixture", enum_leaf, "len_missing"],
        vec![enum_ref.clone()],
        out.clone(),
        unstructured(
            1,
            vec![
                named_local(0, None, &out),
                named_local(1, Some("self"), &enum_ref),
                named_local(2, None, &disc_ty),
            ],
            vec![
                regular(
                    vec![assign(disc.clone(), json!({"Discriminant": deref}))],
                    switch_int(disc, &[(0, 1)], 2),
                ),
                regular(vec![], call_regular(len_fun_id, field, ret, 3, 2)),
                regular(vec![], json!({"Abort": "UndefinedBehavior"})),
                ret_block(vec![]),
            ],
        ),
    )
}

fn vec_len_fun(id: u64) -> Value {
    super::window_opaque_fun(
        id,
        &["alloc", "vec", "<Impl>", "len"],
        vec![vec_adt_of(1, 2)],
        usize_ty(),
    )
}

#[test]
fn exception_class_vec_string_attr_coerces_rust_heap_on_write() {
    let vec_ty = vec_adt_of(1, 2);
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[vec_ty.clone()],
        vec![copy_of(local_place(1, &vec_ty))],
    );
    let llbc = enum_vec_string_field_llbc("BadVec", true, vec![build]);
    let _list_ll = assert_rpy_list_of_str_is_fixed_list();
    let graph = lower(&llbc, "build_s");
    let source = sole_input(&graph);
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert_eq!(
        writes[0].0,
        ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string())),
        "front stores the attr as the list word; graph={}",
        graph.dump()
    );
    assert!(
        matches!(&writes[0].1, LinkArg::Value(value) if value == &source),
        "the operand is still the rust-heap local; graph={}",
        graph.dump()
    );
    assert!(
        !names_list_of_str_helper(&graph),
        "the copy is a jtransform rewrite; graph={}",
        graph.dump()
    );
    let found = inputs(&graph);
    assert!(
        found.iter().any(|(ty, root)| {
            matches!(ty, ValueType::Ref(Some(name)) if name == "Vec<String>")
                || root.as_deref() == Some("Vec<String>")
        }),
        "the local stays a rust-heap header so the copy has a source; inputs={found:?}"
    );
    let transformed = transform_graph(&graph);
    assert_list_coerce_then_setfield(&transformed, &source);
}

#[test]
fn exception_class_vec_string_len_is_fixed_list_length() {
    let list_ll = assert_rpy_list_of_str_is_fixed_list();
    let len_fun = vec_len_fun(0);
    let reader = vec_string_len_reader("BadVec", 1, 0);
    let llbc = enum_vec_string_field_llbc("BadVec", true, vec![len_fun, reader]);
    let graph = lower(&llbc, "len_missing");
    let reads: Vec<&crate::model::SpaceOperation> = super::window_graph_ops(&graph)
        .into_iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "missing" && field.vec_part.is_none()
            )
        })
        .collect();
    assert_eq!(reads.len(), 1, "{}", graph.dump());
    let OpKind::FieldRead { field, ty, .. } = &reads[0].kind else {
        unreachable!("filtered to FieldRead");
    };
    assert!(
        !field.inline_vec,
        "the list word is not an inline Vec header"
    );
    assert_eq!(
        *ty,
        ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string()))
    );
    let produced = reads[0].result.as_ref().expect("field read result");
    assert_eq!(
        produced.concretetype().as_ref(),
        Some(&list_ll),
        "the getattr word is the FixedSizeListRepr pointer; graph={}",
        graph.dump()
    );
    let len_calls: Vec<&crate::model::SpaceOperation> = super::window_graph_ops(&graph)
        .into_iter()
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("__len"),
            _ => false,
        })
        .collect();
    assert_eq!(len_calls.len(), 1, "{}", graph.dump());
    let OpKind::Call { args, .. } = &len_calls[0].kind else {
        unreachable!("filtered to Call");
    };
    assert!(
        matches!(args.first(), Some(LinkArg::Value(value)) if Some(value) == reads[0].result.as_ref()),
        "len's receiver is the list word; graph={}",
        graph.dump()
    );
    let len_parts = super::window_graph_ops(&graph).into_iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. }
                if field.name == "missing" && field.vec_part == Some(crate::model::VecFieldPart::Len)
        )
    });
    assert!(
        !len_parts,
        "missing.len() is not the rust header length word; graph={}",
        graph.dump()
    );
}

fn slice_join_fun(id: u64) -> Value {
    super::window_opaque_fun(
        id,
        &["alloc", "slice", "<Impl>", "join"],
        vec![vec_adt_of(1, 2), str_ref_ty()],
        str_ref_ty(),
    )
}

fn vec_string_join_reader(enum_leaf: &str, fun_id: u64, join_fun_id: u64) -> Value {
    let enum_ty = adt(0);
    let enum_ref = shared_ref(&enum_ty);
    let vec_ty = vec_adt_of(1, 2);
    let out = str_ref_ty();
    let disc_ty = isize_disc_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, &enum_ref), "Deref"]}),
        &enum_ty,
    );
    let field = downcast_field(deref.clone(), &enum_ty, 0, 0, &vec_ty);
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &out);
    let sep = json!({"Const": [{"Str": ", "}, str_ref_ty()]});
    fun(
        fun_id,
        &["fixture", enum_leaf, "getmsg"],
        vec![enum_ref.clone()],
        out.clone(),
        unstructured(
            1,
            vec![
                named_local(0, None, &out),
                named_local(1, Some("self"), &enum_ref),
                named_local(2, None, &disc_ty),
            ],
            vec![
                regular(
                    vec![assign(disc.clone(), json!({"Discriminant": deref}))],
                    switch_int(disc, &[(0, 1)], 2),
                ),
                regular(
                    vec![],
                    json!({"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": join_fun_id},
                            "generics": super::window_empty_generics()
                        }},
                        "args": [{"Read": field}, sep]
                    }, "dest": ret, "target": 3, "on_unwind": 2}}),
                ),
                regular(vec![], json!({"Abort": "UndefinedBehavior"})),
                ret_block(vec![]),
            ],
        ),
    )
}

/// `posonly_kwds.join(", ")` on a marked exception's `Vec<String>` attr
/// is `<[S]>::join`. The list word stays `list_of_str`, and the adapter
/// rewrites that call to `str.join` (`rtype_method_join`).
#[test]
fn exception_class_vec_string_join_lowers_to_str_join() {
    use crate::flowspace::model::Hlvalue;
    let list_ll = assert_rpy_list_of_str_is_fixed_list();
    let join_fun = slice_join_fun(0);
    let reader = vec_string_join_reader("BadVec", 1, 0);
    let llbc = enum_vec_string_field_llbc("BadVec", true, vec![join_fun, reader]);
    let graph = lower(&llbc, "getmsg");
    let join_calls: Vec<&crate::model::SpaceOperation> = super::window_graph_ops(&graph)
        .into_iter()
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                let n = segments.len();
                n >= 3
                    && segments[n - 3] == "slice"
                    && segments[n - 2] == "<Impl>"
                    && segments[n - 1] == "join"
            }
            _ => false,
        })
        .collect();
    assert_eq!(join_calls.len(), 1, "{}", graph.dump());
    let OpKind::Call { args, .. } = &join_calls[0].kind else {
        unreachable!("filtered to Call");
    };
    let Some(LinkArg::Value(receiver)) = args.first() else {
        panic!("join receiver is the list word; graph={}", graph.dump());
    };
    assert_eq!(
        receiver.concretetype().as_ref(),
        Some(&list_ll),
        "join receiver concretetype {}; graph={}",
        receiver
            .concretetype()
            .map(|ty| format!("{ty:?}"))
            .unwrap_or_else(|| "none".to_string()),
        graph.dump()
    );

    let registry = crate::translator::rtyper::call_registry::CallRegistry::new(std::rc::Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    let mut value_map = std::collections::HashMap::new();
    let result = join_calls[0].result.clone().expect("join result");
    let result_hl = Hlvalue::Variable(Variable::new());
    value_map.insert(result, result_hl.clone());
    let mut arg_hls = Vec::new();
    for arg in args {
        match arg {
            LinkArg::Value(var) => {
                let hl = Hlvalue::Variable(Variable::new());
                value_map.insert(var.clone(), hl.clone());
                arg_hls.push(hl);
            }
            LinkArg::Const(constant) => arg_hls.push(Hlvalue::Constant(constant.clone())),
        }
    }
    let translated = crate::translator::rtyper::flowspace_adapter::translate_op(
        join_calls[0],
        &value_map,
        &registry,
    )
    .unwrap_or_else(|err| panic!("{err}; graph={}", graph.dump()));
    assert_eq!(translated.len(), 2, "{}", graph.dump());
    assert_eq!(translated[0].opname, "getattr");
    assert_eq!(translated[0].args[0], arg_hls[1]);
    let Hlvalue::Constant(method) = &translated[0].args[1] else {
        panic!("method name is a Constant");
    };
    assert_eq!(method.value.as_text(), Some("join"));
    assert_eq!(translated[1].opname, "simple_call");
    assert_eq!(translated[1].args[0], translated[0].result);
    assert_eq!(translated[1].args[1], arg_hls[0]);
    assert_eq!(translated[1].result, result_hl);
}

#[test]
fn exception_class_unmarked_vec_string_field_stays_three_words() {
    let vec_ty = vec_adt_of(1, 2);
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[vec_ty.clone()],
        vec![copy_of(local_place(1, &vec_ty))],
    );
    let len_fun = vec_len_fun(1);
    let reader = vec_string_len_reader("Plain", 2, 1);
    let unmarked = enum_vec_string_field_llbc("Plain", false, vec![build, len_fun, reader]);
    let (_, fields, _, _, attrs) = program_metadata(&unmarked);
    let mut saw_row = false;
    for key in ["Plain::S", "fixture::Plain::S", "synth::Plain::S"] {
        let Some(rows) = row_names(&fields, key) else {
            continue;
        };
        for (name, ty, _) in &rows {
            if name == "missing" {
                saw_row = true;
                assert_eq!(ty.as_str(), "Vec<String>", "{key} row {rows:?}");
            }
        }
        let Some(attr_rows) = attrs.get(key) else {
            continue;
        };
        for (name, ty) in attr_rows {
            if name == "missing" {
                assert_ne!(
                    *ty,
                    ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string())),
                    "{key} attr {attr_rows:?}"
                );
            }
        }
    }
    assert!(saw_row, "unmarked Vec<String> row absent");
    let (flag, kind, size) =
        crate::codewriter::call::get_type_flag("Vec<String>", &std::collections::HashSet::new());
    assert_eq!(flag, majit_ir::descr::ArrayFlag::Struct);
    assert_eq!(kind, majit_ir::value::Type::Ref);
    assert_eq!(size, 3 * crate::layout::target_word_size());

    let graph = lower(&unmarked, "build_s");
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert!(
        writes[0].0 != ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string())),
        "an unmarked enum keeps the rust-heap store; got {writes:?} graph={}",
        graph.dump()
    );
    let transformed = transform_graph(&graph);
    assert!(
        !names_list_of_str_helper(&transformed),
        "no coerce without the exception-class flag; graph={}",
        transformed.dump()
    );

    let len_graph = lower(&unmarked, "len_missing");
    // `.len()` of an unmarked inline `Vec` is the header length word.
    // The whole-header read is not kept when nothing else consumes it.
    let inline_header = super::window_graph_ops(&len_graph).into_iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, ty, .. }
                if field.name == "missing"
                    && field.inline_vec
                    && *ty != ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string()))
        )
    });
    assert!(
        inline_header,
        "unmarked read keeps the inline Vec; graph={}",
        len_graph.dump()
    );
    let len_part = super::window_graph_ops(&len_graph).into_iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. }
                if field.name == "missing" && field.vec_part == Some(crate::model::VecFieldPart::Len)
        )
    });
    assert!(
        len_part,
        "unmarked len is the header length word; graph={}",
        len_graph.dump()
    );
    let calls_len = super::window_graph_ops(&len_graph)
        .iter()
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("__len"),
            _ => false,
        });
    assert!(
        !calls_len,
        "unmarked len is not Call __len; graph={}",
        len_graph.dump()
    );

    let holder_ty = adt(0);
    let holder = super::interior_struct(
        0,
        &["fixture", "Holder"],
        &[("missing", vec_ty.clone())],
        24,
        &[0],
        false,
    );
    let build_holder = build_struct_aggregate(
        0,
        "build_holder",
        0,
        &holder_ty,
        &[vec_ty.clone()],
        vec![copy_of(local_place(1, &vec_ty))],
    );
    let plain = super::window_llbc_with_globals(
        vec![holder, opaque_vec_decl(1), opaque_string_decl(2)],
        vec![build_holder],
        Vec::new(),
    );
    let (_, fields, _, _, attrs) = program_metadata(&plain);
    for key in ["Holder", "fixture::Holder"] {
        if let Some(rows) = row_names(&fields, key) {
            for (name, ty, _) in &rows {
                if name == "missing" {
                    assert_eq!(ty.as_str(), "Vec<String>", "{key} row {rows:?}");
                }
            }
        }
        if let Some(rows) = attrs.get(key) {
            for (name, ty) in rows {
                if name == "missing" {
                    assert_ne!(
                        *ty,
                        ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string())),
                        "{key} attr {rows:?}"
                    );
                }
            }
        }
    }
    let graph = lower(&plain, "build_holder");
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert!(
        writes[0].0 != ValueType::Ref(Some(crate::front::mir::RPY_LIST_OF_STR.to_string())),
        "a plain struct keeps the rust-heap store; got {writes:?} graph={}",
        graph.dump()
    );
    let transformed = transform_graph(&graph);
    assert!(
        !names_list_of_str_helper(&transformed),
        "a plain Vec<String> field is not a list-of-str attr; graph={}",
        transformed.dump()
    );
}

fn signature_string_decl() -> Value {
    json!({
        "def_id": 2,
        "item_meta": super::window_meta(&["alloc", "string", "String"], false),
        "kind": "Opaque"
    })
}

/// One-word layout. The field is `alloc::string::String`, a GC container,
/// so the struct's `_gckind` stays Gc and `*mut Signature` is the
/// flavor-raw class root (`ValueType::Ref` on the attr and the read).
/// The raw-owned store writes that pointer as `ValueType::Unsigned`
/// (`descr.py get_type_flag`).
fn signature_struct() -> Value {
    super::interior_struct(
        1,
        &["fixture", "Signature"],
        &[("word", adt(2))],
        8,
        &[0],
        false,
    )
}

fn signature_types() -> Vec<Value> {
    vec![signature_struct(), signature_string_decl()]
}

fn signature_raw_marker() -> Value {
    marker(1, &["fixture", "_alloc_flavor_Signature"], "raw")
}

fn missing_field_row<'a>(
    fields: &'a crate::front::semantic::StructFieldRegistry,
    key: &str,
) -> &'a crate::front::semantic::FieldRow {
    fields
        .fields
        .get(key)
        .and_then(|rows| rows.iter().find(|row| row.name == "missing"))
        .unwrap_or_else(|| panic!("{key} has no missing row"))
}

fn names_raw_box(graph: &FunctionGraph) -> bool {
    super::window_graph_ops(graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::ConstFnAddr { path, .. } if path.contains("jit_ll_box_raw_moved")
        )
    })
}

fn transform_with_fields(
    graph: &FunctionGraph,
    fields: crate::front::semantic::StructFieldRegistry,
    helper: Option<&str>,
) -> FunctionGraph {
    let mut cc = crate::call::CallControl::new();
    cc.set_struct_fields(fields);
    let mut config = crate::codewriter::jtransform::GraphTransformConfig::default();
    if let Some(helper) = helper {
        config.raw_owned_box_helper = helper.to_string();
    }
    crate::codewriter::jtransform::Transformer::new(&config)
        .with_callcontrol(&mut cc)
        .transform(graph)
        .graph
}

fn const_int_of(ops: &[crate::model::SpaceOperation], var: &Variable) -> Option<(usize, i64)> {
    ops.iter()
        .enumerate()
        .find_map(|(index, op)| match &op.kind {
            OpKind::ConstInt(value) if op.result.as_ref() == Some(var) => Some((index, *value)),
            _ => None,
        })
}

fn assert_raw_box_then_setfield(graph: &FunctionGraph, source: &Variable, size: i64, align: i64) {
    let mut found = false;
    for block in &graph.blocks {
        let ops = &block.operations;
        for (index, op) in ops.iter().enumerate() {
            let OpKind::CallResidual {
                args_r,
                args_i,
                funcptr,
                descriptor,
                result_kind,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args_r.is_empty() || args_i.len() != 3 || &args_i[0] != source {
                continue;
            }
            let crate::model::CallFuncPtr::Value(func_var) = funcptr else {
                continue;
            };
            let Some(def_index) = ops[..index]
                .iter()
                .rposition(|earlier| earlier.result.as_ref() == Some(func_var))
            else {
                continue;
            };
            let OpKind::ConstFnAddr { path, .. } = &ops[def_index].kind else {
                continue;
            };
            if !path.contains("jit_ll_box_raw_moved") {
                continue;
            }
            let Some((size_index, size_value)) = const_int_of(&ops[..index], &args_i[1]) else {
                continue;
            };
            let Some((align_index, align_value)) = const_int_of(&ops[..index], &args_i[2]) else {
                continue;
            };
            assert_eq!(size_value, size, "{}", graph.dump());
            assert_eq!(align_value, align, "{}", graph.dump());
            assert!(
                size_index < align_index && align_index < def_index,
                "size, align, then the helper; graph={}",
                graph.dump()
            );
            assert_eq!(*result_kind, 'i', "{}", graph.dump());
            assert_eq!(descriptor.result_type, 'i', "{}", graph.dump());
            assert_eq!(
                descriptor.result_ir_type(),
                majit_ir::value::Type::Int,
                "{}",
                graph.dump()
            );
            assert_eq!(descriptor.arg_classes, "iii", "{}", graph.dump());
            assert!(
                index + 2 < ops.len(),
                "helper, live, then the store; graph={}",
                graph.dump()
            );
            assert!(
                matches!(ops[index + 1].kind, OpKind::Live { .. }),
                "graph={}",
                graph.dump()
            );
            let OpKind::FieldWrite {
                field, value, ty, ..
            } = &ops[index + 2].kind
            else {
                panic!("the store follows live; graph={}", graph.dump());
            };
            let boxed = op.result.as_ref().expect("boxed pointer");
            assert_eq!(
                FunctionGraph::concretetype_of(boxed),
                crate::model::ConcreteType::Signed,
                "raw pointer is int kind; graph={}",
                graph.dump()
            );
            assert_eq!(field.name, "missing");
            assert!(
                matches!(value, LinkArg::Value(stored) if stored == boxed),
                "the store operand is the boxed pointer; graph={}",
                graph.dump()
            );
            assert_eq!(
                *ty,
                ValueType::Unsigned,
                "stored ty {ty:?}; graph={}",
                graph.dump()
            );
            found = true;
        }
    }
    assert!(
        found,
        "raw-owned store boxes the moved value; graph={}",
        graph.dump()
    );
    assert!(
        !field_writes(graph).iter().any(|(name, _)| name == "word"),
        "the projection does not copy the struct leaf; graph={}",
        graph.dump()
    );
}

fn read_missing_fun(enum_leaf: &str, by_value: bool) -> Value {
    let enum_ty = adt(0);
    let enum_ref = shared_ref(&enum_ty);
    let sig_ty = adt(1);
    let field_ty = if by_value {
        sig_ty.clone()
    } else {
        super::interior_raw_mut(sig_ty.clone())
    };
    let borrow_ty = shared_ref(&sig_ty);
    let out = usize_ty();
    let disc_ty = isize_disc_ty();
    let deref = super::window_place(
        json!({"Projection": [local_place(1, &enum_ref), "Deref"]}),
        &enum_ty,
    );
    let field = downcast_field(deref.clone(), &enum_ty, 0, 0, &field_ty);
    let place = if by_value {
        field
    } else {
        super::window_place(json!({"Projection": [field, "Deref"]}), &sig_ty)
    };
    let borrowed = json!({"Ref": {"place": place, "kind": "Shared", "ptr_metadata": null}});
    let disc = local_place(2, &disc_ty);
    let ret = local_place(0, &out);
    let tmp = local_place(3, &borrow_ty);
    fun(
        1,
        &["fixture", enum_leaf, "read_missing"],
        vec![enum_ref.clone()],
        out.clone(),
        unstructured(
            1,
            vec![
                named_local(0, None, &out),
                named_local(1, Some("self"), &enum_ref),
                named_local(2, None, &disc_ty),
                named_local(3, None, &borrow_ty),
            ],
            vec![
                regular(
                    vec![assign(disc.clone(), json!({"Discriminant": deref}))],
                    switch_int(disc, &[(0, 1)], 2),
                ),
                regular(
                    vec![assign(tmp.clone(), borrowed)],
                    call_regular(0, tmp, ret, 3, 2),
                ),
                regular(vec![], json!({"Abort": "UndefinedBehavior"})),
                ret_block(vec![]),
            ],
        ),
    )
}

struct SigFieldRead {
    ty: ValueType,
    inline_vec: bool,
    taken_by_address: bool,
    concretetype: Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType>,
    result: Variable,
}

fn sig_field_read(graph: &FunctionGraph) -> SigFieldRead {
    let reads: Vec<&crate::model::SpaceOperation> = super::window_graph_ops(graph)
        .into_iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "missing"
            )
        })
        .collect();
    assert_eq!(reads.len(), 1, "{}", graph.dump());
    let OpKind::FieldRead { field, ty, .. } = &reads[0].kind else {
        unreachable!("filtered to FieldRead");
    };
    let result = reads[0].result.clone().expect("field read result");
    SigFieldRead {
        ty: ty.clone(),
        inline_vec: field.inline_vec,
        taken_by_address: field.taken_by_address,
        concretetype: result.concretetype(),
        result,
    }
}

fn use_sig_receiver(graph: &FunctionGraph) -> Variable {
    super::window_graph_ops(graph)
        .into_iter()
        .find_map(|op| match &op.kind {
            OpKind::Call { target, args, .. } => {
                let leaf = match target {
                    CallTarget::FunctionPath { segments, .. } => {
                        segments.last().map(String::as_str)
                    }
                    CallTarget::Method { name, .. } => Some(name.as_str()),
                    _ => None,
                };
                if leaf != Some("use_sig") {
                    return None;
                }
                match args.first() {
                    Some(LinkArg::Value(value)) => Some(value.clone()),
                    _ => None,
                }
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("use_sig call; graph={}", graph.dump()))
}

#[test]
fn exception_class_raw_owned_attr_boxes_on_write() {
    let sig_ty = adt(1);
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[sig_ty.clone()],
        vec![copy_of(local_place(1, &sig_ty))],
    );
    let llbc = bad_enum_with_funs(
        "BadSig",
        sig_ty,
        signature_types(),
        vec![signature_raw_marker()],
        vec![build],
    );
    let graph = lower(&llbc, "build_s");
    let source = sole_input(&graph);
    let writes = missing_field_writes(&graph);
    assert_eq!(writes.len(), 1, "{}", graph.dump());
    assert_eq!(
        writes[0].0,
        ValueType::Unsigned,
        "front stores the raw pointer as an unsigned word; graph={}",
        graph.dump()
    );
    assert!(
        matches!(&writes[0].1, LinkArg::Value(value) if value == &source),
        "the operand is still the moved value; graph={}",
        graph.dump()
    );
    assert!(
        !field_writes(&graph).iter().any(|(name, _)| name == "word"),
        "front does not copy the struct leaf; graph={}",
        graph.dump()
    );
    assert!(
        !names_raw_box(&graph),
        "the box is a jtransform rewrite; graph={}",
        graph.dump()
    );
    let owner = field_writes(&graph)
        .into_iter()
        .find(|(name, _)| name == "missing")
        .and_then(|(_, owner)| owner);
    assert_eq!(owner.as_deref(), Some("BadSig::S"));

    let (_, fields, _, _, _) = program_metadata(&llbc);
    assert!(missing_field_row(&fields, "BadSig::S").raw_owned.is_some());
    let transformed = transform_with_fields(&graph, fields.clone(), None);
    assert_raw_box_then_setfield(&transformed, &source, 8, 8);

    let plain = transform_with_fields(&graph, fields, Some(""));
    let plain_writes = missing_field_writes(&plain);
    assert_eq!(plain_writes.len(), 1, "{}", plain.dump());
    assert!(
        matches!(&plain_writes[0].1, LinkArg::Value(value) if value == &source),
        "an empty helper name leaves the moved value; graph={}",
        plain.dump()
    );
    assert!(
        !names_raw_box(&plain),
        "an empty helper name does not box; graph={}",
        plain.dump()
    );
}

#[test]
fn exception_class_declared_raw_pointer_store_is_not_reboxed() {
    let ptr_ty = super::interior_raw_mut(adt(1));
    let enum_ty = adt(0);
    let build = build_aggregate(
        0,
        "build_s",
        0,
        0,
        &enum_ty,
        &[ptr_ty.clone()],
        vec![copy_of(local_place(1, &ptr_ty))],
    );
    let llbc = bad_enum_with_funs(
        "OkRawPtr",
        ptr_ty,
        signature_types(),
        vec![signature_raw_marker()],
        vec![build],
    );
    let (_, fields, _, _, _) = program_metadata(&llbc);
    assert!(
        missing_field_row(&fields, "OkRawPtr::S")
            .raw_owned
            .is_none()
    );
    let graph = lower(&llbc, "build_s");
    let source = sole_input(&graph);
    let transformed = transform_with_fields(&graph, fields, None);
    assert!(
        !names_raw_box(&transformed),
        "a declared pointer is already the word; graph={}",
        transformed.dump()
    );
    let writes = missing_field_writes(&transformed);
    assert_eq!(writes.len(), 1, "{}", transformed.dump());
    assert!(
        matches!(&writes[0].1, LinkArg::Value(value) if value == &source),
        "the operand stays the pointer; graph={}",
        transformed.dump()
    );
}

#[test]
fn exception_class_raw_owned_attr_read_matches_raw_pointer() {
    let use_sig = super::window_opaque_fun(
        0,
        &["fixture", "use_sig"],
        vec![shared_ref(&adt(1))],
        usize_ty(),
    );
    let by_value = bad_enum_with_funs(
        "BadSig",
        adt(1),
        signature_types(),
        vec![signature_raw_marker()],
        vec![use_sig.clone(), read_missing_fun("BadSig", true)],
    );
    let declared = bad_enum_with_funs(
        "OkRawPtr",
        super::interior_raw_mut(adt(1)),
        signature_types(),
        vec![signature_raw_marker()],
        vec![use_sig, read_missing_fun("OkRawPtr", false)],
    );
    let by_graph = lower(&by_value, "read_missing");
    let ptr_graph = lower(&declared, "read_missing");
    let by_read = sig_field_read(&by_graph);
    let ptr_read = sig_field_read(&ptr_graph);
    let root = ValueType::Ref(Some("Signature".to_string()));
    assert_eq!(by_read.ty, root, "{}", by_graph.dump());
    assert_eq!(by_read.ty, ptr_read.ty, "{}", ptr_graph.dump());
    assert!(!by_read.inline_vec, "{}", by_graph.dump());
    assert!(!ptr_read.inline_vec, "{}", ptr_graph.dump());
    assert!(!by_read.taken_by_address, "{}", by_graph.dump());
    assert!(!ptr_read.taken_by_address, "{}", ptr_graph.dump());
    assert_eq!(
        by_read.concretetype,
        ptr_read.concretetype,
        "by={} ptr={}",
        by_graph.dump(),
        ptr_graph.dump()
    );
    assert_eq!(
        use_sig_receiver(&by_graph),
        by_read.result,
        "{}",
        by_graph.dump()
    );
    assert_eq!(
        use_sig_receiver(&ptr_graph),
        ptr_read.result,
        "{}",
        ptr_graph.dump()
    );
}

#[test]
fn exception_class_raw_owned_destructor_includes_inherited_attr() {
    let mut registry = crate::front::semantic::StructFieldRegistry::default();
    registry.fields.insert(
        "Arg::TooMany".to_string(),
        [
            crate::front::semantic::FieldRow::named("signature", "*mut Signature").with_raw_owned(
                crate::front::semantic::RawOwnedField {
                    type_path: "fixture::Signature".to_string(),
                    size: 8,
                    align: 8,
                },
            ),
        ],
    );
    registry.fields.insert(
        "Arg::TooManyMethod".to_string(),
        [crate::front::semantic::FieldRow::named("extra", "usize")],
    );
    let hint = crate::front::llbc_hints::ExceptionClassHint::new(
        "Arg",
        "TooManyMethod(TooMany)",
        vec!["TooMany".into(), "TooManyMethod".into()],
    );
    let mut rows =
        crate::front::llbc_hints::minted_exception_class_rows(std::slice::from_ref(&hint));
    let header_id = majit_ir::descr::StructId::from_canonical(
        crate::codewriter::assembler::host_header_owner(),
    );
    let mut ids = std::collections::HashMap::new();
    ids.insert(
        crate::codewriter::assembler::host_header_owner().to_string(),
        Some(header_id),
    );
    let _registry = crate::test_support::register_struct_ids_serialized(ids);
    registry.fields.insert(
        crate::codewriter::assembler::host_header_owner().to_string(),
        crate::codewriter::assembler::host_header_rows(),
    );
    let mut cc = crate::call::CallControl::new();
    cc.set_struct_layout(
        header_id,
        crate::codewriter::assembler::host_header_layout(),
    );
    cc.set_struct_fields(registry);
    cc.exception_class_hints = vec![hint];
    crate::codewriter::assembler::attach_minted_exception_raw_owned_drops(&cc, &mut rows);

    let word = crate::layout::target_word_size();
    let header = cc
        .struct_layout_for(crate::codewriter::assembler::host_header_owner())
        .expect("host header");
    assert_eq!(header.size, 2 * word);
    let row = |name: &str| {
        rows.iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("{name} missing from {rows:?}"))
    };
    let base = row("Arg");
    let parent = row("Arg::TooMany");
    let child = row("Arg::TooManyMethod");
    assert!(base.raw_owned.is_empty(), "{:?}", base.raw_owned);
    assert_eq!(parent.raw_owned.len(), 1);
    assert_eq!(child.raw_owned.len(), 1);
    assert_eq!(parent.raw_owned[0].offset, header.size);
    assert_eq!(child.raw_owned[0].offset, header.size);
    assert_eq!(base.size, header.size);
    assert_eq!(parent.size, header.size + word);
    assert!(base.gc_ptr_offsets.is_empty());
    assert!(parent.gc_ptr_offsets.is_empty());
    assert!(child.gc_ptr_offsets.is_empty());
    assert_eq!(parent.raw_owned[0].type_path, "fixture::Signature");
    assert_eq!(child.raw_owned[0].type_path, "fixture::Signature");

    let src = crate::front::llbc_hints::render_minted_exception_classes_rs(&rows);
    assert_eq!(
        src.matches("drop_raw_owned_attr::<fixture::Signature>")
            .count(),
        2,
        "{src}"
    );
    assert!(src.contains(&format!(
        "drop_raw_owned_attr::<fixture::Signature>(obj_addr, {})",
        header.size
    )));
    assert!(src.contains("unsafe fn MINTED_EXC_Arg__TooMany_destructor"));
    assert!(src.contains("unsafe fn MINTED_EXC_Arg__TooManyMethod_destructor"));
    assert!(src.contains("destructor: Some(MINTED_EXC_Arg__TooMany_destructor)"));
    assert!(src.contains("destructor: Some(MINTED_EXC_Arg__TooManyMethod_destructor)"));
    assert!(src.contains("unsafe extern \"C\" fn MINTED_EXC_Arg__TooMany_alloc_with_del"));
    assert!(src.contains("unsafe extern \"C\" fn MINTED_EXC_Arg__TooManyMethod_alloc_with_del"));
    assert!(src.contains("build_ll_0_alloc_with_del::Arg::TooMany"));
    assert!(src.contains("build_ll_0_alloc_with_del::Arg::TooManyMethod"));
    assert!(
        !src.contains("fn MINTED_EXC_Arg_alloc_with_del"),
        "the base owns no raw attr; src={src}"
    );
    assert!(src.contains(&format!(
        "MintedExceptionClass {{ name: \"Arg\", pytype: &MINTED_EXC_Arg, parent: None, size: {}, gc_ptr_offsets: &[], destructor: None, alloc_with_del: None }}",
        base.size
    )));
    assert!(
        !src.contains("fn MINTED_EXC_Arg_destructor"),
        "the base owns no raw attr; src={src}"
    );
}

/// A layout sidecar is the same crate's type decls with no function or
/// global bodies. The defining artefact's `_alloc_flavor_` marker is the
/// program-wide set `DuplicateLeafFacts::discover` and `prelink_crate`
/// register exception classes with (`rclass.py` `getgcflavor`).
#[test]
fn layout_sidecar_raw_flavor_attr_uses_the_program_set() {
    let wide = super::interior_struct(
        1,
        &["fixture", "WideSig"],
        &[
            ("a", usize_ty()),
            ("b", usize_ty()),
            ("c", usize_ty()),
            ("d", usize_ty()),
        ],
        32,
        &[0, 8, 16, 24],
        false,
    );
    let enum_ty = enum_decl(
        0,
        &["fixture", "WideErr"],
        vec![variant("Too", vec![named_field("signature", &adt(1))], 0)],
    );
    let types = vec![enum_ty, wide];
    let host = super::window_llbc_with_globals(
        types.clone(),
        Vec::new(),
        vec![
            marker(0, &["fixture", "_exception_class_WideErr"], ""),
            marker(1, &["fixture", "_alloc_flavor_WideSig"], "raw"),
        ],
    );
    let sidecar = || super::window_llbc_with_globals(types.clone(), Vec::new(), Vec::new());
    let program_raw =
        crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&host));
    assert!(
        program_raw.contains("WideSig"),
        "defining artefact harvest: {program_raw:?}"
    );
    assert!(
        32 > crate::layout::target_word_size() as u64,
        "the fixture struct has to miss the one-word size fallback"
    );

    let (_, host_fields, _, _, _) = program_metadata(&host);
    let host_row = host_fields
        .fields
        .get("WideErr::Too")
        .and_then(|rows| rows.iter().find(|row| row.name == "signature"))
        .unwrap_or_else(|| {
            panic!(
                "defining artefact registers WideErr::Too.signature, keys {:?}",
                host_fields.fields.keys().collect::<Vec<_>>()
            )
        });
    assert_eq!(host_row.ty, "*mut WideSig");
    let host_owned = host_row
        .raw_owned
        .as_ref()
        .expect("defining artefact stores the moved struct");
    assert_eq!(host_owned.type_path, "fixture::WideSig");
    assert_eq!(host_owned.size, 32);
    assert_eq!(host_owned.align, 8);

    let bare = sidecar();
    assert_eq!(bare.iter_fun_decls().count(), 0);
    assert_eq!(bare.iter_global_decls().count(), 0);
    let hints = crate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(
        std::slice::from_ref(&host),
    );
    super::super::seed_program_exception_class_hints(&bare, hints.clone());
    let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ =
            super::super::DuplicateLeafFacts::discover(&bare, &std::collections::HashSet::new());
    }));
    let Err(payload) = caught else {
        panic!("a layout copy with an empty raw set must fail the one-word check");
    };
    let message = panic_text(payload);
    let expected =
        "exception class WideErr::Too.signature: field type WideSig is not a one-word attr";
    assert!(
        message.contains(expected),
        "panic `{message}` must contain `{expected}`"
    );

    let copied = sidecar();
    super::super::seed_program_exception_class_hints(&copied, hints);
    let facts = super::super::DuplicateLeafFacts::discover(&copied, &program_raw);
    let row = facts
        .struct_fields
        .fields
        .get("WideErr::Too")
        .and_then(|rows| rows.iter().find(|row| row.name == "signature"))
        .unwrap_or_else(|| {
            panic!(
                "layout copy registers WideErr::Too.signature from the program set, keys {:?}",
                facts.struct_fields.fields.keys().collect::<Vec<_>>()
            )
        });
    assert_eq!(row.ty, "*mut WideSig");
    let owned = row
        .raw_owned
        .as_ref()
        .expect("layout copy stores the moved struct");
    assert_eq!(owned.type_path, "fixture::WideSig");
    assert_eq!(owned.size, 32);
    assert_eq!(owned.align, 8);
    let _ = super::super::prelink_crate(&copied, &std::collections::HashSet::new(), &program_raw);
}
