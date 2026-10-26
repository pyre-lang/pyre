//! Census of non-dereferenced `PyFrame` virtualizable-field projections.
//!
//! Lives in `pyre-jit-trace` because that crate already owns
//! [`PYFRAME_VABLE_FIELDS`] / [`PYFRAME_VABLE_ARRAYS`] and already depends
//! on `majit-translate` (dev + prepass). `majit-translate` cannot take a
//! dependency the other way without a cycle through the prepass.
//!
//! `Place::is_deref_projection` is the front-end rule that sets
//! `FieldDescriptor::base_is_deref` (`front/mir.rs`). The corpus test
//! classifies every field projection with that method rather than a second
//! predicate.

use majit_charon_reader::{
    Llbc,
    ullbc::{
        CallFunc, FunDecl, NameSeg, Operand, Place, PlaceKind, Rvalue, StmtKind, TermKind,
        TypeDeclKind, Unstructured, builtin_path_label,
    },
};
use pyre_jit_trace::virtualizable_spec::{
    PYFRAME_VABLE_ARRAYS, PYFRAME_VABLE_FIELDS, PYFRAME_VABLE_OWNER_ROOT,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// The only function permitted to project a virtualizable field off a
/// non-dereferenced base. It takes the frame by value, so every
/// `frame.<field>` is a projection off a local aggregate rather than off a
/// dereference.
const ALLOWED: &[&str] = &["pyre_interpreter::pyframe::{FrameBox}::new"];

const CORPUS_FILES: &[&str] = &[
    "pyre-object.ullbc",
    "pyre-interpreter.ullbc",
    "pyre-jit.ullbc",
];

#[derive(Default)]
struct Census {
    /// Non-deref virtualizable-field projections, reads and writes together.
    non_deref: BTreeMap<String, usize>,
    unclassified: Vec<String>,
}

fn vable_field_names() -> HashSet<&'static str> {
    PYFRAME_VABLE_FIELDS
        .iter()
        .chain(PYFRAME_VABLE_ARRAYS.iter())
        .map(|(name, _)| *name)
        .collect()
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_ullbc(name: &str) -> PathBuf {
    let path = repo_root().join("target/llbc").join(name);
    assert!(
        path.is_file(),
        "{} is missing — run `python3 scripts/extract-llbc.py` to produce the extracted LLBC",
        path.display()
    );
    path
}

fn pyframe_def_ids(llbc: &Llbc) -> HashSet<u64> {
    llbc.file
        .translated
        .type_decls
        .iter()
        .flatten()
        .filter_map(|td| {
            let leaf = td.item_meta.name_path_str().rsplit("::").next()?;
            matches!(&td.kind, TypeDeclKind::Struct(_))
                .then_some(td.def_id)
                .filter(|_| leaf == PYFRAME_VABLE_OWNER_ROOT)
        })
        .collect()
}

fn tyref_node<'a>(ty: &'a majit_charon_reader::ullbc::TyRef, llbc: &'a Llbc) -> Option<&'a Value> {
    match ty {
        majit_charon_reader::ullbc::TyRef::Inline { value: (_, v) } => Some(v),
        majit_charon_reader::ullbc::TyRef::Other(v) => Some(v),
        majit_charon_reader::ullbc::TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

/// Peel `Deduplicated` / `Value` / `Ref` the way `front/mir.rs`
/// `strip_ty_wrappers` does, then take a nominal ADT id the way
/// `type_decl_ref_adt_id` does.
fn tyref_adt_def_id(ty: &majit_charon_reader::ullbc::TyRef, llbc: &Llbc) -> Option<u64> {
    adt_def_id_from_node(tyref_node(ty, llbc)?, llbc)
}

fn adt_def_id_from_node<'a>(mut node: &'a Value, llbc: &'a Llbc) -> Option<u64> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(Value::as_array) {
            node = arr.get(1)?;
            continue;
        }
        let adt = obj.get("Adt")?.as_object()?;
        return match adt.get("builtin").and_then(Value::as_str) {
            None | Some("Box") => adt.get("id")?.as_u64(),
            Some(_) => None,
        };
    }
    None
}

fn struct_field_index(payload: &Value) -> Option<usize> {
    let arr = payload.as_array()?;
    if arr.len() != 2 || !arr[0].is_null() {
        return None;
    }
    arr[1].as_u64().map(|i| i as usize)
}

fn impl_owner_adt_id(llbc: &Llbc, impl_payload: &Value) -> Option<u64> {
    let ty = impl_payload.get("Ty")?;
    let sb = ty.get("skip_binder")?;
    if let Some(arr) = sb.get("Value").and_then(Value::as_array)
        && let Some(body) = arr.get(1)
    {
        return adt_def_id_from_node(body, llbc);
    }
    if let Some(id) = sb.get("Deduplicated").and_then(Value::as_u64) {
        return llbc.dedup_to_adt_def_id(id).or_else(|| {
            llbc.dedup_body(id)
                .and_then(|b| adt_def_id_from_node(b, llbc))
        });
    }
    adt_def_id_from_node(sb, llbc)
}

fn impl_owner_leaf(llbc: &Llbc, impl_payload: &Value) -> Option<String> {
    let adt_id = impl_owner_adt_id(llbc, impl_payload)?;
    let td = llbc.type_by_id(adt_id)?;
    td.item_meta
        .name_path_str()
        .rsplit("::")
        .next()
        .map(str::to_string)
}

/// Charon pretty-print spelling, e.g. `pyre_interpreter::pyframe::{FrameBox}::new`.
fn display_fn_name(llbc: &Llbc, fd: &FunDecl) -> String {
    let mut out = String::new();
    for seg in fd.item_meta.template_name() {
        if !out.is_empty() {
            out.push_str("::");
        }
        match seg {
            NameSeg::Ident {
                ident: (s, disambiguator),
            } => {
                out.push_str(s);
                if s == "closure" && *disambiguator > 0 {
                    out.push('#');
                    out.push_str(&disambiguator.to_string());
                }
            }
            NameSeg::Other(v) => {
                if let Some(impl_p) = v.get("Impl") {
                    if let Some(leaf) = impl_owner_leaf(llbc, impl_p) {
                        out.push('{');
                        out.push_str(&leaf);
                        out.push('}');
                    } else {
                        out.push_str("<Impl>");
                    }
                } else if let Some(label) = builtin_path_label(v) {
                    out.push_str(&label);
                } else {
                    let label = v
                        .as_object()
                        .and_then(|m| m.keys().next())
                        .map(String::as_str)
                        .unwrap_or("?");
                    out.push('<');
                    out.push_str(label);
                    out.push('>');
                }
            }
        }
    }
    if out.is_empty() {
        fd.item_meta.name_path()
    } else {
        out
    }
}

fn classify_place(
    place: &Place,
    is_write: bool,
    fn_name: &str,
    llbc: &Llbc,
    pyframe_ids: &HashSet<u64>,
    vable: &HashSet<&str>,
    census: &mut Census,
) {
    match &place.kind {
        PlaceKind::Projection(inner, elem) => {
            if let Some(payload) = elem.field_payload() {
                classify_field(
                    inner,
                    payload,
                    is_write,
                    fn_name,
                    llbc,
                    pyframe_ids,
                    vable,
                    census,
                );
            }
            classify_place(inner, false, fn_name, llbc, pyframe_ids, vable, census);
        }
        PlaceKind::Local(_) | PlaceKind::Global { .. } | PlaceKind::Unknown => {}
    }
}

fn classify_field(
    inner: &Place,
    payload: &Value,
    is_write: bool,
    fn_name: &str,
    llbc: &Llbc,
    pyframe_ids: &HashSet<u64>,
    vable: &HashSet<&str>,
    census: &mut Census,
) {
    let Some(def_id) = tyref_adt_def_id(&inner.ty, llbc) else {
        return;
    };
    if !pyframe_ids.contains(&def_id) {
        return;
    }
    let Some(td) = llbc.type_by_id(def_id) else {
        census.unclassified.push(format!(
            "{fn_name}: PyFrame field projection with unresolved type decl"
        ));
        return;
    };
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return;
    };
    let Some(idx) = struct_field_index(payload) else {
        census.unclassified.push(format!(
            "{fn_name}: PyFrame field projection with unindexed payload {payload}"
        ));
        return;
    };
    let Some(field) = fields.get(idx) else {
        census
            .unclassified
            .push(format!("{fn_name}: PyFrame field index {idx} out of range"));
        return;
    };
    let Some(name) = field.name.as_deref() else {
        return;
    };
    if !vable.contains(name) {
        return;
    }
    if matches!(inner.kind, PlaceKind::Unknown) {
        census.unclassified.push(format!(
            "{fn_name}: virtualizable field `{name}` on an unknown base"
        ));
        return;
    }
    // Reads and writes both count: a reads-only census would pass when the
    // write side regresses, and an unsuppressed `setfield_vable_*` against
    // a stack aggregate is the more dangerous direction.
    let _ = is_write;
    if !inner.is_deref_projection() {
        *census.non_deref.entry(fn_name.to_string()).or_insert(0) += 1;
    }
}

fn walk_operand(
    operand: &Operand,
    fn_name: &str,
    llbc: &Llbc,
    pyframe_ids: &HashSet<u64>,
    vable: &HashSet<&str>,
    census: &mut Census,
) {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            classify_place(place, false, fn_name, llbc, pyframe_ids, vable, census);
        }
        Operand::Const(_) => {}
    }
}

fn walk_rvalue(
    rvalue: &Rvalue,
    fn_name: &str,
    llbc: &Llbc,
    pyframe_ids: &HashSet<u64>,
    vable: &HashSet<&str>,
    census: &mut Census,
) {
    match rvalue {
        Rvalue::Use(operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::Repeat(operand, _, _, _)
        | Rvalue::ShallowInitBox(operand, _) => {
            walk_operand(operand, fn_name, llbc, pyframe_ids, vable, census);
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            walk_operand(lhs, fn_name, llbc, pyframe_ids, vable, census);
            walk_operand(rhs, fn_name, llbc, pyframe_ids, vable, census);
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            classify_place(place, false, fn_name, llbc, pyframe_ids, vable, census);
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                walk_operand(operand, fn_name, llbc, pyframe_ids, vable, census);
            }
        }
        Rvalue::Discriminant(place) | Rvalue::Len(place) => {
            classify_place(place, false, fn_name, llbc, pyframe_ids, vable, census);
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

fn walk_body(
    body: &Unstructured,
    fn_name: &str,
    llbc: &Llbc,
    pyframe_ids: &HashSet<u64>,
    vable: &HashSet<&str>,
    census: &mut Census,
) {
    for bb in &body.body {
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(dest, rvalue)) => {
                    classify_place(dest, true, fn_name, llbc, pyframe_ids, vable, census);
                    walk_rvalue(rvalue, fn_name, llbc, pyframe_ids, vable, census);
                }
                Ok(StmtKind::PlaceMention(place)) => {
                    classify_place(place, false, fn_name, llbc, pyframe_ids, vable, census);
                }
                Ok(StmtKind::Assert(assert)) => {
                    walk_operand(&assert.cond, fn_name, llbc, pyframe_ids, vable, census);
                }
                Ok(
                    StmtKind::StorageLive(_)
                    | StmtKind::StorageDead(_)
                    | StmtKind::Borrowck(_)
                    | StmtKind::Unknown,
                )
                | Err(_) => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Switch { discr, .. }) => {
                walk_operand(discr, fn_name, llbc, pyframe_ids, vable, census);
            }
            Ok(TermKind::Call { call, .. }) => {
                classify_place(&call.dest, true, fn_name, llbc, pyframe_ids, vable, census);
                for arg in &call.args {
                    walk_operand(arg, fn_name, llbc, pyframe_ids, vable, census);
                }
                if let CallFunc::Dynamic(operand) = &call.func {
                    walk_operand(operand, fn_name, llbc, pyframe_ids, vable, census);
                }
            }
            Ok(TermKind::Assert { assert, .. }) => {
                walk_operand(&assert.cond, fn_name, llbc, pyframe_ids, vable, census);
            }
            Ok(TermKind::Drop { place, .. }) => {
                classify_place(place, false, fn_name, llbc, pyframe_ids, vable, census);
            }
            Ok(_) | Err(_) => {}
        }
    }
}

fn census_llbc(llbc: &Llbc, vable: &HashSet<&str>, census: &mut Census) {
    let pyframe_ids = pyframe_def_ids(llbc);
    for fd in llbc.iter_fun_decls() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let name = display_fn_name(llbc, fd);
        walk_body(&body, &name, llbc, &pyframe_ids, vable, census);
    }
}

fn evaluate(census: &Census, allowed: &[&str]) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for (fn_name, n) in &census.non_deref {
        if !allowed.contains(&fn_name.as_str()) {
            failures.push(format!(
                "{fn_name}: {n} non-dereferenced virtualizable-field \
                 projection(s); the virtualizable lowering is suppressed for \
                 these, so only the by-value frame constructor may have them"
            ));
        }
    }
    if census.non_deref.is_empty() {
        failures.push(
            "no non-dereferenced projections found at all — the census is no \
             longer matching the shape rather than the tree being clean; if \
             the frame constructor genuinely stopped taking the frame by \
             value, update ALLOWED"
                .into(),
        );
    }
    for line in &census.unclassified {
        failures.push(format!("unclassified projection shape — {line}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn census_corpus(allowed: &[&str]) -> (Census, Result<(), Vec<String>>) {
    let vable = vable_field_names();
    let mut census = Census::default();
    for name in CORPUS_FILES {
        let path = require_ullbc(name);
        let llbc = Llbc::load(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
        census_llbc(&llbc, &vable, &mut census);
    }
    let result = evaluate(&census, allowed);
    (census, result)
}

/// `FrameBox::new` taking the frame by value is the standing witness that
/// the census still matches the shape. A function outside ALLOWED acquiring
/// the same projection, or the constructor losing it, is the tripwire.
///
/// Ignored in the default `cargo test --all` so the artefact walk does not
/// share a process with the parallel suite. CI runs it from the dedicated
/// cargo-test step.
#[test]
#[ignore = "loads target/llbc/{pyre-object,pyre-interpreter,pyre-jit}.ullbc"]
fn framebox_new_is_the_only_non_deref_pyframe_vable_projection() {
    let (census, result) = census_corpus(ALLOWED);
    if let Err(failures) = result {
        let mut msg = String::from("FAILED:\n");
        for f in &failures {
            msg.push_str("  ");
            msg.push_str(f);
            msg.push('\n');
        }
        msg.push_str("\nnon-deref projections by function:\n");
        for (fn_name, n) in &census.non_deref {
            let mark = if ALLOWED.contains(&fn_name.as_str()) {
                "ok"
            } else {
                "FAIL"
            };
            msg.push_str(&format!("  [{mark}] {fn_name}: {n}\n"));
        }
        panic!("{msg}");
    }
    eprintln!("non-deref projections by function:");
    for (fn_name, n) in &census.non_deref {
        eprintln!("  [ok] {fn_name}: {n}");
    }
    assert!(
        census
            .non_deref
            .keys()
            .any(|name| name.as_str() == ALLOWED[0]),
        "ALLOWED witness missing from the census: {:?}",
        census.non_deref
    );
}

fn span() -> Value {
    serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    })
}

fn ident(name: &str) -> Value {
    serde_json::json!({"Ident": [name, 0]})
}

fn item_meta(name: Vec<Value>) -> Value {
    serde_json::json!({
        "name": name,
        "span": span(),
        "source_text": null,
        "attr_info": {
            "attributes": [],
            "inline": null,
            "rename": null,
            "public": true
        },
        "is_local": true
    })
}

fn adt_ty(id: u64) -> Value {
    serde_json::json!({"Adt": {"id": id, "generics": {"types": []}}})
}

fn field_decl(name: &str) -> Value {
    serde_json::json!({
        "name": name,
        "ty": {"Literal": "Usize"},
        "attr_info": null
    })
}

fn struct_decl(def_id: u64, path: &[&str], fields: &[&str]) -> Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta(path.iter().copied().map(ident).collect()),
        "kind": {"Struct": fields.iter().copied().map(field_decl).collect::<Vec<_>>()}
    })
}

fn local(index: u64, name: Option<&str>, ty: Value) -> Value {
    serde_json::json!({
        "index": index,
        "name": name,
        "span": span(),
        "ty": ty
    })
}

fn place_local(index: u64, ty: Value) -> Value {
    serde_json::json!({"kind": {"Local": index}, "ty": ty})
}

fn place_field(inner: Value, field_idx: u64, ty: Value) -> Value {
    serde_json::json!({
        "kind": {"Projection": [inner, {"Field": [null, field_idx]}]},
        "ty": ty
    })
}

fn place_deref(inner: Value, ty: Value) -> Value {
    serde_json::json!({
        "kind": {"Projection": [inner, "Deref"]},
        "ty": ty
    })
}

fn assign_stmt(dest: Value, rvalue: Value) -> Value {
    serde_json::json!({
        "kind": {"Assign": [dest, rvalue]},
        "span": span()
    })
}

fn copy_use(place: Value) -> Value {
    serde_json::json!({"Use": [{"Copy": place}, "No"]})
}

fn fn_decl(def_id: u64, name: Vec<Value>, arg_ty: Value, stmts: Vec<Value>) -> Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta(name),
        "signature": {
            "is_unsafe": false,
            "inputs": [arg_ty.clone()],
            "output": {"Literal": "Bool"}
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, serde_json::json!({"Literal": "Bool"})),
                        local(1, Some("frame"), arg_ty)
                    ]
                },
                "body": [{
                    "statements": stmts,
                    "terminator": {
                        "span": span(),
                        "kind": "Return"
                    }
                }]
            }
        }
    })
}

fn framebox_impl_new(stmts: Vec<Value>) -> Value {
    let impl_seg = serde_json::json!({
        "Impl": {
            "Ty": {
                "skip_binder": adt_ty(1),
                "kind": "InherentImplBlock"
            }
        }
    });
    fn_decl(
        0,
        vec![
            ident("pyre_interpreter"),
            ident("pyframe"),
            impl_seg,
            ident("new"),
        ],
        adt_ty(0),
        stmts,
    )
}

fn load_fixture(type_decls: Vec<Value>, fun_decls: Vec<Value>) -> Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": fun_decls,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

fn pyframe_and_framebox() -> Vec<Value> {
    vec![
        struct_decl(
            0,
            &["pyre_interpreter", "pyframe", "PyFrame"],
            &["valuestackdepth", "pycode"],
        ),
        struct_decl(1, &["pyre_interpreter", "pyframe", "FrameBox"], &["ptr"]),
    ]
}

fn census_fixture(llbc: &Llbc) -> Census {
    let vable = vable_field_names();
    let mut census = Census::default();
    census_llbc(llbc, &vable, &mut census);
    census
}

#[test]
fn deref_of_mut_self_is_not_counted() {
    let self_ty = serde_json::json!({"Ref": ["Mut", adt_ty(0), "No"]});
    let deref = place_deref(place_local(1, self_ty.clone()), adt_ty(0));
    let read = place_field(deref, 0, serde_json::json!({"Literal": "Usize"}));
    let llbc = load_fixture(
        pyframe_and_framebox(),
        vec![fn_decl(
            0,
            vec![ident("settopvalue")],
            self_ty,
            vec![assign_stmt(
                place_local(0, serde_json::json!({"Literal": "Bool"})),
                copy_use(read),
            )],
        )],
    );
    let census = census_fixture(&llbc);
    assert!(census.non_deref.is_empty(), "{:?}", census.non_deref);
    assert!(census.unclassified.is_empty(), "{:?}", census.unclassified);
}

#[test]
fn by_value_frame_constructor_is_counted() {
    let frame = place_local(1, adt_ty(0));
    let pycode = place_field(frame.clone(), 1, serde_json::json!({"Literal": "Usize"}));
    let depth = place_field(frame, 0, serde_json::json!({"Literal": "Usize"}));
    let llbc = load_fixture(
        pyframe_and_framebox(),
        vec![framebox_impl_new(vec![
            assign_stmt(
                place_local(0, serde_json::json!({"Literal": "Bool"})),
                copy_use(pycode),
            ),
            assign_stmt(depth, serde_json::json!({"Use": [{"Const": 0}, "No"]})),
        ])],
    );
    let census = census_fixture(&llbc);
    assert_eq!(
        census
            .non_deref
            .get("pyre_interpreter::pyframe::{FrameBox}::new"),
        Some(&2),
        "{:?}",
        census.non_deref
    );
    assert!(census.unclassified.is_empty(), "{:?}", census.unclassified);
    evaluate(&census, ALLOWED).unwrap();
}

#[test]
fn same_named_field_on_a_non_pyframe_struct_is_not_counted() {
    let other = struct_decl(2, &["fixture", "Other"], &["valuestackdepth"]);
    let mut decls = pyframe_and_framebox();
    decls.push(other);
    let field = place_field(
        place_local(1, adt_ty(2)),
        0,
        serde_json::json!({"Literal": "Usize"}),
    );
    let llbc = load_fixture(
        decls,
        vec![fn_decl(
            0,
            vec![ident("touch_other")],
            adt_ty(2),
            vec![assign_stmt(
                place_local(0, serde_json::json!({"Literal": "Bool"})),
                copy_use(field),
            )],
        )],
    );
    let census = census_fixture(&llbc);
    assert!(census.non_deref.is_empty(), "{:?}", census.non_deref);
    assert!(census.unclassified.is_empty(), "{:?}", census.unclassified);
}

#[test]
fn function_outside_allowed_fails() {
    let frame = place_local(1, adt_ty(0));
    let depth = place_field(frame, 0, serde_json::json!({"Literal": "Usize"}));
    let llbc = load_fixture(
        pyframe_and_framebox(),
        vec![fn_decl(
            0,
            vec![ident("settopvalue")],
            adt_ty(0),
            vec![assign_stmt(
                depth,
                serde_json::json!({"Use": [{"Const": 0}, "No"]}),
            )],
        )],
    );
    let census = census_fixture(&llbc);
    let err = evaluate(&census, ALLOWED).expect_err("settopvalue must fail ALLOWED");
    assert!(
        err.iter()
            .any(|f| f.starts_with("settopvalue: 1 non-dereferenced virtualizable-field")),
        "{err:?}"
    );
}

#[test]
fn panic_const_string_is_not_a_projection() {
    let self_ty = serde_json::json!({"Ref": ["Mut", adt_ty(0), "No"]});
    let deref = place_deref(place_local(1, self_ty.clone()), adt_ty(0));
    let read = place_field(deref, 0, serde_json::json!({"Literal": "Usize"}));
    let panic = serde_json::json!({
        "kind": {
            "Assign": [
                place_local(0, serde_json::json!({"Literal": "Bool"})),
                {"Use": [{"Const": "assertion failed: index < self.valuestackdepth"}, "No"]}
            ]
        },
        "span": span()
    });
    let llbc = load_fixture(
        pyframe_and_framebox(),
        vec![fn_decl(
            0,
            vec![ident("settopvalue")],
            self_ty,
            vec![
                assign_stmt(
                    place_local(0, serde_json::json!({"Literal": "Bool"})),
                    copy_use(read),
                ),
                panic,
            ],
        )],
    );
    let census = census_fixture(&llbc);
    assert!(census.non_deref.is_empty(), "{:?}", census.non_deref);
    assert!(census.unclassified.is_empty(), "{:?}", census.unclassified);
}
