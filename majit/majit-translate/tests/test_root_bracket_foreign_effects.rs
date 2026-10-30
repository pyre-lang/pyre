//! A bracket in a crate that only imports the root-stack API can be erased
//! once the effects of the crates it links against are published.
//!
//! Without them every body-less callee in `pyre-interpreter` -- `core`'s
//! `ptr::is_null` as much as `pyre_object`'s `is_tagged_int` -- is an unknown
//! that may leave the stack changed, so no bracket there whose body calls
//! anything foreign is ever proved balanced.  The harvest publishes
//! `pyre-object`'s own answers; `core` and the other crates outside the
//! translation input cannot name the API at all.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};
use majit_translate::front::mir::{
    RootStackState, apply_published_structs, erased_root_bracket_guards, fn_returns_owned_scope,
    harvest_published_structs, harvest_root_stack_touching_paths, harvest_scope_owning_identities,
    lower_function, mark_foreign_scope_owners, scope_owner_identity,
};
use majit_translate::model::OpKind;

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
);
const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-interpreter.ullbc"
);
const MODULE_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-module.ullbc"
);

/// Bodies that open a bracket, sampled: every `erased_root_bracket_guards`
/// call starts a fresh root-stack analysis, so the whole population costs
/// minutes.
const SAMPLE: usize = 200;

fn callee_name_paths<'a>(
    llbc: &'a Llbc,
    body: &'a majit_charon_reader::ullbc::Unstructured,
) -> impl Iterator<Item = String> + 'a {
    body.body.iter().filter_map(move |bb| {
        let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
            return None;
        };
        let CallFunc::Regular(reg) = &call.func else {
            return None;
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        llbc.fn_by_id(*id).map(|f| f.item_meta.name_path())
    })
}

fn path_is_gc_roots_leaf(path: &str, leaf: &str) -> bool {
    path.rsplit("::").next() == Some(leaf) && path.split("::").any(|s| s == "gc_roots")
}

/// True when every terminator callee outside `gc_roots` lives in a crate the
/// harvest can name. An interpreter-local helper is analysed from its own
/// body, so publishing `pyre-object` cannot change whether it touches.
fn only_foreign_non_root_callees(paths: &[String]) -> bool {
    paths.iter().all(|path| {
        path.split("::").any(|s| s == "gc_roots")
            || path.split("::").next() != Some("pyre_interpreter")
    })
}

/// Bodies holding at least one erased bracket, over the first [`SAMPLE`]
/// bodies that open one and whose other callees are all foreign. Those are
/// the bodies whose unknown `pyre-object` / `core` callees the harvest can
/// answer.
fn erased_bodies(llbc: &Llbc) -> (usize, usize) {
    let (mut opening, mut erased) = (0, 0);
    let root_stack_state = RootStackState::new(llbc);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let paths: Vec<String> = callee_name_paths(llbc, &body).collect();
        if !paths.iter().any(|p| path_is_gc_roots_leaf(p, "push_roots")) {
            continue;
        }
        if !only_foreign_non_root_callees(&paths) {
            continue;
        }
        if opening == SAMPLE {
            break;
        }
        opening += 1;
        if !erased_root_bracket_guards(llbc, fd, &body, &root_stack_state).is_empty() {
            erased += 1;
        }
    }
    (opening, erased)
}

#[test]
fn published_dependency_effects_let_an_importing_crate_erase_its_brackets() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let touching = harvest_root_stack_touching_paths(&object);
    assert!(
        touching
            .iter()
            .any(|p| p == "pyre_object::gc_roots::<Impl>::pin_roots"),
        "the root-stack API itself must stay a touching body"
    );
    assert!(
        !touching
            .iter()
            .any(|p| p == "pyre_object::intobject::w_int_new"),
        "an allocation leaves the root stack as it found it"
    );
    let crate_name = object.crate_name().to_string();
    drop(object);

    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let (opening, before) = erased_bodies(&interpreter);
    interpreter.set_root_stack_effects(vec![crate_name], touching);
    let (_, after) = erased_bodies(&interpreter);
    eprintln!("bodies opening a bracket: {opening}; erased before {before}, after {after}");
    assert!(
        after > before,
        "publishing pyre-object's effects must let more interpreter brackets go"
    );
}

#[test]
fn a_bodyless_rooted_items_new_returns_the_scope_its_defining_crate_published() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let owning = harvest_scope_owning_identities(&object);
    assert!(
        owning.iter().any(|p| p.contains("RootedItems")),
        "RootedItems::new opens push_roots and returns that guard: {owning:?}"
    );
    assert!(
        owning.iter().any(|p| p.contains("DictOperationGuard")),
        "DictOperationGuard::new returns the bracket it opened: {owning:?}"
    );
    drop(object);

    let mut interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let external_id = interpreter
        .iter_fun_decls()
        .find(|fd| {
            !fd.item_meta.is_local
                && fd.item_meta.name_path().ends_with("::new")
                && scope_owner_identity(&interpreter, fd).contains("RootedItems")
        })
        .expect("interpreter names RootedItems::new")
        .def_id;
    for fd in interpreter.file.translated.fun_decls.iter_mut().flatten() {
        if fd.def_id == external_id {
            fd.body = None;
        }
    }
    assert!(
        interpreter
            .fn_by_id(external_id)
            .is_some_and(|fd| fd.body.is_none()),
        "the declaration under test has no body"
    );
    assert!(
        !fn_returns_owned_scope(&interpreter, external_id),
        "without the published constructors the opaque new is not a guard"
    );
    mark_foreign_scope_owners(&interpreter, &owning);
    assert!(
        fn_returns_owned_scope(&interpreter, external_id),
        "the published RootedItems::new is the guard its caller holds"
    );
}

#[test]
fn an_opaque_pyerror_object_field_reads_the_defining_crates_field() {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file()
        || !std::path::Path::new(MODULE_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    // `PyError` is a one-field tuple over the raised value. The named
    // fields live on `PyErrorObject`, which `Deref` exposes. `parse_filter_spec`
    // reads `error.kind` as that struct's field.
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let fields = harvest_published_structs(&interpreter);
    let pyerror_object = fields
        .iter()
        .find(|body| body.path.ends_with("::PyErrorObject"))
        .expect("interpreter publishes PyErrorObject's fields");
    assert!(
        pyerror_object
            .fields
            .iter()
            .any(|field| field.name.as_deref() == Some("kind")),
        "PyErrorObject publishes a named kind field: {:?}",
        pyerror_object
            .fields
            .iter()
            .map(|field| field.name.as_deref())
            .collect::<Vec<_>>()
    );
    drop(interpreter);

    let mut module = Llbc::load(MODULE_LLBC).expect("load pyre-module");
    let opaque = module.iter_type_decls().any(|td| {
        td.item_meta.name_path().ends_with("::PyErrorObject")
            && matches!(td.kind, majit_charon_reader::ullbc::TypeDeclKind::Opaque)
    });
    assert!(opaque, "pyre-module sees PyErrorObject as an opaque struct");
    apply_published_structs(&mut module, &fields);
    assert!(
        module.iter_type_decls().any(|td| {
            td.item_meta.name_path().ends_with("::PyErrorObject") && td.published_struct_copy
        }),
        "PyErrorObject's copied fields stay an import, not a second class"
    );
    let graph = lower_function(&module, "parse_filter_spec")
        .expect("parse_filter_spec lowers once PyErrorObject.kind is a field");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "kind"
                        && field
                            .owner_root
                            .as_deref()
                            .is_some_and(|root| root.contains("PyErrorObject"))
            )
        })
    });
    assert!(
        saw,
        "error.kind must be a FieldRead of PyErrorObject.kind, not the error value"
    );
}

/// `rclass.py InstanceRepr.new_instance` / `setfield`: an imported class
/// keeps the defining owner's fields when constructing an inline header.
#[test]
fn imported_pyobject_header_construction_keeps_the_defining_fields() {
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let fields = harvest_published_structs(&object);
    drop(object);
    let mut module = Llbc::load(MODULE_LLBC).expect("load pyre-module");
    apply_published_structs(&mut module, &fields);
    assert!(module.iter_type_decls().any(|td| {
        td.item_meta.name_path() == "pyre_object::pyobject::PyObject" && td.published_struct_copy
    }));
    let fd = module
        .iter_fun_decls()
        .find(|fd| {
            let path = fd.item_meta.name_path();
            path.contains("::_ssl::memory_bio_methods::") && path.ends_with("::__new__")
        })
        .expect("MemoryBIO.__new__ declaration");
    let graph = majit_translate::front::mir::lower_fun_decl(
        &majit_translate::front::mir::LowerContext::new(&module),
        fd,
    )
    .expect("lower MemoryBIO.__new__ with published PyObject fields");
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if let OpKind::FieldWrite { field, .. } | OpKind::FieldRead { field, .. } = &op.kind {
            assert_ne!(
                field.owner_root.as_deref(),
                Some("Adt"),
                "imported PyObject header must not mint an anonymous positional field: {op:?}"
            );
        }
    }
}
