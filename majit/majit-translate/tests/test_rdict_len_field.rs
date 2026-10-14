//! A residual call to a generic `dont_look_inside` helper must name the
//! instantiation's own funcptr (`annlowlevel.specialize` /
//! `getfunctionptr`). Identity-keyed and module-str-keyed look-inside
//! wrappers therefore residualize under distinct `__inst_` suffixes.
//!
//! `lower_function` has no `SpecQueue`, so this fixture builds the
//! program (`FunctionDesc.cachedgraph` copies) from the two lookup
//! roots that reach `ll_dict_lookup_trampoline`. Dump graphs when
//! `DUMP_RDICT_LEN` is set.

use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names;
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};
use majit_translate::{HostStaticAddrs, SemanticProgram};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

fn object_llbc() -> Option<&'static Llbc> {
    static LLBC: OnceLock<Option<Llbc>> = OnceLock::new();
    LLBC.get_or_init(|| {
        if !std::path::Path::new(OBJECT_LLBC).is_file() {
            eprintln!("skipping: {OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
            return None;
        }
        Some(Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc"))
    })
    .as_ref()
}

fn dump_if_requested(name: &str, graph: &FunctionGraph) {
    let Ok(dir) = std::env::var("DUMP_RDICT_LEN") else {
        return;
    };
    let path = std::path::Path::new(&dir).join(format!("{name}.txt"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, graph.dump()).expect("write residual inst dump");
}

fn inst_leaves(graph: &FunctionGraph) -> Vec<String> {
    let mut leaves = Vec::new();
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        let CallTarget::FunctionPath { segments, .. } = target else {
            continue;
        };
        let Some(leaf) = segments.last() else {
            continue;
        };
        if leaf.contains("__inst_") {
            leaves.push(leaf.clone());
        }
    }
    leaves
}

fn program_inst_leaves(program: &SemanticProgram) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for f in &program.functions {
        let graph = f.graph();
        dump_if_requested(&f.name, graph);
        for leaf in inst_leaves(graph) {
            out.push((f.name.clone(), leaf));
        }
    }
    out
}

fn find_inst_leaf<'a>(leaves: &'a [(String, String)], needle: &str) -> Option<&'a str> {
    leaves
        .iter()
        .find(|(_, leaf)| leaf.contains(needle))
        .map(|(_, leaf)| leaf.as_str())
}

#[test]
fn identity_and_module_lookup_residualize_distinct_instantiations() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(llbc),
        HostStaticAddrs::default(),
        &["identitydict", "celldict", "rordereddict"],
        &[
            "_orig_w_dict_lookup_identity_strategy",
            "_orig_module_dict_entries_get",
        ],
    )
    .expect("lower identity and module lookup graphs");
    let leaves = program_inst_leaves(&program);
    let identity_leaf = find_inst_leaf(&leaves, "IdentityKey").unwrap_or_else(|| {
        panic!(
            "identity instantiation must residualize a distinct __inst_ path\nleaves={leaves:?}\nspecs={:?}",
            program
                .functions
                .iter()
                .filter(|f| f.name.contains("__spec_"))
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
        )
    });
    let module_leaf = find_inst_leaf(&leaves, "StrKey").unwrap_or_else(|| {
        panic!(
            "module-str instantiation must residualize a distinct __inst_ path\nleaves={leaves:?}\nspecs={:?}",
            program
                .functions
                .iter()
                .filter(|f| f.name.contains("__spec_"))
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
        )
    });
    assert_ne!(
        identity_leaf, module_leaf,
        "distinct instantiations must not share a residual funcptr\nidentity={identity_leaf}\nmodule={module_leaf}"
    );
}
