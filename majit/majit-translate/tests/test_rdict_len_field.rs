//! A residual call to a generic `dont_look_inside` helper must name the
//! instantiation's own funcptr (`annlowlevel.specialize` /
//! `getfunctionptr`). Identity-keyed and module-str-keyed look-inside
//! wrappers therefore residualize under distinct `__inst_` suffixes.
//!
//! `lower_function` has no `SpecQueue`, so this fixture builds the
//! program (`FunctionDesc.cachedgraph` copies) from the two lookup
//! roots that reach `ll_dict_lookup_trampoline`. Dump graphs when
//! `DUMP_RDICT_LEN` is set.
//!
//! `IdentityDictStrategy::length` is `identity_storage(w_dict).len()`.
//! `ModuleDictStrategy::length` is `len(self.unerase(w_dict.dstorage))`
//! (`celldict.py`). `ModuleDictStorage` is an `RDict` alias (`erase({})`),
//! so that `len` is `RDict::len` / `ll_dict_len`.
//! `rordereddict.rtype_len` / `ll_dict_len` emit
//! `getfield(d, "num_live_items")` when the table has no custom eq/hash, so
//! the identity `dicttable` (`mark_dict_non_null`) and the module-dict
//! table length are that header word, not `ll_vec_length` of the first
//! storage word.

use std::collections::HashSet;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{
    LowerContext, build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    lower_fun_decl, lower_function,
};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};
use majit_translate::{HostStaticAddrs, SemanticProgram};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
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

fn is_ll_vec_length(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().is_some_and(|leaf| {
            leaf == "ll_vec_length_r" || leaf == "ll_vec_length_i" || leaf == "ll_vec_length_f"
        }),
        _ => false,
    }
}

fn callee_fun_decl_ids(graph: &FunctionGraph) -> Vec<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call { target, .. } => target.fun_decl_id(),
            _ => None,
        })
        .collect()
}

fn count_length_ops(graph: &FunctionGraph, live_reads: &mut usize, ll_vec_length: &mut usize) {
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "num_live_items" => {
                *live_reads += 1;
            }
            OpKind::Call { target, .. } if is_ll_vec_length(target) => {
                *ll_vec_length += 1;
            }
            _ => {}
        }
    }
}

#[test]
fn identity_length_reads_num_live_items() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "identitydict::<Impl>::length")
        .expect("lower identitydict::<Impl>::length");
    dump_if_requested("identitydict_length", &graph);
    let mut live_reads = 0usize;
    let mut ll_vec_length = 0usize;
    count_length_ops(&graph, &mut live_reads, &mut ll_vec_length);
    let mut seen = HashSet::new();
    let mut pending = callee_fun_decl_ids(&graph);
    let context = LowerContext::new(llbc);
    let mut dumps = vec![graph.dump()];
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        let path = fd.item_meta.name_path();
        if !(path.contains("rordereddict") || path.contains("identitydict")) {
            continue;
        }
        let Ok(callee) = lower_fun_decl(&context, fd) else {
            continue;
        };
        count_length_ops(&callee, &mut live_reads, &mut ll_vec_length);
        pending.extend(callee_fun_decl_ids(&callee));
        dumps.push(callee.dump());
    }
    let dump = dumps.join("\n--- callee ---\n");
    assert!(
        live_reads > 0,
        "identity length must getfield num_live_items (`rordereddict.rtype_len`) somewhere in its call chain\n{dump}"
    );
    assert_eq!(
        ll_vec_length, 0,
        "identity length must not call ll_vec_length of the first storage word\n{dump}"
    );
}

#[test]
fn module_dict_len_reads_num_live_items() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    // Storage `len` used to live on `ModuleDictStorage` (`celldict::<Impl>::len`).
    // That type is now `type ModuleDictStorage = RDict<…>`, so the module-dict
    // length that remains in `celldict` is `ModuleDictStrategy::length`.
    let graph =
        lower_function(llbc, "celldict::<Impl>::length").expect("lower celldict::<Impl>::length");
    dump_if_requested("celldict_length", &graph);
    let mut live_reads = 0usize;
    let mut ll_vec_length = 0usize;
    count_length_ops(&graph, &mut live_reads, &mut ll_vec_length);
    let mut seen = HashSet::new();
    let mut pending = callee_fun_decl_ids(&graph);
    let context = LowerContext::new(llbc);
    let mut dumps = vec![graph.dump()];
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        let path = fd.item_meta.name_path();
        if !(path.contains("rordereddict") || path.contains("celldict")) {
            continue;
        }
        let Ok(callee) = lower_fun_decl(&context, fd) else {
            continue;
        };
        count_length_ops(&callee, &mut live_reads, &mut ll_vec_length);
        pending.extend(callee_fun_decl_ids(&callee));
        dumps.push(callee.dump());
    }
    let dump = dumps.join("\n--- callee ---\n");
    assert!(
        live_reads > 0,
        "module dict length must getfield num_live_items (`rordereddict.rtype_len`) somewhere in its call chain\n{dump}"
    );
    assert_eq!(
        ll_vec_length, 0,
        "module dict length must not call ll_vec_length of the first storage word\n{dump}"
    );
}
