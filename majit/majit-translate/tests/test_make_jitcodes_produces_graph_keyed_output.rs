//! Structural anchor for portal-closure `make_jitcodes` output.
//!
//! Runs as
//! `cargo test -p majit-translate test_make_jitcodes_produces_graph_keyed_output`.
//! The compact canonical-pipeline fixture lives in the crate unit test
//! `portal_driver_tests::make_jitcodes_compiles_a_registered_portal_graph`;
//! this integration test retains the release-only full-LLBC acceptance check.
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/codewriter.py make_jitcodes` — the
//!   main driver that pops from `callcontrol.enum_pending_graphs()`
//!   and produces one `JitCode` per graph.
//! - `rpython/jit/codewriter/codewriter.py transform_graph_to_jitcode` —
//!   the 4-step pipeline (jtransform → regalloc → flatten → assemble)
//!   that turns one FunctionGraph into one JitCode.
//! - `rpython/jit/codewriter/call.py CallControl.__init__ self.jitcodes = {}` — the
//!   graph-keyed dict that `AllJitCodes::by_path` mirrors.
//! - `rpython/jit/codewriter/call.py self.all_jitcodes = []` — the
//!   alloc-order list that `AllJitCodes::in_order` mirrors, with the
//!   `all_jitcodes[i].index == i` invariant from `codewriter.py`.
//!
//! ## What this test anchors
//!
//! `make_jitcodes` output is graph-keyed: one `JitCode` per `CallPath`, with
//! no `Instruction`-variant tables or opcode-to-fragment lookup. Portal and
//! dispatch functions participate as ordinary graphs in the same closure.
//! The structural prohibition is that output remains `{graph: JitCode}` only,
//! with no variant-keyed map or synthetic dispatch root.

use majit_translate::{
    CallPath, HostStaticAddrs,
    codewriter::call::access_directly_origin_path,
    flowspace::model::ConstValue,
    front::{
        mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths,
        semantic::MirGraphLookup,
    },
    generated::{AllJitCodes, with_all_jitcodes},
    jitcode::JitCode,
    model::{ExitCase, ExitSwitch},
};
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

#[test]
fn test_make_jitcodes_produces_graph_keyed_output() {
    // The default-CI canonical pipeline witness is the compact unit fixture
    // named above. This integration-level floor only guards against restoring
    // a parallel instruction-keyed output representation.
    assert_no_instruction_keyed_output_map();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_make_jitcodes_produces_graph_keyed_output`"
)]
fn slow_generated_jitcodes_preserve_complete_dispatcher_graph() {
    if !ensure_workspace_llbc_env() {
        eprintln!(
            "skipping: target/llbc/{{pyre-object,pyre-interpreter,pyre-jit}}.ullbc missing — \
             run `scripts/extract-llbc.py` to enable this test"
        );
        return;
    }

    // The runtime artifact stays graph-keyed, but the acceptance test must
    // still prove that lowering preserved the complete interpreter dispatch.
    // Cross-check source match arms against target-sharing MIR switch exits:
    // this catches a dropped switch branch, an accidentally split/merged
    // Or-pattern, or a missing wildcard without publishing an opcode side
    // table to production consumers.
    let llbc_paths: Vec<std::path::PathBuf> = std::env::split_paths(
        &std::env::var_os("MAJIT_MIR_FRONTEND_LLBC")
            .expect("ensure_workspace_llbc_env installs the LLBC path list"),
    )
    .collect();
    let mut ordered_paths = llbc_paths;
    ordered_paths.sort_by_key(|path| {
        !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("pyre-interpreter"))
    });
    let mut dispatcher_program = None;
    for path in ordered_paths {
        let llbc = majit_charon_reader::Llbc::load(&path)
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()));
        if llbc.local_fn("execute_opcode_step").is_some() {
            dispatcher_program = Some(
                build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
                    &[llbc],
                    HostStaticAddrs::default(),
                    &["pyopcode"],
                )
                .expect("lower pyopcode dispatcher from interpreter LLBC"),
            );
            break;
        }
    }
    let program = dispatcher_program.expect("an LLBC input must define execute_opcode_step");
    let lookup = MirGraphLookup::from_program(&program);
    let dispatcher = lookup
        .lookup_free("execute_opcode_step")
        .and_then(|graph| graph.get())
        .expect("execute_opcode_step must lower to one unambiguous free-function graph");
    let switch = dispatcher.block(dispatcher.startblock);
    assert!(
        matches!(switch.exitswitch, Some(ExitSwitch::Value(_))),
        "execute_opcode_step must start with an Instruction discriminant switch",
    );
    let variants = program
        .enum_variant_by_discriminant
        .get("Instruction")
        .expect("Instruction discriminants must be present in the semantic program");
    let mut seen_discriminants = HashSet::new();
    let mut mir_groups: Vec<(majit_translate::BlockId, BTreeSet<String>)> = Vec::new();
    let mut mir_wildcards = 0usize;
    let mut mir_wildcard_target = None;
    for link in &switch.exits {
        let discriminant = match &link.exitcase {
            Some(ExitCase::Const(ConstValue::Int(discriminant))) => *discriminant,
            // An unsigned `Instruction` repr lowers its cases as `r_uint`.
            Some(ExitCase::Const(ConstValue::UInt(discriminant))) => {
                i64::try_from(*discriminant).expect("Instruction discriminant fits in i64")
            }
            Some(ExitCase::Const(ConstValue::UniStr(value))) if value == "default" => {
                mir_wildcards += 1;
                assert!(
                    mir_wildcard_target.replace(link.target).is_none(),
                    "the MIR switch contains more than one wildcard target",
                );
                continue;
            }
            other => panic!("unexpected execute_opcode_step switch case: {other:?}"),
        };
        assert!(
            seen_discriminants.insert(discriminant),
            "Instruction discriminant {discriminant} appears twice in the MIR switch",
        );
        let variant = variants.get(&discriminant).unwrap_or_else(|| {
            panic!("MIR switch discriminant {discriminant} has no Instruction variant name")
        });
        if let Some((_, group)) = mir_groups
            .iter_mut()
            .find(|(target, _)| *target == link.target)
        {
            assert!(group.insert(variant.clone()));
        } else {
            mir_groups.push((link.target, BTreeSet::from([variant.clone()])));
        }
    }

    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let source =
        std::fs::read_to_string(workspace_root.join("pyre/pyre-interpreter/src/pyopcode.rs"))
            .expect("read execute_opcode_step source");
    let syntax = syn::parse_file(&source).expect("parse pyopcode.rs");
    let function = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "execute_opcode_step" => {
                Some(function)
            }
            _ => None,
        })
        .expect("pyopcode.rs defines execute_opcode_step");
    let match_expr = function
        .block
        .stmts
        .iter()
        .find_map(|statement| match statement {
            syn::Stmt::Expr(syn::Expr::Match(match_expr), _) => Some(match_expr),
            _ => None,
        })
        .expect("execute_opcode_step contains its dispatcher match");
    let mut source_groups = Vec::new();
    let mut source_wildcards = 0usize;
    for arm in &match_expr.arms {
        let mut pending = vec![&arm.pat];
        let mut group = BTreeSet::new();
        let mut wildcard = false;
        while let Some(pattern) = pending.pop() {
            match pattern {
                syn::Pat::Or(or_pattern) => pending.extend(or_pattern.cases.iter()),
                syn::Pat::Paren(paren) => pending.push(&paren.pat),
                syn::Pat::Path(path) => {
                    let variant = path
                        .path
                        .segments
                        .last()
                        .expect("Instruction pattern path has a variant")
                        .ident
                        .to_string();
                    assert!(group.insert(variant));
                }
                syn::Pat::Struct(struct_pattern) => {
                    let variant = struct_pattern
                        .path
                        .segments
                        .last()
                        .expect("Instruction struct pattern has a variant")
                        .ident
                        .to_string();
                    assert!(group.insert(variant));
                }
                syn::Pat::TupleStruct(tuple_pattern) => {
                    let variant = tuple_pattern
                        .path
                        .segments
                        .last()
                        .expect("Instruction tuple pattern has a variant")
                        .ident
                        .to_string();
                    assert!(group.insert(variant));
                }
                syn::Pat::Wild(_) => wildcard = true,
                other => panic!(
                    "unexpected execute_opcode_step source pattern kind: {:?}",
                    std::mem::discriminant(other),
                ),
            }
        }
        if wildcard {
            source_wildcards += 1;
            assert!(group.is_empty(), "wildcard arm must not mix named variants");
        } else {
            assert!(
                !group.is_empty(),
                "dispatcher arm has no Instruction variants"
            );
            source_groups.push(group);
        }
    }

    assert!(
        mir_groups
            .iter()
            .all(|(target, _)| Some(*target) != mir_wildcard_target),
        "the wildcard and an explicit Instruction case share a target",
    );
    let mut mir_groups: Vec<Vec<String>> = mir_groups
        .into_iter()
        .map(|(_, group)| group.into_iter().collect())
        .collect();
    let mut source_groups: Vec<Vec<String>> = source_groups
        .into_iter()
        .map(|group| group.into_iter().collect())
        .collect();
    mir_groups.sort();
    source_groups.sort();
    assert_eq!(mir_wildcards, 1, "MIR dispatcher must retain one wildcard");
    assert_eq!(
        source_wildcards, 1,
        "source dispatcher must declare one wildcard"
    );
    // Floors rather than exact totals: the dispatcher grows whenever an opcode
    // is added, so an equality here fails on every legitimate addition while
    // still only catching removals. Shrinkage is the signal. Same rationale as
    // the `reg.in_order.len() >= 28` floor below. Raise these when the
    // dispatcher grows; lowering one means naming the opcodes it dropped.
    assert!(
        source_groups.len() + source_wildcards >= 115,
        "source dispatcher lost arms: {} < 115",
        source_groups.len() + source_wildcards,
    );
    assert!(
        seen_discriminants.len() >= 119,
        "MIR switch lost Instruction discriminants: {} < 119",
        seen_discriminants.len(),
    );
    // The exact anti-drop check. Both sides are sorted arm groups, so this
    // pins the arm count, every variant's arm membership, and which variants
    // share one arm — a dropped switch branch or a split/merged Or-pattern
    // shows up here. The per-side totals above stay floors because of that.
    assert_eq!(
        mir_groups, source_groups,
        "lowered MIR dispatcher groups differ from the source match",
    );

    with_all_jitcodes(|reg| {
        assert_registry_is_graph_keyed(reg);

        // Lower floor is intentional: the upper bound grows as more ordinary
        // helpers become reachable from the portal. A count regression is the
        // signal; growth is valid.
        assert!(
            reg.in_order.len() >= 28,
            "portal closure unexpectedly small"
        );
        let portal_path = CallPath::from_segments(["eval", "eval_loop_jit"]);
        let portal = reg
            .by_path
            .get(&portal_path)
            .expect("the exact configured eval::eval_loop_jit portal must be compiled");
        assert!(portal.jitdriver_sd().is_some());

        // PyPy's portal `dispatch` hints the frame `access_directly=True`
        // (`interp_jit.py dispatch`). `specialize.py default_specialize`
        // then compiles callees reached with that frame as the
        // `(AccessDirect, key)` graph; `cachedgraph` registers it under
        // alt_name `{name}__AccessDirect_None`. The regular graph is only
        // in the closure when an unflagged frame also calls it.
        for required_leaf in ["execute_opcode_step", "execute_pop_top"] {
            assert!(
                reg.by_path.keys().any(|path| {
                    access_directly_origin_path(path)
                        .as_ref()
                        .unwrap_or(path)
                        .last_segment()
                        == Some(required_leaf)
                }),
                "ordinary portal closure must contain `{required_leaf}`",
            );
        }
        assert!(
            reg.by_path.keys().all(|path| {
                path.segments
                    .first()
                    .is_none_or(|segment| segment != "__opcode_dispatch__")
            }),
            "portal closure must not contain synthetic dispatch roots",
        );
    });

    assert_no_instruction_keyed_output_map();
}

fn assert_registry_is_graph_keyed(reg: &AllJitCodes) {
    // `AllJitCodes::by_path` is keyed by `CallPath` (graph identity),
    // matching upstream `call.py:87 self.jitcodes`. The field's type
    // ensures no Instruction-variant key can land here; the runtime checks
    // below pin the registry's two views to the same JitCode objects.
    assert!(!reg.in_order.is_empty(), "registry should not be empty");

    // Invariant 1: every `in_order` entry is also reachable through
    // `by_path`. Together with invariant 2 they pin the 1:1 mapping
    // between CallPath and JitCode (upstream `codewriter.py:80-81`).
    let in_order_ptrs: HashSet<usize> = reg
        .in_order
        .iter()
        .map(|jc: &Arc<JitCode>| Arc::as_ptr(jc) as usize)
        .collect();
    let by_path_ptrs: HashSet<usize> = reg
        .by_path
        .values()
        .map(|jc| Arc::as_ptr(jc) as usize)
        .collect();
    assert!(
        in_order_ptrs.is_subset(&by_path_ptrs),
        "every JitCode in `in_order` must be reachable via `by_path` \
         (RPython `call.py:87-88` parity)"
    );

    // Invariant 2: each JitCode Arc appears in `by_path` under exactly
    // one CallPath. Duplicate entries would mean a graph was registered
    // under two different paths, which breaks upstream's `{graph:
    // JitCode}` identity contract at `call.py:157-165`.
    let mut seen: HashSet<usize> = HashSet::new();
    for (path, jc) in &reg.by_path {
        let ptr = Arc::as_ptr(jc) as usize;
        assert!(
            seen.insert(ptr),
            "JitCode `{}` appears in `by_path` under multiple CallPath \
             keys — last seen at {path:?}",
            jc.name
        );
    }
}

fn assert_no_instruction_keyed_output_map() {
    // Registry output is not keyed by Instruction. `by_path` enforces this at
    // the type level; the source scan additionally prevents a parallel
    // variant-keyed output map or synthetic dispatch namespace from returning.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    walk_rs_files(&root, &mut |path, contents| {
        for (i, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                continue;
            }
            if line.contains("HashMap<Instruction") || line.contains("__opcode_dispatch__") {
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "JitCode output must be graph-keyed only:\n{}",
        offenders.join("\n")
    );
}

/// Resolve workspace LLBC artefacts and export `MAJIT_MIR_FRONTEND_LLBC`
/// when they exist. The compact `JIT_GRAPH_MODULES` fixture cannot rely on
/// production auto-discovery, so this release-only test opts in explicitly.
/// Returns `false` when the artefacts are absent so the caller can skip.
fn ensure_workspace_llbc_env() -> bool {
    if std::env::var_os("MAJIT_MIR_FRONTEND_LLBC").is_some() {
        return true;
    }
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = workspace_root.join("target").join("llbc");
    let required = [
        "pyre-object.ullbc",
        "pyre-interpreter.ullbc",
        "pyre-jit.ullbc",
    ];
    let mut paths = Vec::with_capacity(required.len());
    for name in required {
        let p = llbc_dir.join(name);
        if !p.exists() {
            return false;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    let joined = std::env::join_paths(&paths).expect("join llbc paths");
    // `join_paths` uses the OS path-list separator (`;` on Windows, `:`
    // elsewhere), so Windows drive letters survive lib.rs's `split_paths`.
    //
    // SAFETY: `set_var` is unsafe in Rust 2024 because concurrent environment
    // mutation races. This release-only integration test configures the
    // process before the thread-local generated registry is initialized.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };
    true
}

fn walk_rs_files<F: FnMut(&std::path::Path, &str)>(dir: &std::path::Path, f: &mut F) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && let Ok(contents) = std::fs::read_to_string(&path)
        {
            f(&path, &contents);
        }
    }
}
