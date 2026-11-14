//! Shared helpers for the majit-translate integration tests.
//!
//! Each test crate includes this module and uses a subset of it.
#![expect(dead_code)]

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::flowspace::bytecode::ConstantData;
use majit_translate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs;
use majit_translate::front::mir::{LowerContext, LowerError, lower_fun_decl_with_static_addrs};
use majit_translate::front::semantic::SemanticProgram;
use majit_translate::model::FunctionGraph;
use rustpython_compiler::{Mode, compile as rp_compile};
use rustpython_compiler_core::bytecode::CodeObject;
use std::sync::OnceLock;

pub const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-interpreter.ullbc"
);
pub const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
);
pub const MODULE_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-module.ullbc"
);
pub const RLIB_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/majit-rlib.ullbc"
);
pub const JIT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-jit.ullbc"
);
pub const CORPUS_LLBC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");

/// Compile `src` as an exec-mode module and return the first nested code
/// object (the first function body). Panics if compilation fails or the
/// source contains no function body.
pub fn compile_first_code(src: &str) -> CodeObject {
    let module =
        rp_compile(src, Mode::Exec, "<flow>", Default::default()).expect("compile should succeed");
    module
        .constants
        .iter()
        .find_map(|c| match c {
            ConstantData::Code { code } => Some((**code).clone()),
            _ => None,
        })
        .expect("source should contain at least one function body")
}

fn llbc_slot(path: &'static str) -> &'static OnceLock<Llbc> {
    static INTERPRETER: OnceLock<Llbc> = OnceLock::new();
    static OBJECT: OnceLock<Llbc> = OnceLock::new();
    static MODULE: OnceLock<Llbc> = OnceLock::new();
    static RLIB: OnceLock<Llbc> = OnceLock::new();
    static JIT: OnceLock<Llbc> = OnceLock::new();
    static CORPUS: OnceLock<Llbc> = OnceLock::new();
    match path {
        INTERPRETER_LLBC => &INTERPRETER,
        OBJECT_LLBC => &OBJECT,
        MODULE_LLBC => &MODULE,
        RLIB_LLBC => &RLIB,
        JIT_LLBC => &JIT,
        CORPUS_LLBC => &CORPUS,
        other => panic!("load_llbc only knows the production artefacts, got {other}"),
    }
}

fn context_slot(path: &'static str) -> &'static OnceLock<LowerContext<'static>> {
    static INTERPRETER: OnceLock<LowerContext<'static>> = OnceLock::new();
    static OBJECT: OnceLock<LowerContext<'static>> = OnceLock::new();
    static MODULE: OnceLock<LowerContext<'static>> = OnceLock::new();
    static RLIB: OnceLock<LowerContext<'static>> = OnceLock::new();
    static JIT: OnceLock<LowerContext<'static>> = OnceLock::new();
    static CORPUS: OnceLock<LowerContext<'static>> = OnceLock::new();
    match path {
        INTERPRETER_LLBC => &INTERPRETER,
        OBJECT_LLBC => &OBJECT,
        MODULE_LLBC => &MODULE,
        RLIB_LLBC => &RLIB,
        JIT_LLBC => &JIT,
        CORPUS_LLBC => &CORPUS,
        other => panic!("lower_context_for only knows the production artefacts, got {other}"),
    }
}

const PRODUCTION_LLBC_PATHS: &[&str] = &[
    INTERPRETER_LLBC,
    OBJECT_LLBC,
    MODULE_LLBC,
    RLIB_LLBC,
    JIT_LLBC,
    CORPUS_LLBC,
];

fn as_production_llbc(llbc: &Llbc) -> Option<&'static Llbc> {
    for path in PRODUCTION_LLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return Some(loaded);
        }
    }
    None
}

fn context_slot_for(llbc: &'static Llbc) -> &'static OnceLock<LowerContext<'static>> {
    for path in PRODUCTION_LLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return context_slot(path);
        }
    }
    panic!("llbc is not a production artefact from load_llbc")
}

fn make_lower_context(llbc: &'static Llbc) -> LowerContext<'static> {
    for path in PRODUCTION_LLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return if *path == INTERPRETER_LLBC || *path == MODULE_LLBC {
                LowerContext::new_with_dependency_raw_flavors(
                    llbc,
                    &harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(load_llbc(
                        OBJECT_LLBC,
                    ))),
                )
            } else {
                LowerContext::new(llbc)
            };
        }
    }
    panic!("llbc is not a production artefact from load_llbc")
}

fn cached_lower_context(llbc: &'static Llbc) -> &'static LowerContext<'static> {
    context_slot_for(llbc)
        .get()
        .expect("ensure_link_baseline before reading the cached LowerContext")
}

/// Publish this artefact's own stack classification and `CodeFlags` once,
/// then fill the cached [`LowerContext`]. Holds the write lock without
/// restoring so the published baseline stays.
pub fn ensure_link_baseline(llbc: &'static Llbc) {
    let slot = context_slot_for(llbc);
    if slot.get().is_some() {
        return;
    }
    majit_charon_reader::with_llbc_link_write(std::iter::empty::<&Llbc>(), || {
        if slot.get().is_some() {
            return;
        }
        let _ = slot.get_or_init(|| make_lower_context(llbc));
    });
}

/// `ensure_link_baseline` then the cached context, under the read lock.
pub fn with_shared_lowering<R>(
    llbc: &'static Llbc,
    f: impl FnOnce(&'static LowerContext<'static>) -> R,
) -> R {
    ensure_link_baseline(llbc);
    with_llbc_link_read(std::iter::once(llbc), || f(cached_lower_context(llbc)))
}

/// Shared artefacts use the cached context under the read lock; fixtures
/// build a fresh [`LowerContext`].
pub fn with_lowering<R>(llbc: &Llbc, f: impl FnOnce(&LowerContext<'_>) -> R) -> R {
    if let Some(shared) = as_production_llbc(llbc) {
        with_shared_lowering(shared, f)
    } else {
        let ctx = LowerContext::new(llbc);
        f(&ctx)
    }
}

/// `ensure_link_baseline` for each artefact, then the cached contexts under
/// one read lock. Census workers may copy the `'static` refs; they must not
/// take the write lock while this read is held.
pub fn with_shared_lowering_many<R>(
    llbcs: &[&'static Llbc],
    f: impl FnOnce(&[&'static LowerContext<'static>]) -> R,
) -> R {
    for llbc in llbcs {
        ensure_link_baseline(llbc);
    }
    with_llbc_link_read(llbcs.iter().copied(), || {
        let ctxs: Vec<&'static LowerContext<'static>> =
            llbcs.iter().copied().map(cached_lower_context).collect();
        f(&ctxs)
    })
}

pub fn with_llbc_link_write<'a, R>(
    llbcs: impl IntoIterator<Item = &'a Llbc>,
    f: impl FnOnce() -> R,
) -> R {
    majit_charon_reader::with_llbc_link_write(llbcs, f)
}

pub fn with_llbc_link_read<'a, R>(
    llbcs: impl IntoIterator<Item = &'a Llbc>,
    f: impl FnOnce() -> R,
) -> R {
    majit_charon_reader::with_llbc_link_read(llbcs, f)
}

/// Snapshot `llbc`'s link side tables, run `f`, and restore them afterwards
/// (including when `f` panics).
pub fn with_llbc_link_state<R>(llbc: &'static Llbc, f: impl FnOnce() -> R) -> R {
    ensure_link_baseline(llbc);
    with_llbc_link_write(std::iter::once(llbc), f)
}

pub fn with_llbc_link_state_many<R>(llbcs: &[&'static Llbc], f: impl FnOnce() -> R) -> R {
    for llbc in llbcs {
        ensure_link_baseline(llbc);
    }
    with_llbc_link_write(llbcs.iter().copied(), f)
}

fn with_shared_link_write<'a, R>(
    llbcs: impl IntoIterator<Item = &'a Llbc>,
    f: impl FnOnce() -> R,
) -> R {
    let llbcs: Vec<&Llbc> = llbcs.into_iter().collect();
    let shared: Vec<&'static Llbc> = llbcs
        .iter()
        .copied()
        .filter_map(as_production_llbc)
        .collect();
    if shared.is_empty() {
        f()
    } else {
        for llbc in &shared {
            ensure_link_baseline(llbc);
        }
        with_llbc_link_write(shared, f)
    }
}

/// Parse `path` once per test binary and share the artefact.
pub fn load_llbc(path: &'static str) -> &'static Llbc {
    llbc_slot(path).get_or_init(|| {
        Llbc::load(path).unwrap_or_else(|err| {
            panic!("{path} is missing or unreadable; run `python3 scripts/extract-llbc.py`: {err}")
        })
    })
}

/// `None` when the artefact has not been extracted, so the caller can skip.
pub fn load_llbc_if_present(path: &'static str) -> Option<&'static Llbc> {
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
        return None;
    }
    Some(load_llbc(path))
}

/// Lower one named declaration, reusing the artefact's shared context.
pub fn lower_named(llbc: &'static Llbc, function_name: &str) -> Result<FunctionGraph, LowerError> {
    lower_named_with_static_addrs(llbc, function_name, HostStaticAddrs::default())
}

pub fn lower_named_with_static_addrs(
    llbc: &'static Llbc,
    function_name: &str,
    static_addrs: HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    with_shared_lowering(llbc, |ctx| {
        let fd = llbc
            .local_fn(function_name)
            .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
        lower_fun_decl_with_static_addrs(ctx, fd, static_addrs)
    })
}

pub fn lower_function(llbc: &Llbc, function_name: &str) -> Result<FunctionGraph, LowerError> {
    lower_function_with_static_addrs(llbc, function_name, HostStaticAddrs::default())
}

pub fn lower_function_with_static_addrs(
    llbc: &Llbc,
    function_name: &str,
    static_addrs: HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    if let Some(shared) = as_production_llbc(llbc) {
        with_shared_lowering(shared, |ctx| {
            let fd = shared
                .local_fn(function_name)
                .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
            lower_fun_decl_with_static_addrs(ctx, fd, static_addrs)
        })
    } else {
        majit_translate::front::mir::lower_function_with_static_addrs(
            llbc,
            function_name,
            static_addrs,
        )
    }
}

pub fn build_semantic_program_from_llbc(llbc: &Llbc) -> Result<SemanticProgram, LowerError> {
    with_shared_link_write(std::iter::once(llbc), || {
        majit_translate::front::mir::build_semantic_program_from_llbc(llbc)
    })
}

pub fn build_semantic_program_from_llbcs_with_static_addrs_and_function_names<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: HostStaticAddrs<'_>,
    module_paths: &[&str],
    function_names: &[&str],
) -> Result<SemanticProgram, LowerError> {
    with_shared_link_write(llbcs.iter().map(std::borrow::Borrow::borrow), || {
        majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
            llbcs,
            static_addrs,
            module_paths,
            function_names,
        )
    })
}

pub fn build_semantic_program_from_llbcs_with_static_addrs_and_module_paths<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: HostStaticAddrs<'_>,
    module_paths: &[&str],
) -> Result<SemanticProgram, LowerError> {
    with_shared_link_write(llbcs.iter().map(std::borrow::Borrow::borrow), || {
        majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
            llbcs,
            static_addrs,
            module_paths,
        )
    })
}

pub fn interpreter_llbc() -> &'static Llbc {
    load_llbc(INTERPRETER_LLBC)
}

pub fn object_llbc() -> &'static Llbc {
    load_llbc(OBJECT_LLBC)
}

pub fn module_llbc() -> &'static Llbc {
    load_llbc(MODULE_LLBC)
}

pub fn jit_llbc() -> &'static Llbc {
    load_llbc(JIT_LLBC)
}

/// rbigint bodies live in `majit-rlib.ullbc`; their callers live in
/// `pyre-object.ullbc`. Owner first, matching the front-end merge.
pub fn rbigint_llbcs() -> &'static [&'static Llbc] {
    static SLOT: OnceLock<Vec<&'static Llbc>> = OnceLock::new();
    SLOT.get_or_init(|| vec![load_llbc(RLIB_LLBC), load_llbc(OBJECT_LLBC)])
        .as_slice()
}
