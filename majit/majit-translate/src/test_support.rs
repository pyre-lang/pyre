//! Helpers shared by this crate's `#[test]` bodies.
//!
//! Nothing here is compiled into the library; the module exists so that two
//! tests in different files can share one synchronisation point, which a
//! helper private to one `mod tests` cannot provide.

use std::collections::HashMap;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;

use crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs;
use crate::front::mir::{LowerContext, LowerError, lower_fun_decl_with_static_addrs};
use crate::model::FunctionGraph;
use crate::runtime_names::artifacts;

static STRUCT_REGISTRY_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// One parsed artefact per test process. Keyed by path *suffix*, not by
/// `&'static str` pointer identity: each `concat!` site is a distinct
/// static, and matching those pointers would re-parse the same file.
/// Mirrors `tests/common/mod.rs` `llbc_slot`, plus jit and corpus.
fn llbc_slot(path: &str) -> &'static OnceLock<Llbc> {
    static INTERPRETER: OnceLock<Llbc> = OnceLock::new();
    static OBJECT: OnceLock<Llbc> = OnceLock::new();
    static MODULE: OnceLock<Llbc> = OnceLock::new();
    static JIT: OnceLock<Llbc> = OnceLock::new();
    static RLIB: OnceLock<Llbc> = OnceLock::new();
    static CORPUS: OnceLock<Llbc> = OnceLock::new();
    // `*.layouts.ullbc` sidecars share a crate stem; they are not these
    // artefacts (`layout.rs` `layouts_ullbc_filename`).
    if path.contains("layouts") {
        panic!("load_ullbc only knows the production artefacts, got {path}");
    }
    if path.ends_with("pyre-interpreter.ullbc") {
        &INTERPRETER
    } else if path.ends_with("pyre-object.ullbc") {
        &OBJECT
    } else if path.ends_with("pyre-module.ullbc") {
        &MODULE
    } else if path.ends_with("pyre-jit.ullbc") {
        &JIT
    } else if path.ends_with("majit-rlib.ullbc") {
        &RLIB
    } else if path.ends_with("corpus.ullbc") {
        &CORPUS
    } else {
        panic!("load_ullbc only knows the production artefacts, got {path}");
    }
}

fn context_slot(path: &str) -> &'static OnceLock<LowerContext<'static>> {
    static INTERPRETER: OnceLock<LowerContext<'static>> = OnceLock::new();
    static OBJECT: OnceLock<LowerContext<'static>> = OnceLock::new();
    static MODULE: OnceLock<LowerContext<'static>> = OnceLock::new();
    static JIT: OnceLock<LowerContext<'static>> = OnceLock::new();
    static RLIB: OnceLock<LowerContext<'static>> = OnceLock::new();
    static CORPUS: OnceLock<LowerContext<'static>> = OnceLock::new();
    if path.contains("layouts") {
        panic!("lower_context_for only knows the production artefacts, got {path}");
    }
    if path.ends_with("pyre-interpreter.ullbc") {
        &INTERPRETER
    } else if path.ends_with("pyre-object.ullbc") {
        &OBJECT
    } else if path.ends_with("pyre-module.ullbc") {
        &MODULE
    } else if path.ends_with("pyre-jit.ullbc") {
        &JIT
    } else if path.ends_with("majit-rlib.ullbc") {
        &RLIB
    } else if path.ends_with("corpus.ullbc") {
        &CORPUS
    } else {
        panic!("lower_context_for only knows the production artefacts, got {path}");
    }
}

/// Parse `path` once per test binary and share the artefact.
pub(crate) fn load_ullbc(path: &'static str) -> &'static Llbc {
    llbc_slot(path)
        .get_or_init(|| Llbc::load(path).unwrap_or_else(|err| panic!("load {path}: {err}")))
}

/// `None` when the artefact has not been extracted, so the caller can skip.
pub(crate) fn load_ullbc_if_present(path: &'static str) -> Option<&'static Llbc> {
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
        return None;
    }
    Some(load_ullbc(path))
}

const PRODUCTION_ULLBC_PATHS: &[&str] = &[
    artifacts::INTERPRETER_ULLBC,
    artifacts::OBJECT_ULLBC,
    artifacts::MODULE_ULLBC,
    artifacts::JIT_ULLBC,
    artifacts::MAJIT_RLIB_ULLBC,
    artifacts::CHARON_CORPUS_ULLBC,
];

fn as_production_llbc(llbc: &Llbc) -> Option<&'static Llbc> {
    for path in PRODUCTION_ULLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return Some(loaded);
        }
    }
    None
}

fn context_slot_for(llbc: &'static Llbc) -> &'static OnceLock<LowerContext<'static>> {
    for path in PRODUCTION_ULLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return context_slot(path);
        }
    }
    panic!("llbc is not a production artefact from load_ullbc")
}

fn make_lower_context(llbc: &'static Llbc) -> LowerContext<'static> {
    for path in PRODUCTION_ULLBC_PATHS {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return if path.ends_with("pyre-interpreter.ullbc")
                || path.ends_with("pyre-module.ullbc")
            {
                LowerContext::new_with_dependency_raw_flavors(
                    llbc,
                    &harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(load_ullbc(
                        artifacts::OBJECT_ULLBC,
                    ))),
                )
            } else {
                LowerContext::new(llbc)
            };
        }
    }
    panic!("llbc is not a production artefact from load_ullbc")
}

fn cached_lower_context(llbc: &'static Llbc) -> &'static LowerContext<'static> {
    context_slot_for(llbc)
        .get()
        .expect("ensure_link_baseline before reading the cached LowerContext")
}

/// Publish this artefact's own stack classification and `CodeFlags` once,
/// then fill the cached [`LowerContext`]. Holds the write lock without
/// restoring so the published baseline stays.
pub(crate) fn ensure_link_baseline(llbc: &'static Llbc) {
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
pub(crate) fn with_shared_lowering<R>(
    llbc: &'static Llbc,
    f: impl FnOnce(&'static LowerContext<'static>) -> R,
) -> R {
    ensure_link_baseline(llbc);
    with_llbc_link_read(std::iter::once(llbc), || f(cached_lower_context(llbc)))
}

/// Shared artefacts use the cached context under the read lock; fixtures
/// build a fresh [`LowerContext`].
pub(crate) fn with_lowering<R>(llbc: &Llbc, f: impl FnOnce(&LowerContext<'_>) -> R) -> R {
    if let Some(shared) = as_production_llbc(llbc) {
        with_shared_lowering(shared, f)
    } else {
        let ctx = LowerContext::new(llbc);
        f(&ctx)
    }
}

pub(crate) fn with_llbc_link_write<'a, R>(
    llbcs: impl IntoIterator<Item = &'a Llbc>,
    f: impl FnOnce() -> R,
) -> R {
    majit_charon_reader::with_llbc_link_write(llbcs, f)
}

pub(crate) fn with_llbc_link_read<'a, R>(
    llbcs: impl IntoIterator<Item = &'a Llbc>,
    f: impl FnOnce() -> R,
) -> R {
    majit_charon_reader::with_llbc_link_read(llbcs, f)
}

/// Snapshot/restore shared artefacts under write. Fixtures run `f` as-is.
pub(crate) fn with_shared_link_write<'a, R>(
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

pub(crate) fn lower_function(
    llbc: &Llbc,
    function_name: &str,
) -> Result<FunctionGraph, LowerError> {
    lower_function_with_static_addrs(llbc, function_name, crate::HostStaticAddrs::default())
}

pub(crate) fn lower_function_with_static_addrs(
    llbc: &Llbc,
    function_name: &str,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    if let Some(shared) = as_production_llbc(llbc) {
        with_shared_lowering(shared, |ctx| {
            let fd = shared
                .local_fn(function_name)
                .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
            lower_fun_decl_with_static_addrs(ctx, fd, static_addrs)
        })
    } else {
        crate::front::mir::lower_function_with_static_addrs(llbc, function_name, static_addrs)
    }
}

pub(crate) fn build_semantic_program_from_llbc(
    llbc: &Llbc,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    with_shared_link_write(std::iter::once(llbc), || {
        crate::front::mir::build_semantic_program_from_llbc(llbc)
    })
}

pub(crate) fn build_semantic_program_from_llbcs(
    llbcs: &[Llbc],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    with_shared_link_write(llbcs.iter(), || {
        crate::front::mir::build_semantic_program_from_llbcs(llbcs)
    })
}

pub(crate) fn build_semantic_program_from_llbcs_with_static_addrs_and_function_names<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
    function_names: &[&str],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    with_shared_link_write(llbcs.iter().map(std::borrow::Borrow::borrow), || {
        crate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
            llbcs,
            static_addrs,
            module_paths,
            function_names,
        )
    })
}

#[allow(dead_code)]
pub(crate) fn build_semantic_program_from_llbcs_with_static_addrs_and_module_paths<
    L: std::borrow::Borrow<Llbc>,
>(
    llbcs: &[L],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    with_shared_link_write(llbcs.iter().map(std::borrow::Borrow::borrow), || {
        crate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
            llbcs,
            static_addrs,
            module_paths,
        )
    })
}

pub(crate) fn corpus_ullbc() -> &'static Llbc {
    load_ullbc(artifacts::CHARON_CORPUS_ULLBC)
}

/// Publish `table` into the process-global name → `StructId` map and hold
/// every other registering test out until the returned guard drops.
///
/// `register_struct_ids` REPLACES the whole table (`majit-ir/src/descr.rs`,
/// `*guard = table`), which is right for production — the front end populates
/// it once per program, and merging would leak one program's names into the
/// next — and hostile to `cargo test`'s default parallelism, where a second
/// registering test wipes the first one's only entry while that test is still
/// running. The victim does not fail where it registered: its lookup silently
/// takes a different arm and returns a wrong answer.
///
/// Measured, not inferred: two registering tests, run as a pair with default
/// threads, failed 9 of 12 runs; in the full lib-test binary the same race
/// surfaced once in 6, because the scheduler rarely puts them adjacent. Both
/// pass in isolation and under `--test-threads=1`, so neither the isolated run
/// nor the serial run can see this.
///
/// Scope is one test BINARY, which is the whole racing population: other
/// crates' tests run in their own processes and cannot reach this table. That
/// is also why the lock lives here rather than in one file's `mod tests` — a
/// second file's writer would otherwise take a different lock and race anyway.
///
/// The lock is poison-tolerant. A test that panics mid-body would otherwise
/// convert one real failure into a cascade of unrelated ones.
///
/// Bind the result to a NAMED local (`let _registry = …`). A bare `let _ = …`
/// drops the guard at once and restores exactly the race this exists to
/// remove — while still compiling, and still passing in isolation.
#[must_use = "the returned guard holds the registry lock for the rest of \
              the test; dropping it immediately re-opens the race"]
/// Do not hold this guard while calling another registrar: the lock is not
/// reentrant. A test that must publish both tables should use
/// [`register_struct_registries_serialized`].
pub(crate) fn register_struct_ids_serialized(
    table: HashMap<String, Option<majit_ir::descr::StructId>>,
) -> parking_lot::MutexGuard<'static, ()> {
    let guard = STRUCT_REGISTRY_TEST_LOCK.lock();
    majit_ir::descr::register_struct_ids(table);
    guard
}

/// Publish an origin table while excluding tests that replace either global
/// struct-name registry. Production publishes both together for one program;
/// tests must likewise prevent one table from changing under another test's
/// descriptor construction.
#[must_use = "the returned guard holds the registry lock for the rest of the test"]
pub(crate) fn register_struct_origins_serialized(
    table: HashMap<String, String>,
) -> parking_lot::MutexGuard<'static, ()> {
    let guard = STRUCT_REGISTRY_TEST_LOCK.lock();
    majit_ir::descr::register_struct_origins(table);
    guard
}

/// Publish both global struct-name registries under one acquisition of their
/// non-reentrant test lock. A test that needs both tables must use this helper
/// instead of retaining one single-table guard while requesting the other.
#[must_use = "the returned guard holds the registry lock for the rest of the test"]
pub(crate) fn register_struct_registries_serialized(
    ids: HashMap<String, Option<majit_ir::descr::StructId>>,
    origins: HashMap<String, String>,
) -> parking_lot::MutexGuard<'static, ()> {
    let guard = STRUCT_REGISTRY_TEST_LOCK.lock();
    majit_ir::descr::register_struct_ids(ids);
    majit_ir::descr::register_struct_origins(origins);
    guard
}
