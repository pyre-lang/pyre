//! Helpers shared by this crate's `#[test]` bodies.
//!
//! Nothing here is compiled into the library; the module exists so that two
//! tests in different files can share one synchronisation point, which a
//! helper private to one `mod tests` cannot provide.

use std::collections::HashMap;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;

use crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs;
use crate::front::mir::LowerContext;
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

/// Derive [`LowerContext`] once per artefact in this test binary.
///
/// Mirrors `tests/common/mod.rs` `lower_context_for`. `LowerContext` is
/// `Sync` so tests can park it in `OnceLock` (`front::mir::LowerContext`).
pub(crate) fn lower_context_for(llbc: &'static Llbc) -> &'static LowerContext<'static> {
    for path in [
        artifacts::INTERPRETER_ULLBC,
        artifacts::OBJECT_ULLBC,
        artifacts::MODULE_ULLBC,
        artifacts::JIT_ULLBC,
        artifacts::MAJIT_RLIB_ULLBC,
        artifacts::CHARON_CORPUS_ULLBC,
    ] {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return context_slot(path).get_or_init(|| {
                if path.ends_with("pyre-interpreter.ullbc") || path.ends_with("pyre-module.ullbc") {
                    LowerContext::new_with_dependency_raw_flavors(
                        llbc,
                        &harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(load_ullbc(
                            artifacts::OBJECT_ULLBC,
                        ))),
                    )
                } else {
                    LowerContext::new(llbc)
                }
            });
        }
    }
    panic!("lower_context_for: llbc is not a production artefact from load_ullbc")
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
