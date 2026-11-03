//! Shared helpers for the majit-translate integration tests.
//!
//! Each test crate includes this module and uses a subset of it.
#![expect(dead_code)]

use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::flowspace::bytecode::ConstantData;
use majit_translate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs;
use majit_translate::front::mir::{LowerContext, LowerError, lower_fun_decl_with_static_addrs};
use majit_translate::model::FunctionGraph;
use rustpython_compiler::{Mode, compile as rp_compile};
use rustpython_compiler_core::bytecode::CodeObject;
use std::cell::Cell;
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

static LLBC_LINK_STATE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
thread_local! {
    static LLBC_LINK_STATE_HELD: Cell<bool> = const { Cell::new(false) };
}

/// Hold the process-wide link-state lock for the duration of `f`.
///
/// Nested calls on the same thread skip a second acquire so a census worker
/// parent can wrap several artefacts, and so [`with_llbc_link_state`] can
/// snapshot more than one parse under one lock.
pub fn with_llbc_link_lock<R>(f: impl FnOnce() -> R) -> R {
    LLBC_LINK_STATE_HELD.with(|held| {
        if held.get() {
            f()
        } else {
            let _mutex = LLBC_LINK_STATE_LOCK.lock();
            held.set(true);
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    LLBC_LINK_STATE_HELD.with(|held| held.set(false));
                }
            }
            let _reset = Reset;
            f()
        }
    })
}

/// Snapshot `llbc`'s link side tables, run `f`, and restore them afterwards
/// (including when `f` panics). Holds [`with_llbc_link_lock`] for the call.
pub fn with_llbc_link_state<R>(llbc: &'static Llbc, f: impl FnOnce() -> R) -> R {
    with_llbc_link_lock(|| {
        let snapshot = llbc.snapshot_link_state();
        struct Restore {
            llbc: &'static Llbc,
            snapshot: Option<majit_charon_reader::LinkStateSnapshot>,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                if let Some(snapshot) = self.snapshot.take() {
                    self.llbc.restore_link_state(snapshot);
                }
            }
        }
        let _restore = Restore {
            llbc,
            snapshot: Some(snapshot),
        };
        f()
    })
}

/// Parse `path` once per test binary and share the artefact.
pub fn load_llbc(path: &'static str) -> &'static Llbc {
    llbc_slot(path)
        .get_or_init(|| Llbc::load(path).unwrap_or_else(|err| panic!("load {path}: {err}")))
}

/// `None` when the artefact has not been extracted, so the caller can skip.
pub fn load_llbc_if_present(path: &'static str) -> Option<&'static Llbc> {
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
        return None;
    }
    Some(load_llbc(path))
}

/// Derive [`LowerContext`] once per artefact in this test binary.
pub fn lower_context_for(llbc: &'static Llbc) -> &'static LowerContext<'static> {
    for path in [
        INTERPRETER_LLBC,
        OBJECT_LLBC,
        MODULE_LLBC,
        RLIB_LLBC,
        JIT_LLBC,
        CORPUS_LLBC,
    ] {
        if let Some(loaded) = llbc_slot(path).get()
            && std::ptr::eq(llbc, loaded)
        {
            return context_slot(path).get_or_init(|| {
                if path == INTERPRETER_LLBC || path == MODULE_LLBC {
                    LowerContext::new_with_dependency_raw_flavors(
                        llbc,
                        &harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(load_llbc(
                            OBJECT_LLBC,
                        ))),
                    )
                } else {
                    LowerContext::new(llbc)
                }
            });
        }
    }
    panic!("lower_context_for: llbc is not a production artefact from load_llbc")
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
    with_llbc_link_lock(|| {
        let fd = llbc
            .local_fn(function_name)
            .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
        lower_fun_decl_with_static_addrs(lower_context_for(llbc), fd, static_addrs)
    })
}

pub fn interpreter_llbc() -> Option<&'static Llbc> {
    load_llbc_if_present(INTERPRETER_LLBC)
}

pub fn object_llbc() -> Option<&'static Llbc> {
    load_llbc_if_present(OBJECT_LLBC)
}

pub fn module_llbc() -> Option<&'static Llbc> {
    load_llbc_if_present(MODULE_LLBC)
}

pub fn jit_llbc() -> Option<&'static Llbc> {
    load_llbc_if_present(JIT_LLBC)
}

/// rbigint bodies live in `majit-rlib.ullbc`; their callers live in
/// `pyre-object.ullbc`. Owner first, matching the front-end merge.
pub fn rbigint_llbcs() -> Option<&'static [&'static Llbc]> {
    static SLOT: OnceLock<Option<Vec<&'static Llbc>>> = OnceLock::new();
    SLOT.get_or_init(|| {
        for path in [RLIB_LLBC, OBJECT_LLBC] {
            if !std::path::Path::new(path).is_file() {
                eprintln!(
                    "skipping: {path} is missing; run \
                     `python3 scripts/extract-llbc.py majit-rlib pyre-object`"
                );
                return None;
            }
        }
        Some(vec![load_llbc(RLIB_LLBC), load_llbc(OBJECT_LLBC)])
    })
    .as_deref()
}
