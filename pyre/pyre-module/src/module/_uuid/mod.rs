//! `_uuid` — the private extension `uuid` reads a stable node from.
//!
//! PyPy has no `_uuid` module.  `_uuidmodule.c` builds a libuuid door on POSIX
//! and an rpcrt4 one on Windows, and configure leaves the module out of the
//! build entirely where neither is available; `uuid.py` catches the resulting
//! `ImportError` and falls back to its Python getters.  Only the `MS_WINDOWS`
//! half is ported here, so every other target keeps that fallback.
//!
//! Without it `uuid.getnode()` cannot reach a MAC address on Windows at all:
//! `uuid.py` leaves `_OS_GETTERS` empty on win32, so `_windll_getnode` is the
//! only non-random getter and it declines unless `_uuid` reports a stable
//! extractable node.  The answer is then a fresh random multicast node on
//! every call.

use pyre_object::{PY_NULL, PyObjectRef};

fn sequential_uuid() -> ([u8; 16], i32) {
    let uuid = rustpython_host_env::uuid::create_sequential();
    (uuid.bytes, uuid.status)
}

fn status_ok() -> i32 {
    rustpython_host_env::uuid::STATUS_OK
}
fn status_local_only() -> i32 {
    rustpython_host_env::uuid::STATUS_LOCAL_ONLY
}
fn status_no_address() -> i32 {
    rustpython_host_env::uuid::STATUS_NO_ADDRESS
}

/// `py_windows_has_stable_node`: only `RPC_S_OK` means the node came from a
/// network card.  The two local-only statuses report a random node, which is
/// no more stable than the one `uuid.py` makes for itself.
fn has_stable_node() -> bool {
    sequential_uuid().1 == status_ok()
}

/// `py_UuidCreate($module, /)`.
fn uuid_create(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let (bytes, status) = sequential_uuid();
    // The two local-only statuses are successes that say the node is random
    // rather than MAC-derived.  If the OS cannot tell, neither can we, so the
    // UUID is taken anyway.
    if status != status_ok() && status != status_local_only() && status != status_no_address() {
        return Err(pyre_interpreter::PyError::os_error_win32_syscall2(
            status, PY_NULL, PY_NULL,
        ));
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
}

pyre_interpreter::py_module! {
    "_uuid",
    extra_init: |ns| {
        pyre_interpreter::__pyre_store!(ns, "UuidCreate", pyre_interpreter::gateway::with_module(
                "_uuid",
                pyre_interpreter::make_module_builtin_function_with_arity("UuidCreate", uuid_create, 0),
            ));
        // `generate_time_safe` is the libuuid entry point and is not in the
        // method table of a Windows build, so the flag that guards it is 0.
        pyre_interpreter::__pyre_store!(ns, "has_uuid_generate_time_safe", pyre_object::w_int_new(0));
        pyre_interpreter::__pyre_store!(ns, "has_stable_extractable_node", pyre_object::w_int_new(i64::from(has_stable_node())));
    },
}
