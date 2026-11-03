//! `_contextvars` module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! `Context`, `ContextVar`, `Token`, and `copy_context` are the app-level
//! line-by-line port.  ExecutionContext state is still read and written
//! through `__pypy__.get_contextvar_context` / `set_contextvar_context`.

pyre_interpreter::py_module! {
    "_contextvars",
    appleveldefs: {
        "_contextvars_app.py" => ["Context", "ContextVar", "Token", "copy_context"],
    },
    extra_init: |ns| {
        // [3.15-spec] PyPy keeps Context/ContextVar/Token as ordinary
        // app-level classes in lib_pypy/_contextvars.py (Unsubclassable).
        // CPython exposes PyContext_Type / PyContextVar_Type /
        // PyContextToken_Type as static immutable types.  No @jit.*,
        // _immutable_fields_, or runtime reader in PyPy's class definition
        // depends on the public owner flags, so project only the observable
        // axes here.
        for name in ["Context", "ContextVar", "Token"] {
            let t = pyre_interpreter::module_ns_get(ns, name).unwrap_or_else(|| {
                panic!("_contextvars.{name} must be installed by appleveldefs")
            });
            unsafe {
                pyre_object::w_type_set_cpython_type_flags(t, false, true, true);
                pyre_object::w_type_suppress_cpython_basetype(t);
            }
        }
    },
}
