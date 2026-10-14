//! `pypyjit` module — PyPy: `pypy/module/pypyjit/moduledef.py`.
//!
//! The module dict binds `set_param` to `interp_jit.set_param`. The JIT
//! itself lives in the higher `pyre-jit` crate, so that function routes
//! through the `SET_JIT_PARAM_STRING_HOOK` / `SET_JIT_PARAM_HOOK` that
//! pyre-jit registers at boot (`call.rs`). Because the hooks are in-process
//! function pointers rather than an env lever, a `pypyjit.set_param(...)`
//! call configures the warmstate on every backend including the wasm guest
//! (which sees no environment).

pub mod interp_jit;

pyre_interpreter::py_module! {
    "pypyjit",
    interpleveldefs: {
        "set_param" => pyre_interpreter::gateway::make_module_builtin_function_passthrough0(
            "set_param",
            interp_jit::set_param,
            interp_jit::set_param_args,
        ),
    },
}
