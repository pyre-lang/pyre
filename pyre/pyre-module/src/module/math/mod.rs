//! math module — PyPy: pypy/module/math/
//!
//! Function bodies live in `interp_math`; this declarative table mirrors
//! `moduledef.py` interpleveldefs.

pub mod interp_math;

use interp_math as m;

pyre_interpreter::py_module! {
    "math",
    interpleveldefs: {
        "e"   => pyre_object::floatobject::w_float_new(pymath::math::E),
        "pi"  => pyre_object::floatobject::w_float_new(pymath::math::PI),
        "tau" => pyre_object::floatobject::w_float_new(pymath::math::TAU),
        "inf" => pyre_object::floatobject::w_float_new(pymath::math::INF),
        "nan" => pyre_object::floatobject::w_float_new(pymath::math::NAN),
    },
    module_functions: {
        // Trigonometric. The one- and two-argument float builtins are
        // installed in `extra_init` as `__majit_wrap_math_*`.

        // Exponential / logarithmic. sqrt/cbrt/log/log1p/exp/exp2/expm1/pow
        // are installed in `extra_init`.
        "log2"  / 1 = m::log2,
        "log10" / 1 = m::log10,

        // Gamma / error. erf/erfc/gamma/lgamma are installed in `extra_init`.

        // Rounding / truncation: floor/ceil/trunc are installed in `extra_init`.

        // Floating-point manipulation. `fabs`, `ulp`, `frexp`, `ldexp`,
        // `fmod`, `copysign`, `remainder`, `fmax` and `fmin` are installed
        // in `extra_init`.
        "modf"      / 1 = m::modf,
        "nextafter" / * = m::nextafter; pyre_interpreter::Signature::new(vec!["x", "y", "steps"], None, None, 1, 2),
        "fma"       / 3 = m::fma,

        // Classification. `isclose` is installed in `extra_init`.
        "isinf"        / 1 = m::isinf,
        "isnan"        / 1 = m::isnan,
        "isfinite"     / 1 = m::isfinite,
        "isnormal"     / 1 = m::isnormal,
        "issubnormal"  / 1 = m::issubnormal,
        "signbit"      / 1 = m::signbit,

        // Conversion. `degrees` and `radians` are installed in `extra_init`.

        // Multi-dimensional
        "hypot" / * = m::hypot; pyre_interpreter::Signature::new(vec![], Some("args"), None, 0, 0),
        "dist"  / 2 = m::dist,

        // Aggregation
        "fsum"    / 1 = m::fsum,
        "prod"    / * = m::prod; pyre_interpreter::Signature::new(vec!["iterable", "start"], None, None, 1, 1),
        "sumprod" / 2 = m::sumprod,

        // Integer math
        "factorial" / 1 = m::factorial,
        "gcd"   / * = m::gcd; pyre_interpreter::Signature::new(vec![], Some("args"), None, 0, 0),
        "lcm"   / * = m::lcm; pyre_interpreter::Signature::new(vec![], Some("args"), None, 0, 0),
        "comb"  / 2 = m::comb,
        "perm"  / * = m::perm; pyre_interpreter::Signature::new(vec!["n", "k"], None, None, 0, 2),
        // `isqrt` is installed in `extra_init`.
    },
    extra_init: |ns| {
        // Module builtin, not a method descriptor: `BuiltinCode.func` is the
        // gateway itself so builtin-call descent walks it.
        let install = |mut ns: pyre_object::PyObjectRef,
                       name: &'static str,
                       func: pyre_interpreter::BuiltinCodeFn| {
            pyre_interpreter::__pyre_store!(
                ns,
                name,
                pyre_interpreter::make_module_builtin_function_with_arity(name, func, 1)
            );
            ns
        };
        ns = install(ns, "sqrt", m::__majit_wrap_math_sqrt);
        ns = install(ns, "sin", m::__majit_wrap_math_sin);
        ns = install(ns, "cos", m::__majit_wrap_math_cos);
        ns = install(ns, "tan", m::__majit_wrap_math_tan);
        ns = install(ns, "asin", m::__majit_wrap_math_asin);
        ns = install(ns, "acos", m::__majit_wrap_math_acos);
        ns = install(ns, "atan", m::__majit_wrap_math_atan);
        ns = install(ns, "tanh", m::__majit_wrap_math_tanh);
        ns = install(ns, "asinh", m::__majit_wrap_math_asinh);
        ns = install(ns, "acosh", m::__majit_wrap_math_acosh);
        ns = install(ns, "atanh", m::__majit_wrap_math_atanh);
        ns = install(ns, "log1p", m::__majit_wrap_math_log1p);
        ns = install(ns, "cbrt", m::__majit_wrap_math_cbrt);
        ns = install(ns, "erf", m::__majit_wrap_math_erf);
        ns = install(ns, "erfc", m::__majit_wrap_math_erfc);
        ns = install(ns, "ulp", m::__majit_wrap_math_ulp);
        ns = install(ns, "degrees", m::__majit_wrap_math_degrees);
        ns = install(ns, "radians", m::__majit_wrap_math_radians);
        ns = install(ns, "fabs", m::__majit_wrap_math_fabs);
        ns = install(ns, "exp", m::__majit_wrap_math_exp);
        ns = install(ns, "exp2", m::__majit_wrap_math_exp2);
        ns = install(ns, "expm1", m::__majit_wrap_math_expm1);
        ns = install(ns, "sinh", m::__majit_wrap_math_sinh);
        ns = install(ns, "cosh", m::__majit_wrap_math_cosh);
        ns = install(ns, "gamma", m::__majit_wrap_math_gamma);
        ns = install(ns, "lgamma", m::__majit_wrap_math_lgamma);
        ns = install(ns, "floor", m::__majit_wrap_math_floor);
        ns = install(ns, "ceil", m::__majit_wrap_math_ceil);
        ns = install(ns, "trunc", m::__majit_wrap_math_trunc);
        ns = install(ns, "frexp", m::__majit_wrap_math_frexp);
        ns = install(ns, "isqrt", m::__majit_wrap_math_isqrt);
        // Optional base, or keyword tolerances: not a fixed arity-1 builtin.
        pyre_interpreter::__pyre_store!(
            ns,
            "log",
            pyre_interpreter::gateway::function_set_null_positional_defaults(
                pyre_interpreter::make_module_builtin_function_with_arity_and_sig(
                    "log",
                    m::__majit_wrap_math_log,
                    pyre_interpreter::HOPELESS,
                    pyre_interpreter::Signature::new(vec!["x", "base"], None, None, 0, 2),
                ),
                1,
            )
        );
        let mut isclose = pyre_object::with_roots!(ns => pyre_interpreter::function_set_kwonly_defaults!(
            pyre_interpreter::make_module_builtin_function_with_arity_and_sig(
                "isclose",
                m::__majit_wrap_math_isclose,
                pyre_interpreter::HOPELESS,
                pyre_interpreter::Signature::new(
                    vec!["a", "b", "rel_tol", "abs_tol"],
                    None,
                    None,
                    2,
                    0,
                ),
            ),
            "rel_tol" => pyre_object::floatobject::w_float_new(1e-9),
            "abs_tol" => pyre_object::floatobject::w_float_new(0.0),
        ));
        pyre_interpreter::__pyre_store!(ns, "isclose", isclose);
        let install2 = |mut ns: pyre_object::PyObjectRef,
                        name: &'static str,
                        func: pyre_interpreter::BuiltinCodeFn| {
            pyre_interpreter::__pyre_store!(
                ns,
                name,
                pyre_interpreter::make_module_builtin_function_with_arity(name, func, 2)
            );
            ns
        };
        ns = install2(ns, "pow", m::__majit_wrap_math_pow);
        ns = install2(ns, "fmod", m::__majit_wrap_math_fmod);
        ns = install2(ns, "copysign", m::__majit_wrap_math_copysign);
        ns = install2(ns, "remainder", m::__majit_wrap_math_remainder);
        ns = install2(ns, "fmax", m::__majit_wrap_math_fmax);
        ns = install2(ns, "fmin", m::__majit_wrap_math_fmin);
        ns = install2(ns, "atan2", m::__majit_wrap_math_atan2);
        ns = install2(ns, "ldexp", m::__majit_wrap_math_ldexp);
        // Trailing defaults live on `Function.defs_w` / `w_kw_defs`
        // (`interp2app._getdefaults`). `perm` `k`, `prod` `start=1`,
        // `nextafter` `steps=None`.
        if let Some(mut function) = unsafe { pyre_object::w_dict_getitem_str(ns, "perm") } {
            function = pyre_interpreter::gateway::function_set_null_positional_defaults(function, 1);
            pyre_interpreter::__pyre_store!(ns, "perm", function);
        }
        if let Some(mut function) = unsafe { pyre_object::w_dict_getitem_str(ns, "prod") } {
            function = pyre_object::with_roots!(ns => pyre_interpreter::function_set_kwonly_defaults!(
                function,
                "start" => pyre_object::w_int_new(1),
            ));
            pyre_interpreter::__pyre_store!(ns, "prod", function);
        }
        if let Some(mut function) = unsafe { pyre_object::w_dict_getitem_str(ns, "nextafter") } {
            function = pyre_object::with_roots!(ns => {
                pyre_interpreter::function_set_kwonly_default_pairs(
                    function,
                    &[("steps", pyre_object::w_none())],
                )
            });
            pyre_interpreter::__pyre_store!(ns, "nextafter", function);
        }
    },
}
