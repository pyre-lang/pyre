//! cmath module — PyPy: pypy/module/cmath/
//!
//! Function bodies live in `interp_cmath`; this declarative table mirrors
//! `moduledef.py` interpleveldefs.

pub mod interp_cmath;

use interp_cmath as m;

pyre_interpreter::py_module! {
    "cmath",
    interpleveldefs: {
        "pi"   => pyre_object::floatobject::w_float_new(pymath::math::PI),
        "e"    => pyre_object::floatobject::w_float_new(pymath::math::E),
        "tau"  => pyre_object::floatobject::w_float_new(pymath::math::TAU),
        "inf"  => pyre_object::floatobject::w_float_new(pymath::math::INF),
        "nan"  => pyre_object::floatobject::w_float_new(pymath::math::NAN),
        "infj" => pyre_object::complexobject::w_complex_new(0.0, pymath::math::INF),
        "nanj" => pyre_object::complexobject::w_complex_new(0.0, pymath::math::NAN),
    },
    functions: {
        "sqrt"  / 1 = m::sqrt,
        "exp"   / 1 = m::exp,
        "log10" / 1 = m::log10,
        "sin"   / 1 = m::sin,
        "cos"   / 1 = m::cos,
        "tan"   / 1 = m::tan,
        "asin"  / 1 = m::asin,
        "acos"  / 1 = m::acos,
        "atan"  / 1 = m::atan,
        "sinh"  / 1 = m::sinh,
        "cosh"  / 1 = m::cosh,
        "tanh"  / 1 = m::tanh,
        "asinh" / 1 = m::asinh,
        "acosh" / 1 = m::acosh,
        "atanh" / 1 = m::atanh,
        "log"   / * = m::log; pyre_interpreter::Signature::new(vec!["z", "base"], None, None, 0, 2),

        "phase" / 1 = m::phase,
        "polar" / 1 = m::polar,
        "rect"  / 2 = m::rect,

        "isfinite" / 1 = m::isfinite,
        "isinf"    / 1 = m::isinf,
        "isnan"    / 1 = m::isnan,
        "isclose"  / * = m::isclose; pyre_interpreter::Signature::new(vec!["a", "b", "rel_tol", "abs_tol"], None, None, 2, 2),
    },
}
