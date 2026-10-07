//! Float/complex `repr` residual-call retargets.
//!
//! The two arms mirror `float2string` / `float_repr`
//! (`pypy/objspace/std/floatobject.py`) and `repr_format` /
//! `format_float` (`pypy/objspace/std/complexobject.py`).  Neither
//! ports `rpython.rlib.rfloat.formatd` itself: upstream reaches it only
//! indirectly (`float_repr` -> `float2string` -> `formatd`, and only on the
//! `isfinite` branch), while these arms retarget RustPython/pyre formatters.
//!
//! What the retarget is for: the translated `formatd` result is one `STR` GC
//! pointer, but RustPython's formatter returns a three-word Rust `String`,
//! which the residual-call ABI cannot return.  The interpreter publishes an
//! exact one-word `Utf8Str*` (`rstr.py` `rpy_string`) wrapper; this module
//! selects that wrapper without teaching the parity layers a Rust ABI.
//! Structurally this is the [`crate::front::rbigint_call`] residual-retarget
//! family.

const FLOAT_TO_STRING: &[&str] = &["rustpython_literal", "float", "to_string"];
const FLOAT_RESIDUAL: &[&str] = &["pyre_interpreter", "display", "jit_format_float_repr_rstr"];
const COMPLEX_COMPONENT: &[&str] = &[
    "pyre_interpreter",
    "typedef",
    "format_complex_component_repr",
];
const COMPLEX_COMPONENT_RESIDUAL: &[&str] = &[
    "pyre_interpreter",
    "typedef",
    "jit_format_complex_component_repr_rstr",
];

pub(crate) fn repr_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    let residual = if path == FLOAT_TO_STRING {
        FLOAT_RESIDUAL
    } else if path == COMPLEX_COMPONENT {
        COMPLEX_COMPONENT_RESIDUAL
    } else {
        return None;
    };
    Some(residual.iter().map(|part| (*part).to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_string()).collect()
    }

    #[test]
    fn retargets_only_the_exact_rustpython_float_repr_formatter() {
        assert_eq!(
            repr_residual_path(&segs(FLOAT_TO_STRING)),
            Some(segs(FLOAT_RESIDUAL))
        );
        assert!(repr_residual_path(&segs(&["float", "to_string"])).is_none());
        assert!(
            repr_residual_path(&segs(&["rustpython_literal", "complex", "to_string"])).is_none()
        );
    }

    #[test]
    fn retargets_only_the_pypy_complex_component_formatter_boundary() {
        assert_eq!(
            repr_residual_path(&segs(COMPLEX_COMPONENT)),
            Some(segs(COMPLEX_COMPONENT_RESIDUAL))
        );
        assert!(repr_residual_path(&segs(&["typedef", "format_complex_component_repr"])).is_none());
    }
}
