//! RPython `rbigint.fromint`/`from_u128` call → GC-reference residual.
//!
//! RPython represents every `rbigint` value as one GC reference.  Rust's
//! `RBigInt` source struct is intentionally available to the translator, but
//! its native by-value return ABI is not the JIT ABI.  Calls that construct an
//! `RBigInt` from a machine word therefore target a `dont_look_inside` wrapper
//! returning `*mut RBigInt`, exactly like the arithmetic residuals in
//! [`crate::front::bigint_binop`].

const RESIDUAL_MODULE: [&str; 2] = [crate::runtime_names::crates::OBJECT, "longobject"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarResult {
    Int,
    Bool,
    /// One GC reference, the translated `RBigInt` / `BigInt` result.
    Ref,
}

/// Return the pointer-ABI constructor residual for an RBigInt constructor call.
///
/// The caller proves the destination is the exact local `RBigInt` ADT and that
/// the source is a supported machine-word integer.  `i128`/`u128` deliberately
/// decline: RPython's JIT has no longlonglong register kind, so those paths
/// must not be silently narrowed to a word.
pub(crate) fn constructor_residual_path(
    segments: &[String],
    unsigned: bool,
) -> Option<Vec<String>> {
    let leaf = segments.last()?.as_str();
    let matches_constructor = match leaf {
        "from" => true,
        "fromint" | "fromrarith_int" => !unsigned,
        // Unsigned specialization of `@specialize.argtype(0) fromrarith_int`
        // (`rbigint.py`). The leaf is the proof of unsigned; do not wait
        // on the argument-type gate that `from` uses.
        "fromrarith_uint" => true,
        // RPython's JIT has no longlonglong register kind. Keep the decline
        // local to this mapper instead of relying on the caller's width gate.
        "from_u128" => return None,
        _ => false,
    };
    if !matches_constructor
        || !segments
            .windows(2)
            .any(|pair| pair == ["rbigint", "RBigInt"])
    {
        return None;
    }
    let mut path: Vec<String> = RESIDUAL_MODULE
        .iter()
        .map(|part| part.to_string())
        .collect();
    path.push(
        if unsigned || leaf == "fromrarith_uint" {
            "jit_bigint_from_u64"
        } else {
            "jit_bigint_from_i64"
        }
        .to_string(),
    );
    Some(path)
}

/// Translate Rust's shallow handle clone to a fresh one-GC-reference payload.
///
/// The caller proves both the receiver and destination are the exact local
/// RBigInt ADT. The residual preserves `Clone`'s shared immutable digit array
/// while giving the returned by-value handle its own translated GC object.
pub(crate) fn clone_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    if leaf != "clone" {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::OBJECT,
            "longobject",
            "jit_bigint_clone",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// Retarget the Rust ownership spelling `w_long_new(RBigInt)` to the pointer
/// ABI used by translated rbigints.
///
/// At source level an `RBigInt` is a by-value two-word Rust handle and
/// `w_long_new` first allocates its GC payload. After translation that same
/// value is already the one-GC-reference rbigint object, exactly as in
/// RPython. Re-boxing it as a by-value Rust argument would both use the wrong
/// residual ABI and allocate a redundant payload. The caller proves the sole
/// argument is the exact local RBigInt ADT before applying this target swap.
pub(crate) fn long_box_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let leaf = segments.last().map(String::as_str)?;
    if !matches!(
        leaf,
        "w_long_new" | "w_long_new_fresh_rbigint_handle" | "box_bigint_constant"
    ) || segments.iter().rev().nth(1).map(String::as_str) != Some("longobject")
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::OBJECT,
            "longobject",
            "w_long_from_raw",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// Bare-payload residual for scalar RBigInt queries. The caller has already
/// proved exact RBigInt receiver identity.
///
/// Every mapped method returns a plain machine word in Rust, so retargeting it
/// to a one-word residual preserves the call's type.  `bit_length` /
/// `bit_count` are deliberately absent even though `rbigint.bit_length` /
/// `rbigint.bit_count` (`rpython/rlib/rbigint.py`) are `@jit.elidable` like the
/// rest: both raise OverflowError through `ovfcheck`, so their Rust signature
/// is `Result<i64, RBigIntError>` and a one-word residual would ERASE that
/// `Result` under a caller that still destructures it.  Upstream mints no
/// residual for them either — they are ordinary elidable graphs the codewriter
/// traces — so the parity shape and the sound shape agree.  Converging further
/// means `simplify.transform_ovfcheck`'s rewrite, which
/// [`crate::front::checked_arith`] already performs for the `checked_*` +
/// `Option` spelling of the same idiom.
pub(crate) fn scalar_residual_for_method(leaf: &str) -> Option<(Vec<String>, ScalarResult)> {
    let (module, residual, result) = match leaf {
        "get_sign" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_sign_i64",
            ScalarResult::Int,
        ),
        "bits" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_bits",
            ScalarResult::Int,
        ),
        "is_zero" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_is_zero",
            ScalarResult::Bool,
        ),
        "is_one" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_is_one",
            ScalarResult::Bool,
        ),
        "fits_int" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_to_i64_fits",
            ScalarResult::Bool,
        ),
        "tobool" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_tobool",
            ScalarResult::Bool,
        ),
        "hash" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_hash",
            ScalarResult::Int,
        ),
        _ => return None,
    };
    let mut path: Vec<String> = module.iter().map(|part| part.to_string()).collect();
    path.push(residual.to_string());
    Some((path, result))
}

/// Bare-payload residual for an RBigInt operation specialized for one signed
/// machine-word operand.  PyPy's W_LongObject descriptors call these exact
/// `rbigint.int_*` methods instead of first allocating an rbigint for the
/// W_IntObject.  The MIR caller proves the receiver is RBigInt and the second
/// argument is exactly i64 before applying this mapping.
pub(crate) fn int_binop_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual = match leaf {
        "int_add" => "jit_bigint_int_add",
        "int_sub" => "jit_bigint_int_sub",
        "int_mul" => "jit_bigint_int_mul",
        "int_and_" => "jit_bigint_int_and",
        "int_or_" => "jit_bigint_int_or",
        "int_xor" => "jit_bigint_int_xor",
        _ => return None,
    };
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// Scalar-bool counterpart of [`int_binop_residual_for_method`] for PyPy's
/// mixed W_LongObject/W_IntObject rich comparisons.
pub(crate) fn int_comparison_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual = match leaf {
        "int_eq" => "jit_bigint_int_eq",
        "int_ne" => "jit_bigint_int_ne",
        "int_lt" => "jit_bigint_int_lt",
        "int_le" => "jit_bigint_int_le",
        "int_gt" => "jit_bigint_int_gt",
        "int_ge" => "jit_bigint_int_ge",
        _ => return None,
    };
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// `descroperation::bigint_{add,sub,mul,mod}(&RBigInt, &RBigInt) -> *mut RBigInt`
/// are the source spellings of `rbigint.add` / `sub` / `mul` / `mod`. Each
/// call is retargeted to the matching `jit_bigint_*`, whose result the front
/// models as one GC reference. `bigint_mod` targets `jit_bigint_mod_floor`,
/// the same residual as the value-returning `bigint_modulo_nonzero` seam.
pub(crate) fn bigint_add_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_add") => "jit_bigint_add",
        Some("bigint_sub") => "jit_bigint_sub",
        Some("bigint_mul") => "jit_bigint_mul",
        Some("bigint_mod") => "jit_bigint_mod_floor",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// `descroperation::bigint_int_{add,sub,mul}(&RBigInt, i64) -> *mut RBigInt`
/// are the source spellings of `rbigint.int_add` / `int_sub` / `int_mul`.
/// The zero/one alias stays inside the helper; the front retargets the call
/// to the elidable `jit_bigint_int_*` residual.
pub(crate) fn bigint_int_payload_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_int_add") => "jit_bigint_int_add",
        Some("bigint_int_sub") => "jit_bigint_int_sub",
        Some("bigint_int_mul") => "jit_bigint_int_mul",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// `descroperation::bigint_pow_nomod(&RBigInt, &RBigInt) ->
/// Result<RBigInt, PyError>` is Rust's source-level spelling of RPython's
/// elidable `rbigint.pow` call with an implicit MemoryError edge.  The caller
/// rule in `result_exc` removes the `Result` diamond; this target swap supplies
/// the matching one-GC-reference result ABI.
pub(crate) fn pow_nomod_residual_path(segments: &[String]) -> Option<Vec<String>> {
    // `bigint_int_pow_nomod` is the machine-int-exponent leg `descr_pow` takes
    // for a `W_IntObject` exponent; its carrier and pointer result ABI match.
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_pow_nomod") => "jit_bigint_pow_nomod",
        Some("bigint_int_pow_nomod") => "jit_bigint_int_pow_nomod",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// By-value or unregistered leaves of the zero-checked long/int seams.
///
/// `bigint_int_modulo_int_result_nonzero` is ABI-safe (`&BigInt`, `i64` in,
/// `i64` out) but unpublished. `bigint_int_floordiv_nonzero` and
/// `bigint_rshift` return `BigInt` by value. Each already has a
/// `jit_bigint_*` wrapper: the remainder stays a machine word, and the two
/// bigint results come back as the GC reference those wrappers return.
pub(crate) fn nonzero_leaf_residual_path(
    segments: &[String],
) -> Option<(Vec<String>, ScalarResult)> {
    let (residual, result) = match segments.last().map(String::as_str) {
        Some("bigint_int_modulo_int_result_nonzero") => {
            ("jit_bigint_int_mod_int_result", ScalarResult::Int)
        }
        Some("bigint_int_floordiv_nonzero") => ("jit_bigint_int_div_floor", ScalarResult::Ref),
        // `jit_bigint_rshift` takes two payload pointers. This leaf's second
        // argument is the machine shift count, which `jit_bigint_shr` already
        // accepts, returning the GC reference.
        Some("bigint_rshift") => ("jit_bigint_shr", ScalarResult::Ref),
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some((
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        result,
    ))
}

/// `descroperation::bigint_lshift_count(&RBigInt, i64) ->
/// Result<RBigInt, PyError>` carries RPython `rbigint.lshift`'s implicit
/// MemoryError through Rust source. `result_exc` removes that carrier after
/// this target swap supplies the one-GC-reference residual result.
pub(crate) fn lshift_count_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_lshift_count") => "jit_bigint_lshift_count",
        Some("bigint_lshift_int_int_result") => "jit_bigint_lshift_int_int_result",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// The `_make_ovf2long` seams (intobject.py): a machine-int add,
/// subtract, or multiply that overflowed recovers the exact result from the
/// two Signed words in one elidable call.  Both operands are already machine
/// words at source level, so the target swap only supplies the
/// one-GC-reference result ABI.
pub(crate) fn ovf2long_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_add_int_int") => "jit_bigint_add_int_int",
        Some("bigint_sub_int_int") => "jit_bigint_sub_int_int",
        Some("bigint_mul_int_int") => "jit_bigint_mul_int_int",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// RustPython's compiler exposes integer constants as an opaque foreign
/// BigInt.  Retarget the one sanctioned conversion seam to a pointer-returning
/// pure residual so Malachite never enters the translated arithmetic graph.
pub(crate) fn compiler_bigint_residual_path(segments: &[String]) -> Option<Vec<String>> {
    if !segments.ends_with(&[
        crate::runtime_names::crates::INTERPRETER.to_string(),
        "compiler_bigint_to_rbigint".to_string(),
    ]) {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "jit_compiler_bigint_to_rbigint",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// The interpreter's own unary RBigInt wrappers.
///
/// A descent reaches `descroperation::bigint_neg` / `bigint_invert` as an
/// unlowered helper and stops there, so the `<RBigInt as Neg>::neg` call in
/// their bodies is never seen and
/// [`bigint_unop_residual_path`](crate::front::bigint_binop::bigint_unop_residual_path)
/// never applies.  Swapping the wrapper itself reaches the same two residuals
/// the operator spelling reaches, the way [`divmod_projection_residual_path`]
/// does for the binary wrappers.
pub(crate) fn unop_wrapper_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_neg") => "jit_bigint_neg",
        Some("bigint_invert") => "jit_bigint_invert",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        ["pyre_interpreter", "objspace", "descroperation", residual]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

/// Zero-checked interpreter seams for the two projections of
/// `rbigint.divmod`. Each returns a direct RBigInt handle at source level,
/// so the target swap is ABI-identical to the other pointer residual mappings.
pub(crate) fn divmod_projection_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_floordiv_nonzero") => "jit_bigint_div_floor",
        Some("bigint_modulo_nonzero") => "jit_bigint_mod_floor",
        // Machine-int-divisor quotient leg (`_int_floordiv`): the divisor is
        // already a bare word, so this swap stays ABI-identical. The `_int_mod`
        // leg has no entry — `int_mod_int_result` yields a machine int, while
        // every result in this map is modeled as a GC reference.
        Some("bigint_int_floordiv_nonzero") => "jit_bigint_int_div_floor",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            residual,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// `descroperation::bigint_divrem_returns_lhs_remainder(&RBigInt, &RBigInt)
/// -> bool` is the host form of `rbigint._divrem`'s `(NULLRBIGINT, a)`
/// early-return test that `long_mod` and `integer_divmod_pair` consult before
/// reusing the dividend's payload. Retarget it to the elidable
/// `jit_bigint_divrem_returns_lhs_remainder` so the two operands reach the
/// residual as the GC references they are; the boolean result stays in the
/// integer bank like the RBigInt comparisons.
pub(crate) fn divrem_lhs_remainder_residual_path(segments: &[String]) -> Option<Vec<String>> {
    if segments.last().map(String::as_str) != Some("bigint_divrem_returns_lhs_remainder") {
        return None;
    }
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        [
            crate::runtime_names::crates::INTERPRETER,
            "objspace",
            "descroperation",
            "jit_bigint_divrem_returns_lhs_remainder",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_divrem_lhs_remainder_host_form_only_in_descroperation() {
        assert_eq!(
            divrem_lhs_remainder_residual_path(&segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "bigint_divrem_returns_lhs_remainder",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_divrem_returns_lhs_remainder",
            ])),
        );
        // The residual itself and a same-named function elsewhere keep their
        // own bodies.
        assert_eq!(
            divrem_lhs_remainder_residual_path(&segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_divrem_returns_lhs_remainder",
            ])),
            None,
        );
        assert_eq!(
            divrem_lhs_remainder_residual_path(&segs(&[
                "pyre_object",
                "longobject",
                "bigint_divrem_returns_lhs_remainder",
            ])),
            None,
        );
    }

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    #[test]
    fn maps_the_unary_wrappers_and_nothing_else() {
        for (wrapper, residual) in [
            ("bigint_neg", "jit_bigint_neg"),
            ("bigint_invert", "jit_bigint_invert"),
        ] {
            assert_eq!(
                unop_wrapper_residual_path(&segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    wrapper,
                ])),
                Some(segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    residual,
                ])),
            );
        }
        // The owner is load-bearing: a same-named function anywhere else is
        // not this wrapper and keeps its own body.
        assert_eq!(
            unop_wrapper_residual_path(&segs(&["pyre_object", "longobject", "bigint_neg"])),
            None,
        );
        assert_eq!(
            unop_wrapper_residual_path(&segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "bigint_clone",
            ])),
            None,
        );
    }

    #[test]
    fn maps_signed_and_unsigned_word_constructors() {
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromint"]), false),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_from_i64"
            ]))
        );
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "from"]), true),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_from_u64"
            ]))
        );
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromrarith_int"]), false),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_from_i64"
            ]))
        );
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromrarith_uint"]), true),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_from_u64"
            ]))
        );
    }

    #[test]
    fn declines_other_types_and_wrong_signedness() {
        assert!(constructor_residual_path(&segs(&["other", "RBigInt", "from"]), false).is_none());
        assert!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromint"]), true).is_none()
        );
        assert!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "from_u128"]), true).is_none()
        );
        assert!(constructor_residual_path(&segs(&["rbigint", "RBigInt", "add"]), false).is_none());
        assert!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromrarith_int"]), true)
                .is_none()
        );
        // The leaf names the unsigned constructor even if the argument
        // atom was not classified as unsigned.
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromrarith_uint"]), false),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_from_u64"
            ]))
        );
    }

    #[test]
    fn maps_only_the_exact_clone_method() {
        assert_eq!(
            clone_residual_for_method("clone"),
            Some(segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "jit_bigint_clone"
            ]))
        );
        assert!(clone_residual_for_method("clone_from").is_none());
    }

    #[test]
    fn maps_translated_long_boxing_to_pointer_abi() {
        let expected = segs(&[
            crate::runtime_names::crates::OBJECT,
            "longobject",
            "w_long_from_raw",
        ]);
        for leaf in [
            "w_long_new",
            "w_long_new_fresh_rbigint_handle",
            "box_bigint_constant",
        ] {
            assert_eq!(
                long_box_residual_path(&segs(&[
                    crate::runtime_names::crates::OBJECT,
                    "longobject",
                    leaf
                ])),
                Some(expected.clone())
            );
        }
        assert!(
            long_box_residual_path(&segs(&["other", "longobject", "w_long_new"])).is_some(),
            "crate prefix is intentionally irrelevant; module/leaf and exact argument type identify the call"
        );
        assert!(
            long_box_residual_path(&segs(&[
                crate::runtime_names::crates::OBJECT,
                "other",
                "w_long_new"
            ]))
            .is_none()
        );
    }

    #[test]
    fn maps_scalar_queries_with_their_result_bank() {
        assert_eq!(
            scalar_residual_for_method("get_sign"),
            Some((
                segs(&[
                    crate::runtime_names::crates::OBJECT,
                    "longobject",
                    "jit_bigint_sign_i64"
                ]),
                ScalarResult::Int
            ))
        );
        assert_eq!(
            scalar_residual_for_method("is_zero"),
            Some((
                segs(&[
                    crate::runtime_names::crates::OBJECT,
                    "longobject",
                    "jit_bigint_is_zero"
                ]),
                ScalarResult::Bool
            ))
        );
        // `bit_length` / `bit_count` return `Result<i64, RBigIntError>` —
        // `ovfcheck`'s value encoding — so no one-word residual may claim
        // them; see this mapper's note.
        assert_eq!(scalar_residual_for_method("bit_length"), None);
        assert_eq!(scalar_residual_for_method("bit_count"), None);
        assert!(scalar_residual_for_method("add").is_none());
    }

    #[test]
    fn maps_each_machine_int_specialized_operation() {
        for (method, residual) in [
            ("int_add", "jit_bigint_int_add"),
            ("int_sub", "jit_bigint_int_sub"),
            ("int_mul", "jit_bigint_int_mul"),
            ("int_and_", "jit_bigint_int_and"),
            ("int_or_", "jit_bigint_int_or"),
            ("int_xor", "jit_bigint_int_xor"),
        ] {
            assert_eq!(
                int_binop_residual_for_method(method),
                Some(segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        assert!(int_binop_residual_for_method("add").is_none());
        assert!(int_binop_residual_for_method("int_floordiv").is_none());
    }

    #[test]
    fn maps_each_machine_int_specialized_comparison() {
        for (method, residual) in [
            ("int_eq", "jit_bigint_int_eq"),
            ("int_ne", "jit_bigint_int_ne"),
            ("int_lt", "jit_bigint_int_lt"),
            ("int_le", "jit_bigint_int_le"),
            ("int_gt", "jit_bigint_int_gt"),
            ("int_ge", "jit_bigint_int_ge"),
        ] {
            assert_eq!(
                int_comparison_residual_for_method(method),
                Some(segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        assert!(int_comparison_residual_for_method("lt").is_none());
        assert!(int_comparison_residual_for_method("int_add").is_none());
    }

    #[test]
    fn maps_bigint_add_leaf_to_pointer_abi() {
        assert_eq!(
            bigint_add_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_add",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_add",
            ]))
        );
        assert!(
            bigint_add_residual_path(&segs(&["majit_rlib", "rbigint", "RBigInt", "add"])).is_none()
        );
        assert_eq!(
            bigint_add_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_sub",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_sub",
            ]))
        );
        assert_eq!(
            bigint_add_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_mul",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_mul",
            ]))
        );
        assert_eq!(
            bigint_add_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_mod",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_mod_floor",
            ]))
        );
    }

    #[test]
    fn maps_bigint_int_payload_leaves_to_pointer_abi() {
        for (source, residual) in [
            ("bigint_int_add", "jit_bigint_int_add"),
            ("bigint_int_sub", "jit_bigint_int_sub"),
            ("bigint_int_mul", "jit_bigint_int_mul"),
        ] {
            assert_eq!(
                bigint_int_payload_residual_path(&segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    source,
                ])),
                Some(segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        assert!(
            bigint_int_payload_residual_path(&segs(&[
                "majit_rlib",
                "rbigint",
                "RBigInt",
                "int_add",
            ]))
            .is_none()
        );
    }

    #[test]
    fn maps_nomod_pow_result_seam_to_pointer_abi() {
        assert_eq!(
            pow_nomod_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_pow_nomod",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_pow_nomod",
            ]))
        );
        assert!(
            pow_nomod_residual_path(&segs(&["majit_rlib", "rbigint", "RBigInt", "pow",])).is_none()
        );
        assert_eq!(
            pow_nomod_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_int_pow_nomod",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_int_pow_nomod",
            ]))
        );
    }

    #[test]
    fn maps_nonzero_leaves_to_existing_wrappers() {
        assert_eq!(
            nonzero_leaf_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_int_modulo_int_result_nonzero",
            ])),
            Some((
                segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    "jit_bigint_int_mod_int_result",
                ]),
                ScalarResult::Int,
            ))
        );
        assert_eq!(
            nonzero_leaf_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_int_floordiv_nonzero",
            ])),
            Some((
                segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    "jit_bigint_int_div_floor",
                ]),
                ScalarResult::Ref,
            ))
        );
        assert_eq!(
            nonzero_leaf_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_rshift",
            ])),
            Some((
                segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    "jit_bigint_shr",
                ]),
                ScalarResult::Ref,
            ))
        );
        assert!(
            nonzero_leaf_residual_path(&segs(&["majit_rlib", "rbigint", "RBigInt", "rshift",]))
                .is_none()
        );
    }

    #[test]
    fn maps_lshift_result_seam_to_pointer_abi() {
        assert_eq!(
            lshift_count_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_lshift_count",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_lshift_count",
            ]))
        );
        assert_eq!(
            lshift_count_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_lshift_int_int_result",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "jit_bigint_lshift_int_int_result",
            ]))
        );
        assert!(
            lshift_count_residual_path(&segs(&["majit_rlib", "rbigint", "RBigInt", "lshift",]))
                .is_none()
        );
    }

    #[test]
    fn maps_only_the_compiler_bigint_conversion_seam() {
        assert_eq!(
            compiler_bigint_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "compiler_bigint_to_rbigint",
            ])),
            Some(segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "jit_compiler_bigint_to_rbigint",
            ]))
        );
        assert!(
            compiler_bigint_residual_path(&segs(&["malachite_bigint", "BigInt", "to_bytes_le",]))
                .is_none()
        );
    }

    #[test]
    fn maps_zero_checked_divmod_projections() {
        for (source, residual) in [
            ("bigint_floordiv_nonzero", "jit_bigint_div_floor"),
            ("bigint_modulo_nonzero", "jit_bigint_mod_floor"),
            ("bigint_int_floordiv_nonzero", "jit_bigint_int_div_floor"),
        ] {
            assert_eq!(
                divmod_projection_residual_path(&segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    source,
                ])),
                Some(segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
    }

    #[test]
    fn maps_the_ovf2long_seams_and_nothing_else() {
        for (source, residual) in [
            ("bigint_add_int_int", "jit_bigint_add_int_int"),
            ("bigint_sub_int_int", "jit_bigint_sub_int_int"),
            ("bigint_mul_int_int", "jit_bigint_mul_int_int"),
        ] {
            assert_eq!(
                ovf2long_residual_path(&segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    source,
                ])),
                Some(segs(&[
                    crate::runtime_names::crates::INTERPRETER,
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        // The owner is load-bearing, as it is for the unary wrappers.
        assert_eq!(
            ovf2long_residual_path(&segs(&[
                crate::runtime_names::crates::OBJECT,
                "longobject",
                "bigint_mul_int_int",
            ])),
            None,
        );
        assert_eq!(
            ovf2long_residual_path(&segs(&[
                crate::runtime_names::crates::INTERPRETER,
                "objspace",
                "descroperation",
                "bigint_mul",
            ])),
            None,
        );
    }
}
