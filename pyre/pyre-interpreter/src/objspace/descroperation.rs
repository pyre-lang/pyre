//! pypy/objspace/descroperation.py — binary/unary operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This module
//! contains the dispatch layer that routes `+`, `-`, `*`, `//`, `%`,
//! `**`, `<<`, `>>`, `&`, `|`, `^`, comparisons, and unary `+`/`-`/`~`
//! through type-specific fast paths and then the dunder protocol.
#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntGcRoot, live_rbigint};
use num_traits::ToPrimitive;

use pyre_object::unicodeobject::is_str;
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

use crate::baseobjspace::{
    getattr, getitem, is_true, issubtype_w, lookup, lookup_in_type, lookup_in_type_where,
    lookup_where_with_method_cache, p_abstract_issubclass_w,
};
pub use crate::{PyError, PyErrorKind, PyResult};

/// Every zero-divisor `ZeroDivisionError` carries this one message, whatever
/// the operator and whatever the operand types — int, long, float or complex,
/// `/`, `//`, `%` or `divmod`. The per-operator and per-type wordings
/// ("integer division or modulo by zero", "float modulo", …) were unified in
/// 3.12. `0 ** -1` is the one ZeroDivisionError that keeps its own message
/// ("zero to a negative power"), because it is not a division.
const ZERO_DIVISION_MSG: &str = "division by zero";

// ── BigInt helpers ──────────────────────────────────────────────────

/// Box a BigInt result, demoting to W_IntObject if it fits in i64.

pub(crate) fn bigint_result(value: BigInt) -> PyObjectRef {
    // `fits_int` / `toint` only read `_size` and digits; they do not allocate.
    // Boxing is the single allocation, and `w_long_new` roots `_digits` across
    // that payload malloc itself.
    if jit_bigint_to_i64_fits(&value) != 0 {
        w_int_new(jit_bigint_to_i64_value(&value))
    } else {
        w_long_new(value)
    }
}

/// Root `obj`'s `rbigint` for a caller that reads it again after its own
/// collecting call. An arm that reads the value once and hands it to one
/// `rbigint` method reads [`w_long_get_value`] instead: the method roots its
/// receiver itself before it allocates (`_x_int_add`, `int_mod_int_result`),
/// and `descr_binop` holds no livevar across a collecting call
/// (`gctransform/framework.py` `get_livevars_for_roots`).
unsafe fn live_long_num(obj: PyObjectRef) -> RBigIntGcRoot {
    live_rbigint(w_long_get_value(obj))
}

/// CPython `long_invmod` / PyPy three-argument `pow` inverse step.
///
/// Keep the Bézout coefficients as rbigints: the loop is the ordinary
/// extended Euclidean algorithm and never crosses into an opaque host bigint.
fn bigint_mod_inverse(base: &BigInt, modulus: &BigInt) -> Result<BigInt, PyError> {
    let base = RBigIntGcRoot::new(base.translated_alias());
    let modulus = RBigIntGcRoot::new(modulus.translated_alias());
    let mut old_r = RBigIntGcRoot::new(
        base.r#mod(&modulus)
            .map_err(|_| PyError::value_error("pow() 3rd argument cannot be 0"))?,
    );
    let mut r = RBigIntGcRoot::new(modulus.translated_alias());
    let mut old_s = RBigIntGcRoot::new(BigInt::one());
    let mut s = RBigIntGcRoot::new(BigInt::zero());
    while !r.is_zero() {
        let quotient =
            RBigIntGcRoot::new(old_r.floordiv(&r).map_err(|_| {
                PyError::value_error("base is not invertible for the given modulus")
            })?);
        let qr = RBigIntGcRoot::new(quotient.mul(&r));
        let next_r = RBigIntGcRoot::new(old_r.sub(&qr));
        old_r = r;
        r = next_r;
        let qs = RBigIntGcRoot::new(quotient.mul(&s));
        let next_s = RBigIntGcRoot::new(old_s.sub(&qs));
        old_s = s;
        s = next_s;
    }
    if !old_r.int_eq(1) {
        return Err(PyError::value_error(
            "base is not invertible for the given modulus",
        ));
    }
    old_s
        .r#mod(&modulus)
        .map_err(|_| PyError::value_error("pow() 3rd argument cannot be 0"))
}

/// `rbigint.add_int_int_bigint_result`, the sum half of `_make_ovf2long`
/// (intobject.py).  The overflow recovery takes two Signed words and
/// reaches the exact bigint sum in one elidable call, with no rbigint
/// allocated for either operand; the MIR front retargets this seam to
/// `jit_bigint_add_int_int` the way [`bigint_lshift_int_int_result`] is
/// retargeted for the shift arm.
#[majit_macros::jit_elidable]
fn bigint_add_int_int(a: i64, b: i64) -> BigInt {
    BigInt::add_int_int_bigint_result(a, b)
}

/// `rbigint.sub_int_int_bigint_result`; see [`bigint_add_int_int`].
#[majit_macros::jit_elidable]
fn bigint_sub_int_int(a: i64, b: i64) -> BigInt {
    BigInt::sub_int_int_bigint_result(a, b)
}

/// `rbigint.mul_int_int_bigint_result`; see [`bigint_add_int_int`].
#[majit_macros::jit_elidable]
fn bigint_mul_int_int(a: i64, b: i64) -> BigInt {
    BigInt::mul_int_int_bigint_result(a, b)
}

/// Host spelling of the already-zero-checked quotient half of
/// `rbigint.divmod`. The MIR front retargets this exact seam to
/// `jit_bigint_div_floor`, restoring RPython's one-GCREF CALL_PURE result
/// instead of exposing Rust's `Result<(RBigInt, RBigInt), RBigIntError>`.
#[majit_macros::jit_elidable]
fn bigint_floordiv_nonzero(a: &BigInt, b: &BigInt) -> BigInt {
    a.divmod(b).expect("divisor was checked nonzero").0
}

/// Remainder companion of [`bigint_floordiv_nonzero`].
#[majit_macros::jit_elidable]
fn bigint_modulo_nonzero(a: &BigInt, b: &BigInt) -> BigInt {
    a.divmod(b).expect("divisor was checked nonzero").1
}

/// Machine-int-divisor form of [`bigint_floordiv_nonzero`]
/// (`longobject.py _int_floordiv` → `rbigint.int_floordiv`). The dedicated
/// leg divides by a single digit instead of first materializing an rbigint for
/// the `W_IntObject`.
#[majit_macros::jit_elidable]
fn bigint_int_floordiv_nonzero(a: &BigInt, b: i64) -> BigInt {
    a.int_floordiv(b).expect("divisor was checked nonzero")
}

/// Machine-int-divisor form of [`bigint_modulo_nonzero`]
/// (`longobject.py _int_mod` → `rbigint.int_mod_int_result`). The remainder
/// of a long by a machine int always fits a machine int, so the descriptor
/// returns `space.newint` and this leg allocates no result bigint.
#[majit_macros::jit_elidable]
fn bigint_int_modulo_int_result_nonzero(a: &BigInt, b: i64) -> i64 {
    a.int_mod_int_result(b)
        .expect("divisor was checked nonzero")
}

#[majit_macros::elidable]
fn bigint_and(a: BigInt, b: BigInt) -> BigInt {
    a & b
}

#[majit_macros::elidable]
fn bigint_or(a: BigInt, b: BigInt) -> BigInt {
    a | b
}

#[majit_macros::elidable]
fn bigint_xor(a: BigInt, b: BigInt) -> BigInt {
    a ^ b
}

#[majit_macros::elidable]
fn bigint_lshift(a: &BigInt, shift: i64) -> Result<BigInt, majit_rlib::rbigint::RBigIntError> {
    // longobject.py passes the result of rbigint.toint(), an RPython Signed.
    // Keep that word signed and pointer-width independent all the way to
    // rbigint.lshift; narrowing through usize truncates valid counts on wasm32.
    a.lshift(shift)
}

#[majit_macros::elidable]
fn bigint_rshift(a: &BigInt, shift: i64) -> BigInt {
    // Same Signed count shape as bigint_lshift.  The caller has rejected a
    // negative value, so rshift cannot fail here.
    a.rshift(shift, false).expect("nonnegative shift")
}

/// Exact `rbigint._divrem` early-return predicate (`rbigint.py`).
///
/// Upstream deliberately tests only the digit count and most-significant
/// digit here. When it succeeds, the remainder is the input object `a`
/// itself, so pointer-ABI residuals must not allocate a shallow copy.
#[inline]
fn divrem_returns_input_as_remainder(a: &BigInt, b: &BigInt) -> bool {
    let size_a = a.numdigits();
    let size_b = b.numdigits();
    size_a < size_b
        || (size_a == size_b && a.digit((size_a - 1).abs()) < b.digit((size_b - 1).abs()))
}

/// Whether `rbigint.divmod(a, b)` reaches `_divrem`'s literal
/// `(NULLRBIGINT, a)` return without a floored-sign adjustment, so the
/// remainder is `a`'s own payload.
///
/// The MIR front retargets this call to
/// [`jit_bigint_divrem_returns_lhs_remainder`], keeping both operands the GC
/// references they are rather than machine words that outlive a minor
/// collection.
#[majit_macros::dont_look_inside]
fn bigint_divrem_returns_lhs_remainder(a: &BigInt, b: &BigInt) -> bool {
    a.get_sign() != 0
        && a.get_sign() == b.get_sign()
        && b.numdigits() != 1
        && divrem_returns_input_as_remainder(a, b)
}

/// Host form of `rbigint.add` used by `long_add`.
///
/// The MIR front retargets this call to `jit_bigint_add` and models the
/// result as one GC reference, the same way [`bigint_pow_nomod`] becomes
/// `jit_bigint_pow_nomod`. A zero sign returns the other operand's payload;
/// otherwise the sum is a fresh payload.
#[majit_macros::dont_look_inside]
fn bigint_add(a: &BigInt, b: &BigInt) -> *mut BigInt {
    if a.get_sign() == 0 {
        return b as *const BigInt as *mut BigInt;
    }
    if b.get_sign() == 0 {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(a.add(b))
}

/// Host form of `rbigint.sub` (`rbigint.py`). A zero subtrahend returns
/// `self`; the MIR front retargets this call to `jit_bigint_sub`.
#[majit_macros::dont_look_inside]
fn bigint_sub(a: &BigInt, b: &BigInt) -> *mut BigInt {
    if b.get_sign() == 0 {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(a.sub(b))
}

/// Host form of `rbigint.mul` (`rbigint.py`). A zero factor returns the
/// shared zero inside the elidable multiply; the MIR front retargets this
/// call to `jit_bigint_mul`.
#[majit_macros::dont_look_inside]
fn bigint_mul(a: &BigInt, b: &BigInt) -> *mut BigInt {
    pyre_object::longobject::alloc_bigint_nursery(a.mul(b))
}

/// Host form of `rbigint.mod` (`rbigint.py` `mod` / `_divmod` / `_divrem`).
/// A same-sign divisor of more than one digit returns `self` when `|a| < |b|`;
/// a one-digit divisor goes through `int_divmod` and constructs, and opposite
/// signs construct. The MIR front retargets this call to `jit_bigint_mod_floor`.
#[majit_macros::dont_look_inside]
fn bigint_mod(a: &BigInt, b: &BigInt) -> *mut BigInt {
    let selfsign = a.get_sign();
    let othersign = b.get_sign();
    if selfsign != 0
        && othersign != 0
        && selfsign == othersign
        && b.numdigits() != 1
        && divrem_returns_input_as_remainder(a, b)
    {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(
        a.divmod(b).expect("divisor was checked nonzero").1,
    )
}

/// Host form of `rbigint.int_add`. `iother == 0` returns `self` unless
/// `self` is already zero (`rbigint.py` `int_add`). The MIR front retargets
/// this call to `jit_bigint_int_add`.
#[majit_macros::dont_look_inside]
fn bigint_int_add(a: &BigInt, b: i64) -> *mut BigInt {
    if a.get_sign() != 0 && b == 0 {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(a.int_add(b))
}

/// Host form of `rbigint.int_sub`. `iother == 0` returns `self`
/// (`rbigint.py` `int_sub`). Retargeted to `jit_bigint_int_sub`.
#[majit_macros::dont_look_inside]
fn bigint_int_sub(a: &BigInt, b: i64) -> *mut BigInt {
    if b == 0 {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(a.int_sub(b))
}

/// Host form of `rbigint.int_mul`. A positive one digit returns `self`
/// when `self` is nonzero (`rbigint.py` `int_mul`). Retargeted to
/// `jit_bigint_int_mul`.
#[majit_macros::dont_look_inside]
fn bigint_int_mul(a: &BigInt, b: i64) -> *mut BigInt {
    if a.get_sign() != 0 && b == 1 {
        return a as *const BigInt as *mut BigInt;
    }
    pyre_object::longobject::alloc_bigint_nursery(a.int_mul(b))
}

/// Host form of `rbigint.pow(a, b, None)` used by `long_pow`.
///
/// The MIR front erases this Rust `Result` carrier back to RPython's implicit
/// exception edge and retargets the payload call to `jit_bigint_pow_nomod`.
/// Keeping the conversion here lets the ordinary interpreter retain its
/// native `PyResult` contract without exposing `RBigIntError` to translated
/// callers.
#[majit_macros::dont_look_inside]
fn bigint_pow_nomod(a: &BigInt, b: &BigInt) -> Result<BigInt, PyError> {
    // `rbigint.pow` performs several collecting digit-array allocations while
    // both operands remain live. RPython's GC transform roots those incoming
    // handles for the whole call.
    let a = RBigIntGcRoot::new(a.translated_alias());
    let b = RBigIntGcRoot::new(b.translated_alias());
    a.pow(&b, None)
        .map_err(|_| PyError::memory_error("exponent too large"))
}

/// Machine-int-exponent form of [`bigint_pow_nomod`]. `descr_pow` keeps a
/// `W_IntObject` exponent unwrapped and calls `rbigint.int_pow`
/// (`longobject.py:230`); only a long exponent reaches `rbigint.pow`.
#[majit_macros::dont_look_inside]
fn bigint_int_pow_nomod(a: &BigInt, b: i64) -> Result<BigInt, PyError> {
    // `int_pow` also keeps the base live across its intermediate allocations.
    let a = RBigIntGcRoot::new(a.translated_alias());
    a.int_pow(b, None)
        .map_err(|_| PyError::memory_error("exponent too large"))
}

/// Source-level spelling of `rbigint.lshift`'s implicit MemoryError edge.
///
/// `long_lshift` has already validated the count exactly like
/// longobject.py:372-381.  The MIR front retargets this Result carrier to the
/// GCREF-returning `jit_bigint_lshift_count` residual, just as it does for
/// `bigint_pow_nomod`.
#[majit_macros::dont_look_inside]
fn bigint_lshift_count(a: &BigInt, shift: i64) -> Result<BigInt, PyError> {
    a.lshift(shift).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => PyError::value_error("negative shift count"),
    })
}

/// Machine-int-specialized counterpart of [`bigint_lshift_count`].
///
/// intobject.py calls
/// `rbigint.lshift_int_int_bigint_result(a, b)` directly on two Signed words.
/// Rust exposes its implicit MemoryError as `Result`; the MIR front retargets
/// this exact carrier to `jit_bigint_lshift_int_int_result`, preserving the
/// specialized no-input-rbigint-allocation shape.
#[majit_macros::dont_look_inside]
fn bigint_lshift_int_int_result(iself: i64, shift: i64) -> Result<BigInt, PyError> {
    BigInt::lshift_int_int_bigint_result(iself, shift).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => PyError::value_error("negative shift count"),
    })
}

#[majit_macros::elidable]
fn bigint_to_f64(a: BigInt) -> f64 {
    jit_bigint_to_f64_or_inf(&a)
}

/// RPython `_divrem`'s truncated quotient over two bare RBigInt payloads.
/// Allocates the result via the COLLECTING nursery (a gcmap-rooted residual,
/// its operand pointers rooted across the alloc), matching the arithmetic
/// residuals. Returns a freshly heap-allocated `*mut BigInt` encoded as the
/// JIT's uniform i64 word; the MIR retarget keeps the result modeled as GcRef.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_div(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                majit_rlib::rbigint::_divrem(&*a, &*b)
                    .expect("division by zero")
                    .0,
            ),
        )
    }
}

/// `_divrem`'s truncated remainder.
/// See [`jit_bigint_div`]; both project the same upstream helper.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_rem(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        if (&*b).get_sign() != 0 && divrem_returns_input_as_remainder(&*a, &*b) {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                majit_rlib::rbigint::_divrem(&*a, &*b)
                    .expect("division by zero")
                    .1,
            ),
        )
    }
}

/// `rbigint.divmod`'s floored quotient projection. Allocates via the collecting nursery
/// (a gcmap-rooted residual, its operand pointers rooted across the alloc),
/// matching the arithmetic residuals. Returns a freshly heap-allocated
/// `*mut BigInt` encoded as the JIT's uniform i64 word; the MIR retarget keeps
/// the result modeled as a traced GcRef.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_div_floor(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).divmod(&*b).expect("division by zero").0,
            ),
        )
    }
}

/// Machine-int-divisor quotient (`longobject.py _int_floordiv` →
/// `rbigint.int_floordiv`). `b` is a bare machine word, not a payload pointer.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_div_floor(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        // rbigint.int_floordiv: a positive bigint divided by +1 returns
        // `self`, i.e. the identical translated GC reference.
        if b == 1 && (&*a).get_sign() == 1 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).int_floordiv(b).expect("division by zero"),
            ),
        )
    }
}

/// Machine-int-divisor remainder (`longobject.py _int_mod` →
/// `rbigint.int_mod_int_result`). The remainder of a long by a machine int
/// always fits a machine int, so this residual returns the value itself and
/// allocates nothing.
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_int_mod_int_result(a: *const BigInt, b: i64) -> i64 {
    unsafe { (&*a).int_mod_int_result(b).expect("division by zero") }
}

/// Both halves of a machine-int divmod (`longobject.py _int_divmod` →
/// `rbigint.int_divmod`, rbigint.py `@jit.elidable`). `b` is a bare
/// machine word, not a payload pointer.
///
/// One call produces both results: `//` and `%` each reach `_divmod`, so
/// splitting this into two residuals would run the division twice. The result
/// is the RPython `tuple2` the elidable returns, and the caller reads its two
/// halves with `getfield_gc_r` before boxing each as a `W_LongObject` — the
/// wrappers and the interpreter tuple stay in traced code, where they remain
/// virtualizable.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_divmod(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntPairResult {
    unsafe {
        let (div, modulo) = (&*a).int_divmod(b).expect("division by zero");
        pyre_object::longobject::encode_jit_bigint_pair_result(
            pyre_object::longobject::alloc_bigint_pair_nursery_collecting(div, modulo),
        )
    }
}

/// `rbigint.divmod` (`@jit.elidable`) with both halves in one `tuple2`,
/// the two-rbigint counterpart of [`jit_bigint_int_divmod`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_divmod(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntPairResult {
    unsafe {
        let (div, modulo) = (&*a).divmod(&*b).expect("division by zero");
        pyre_object::longobject::encode_jit_bigint_pair_result(
            pyre_object::longobject::alloc_bigint_pair_nursery_collecting(div, modulo),
        )
    }
}

/// `rbigint.divmod`'s floored modulus projection.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mod_floor(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        let selfsign = (&*a).get_sign();
        let othersign = (&*b).get_sign();
        // `divmod` routes one-digit divisors through `int_divmod`, which
        // constructs its remainder. Otherwise `_divmod_small` preserves
        // `_divrem`'s `a` remainder when no opposite-sign correction occurs.
        if selfsign != 0
            && othersign != 0
            && selfsign == othersign
            && (&*b).numdigits() != 1
            && divrem_returns_input_as_remainder(&*a, &*b)
        {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).divmod(&*b).expect("division by zero").1,
            ),
        )
    }
}

// ── BigInt binary-operator residuals ─────────────────────────────────
// Rust trait-operator shells (`<RBigInt as BitAnd>::bitand`, …) obscure the
// RPython method identity in LLBC, so a traced-into caller emits
// an unregistered `<Impl>` FunctionPath call the census Skips. `front::mir`
// retargets each such call — guarded on both operands resolving to the
// opaque `BigInt` ADT — to the matching residual below. Both operands and
// the result are the classdef-less `*mut BigInt` GcRef the front models a
// `BigInt` as. Operands are `*const BigInt`, so `FUNC.ARGS` is Ref —
// `history.getkind` banks a GC pointer as `"ref"`. The result is
// `JitBigIntResult`. wasm32 still passes that pointer in the uniform i64
// word (`WordArg::from_word`); the word is the call ABI, not `FUNC.ARGS`. Each
// allocates the fresh result in the collecting nursery, its operand pointers
// rooted across the alloc. These wrappers retain RPython's elidable effect:
// allocation-only operations use `EF_ELIDABLE_OR_MEMORYERROR`.

/// `rbigint.and_` payload — `&BigInt & &BigInt`. See the module note above.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_and(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::and_(&*a, &*b)),
        )
    }
}

/// `rbigint.or_` payload — `&BigInt | &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_or(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::or_(&*a, &*b)),
        )
    }
}

/// `rbigint.xor_` payload — `&BigInt ^ &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_xor(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::xor(&*a, &*b)),
        )
    }
}

/// `rbigint.sub` payload — `&BigInt - &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        if (&*b).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::sub(&*a, &*b)),
        )
    }
}

/// `rbigint.mul` payload — `&BigInt * &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::mul(&*a, &*b)),
        )
    }
}

/// `rbigint.add` payload — `&BigInt + &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    unsafe {
        if (&*a).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(b as *mut BigInt);
        }
        if (&*b).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(BigInt::add(&*a, &*b)),
        )
    }
}

/// `rbigint.add_int_int_bigint_result` payload — the exact bigint sum of two
/// machine ints, with no rbigint built for either operand. Pointer-ABI form of
/// [`bigint_add_int_int`]; see [`jit_bigint_and`] for the result encoding.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add_int_int(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    pyre_object::longobject::encode_jit_bigint_result(
        pyre_object::longobject::alloc_bigint_nursery_collecting(
            BigInt::add_int_int_bigint_result(a, b),
        ),
    )
}

/// `rbigint.sub_int_int_bigint_result` payload. See [`jit_bigint_add_int_int`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub_int_int(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    pyre_object::longobject::encode_jit_bigint_result(
        pyre_object::longobject::alloc_bigint_nursery_collecting(
            BigInt::sub_int_int_bigint_result(a, b),
        ),
    )
}

/// `rbigint.mul_int_int_bigint_result` payload. See [`jit_bigint_add_int_int`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul_int_int(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    pyre_object::longobject::encode_jit_bigint_result(
        pyre_object::longobject::alloc_bigint_nursery_collecting(
            BigInt::mul_int_int_bigint_result(a, b),
        ),
    )
}

// ── BigInt/machine-int arithmetic residuals ─────────────────────────
//
// pypy/objspace/std/longobject.py:_make_generic_descr_binop and descr_sub
// call rbigint.int_{add,sub,mul,and_,or_,xor} whenever the other operand is
// a W_IntObject.  Those methods are @jit.elidable and return one rbigint GC
// reference.  The interpreter LLBC sees RBigInt as an opaque dependency ADT,
// so front::mir retargets the inherent method calls to these pointer-ABI
// residuals instead of allocating a temporary RBigInt for the machine word.

macro_rules! bigint_int_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt/machine-int residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_or_memerror]
        pub extern "C" fn $name(
            a: *const BigInt,
            b: i64,
        ) -> pyre_object::longobject::JitBigIntResult {
            unsafe {
                pyre_object::longobject::encode_jit_bigint_result(
                    pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).$method(b)),
                )
            }
        }
    };
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_add(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        if b == 0 && (&*a).get_sign() != 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_add(b)),
        )
    }
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_sub(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        if b == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_sub(b)),
        )
    }
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_mul(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        if b == 1 && (&*a).get_sign() != 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_mul(b)),
        )
    }
}

bigint_int_residual!(jit_bigint_int_and, int_and_);
bigint_int_residual!(jit_bigint_int_or, int_or_);
bigint_int_residual!(jit_bigint_int_xor, int_xor);

macro_rules! bigint_int_comparison_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt/machine-int comparison residual using the translated GC-reference ABI."]
        // rbigint.py `_make_int_comparison`: these helpers only inspect the
        // existing digits and return a bool; unlike the arithmetic int_*
        // family above, they allocate nothing and cannot raise.
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(a: *const BigInt, b: i64) -> i64 {
            unsafe { (&*a).$method(b) as i64 }
        }
    };
}

bigint_int_comparison_residual!(jit_bigint_int_eq, int_eq);
bigint_int_comparison_residual!(jit_bigint_int_ne, int_ne);
bigint_int_comparison_residual!(jit_bigint_int_lt, int_lt);
bigint_int_comparison_residual!(jit_bigint_int_le, int_le);
bigint_int_comparison_residual!(jit_bigint_int_gt, int_gt);
bigint_int_comparison_residual!(jit_bigint_int_ge, int_ge);

/// `rbigint.pow(a, b, None)` after `longobject._pow_nomod` has rejected a
/// negative exponent and handled the 0/±1 fast paths.  In that domain the
/// only remaining explicit rbigint failure is MemoryError.  RPython carries
/// it as the implicit exception edge of an `EF_ELIDABLE_OR_MEMORYERROR` call;
/// publish the same backend exception and return an ignored null payload.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_pow_nomod(
    a: *const BigInt,
    b: *const BigInt,
) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a, b);
    // rbigint.pow(..., modulus=None): exponent +1 returns `self`.
    if unsafe { (&*a).get_sign() != 0 && (&*b).int_eq(1) } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    // The residual call can collect inside `rbigint.pow`; mirror the roots
    // inserted for both arguments by RPython's GC transform instead of
    // depending on a backend-specific native call map.
    let a = RBigIntGcRoot::new(unsafe { (&*a).translated_alias() });
    let b = RBigIntGcRoot::new(unsafe { (&*b).translated_alias() });
    match a.pow(&b, None) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// Machine-int-exponent form of [`jit_bigint_pow_nomod`]
/// (`longobject.py:230` → `rbigint.int_pow`). `b` is a bare machine word.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_pow_nomod(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    // rbigint.int_pow(..., modulus=None): exponent 1 returns `self`.
    if b == 1 && unsafe { (&*a).get_sign() != 0 } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    // `int_pow` retains the base across intermediate collecting allocations.
    let a = RBigIntGcRoot::new(unsafe { (&*a).translated_alias() });
    match a.int_pow(b, None) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// `rbigint.lshift(a, machine_count)` after the Python-level sign/range
/// checks. RPython exposes only its implicit MemoryError edge.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_lshift_count(
    a: *const BigInt,
    shift: i64,
) -> pyre_object::longobject::JitBigIntResult {
    // rbigint.lshift returns `self` for both a zero count and a zero base.
    if shift == 0 || unsafe { (&*a).get_sign() == 0 } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    match unsafe { (&*a).lshift(shift) } {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// intobject.py's Signed×Signed specialized overflow leg. This is the direct
/// pointer-ABI form of [`bigint_lshift_int_int_result`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_lshift_int_int_result(
    iself: i64,
    shift: i64,
) -> pyre_object::longobject::JitBigIntResult {
    match BigInt::lshift_int_int_bigint_result(iself, shift) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// `rbigint.neg` payload — `-&BigInt`. A unary operator, so a single operand
/// pointer; the result is a fresh negated `BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_neg(a: *const BigInt) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            // rbigint.py always constructs a fresh rbigint handle,
            // including for zero; only its immutable digits are shared.
            majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting((&*a).neg()),
        )
    }
}

/// Pointer-ABI residual of [`bigint_divrem_returns_lhs_remainder`].
///
/// Keeping it residual avoids rebuilding the opaque RBigInt field graph in
/// `long_mod` while retaining an elidable/cannot-raise operation.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_bigint_divrem_returns_lhs_remainder(
    a: *const BigInt,
    b: *const BigInt,
) -> i64 {
    let (a, b) = (a, b);
    unsafe { bigint_divrem_returns_lhs_remainder(&*a, &*b) as i64 }
}

/// `rbigint.invert` payload. Zero returns the canonical -1 prebuilt; every
/// other input follows `int_add(1)` and sign inversion.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_invert(a: *const BigInt) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).invert()),
        )
    }
}

// ── BigInt shift-by-`usize` residuals ────────────────────────────────
// `<BigInt as Shl<usize>>::shl` / `Shr<usize>::shr` — the shift amount is a
// plain machine integer, NOT a `BigInt`, so `b` is the count value itself
// (not a pointer). `front::mir` retargets these separately, guarded on the
// first operand being the opaque `BigInt` ADT and the second being an
// integer. See the operator-residual module note above.

/// `rbigint.lshift(a, count)`, with the count kept as the `i64` `lshift`
/// takes.  Narrowing it to a machine `usize` first truncates a count of
/// 2**32 or more on a 32-bit target, which answers with the operand shifted
/// by the low word instead of by the count.  A count the result cannot
/// represent publishes MemoryError and answers null, as
/// [`jit_bigint_lshift_count`] does; `Shl` would panic instead.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_shl(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        if b == 0 || (&*a).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        match (&*a).lshift(b) {
            Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
                pyre_object::longobject::alloc_bigint_nursery_collecting(value),
            ),
            Err(_) => {
                crate::runtime_ops::jit_publish_exception(
                    pyre_object::interp_exceptions::memory_error_singleton(),
                );
                pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
            }
        }
    }
}

/// `rbigint.rshift(a, count)`, with the count kept as the `i64` `rshift`
/// takes.  Narrowing it to a machine `usize` first truncates a count of
/// 2**32 or more on a 32-bit target, which answers with the operand
/// unshifted where the shift is a total right shift.  A negative count
/// publishes MemoryError and answers null rather than panicking through
/// `Shr`; the JIT arm guards it non-negative before the call.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_shr(
    a: *const BigInt,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    unsafe {
        if b == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        match (&*a).rshift(b, false) {
            Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
                pyre_object::longobject::alloc_bigint_nursery_collecting(value),
            ),
            Err(_) => {
                crate::runtime_ops::jit_publish_exception(
                    pyre_object::interp_exceptions::memory_error_singleton(),
                );
                pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
            }
        }
    }
}

#[majit_macros::elidable]
fn float_copysign(mag: f64, sign: f64) -> f64 {
    if sign.is_sign_negative() {
        -mag.abs()
    } else {
        mag.abs()
    }
}

#[majit_macros::elidable]
fn bigint_neg(a: &BigInt) -> BigInt {
    a.neg()
}

#[majit_macros::elidable]
fn bigint_clone(a: &BigInt) -> BigInt {
    a.clone()
}

#[majit_macros::elidable]
fn bigint_invert(a: &BigInt) -> BigInt {
    a.invert()
}

#[majit_macros::elidable]
pub(crate) fn bigint_eq(a: BigInt, b: BigInt) -> bool {
    a == b
}

#[majit_macros::elidable]
fn bigint_lt(a: BigInt, b: BigInt) -> bool {
    a < b
}

#[majit_macros::elidable]
fn bigint_gt(a: BigInt, b: BigInt) -> bool {
    a > b
}

/// longobject.py `_truediv` delegates directly to rbigint.truediv and
/// only translates its application-level exceptions.
#[majit_macros::elidable]
fn bigint_truediv(a: &BigInt, b: &BigInt) -> Result<f64, PyError> {
    a.truediv(b).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::DivisionByZero => {
            PyError::zero_division(ZERO_DIVISION_MSG)
        }
        majit_rlib::rbigint::RBigIntError::FloatDivisionOverflow => {
            PyError::overflow_error("integer division result too large for a float")
        }
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => unreachable!("rbigint.truediv has no other exception edge"),
    })
}

// ── Arithmetic operations ─────────────────────────────────────────────

/// Integer addition fast path.
///
/// The JIT will specialize this via:
///   GuardClass(a, &INT_TYPE)
///   GuardClass(b, &INT_TYPE)
///   GetfieldGcI(a, intval_offset) → va
///   GetfieldGcI(b, intval_offset) → vb
///   IntAdd(va, vb) → result
///   New(W_IntObject) + SetfieldGcI(result)

unsafe fn int_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    _int_add(va, vb)
}

/// intobject.py `descr_add` after the two `intval` reads: `ovfcheck` then
/// `space.newint`, or `rbigint` on overflow.
///
/// The success arm boxes with `malloc_typed_managed`, the constructor
/// [`_float_add`] uses, so `fuse_boxing_alloc` rewrites it to
/// `new_with_vtable`. [`w_int_new`]'s collector allocation is
/// [`w_int_gc_alloc`] (`dont_look_inside`); a binop-rewind inline refuses
/// that residual. The overflow arm stays [`_int_add_ovf`] so the success
/// trace only guards it.
#[inline(never)]
pub(crate) fn _int_add(x: i64, y: i64) -> PyResult {
    let Some(r) = x.checked_add(y) else {
        return _int_add_ovf(x, y);
    };
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: r,
    }) as PyObjectRef)
}

/// `_ovf2long` of `descr_add`. Residual so a success trace of [`_int_add`]
/// does not record `rbigint.add`.
#[majit_macros::dont_look_inside]
fn _int_add_ovf(x: i64, y: i64) -> PyResult {
    Ok(w_long_new(bigint_add_int_int(x, y)))
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    _int_sub(va, vb)
}

/// intobject.py `descr_sub` after the two `intval` reads: `ovfcheck` then
/// `wrapint`, or `_make_ovf2long` on overflow. Same split as [`_int_add`].
/// The constructor stays in this leaf: a binop-rewind inline refuses
/// [`w_int_new`]'s [`w_int_gc_alloc`], and `fuse_boxing_alloc` only rewrites
/// a `malloc_typed_managed` it can see in the descended body.
#[inline(never)]
pub(crate) fn _int_sub(x: i64, y: i64) -> PyResult {
    let Some(r) = x.checked_sub(y) else {
        return _int_sub_ovf(x, y);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let w_class = pyre_object::gc_roots::pin_root(get_instantiate(&INT_TYPE));
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class,
        },
        intval: r,
    }) as PyObjectRef)
}

/// `_ovf2long` of `descr_sub`. Residual so a success trace of [`_int_sub`]
/// does not record `rbigint.sub`.
#[majit_macros::dont_look_inside]
fn _int_sub_ovf(x: i64, y: i64) -> PyResult {
    Ok(w_long_new(bigint_sub_int_int(x, y)))
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    _int_mul(va, vb)
}

/// intobject.py `descr_mul` after the two `intval` reads: `ovfcheck` then
/// `wrapint`, or `_make_ovf2long` on overflow. Same split as [`_int_add`].
#[inline(never)]
pub(crate) fn _int_mul(x: i64, y: i64) -> PyResult {
    let Some(r) = x.checked_mul(y) else {
        return _int_mul_ovf(x, y);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let w_class = pyre_object::gc_roots::pin_root(get_instantiate(&INT_TYPE));
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class,
        },
        intval: r,
    }) as PyObjectRef)
}

/// `_ovf2long` of `descr_mul`. Residual so a success trace of [`_int_mul`]
/// does not record `rbigint.mul`.
#[majit_macros::dont_look_inside]
fn _int_mul_ovf(x: i64, y: i64) -> PyResult {
    Ok(w_long_new(bigint_mul_int_int(x, y)))
}

/// intobject.py `_truediv(space, x, y)` success body: zero, then
/// `space.newfloat(float(x)/float(y))`.  The mantissa overflow
/// (`r_uint(abs(n)) >> DBL_MANT_DIG`) stays in [`int_truediv`]:
/// putting that diamond here lowers to a backward `goto` and hung
/// the helper walk.  Charon cannot inline the cross-crate `newfloat`
/// helper, so the success constructor lives here and
/// `fuse_boxing_alloc` rewrites `malloc_typed_managed` to
/// `new_with_vtable`.
#[inline(never)]
pub(crate) fn _truediv(x: i64, y: i64) -> PyResult {
    if y == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let value = (x as f64) / (y as f64);
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: value,
    }) as PyObjectRef)
}

/// `descr_truediv`: unbox, then `_truediv`, catching the mantissa
/// overflow as `_make_ovf2long('truediv')`.  The overflow diamond
/// cannot live in [`_truediv`]: it lowers to a backward `goto` and
/// hung the helper walk.
pub(crate) unsafe fn int_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = int_value(a);
    let y = int_value(b);
    const MANTISSA_LIM: i64 = 1 << 53;
    if x <= -MANTISSA_LIM || x >= MANTISSA_LIM || y <= -MANTISSA_LIM || y >= MANTISSA_LIM {
        return int_truediv_ovf2long(x, y);
    }
    _truediv(x, y)
}

/// `_make_ovf2long('truediv')`: `W_LongObject.fromint` then
/// `descr_truediv`. Residual so a `_truediv` walk does not record
/// `rbigint.truediv`.
#[majit_macros::dont_look_inside]
fn int_truediv_ovf2long(va: i64, vb: i64) -> PyResult {
    // `fromint` of `vb` collects.
    let a = RBigIntGcRoot::new(BigInt::from(va));
    let b = BigInt::from(vb);
    Ok(w_float_new(bigint_truediv(&a, &b)?))
}

/// longobject.py `_truediv` / mixed long-int.  At least one operand is
/// `W_LongObject`; machine-int pairs stay in [`int_truediv`].
unsafe fn long_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `fromint` of a machine-int operand collects before the other operand
    // is read. `descr_truediv` keeps both live (`gct_direct_call` /
    // `get_livevars_for_roots`).
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    if !is_long(b()) && as_float(b()) == 0.0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let va = if is_long(a()) {
        live_long_num(a())
    } else {
        RBigIntGcRoot::new(BigInt::from(int_value(a())))
    };
    let vb = if is_long(b()) {
        live_long_num(b())
    } else {
        RBigIntGcRoot::new(BigInt::from(int_value(b())))
    };
    Ok(w_float_new(bigint_truediv(&va, &vb)?))
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // intobject.py `_floordiv`: `ovfcheck(x // y)` has exactly one
    // non-zero-divisor overflow on a signed machine word.
    if va == -9_223_372_036_854_775_808_i64 && vb == -1 {
        // `fromint` of `vb` collects.
        let va = RBigIntGcRoot::new(BigInt::from(va));
        let vb = BigInt::from(vb);
        return Ok(bigint_result(bigint_floordiv_nonzero(&va, &vb)));
    }
    Ok(w_int_new(ll_int_py_div(va, vb)))
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // intobject.py `_mod`: the matching machine-word overflow is
    // `MIN % -1`; bounce that one case to rbigint like `ovfcheck`.
    if va == -9_223_372_036_854_775_808_i64 && vb == -1 {
        // `fromint` of `vb` collects.
        let va = RBigIntGcRoot::new(BigInt::from(va));
        let vb = BigInt::from(vb);
        return Ok(bigint_result(bigint_modulo_nonzero(&va, &vb)));
    }
    Ok(w_int_new(ll_int_py_mod(va, vb)))
}

/// rint.py `ll_int_py_div`: the floor quotient, from the truncating machine
/// quotient plus a sign correction.  The `int.py_div` oopspec makes the call
/// an elidable residual the optimizer strength-reduces for a constant
/// divisor (`rewrite.py optimize_call_int_py_div`), which is what the
/// generated `//` descent needs to match the retired hand fold's trace.
/// Both callers check `y != 0` and `x != INT_MIN || y != -1` first, so the
/// `wrapping_*` never wrap.
#[majit_macros::oopspec("int.py_div(x, y)")]
pub(crate) fn ll_int_py_div(x: i64, y: i64) -> i64 {
    let r = x.wrapping_div(y);
    let p = r.wrapping_mul(y);
    let u = if y < 0 {
        p.wrapping_sub(x)
    } else {
        x.wrapping_sub(p)
    };
    // INT_BITS_1 (rarithmetic)
    r.wrapping_add(u >> 63)
}

/// rint.py `ll_int_py_mod`, the remainder companion of [`ll_int_py_div`].
#[majit_macros::oopspec("int.py_mod(x, y)")]
pub(crate) fn ll_int_py_mod(x: i64, y: i64) -> i64 {
    let r = x.wrapping_rem(y);
    let u = if y < 0 { r.wrapping_neg() } else { r };
    // INT_BITS_1 (rarithmetic)
    r.wrapping_add(y & (u >> 63))
}

/// `support.py` `_ll_2_int_floordiv`: no-branch reverse of [`ll_int_py_div`].
///
/// `x // y` is [`ll_int_py_div`] (`oopspec` `int.py_div`). The shift is
/// `LONG_BIT - 1` on Signed. `wrapping_*` keeps this body free of panic
/// edges; a zero divisor still panics inside [`ll_int_py_div`].
pub(crate) extern "C" fn _ll_2_int_floordiv(x: i64, y: i64) -> i64 {
    let r = ll_int_py_div(x, y);
    let p = r.wrapping_mul(y);
    r.wrapping_add(((x ^ y) >> 63) & ((p != x) as i64))
}

/// `support.py` `_ll_2_int_mod`. `x % y` is [`ll_int_py_mod`].
pub(crate) extern "C" fn _ll_2_int_mod(x: i64, y: i64) -> i64 {
    let r = ll_int_py_mod(x, y);
    r.wrapping_sub(y & (((x ^ y) & (r | r.wrapping_neg())) >> 63))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py `_make_generic_descr_binop('add')`: one type dispatch,
    // then `rbigint.int_add` or `rbigint.add`. The zero alias lives inside
    // those elidable operations (`bigint_int_add` / `bigint_add`). PyPy's
    // W_BoolObject subclasses W_IntObject; pyre stores bool separately, so
    // the one upstream W_IntObject arm has two storage projections here.
    if is_long(a) && is_bool(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_add(
            w_long_get_value(a),
            w_bool_get_value(b) as i64,
        )));
    }
    if is_long(a) && is_int(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_add(
            w_long_get_value(a),
            w_int_get_value(b),
        )));
    }
    if is_bool(a) && is_long(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_add(
            w_long_get_value(b),
            w_bool_get_value(a) as i64,
        )));
    }
    if is_int(a) && is_long(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_add(
            w_long_get_value(b),
            w_int_get_value(a),
        )));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(pyre_object::longobject::w_long_from_raw(bigint_add(
        w_long_get_value(a),
        w_long_get_value(b),
    )))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py `descr_sub` specializes only `long - int`; `descr_rsub`
    // keeps `int - long` on the ordinary two-rbigint path. The zero alias
    // lives inside `rbigint.int_sub` / `rbigint.sub`.
    if is_long(a) && is_bool(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_sub(
            w_long_get_value(a),
            w_bool_get_value(b) as i64,
        )));
    }
    if is_long(a) && is_int(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_sub(
            w_long_get_value(a),
            w_int_get_value(b),
        )));
    }
    if is_long(a) {
        debug_assert!(is_long(b));
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_sub(
            w_long_get_value(a),
            w_long_get_value(b),
        )));
    }
    // Reflected int/bool - long follows descr_rsub's ordinary rbigint path:
    // only the machine-word left operand needs materialising.
    debug_assert!(is_long(b));
    let vb = live_long_num(b);
    Ok(w_long_new(BigInt::from(int_value(a)).sub(&vb)))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py `_make_generic_descr_binop('mul')`: commutative
    // `rbigint.int_mul`. The one-alias lives inside `bigint_int_mul`.
    if is_long(a) && is_bool(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_mul(
            w_long_get_value(a),
            w_bool_get_value(b) as i64,
        )));
    }
    if is_long(a) && is_int(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_mul(
            w_long_get_value(a),
            w_int_get_value(b),
        )));
    }
    if is_bool(a) && is_long(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_mul(
            w_long_get_value(b),
            w_bool_get_value(a) as i64,
        )));
    }
    if is_int(a) && is_long(b) {
        return Ok(pyre_object::longobject::w_long_from_raw(bigint_int_mul(
            w_long_get_value(b),
            w_int_get_value(a),
        )));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(pyre_object::longobject::w_long_from_raw(bigint_mul(
        w_long_get_value(a),
        w_long_get_value(b),
    )))
}

/// `_int_floordiv`: a machine-int divisor never becomes an rbigint.
/// The guard runs before the receiver is read, then `rbigint.int_floordiv`,
/// then `newlong`. Nothing allocates between entry and that read.
unsafe fn long_int_floordiv(a: PyObjectRef, other: i64) -> PyResult {
    if other == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    debug_assert!(is_long(a));
    let va = w_long_get_value(a);
    // `int_floordiv` returns `self` for a positive receiver and divisor 1.
    // `newlong` wraps that payload.
    if va.get_sign() == 1 && other == 1 {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    Ok(w_long_new(bigint_int_floordiv_nonzero(va, other)))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `_floordiv`. A machine-int divisor is `_int_floordiv`
    // (`long_int_floordiv`) and does not reach here. `W_LongObject.asbigint`
    // returns `self.num` with no malloc; the reflected `int` receiver is
    // coerced the way `descr_rbinop` does, after its machine word is copied.
    debug_assert!(is_long(b));
    if !is_long(a) {
        // `descr_rbinop` coerces the int receiver first; `b`'s value is read
        // after that malloc.
        let va = BigInt::from(int_value(a));
        let vb = w_long_get_value(b);
        if !vb.tobool() {
            return Err(PyError::zero_division(ZERO_DIVISION_MSG));
        }
        return Ok(w_long_new(bigint_floordiv_nonzero(&va, vb)));
    }
    // Both values are read once and handed to `rbigint.floordiv`, which
    // roots its operands before `_divmod` allocates.
    let va = w_long_get_value(a);
    let vb = w_long_get_value(b);
    if !vb.tobool() {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // rbigint.floordiv → _divmod, returning the quotient half (rbigint.py).
    // `_floordiv` `newlong`s the quotient, keeping a long.
    Ok(w_long_new(bigint_floordiv_nonzero(va, vb)))
}

/// `_int_mod`: `rbigint.int_mod_int_result`, then `space.newint`.
/// The remainder of a long by a machine int always fits a machine int.
unsafe fn long_int_mod(a: PyObjectRef, other: i64) -> PyResult {
    if other == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    debug_assert!(is_long(a));
    let va = w_long_get_value(a);
    Ok(w_int_new(bigint_int_modulo_int_result_nonzero(va, other)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `_mod`. A machine-int divisor is `_int_mod` (`long_int_mod`).
    debug_assert!(is_long(b));
    if !is_long(a) {
        // `descr_rbinop` coerces the int receiver first; `b`'s value is read
        // after that malloc.
        let va = BigInt::from(int_value(a));
        let vb = w_long_get_value(b);
        if !vb.tobool() {
            return Err(PyError::zero_division(ZERO_DIVISION_MSG));
        }
        return Ok(w_long_new(bigint_modulo_nonzero(&va, vb)));
    }
    // Both values are read once: the zero and remainder-is-lhs checks
    // allocate nothing, and `rbigint.mod` roots its operands before
    // `_divmod` allocates.
    let va = w_long_get_value(a);
    let vb = w_long_get_value(b);
    if !vb.tobool() {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // `rbigint.mod` returns the input payload when `_divrem` does. The alias
    // stays inside `bigint_mod`; this descriptor only boxes it.
    Ok(pyre_object::longobject::w_long_from_raw(bigint_mod(va, vb)))
}

/// `_int_divmod`: `rbigint.int_divmod`, then `newtuple2` of two `newlong`s.
unsafe fn long_int_divmod(a: PyObjectRef, other: i64) -> PyResult {
    if other == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    debug_assert!(is_long(a));
    let va = live_long_num(a);
    let (q, r) = va.int_divmod(other).expect("divisor was checked nonzero");
    // Two live rbigints before either wrapper is allocated. The GC transform
    // roots both across those collecting allocations.
    let q = RBigIntGcRoot::new(q);
    let r = RBigIntGcRoot::new(r);
    let mut w_q = w_long_new(q.translated_alias());
    let w_r = pyre_object::with_roots!(w_q => w_long_new(r.translated_alias()));
    Ok(pyre_object::wraptuple2(w_q, w_r))
}

/// PyPy longobject.py `_divmod` / `_int_divmod`: compute both halves with one
/// rbigint division and only then box the pair.  Calling `floordiv` and `mod`
/// separately would run `_divmod` twice.
unsafe fn integer_divmod_pair(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    if is_int_like(a) && is_int_like(b) {
        let va = int_value(a);
        let vb = int_value(b);
        debug_assert_ne!(vb, 0, "numeric_divmod checks the divisor");
        // rint.py `ll_int_py_div_ovf`: the overflow test is not
        // short-circuited, so a known divisor removes it entirely.
        if (va == i64::MIN) & (vb == -1) {
            let (q, r) = BigInt::from(va)
                .int_divmod(vb)
                .expect("divisor was checked nonzero");
            let q = RBigIntGcRoot::new(q);
            let r = RBigIntGcRoot::new(r);
            let mut w_q = bigint_result(q.translated_alias());
            let w_r = pyre_object::with_roots!(w_q => bigint_result(r.translated_alias()));
            return Ok(pyre_object::wraptuple2(w_q, w_r));
        }
        // intobject.py `_divmod`: `z = ovfcheck(x // y)`, `m = x % y`, then
        // `space.newtuple2(space.newint(z), space.newint(m))`.
        let z = ll_int_py_div(va, vb);
        let m = ll_int_py_mod(va, vb);
        let mut w_z = w_int_new(z);
        let w_m = pyre_object::with_roots!(w_z => w_int_new(m));
        return Ok(pyre_object::wraptuple2(w_z, w_m));
    }

    // `_make_descr_binop(_divmod, _int_divmod)`: a machine-int divisor is
    // `_int_divmod`. Reflected int/long follows `_divmod`.
    if is_long(a) && is_int_like(b) {
        return long_int_divmod(a, int_value(b));
    }
    let remainder_aliases_a = if is_long(a) && is_long(b) {
        bigint_divrem_returns_lhs_remainder(w_long_get_value(a), w_long_get_value(b))
    } else {
        false
    };
    debug_assert!(is_long(b));
    let va = if is_long(a) {
        live_long_num(a)
    } else {
        RBigIntGcRoot::new(BigInt::from(int_value(a)))
    };
    let vb = live_long_num(b);
    let (q, r) =
        pyre_object::with_roots!(a, b => va.divmod(&vb)).expect("divisor was checked nonzero");
    // `_divmod` produces two live rbigints before either wrapper is
    // allocated. RPython's GC transform roots both across those collecting
    // allocations.
    let q = RBigIntGcRoot::new(q);
    let r = RBigIntGcRoot::new(r);
    if is_long(a) || is_long(b) {
        let mut w_q = pyre_object::with_roots!(a => w_long_new(q.translated_alias()));
        let w_r = if remainder_aliases_a {
            pyre_object::with_roots!(w_q => pyre_object::longobject::w_long_from_raw(w_long_get_raw_value(a)))
        } else {
            pyre_object::with_roots!(w_q => w_long_new(r.translated_alias()))
        };
        Ok(pyre_object::wraptuple2(w_q, w_r))
    } else {
        let mut w_q = bigint_result(q.translated_alias());
        let w_r = pyre_object::with_roots!(w_q => bigint_result(r.translated_alias()));
        Ok(pyre_object::wraptuple2(w_q, w_r))
    }
}

/// `rbigint.floordiv` payload half (`longobject.py _floordiv` →
/// `rbigint.floordiv` → `divmod`, `rbigint.py @jit.elidable`). Elidable
/// but CAN raise ZeroDivisionError on a zero divisor → `EF_ELIDABLE_CAN_RAISE`:
/// the trace records `CALL_PURE` + `GUARD_NO_EXCEPTION`. Returns a bare
/// `*mut BigInt` (Int) on success; on a zero divisor publishes the exception
/// and returns 0 so the trailing `GUARD_NO_EXCEPTION` deopts.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_floordiv_raw(a: PyObjectRef, b: PyObjectRef) -> i64 {
    let (a, b) = unsafe { (w_long_get_value(a), w_long_get_value(b)) };
    bigint_floordiv_core(a, b, false)
}

/// `rbigint.mod` over `W_LongObject` operands (no-collect, record-time). Same
/// `EF_ELIDABLE_CAN_RAISE` contract as [`jit_w_long_floordiv_raw`].
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_mod_raw(a: PyObjectRef, b: PyObjectRef) -> i64 {
    let (a, b) = unsafe { (w_long_get_value(a), w_long_get_value(b)) };
    bigint_mod_core(a, b, false)
}

/// `rbigint.lshift` over `W_LongObject` operands (no-collect, record-time).
/// Elidable but CAN raise ValueError (negative shift) / OverflowError (shift too
/// large and base nonzero) → `EF_ELIDABLE_CAN_RAISE`. Walker-only (the trait
/// path defers shift to the generic residual).
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_lshift_raw(a: PyObjectRef, b: PyObjectRef) -> i64 {
    let (a, b) = unsafe { (w_long_get_value(a), w_long_get_value(b)) };
    bigint_lshift_core(a, b, false)
}

/// `rbigint.rshift` over `W_LongObject` operands (no-collect, record-time). Like
/// [`jit_w_long_lshift_raw`] but a shift too large yields 0 / -1 (all bits
/// shifted out) instead of OverflowError; only a negative shift raises.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_rshift_raw(a: PyObjectRef, b: PyObjectRef) -> i64 {
    let (a, b) = unsafe { (w_long_get_value(a), w_long_get_value(b)) };
    bigint_rshift_core(a, b, false)
}

/// Allocate a result bigint, choosing the COLLECTING nursery (runtime payload
/// helpers, invoked from a gcmap-rooted residual call) or the NO-COLLECT nursery
/// (record-time wrappers, which hold the operands natively so a collection would
/// move them). See `longobject::alloc_bigint_nursery_collecting`.
#[inline]
fn alloc_result_bigint(value: BigInt, collecting: bool) -> i64 {
    if collecting {
        pyre_object::longobject::alloc_bigint_nursery_collecting(value) as i64
    } else {
        pyre_object::longobject::alloc_bigint_nursery(value) as i64
    }
}

/// `rbigint.floordiv`/`mod`/`lshift`/`rshift` cores over `&BigInt` operands,
/// shared by the collecting runtime payload helpers (`jit_bigint_*`) and the
/// no-collect record-time wrappers (`jit_w_long_*_raw`). Each returns a bare
/// `*mut BigInt` (Int, as i64) on success; a zero divisor / out-of-range shift
/// publishes the exception and returns 0 so the trailing `GUARD_NO_EXCEPTION`
/// deopts.
fn bigint_floordiv_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) == 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::zero_division(ZERO_DIVISION_MSG).to_exc_object(),
        );
        return 0;
    }
    // rbigint.floordiv → _divmod, returning the quotient half (rbigint.py).
    alloc_result_bigint(
        a.divmod(b).expect("divisor was checked nonzero").0,
        collecting,
    )
}

fn bigint_mod_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) == 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::zero_division(ZERO_DIVISION_MSG).to_exc_object(),
        );
        return 0;
    }
    let selfsign = pyre_object::longobject::jit_bigint_sign_i64(a);
    let othersign = pyre_object::longobject::jit_bigint_sign_i64(b);
    if selfsign != 0
        && selfsign == othersign
        && b.numdigits() != 1
        && divrem_returns_input_as_remainder(a, b)
    {
        // `_divmod_small` leaves `_divrem`'s literal `a` remainder unchanged.
        return a as *const BigInt as i64;
    }
    // rbigint.mod → divmod, returning the remainder half.
    alloc_result_bigint(
        a.divmod(b).expect("divisor was checked nonzero").1,
        collecting,
    )
}

fn bigint_lshift_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) < 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::value_error("negative shift count").to_exc_object(),
        );
        return 0;
    }
    // `rbigint.toint()` is a *signed* machine int (i64), so a count above
    // i64::MAX overflows here — not at usize::MAX — matching `_lshift`.
    let shift = if jit_bigint_to_i64_fits(b) != 0 {
        jit_bigint_to_i64_value(b)
    } else {
        if pyre_object::longobject::jit_bigint_sign_i64(a) == 0 {
            return a as *const BigInt as i64;
        }
        crate::runtime_ops::jit_publish_exception(
            PyError::overflow_error("shift count too large").to_exc_object(),
        );
        return 0;
    };
    if shift == 0 || pyre_object::longobject::jit_bigint_sign_i64(a) == 0 {
        return a as *const BigInt as i64;
    }
    match bigint_lshift(a, shift) {
        Ok(value) => alloc_result_bigint(value, collecting),
        Err(majit_rlib::rbigint::RBigIntError::Memory) => {
            crate::runtime_ops::jit_publish_exception(PyError::memory_error("").to_exc_object());
            0
        }
        Err(_) => unreachable!("shift sign/range was validated above"),
    }
}

fn bigint_rshift_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) < 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::value_error("negative shift count").to_exc_object(),
        );
        return 0;
    }
    // `toint()` overflow (count > i64::MAX) takes this branch like `_rshift`;
    // for rshift the result (0 / -1) is the same as an actual huge shift.
    let shift = if jit_bigint_to_i64_fits(b) != 0 {
        jit_bigint_to_i64_value(b)
    } else {
        let val = if pyre_object::longobject::jit_bigint_sign_i64(a) < 0 {
            -1
        } else {
            0
        };
        return alloc_result_bigint(BigInt::from(val), collecting);
    };
    if shift == 0 {
        // rbigint.rshift(..., 0) returns `self`.
        return a as *const BigInt as i64;
    }
    alloc_result_bigint(bigint_rshift(a, shift), collecting)
}

/// `rbigint.floordiv`/`mod`/`lshift`/`rshift` payload halves on bare
/// `*const BigInt` operands — the divmod/shift the walker emits after reading
/// each `W_LongObject` operand's immutable `value` via `GetfieldGc`. Taking
/// the payloads (not the wrappers) keeps these elidable calls pure on the
/// immutable bigints so the optimizer never reorders them ahead of the boxing
/// `setfield_gc`. Allocates the result via the COLLECTING nursery (the call is a
/// gcmap-rooted residual `CallR` holding no unrooted pointer across the alloc).
/// `EF_ELIDABLE_CAN_RAISE`.
///
/// # Safety note: both operands are live GC `rbigint`s (`*const BigInt`) for
/// the duration of the call. The `i64` result is that pointer word.
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_floordiv(a: *const BigInt, b: *const BigInt) -> i64 {
    let (a, b) = unsafe { (&*(a), &*(b)) };
    bigint_floordiv_core(a, b, true)
}

/// `rbigint.mod` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_mod(a: *const BigInt, b: *const BigInt) -> i64 {
    let (a, b) = unsafe { (&*(a), &*(b)) };
    bigint_mod_core(a, b, true)
}

/// `rbigint.lshift` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_lshift(a: *const BigInt, b: *const BigInt) -> i64 {
    let (a, b) = unsafe { (&*(a), &*(b)) };
    bigint_lshift_core(a, b, true)
}

/// `rbigint.rshift` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_rshift(a: *const BigInt, b: *const BigInt) -> i64 {
    let (a, b) = unsafe { (&*(a), &*(b)) };
    bigint_rshift_core(a, b, true)
}

/// `rbigint.truediv` payload half (`longobject.py _truediv` →
/// `rbigint.truediv`, `rbigint.py`). Elidable but CAN raise
/// ZeroDivisionError / OverflowError → `EF_ELIDABLE_CAN_RAISE`: `CALL_PURE_F` +
/// `GUARD_NO_EXCEPTION`. Returns the correctly-rounded quotient as an `f64`
/// directly (a `CallPureF`, the float analogue of `rbigint.truediv` returning a
/// float); the walker then boxes it with `wrapfloat` (transparent
/// `new_with_vtable` + `setfield_gc_f`, mirroring `space.newfloat(f)`). On a
/// raising input publishes the exception and returns garbage (the guard
/// deopts). Walker-only, like the shift helpers.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_truediv_raw(a: PyObjectRef, b: PyObjectRef) -> f64 {
    unsafe {
        match bigint_truediv(w_long_get_value(a), w_long_get_value(b)) {
            Ok(f) => f,
            Err(mut e) => {
                crate::runtime_ops::jit_publish_exception(e.to_exc_object());
                0.0
            }
        }
    }
}

// ── Float arithmetic operations ──────────────────────────────────────

/// Coerce an operand to f64. Works for int, long, and float objects.
unsafe fn as_float(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        // long → f64 (may lose precision for very large values)
        jit_bigint_to_f64_or_inf(w_long_get_value(obj))
    }
}

/// Reject an over-range `int` operand of a value-producing float operator:
/// `PyFloat_AsDouble` raises `OverflowError` when an `int`'s magnitude
/// exceeds f64 range. `v` is the already-extracted [`as_float`] value; a
/// genuine float infinity is preserved, only an over-range `int` raises.
///
/// Split from the f64 extraction so the operator bodies pass plain `f64`
/// values — never a `Result<f64, _>` — into the arithmetic. A `Result`
/// payload mixing `Float` (`Ok`) and `Ref` (`Err`) has no single register
/// kind, so the JIT codewriter cannot flatten it (`emit_list_of_kind`).
unsafe fn reject_float_coercion_overflow(obj: PyObjectRef, v: f64) -> Result<(), PyError> {
    if is_long(obj) && !v.is_finite() {
        return Err(PyError::overflow_error("int too large to convert to float"));
    }
    Ok(())
}

/// Both operands of a float operator as doubles (floatobject.py `_to_float`).
///
/// A long operand's conversion allocates (`rbigint.tofloat`), so both
/// operands stay on the shadow stack and are read back after each
/// conversion.  The bracket closes before the overflow check, whose early
/// return then carries no root cleanup.  At most one operand is a long, so
/// the check order cannot choose between two errors.
unsafe fn float_operands(a: PyObjectRef, b: PyObjectRef) -> Result<(f64, f64), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(a);
    let _ = roots.pin_root(b);
    let va = as_float(roots.get(base));
    let a_overflows = is_long(roots.get(base)) && !va.is_finite();
    let vb = as_float(roots.get(base + 1));
    let b_overflows = is_long(roots.get(base + 1)) && !vb.is_finite();
    drop(roots);
    if a_overflows || b_overflows {
        return Err(PyError::overflow_error("int too large to convert to float"));
    }
    Ok((va, vb))
}

/// A `float`/`complex` power coerces each `int` operand to a double before the
/// power is computed, so an over-range `int` (base or exponent) raises
/// OverflowError up front — even for `1.0 ** huge`, which never reaches the
/// arithmetic. Only an over-range `int` raises; a genuine float infinity does
/// not.
pub(crate) unsafe fn reject_pow_operand_overflow(obj: PyObjectRef) -> Result<(), PyError> {
    if is_long(obj) && !jit_bigint_to_f64_or_inf(w_long_get_value(obj)).is_finite() {
        return Err(PyError::overflow_error("int too large to convert to float"));
    }
    Ok(())
}

/// True if both operands are numeric and at least one is float.

unsafe fn is_float_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    let a_num = is_int(a) || is_float(a) || is_long(a);
    let b_num = is_int(b) || is_float(b) || is_long(b);
    a_num && b_num && (is_float(a) || is_float(b))
}

/// floatobject.py `descr_add` / `descr_sub` / `descr_mul` / `descr_div`
/// after `_to_float`: `W_FloatObject(self.floatval ? w_rhs.floatval)`.
/// Charon cannot inline the cross-crate `newfloat`, so the constructor
/// body lives in each leaf the way it lives in [`_truediv`].
#[inline(never)]
pub(crate) fn _float_add(x: f64, y: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x + y,
    }) as PyObjectRef)
}

#[inline(never)]
pub(crate) fn _float_sub(x: f64, y: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x - y,
    }) as PyObjectRef)
}

#[inline(never)]
pub(crate) fn _float_mul(x: f64, y: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x * y,
    }) as PyObjectRef)
}

#[inline(never)]
pub(crate) fn _float_truediv(x: f64, y: f64) -> PyResult {
    if y == 0.0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x / y,
    }) as PyObjectRef)
}

unsafe fn float_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (va, vb) = float_operands(a, b)?;
    _float_add(va, vb)
}

unsafe fn float_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (va, vb) = float_operands(a, b)?;
    _float_sub(va, vb)
}

unsafe fn float_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (va, vb) = float_operands(a, b)?;
    _float_mul(va, vb)
}

unsafe fn float_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (va, vb) = float_operands(a, b)?;
    _float_truediv(va, vb)
}

/// floatobject.py: descr_floordiv → _divmod_w()[0].
unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (x, y) = float_operands(a, b)?;
    let (floordiv, _mod) = float_divmod_w(x, y)?;
    Ok(w_float_new(floordiv))
}

/// floatobject.py: descr_mod with math_fmod + sign correction.
unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (x, y) = float_operands(a, b)?;
    if y == 0.0 {
        // floatobject.py:526
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let mut m = jit_float_fmod(x, y);
    if m != 0.0 {
        // floatobject.py:529-531: ensure remainder has same sign as denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // floatobject.py:536-538: signed zero — copysign(0.0, y)
        m = float_copysign(0.0, y);
    }
    Ok(w_float_new(m))
}

/// floatobject.py: _divmod_w.
fn float_divmod_w(x: f64, y: f64) -> Result<(f64, f64), PyError> {
    if y == 0.0 {
        // floatobject.py:761
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let mut m = jit_float_fmod(x, y);
    // floatobject.py:767: div = (x - mod) / y
    let mut div = (x - m) / y;
    if m != 0.0 {
        // floatobject.py:769-771: sign correction
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        // floatobject.py:776-778: signed zero
        // "mod *= mod" hides "+0" from optimizer, then negate if y < 0
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // floatobject.py:784-790: snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        // floatobject.py:789-790: zero with sign of true quotient
        let d = div * div; // hide from optimizer
        d * x / y
    };
    Ok((floordiv, m))
}

// ── Power ────────────────────────────────────────────────────────────

/// intobject.py `_pow_nomod`: `@jit.look_inside_iff(lambda iv, iw: jit.isconstant(iw))`.
fn int_pow_nomod_iff(_iv: i64, iw: i64) -> bool {
    majit_rlib::jit::isconstant(&iw)
}

/// Exponentiation by squaring for a non-negative machine exponent.
/// A variable exponent stays a residual call of the trampoline.
#[majit_macros::look_inside_iff(int_pow_nomod_iff)]
fn int_pow_nomod(iv: i64, mut iw: i64) -> Result<i64, PyError> {
    let mut temp = iv;
    let mut ix = 1_i64;
    loop {
        if iw & 1 != 0 {
            let Some(value) = ix.checked_mul(temp) else {
                return Err(PyError::overflow_error("integer overflow"));
            };
            ix = value;
        }
        iw >>= 1;
        if iw == 0 {
            return Ok(ix);
        }
        let Some(value) = temp.checked_mul(temp) else {
            return Err(PyError::overflow_error("integer overflow"));
        };
        temp = value;
    }
}

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        // intobject.py _pow_nomod raises ValueError for iw < 0,
        // descr_pow catches it and routes through float pow — which
        // carries the ZeroDivisionError guard from floatobject.py:910-913.
        // `float_pow_raw` is not a prepass subject (its overflow arm reaches
        // `PyError::errno_pair`, whose `io::Error::from_raw_os_error` has no
        // lowering), so the call stays opaque here.
        return Ok(w_float_new(int_pow_negative(va, vb)?));
    }
    // intobject.py:415 / longobject.py:229: x ** 0 == 1 for any x.
    if vb == 0 {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    match va {
        0 => return Ok(w_int_new(0)),
        1 => return Ok(w_int_new(1)),
        -1 => return Ok(w_int_new(if vb % 2 == 0 { 1 } else { -1 })),
        _ => {}
    }
    // intobject.py `_pow_nomod`: exponentiation by squaring with an
    // overflow check at each machine multiplication. `checked_mul` is the
    // Rust source spelling the MIR front lowers back to `int_mul_ovf`.
    match int_pow_nomod(va, vb) {
        Ok(r) => Ok(w_int_new(r)),
        Err(_) => Ok(w_long_new(pow_ovf2long(va, vb)?)),
    }
}

/// intobject.py `_pow_ovf2long`: `W_LongObject.fromint(iv)` then
/// `rbigint.int_pow(iw)`.  The base is built inside this opaque call, so
/// the traced `int_pow` never passes a stack `BigInt` to the
/// `jit_bigint_int_pow_nomod` residual, which reads a heap rbigint pointer.
#[majit_macros::dont_look_inside]
fn pow_ovf2long(iv: i64, iw: i64) -> Result<BigInt, PyError> {
    let base = BigInt::from(iv);
    bigint_int_pow_nomod(&base, iw)
}

/// Negative-exponent `int ** int` float route, kept opaque so `int_pow`
/// does not inline `float_pow_raw`.
#[majit_macros::dont_look_inside]
fn int_pow_negative(base: i64, exp: i64) -> Result<f64, PyError> {
    float_pow_raw(base as f64, exp as f64)
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // Building either `rbigint` allocates a digit block (`fromint` ->
    // `RBigInt::new` -> `Digits::new`), so it collects before the other operand
    // is read.  Both operands stay on the shadow stack for the whole body and
    // every read comes back out of its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let vb = RBigIntGcRoot::new(if is_long(b()) {
        w_long_get_value(b()).translated_alias()
    } else {
        BigInt::from(int_value(b()))
    });
    if vb.get_sign() < 0 {
        // longobject.py calls descr_float on both integer operands
        // before float pow.  RBigInt::tofloat raises on an out-of-range value;
        // do not silently pass the infinity sentinel from as_float onward.
        reject_pow_operand_overflow(a())?;
        reject_pow_operand_overflow(b())?;
        let fa = as_float(a());
        let fb = as_float(b());
        return Ok(w_float_new(float_pow_raw(fa, fb)?));
    }
    // longobject.py: `if not exp_bigint: return int_pow(0)` → 1. `descr_pow`
    // wraps every branch as `W_LongObject`, so a long base keeps the long
    // representation across these trivial-base short-circuits too.
    if vb.get_sign() == 0 {
        return Ok(w_long_new(BigInt::from(1)));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    let va = RBigIntGcRoot::new(if is_long(a()) {
        w_long_get_value(a()).translated_alias()
    } else {
        BigInt::from(int_value(a()))
    });
    // Both rbigint.int_pow(1) and rbigint.pow(ONERBIGINT) return the base
    // reference after the zero-base check. W_LongObject adds only a wrapper.
    if is_long(a()) && va.get_sign() != 0 && vb.int_eq(1) {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a()),
        ));
    }
    if va.get_sign() == 0 {
        return Ok(w_long_new(BigInt::from(0)));
    }
    if va.int_eq(1) {
        return Ok(w_long_new(BigInt::from(1)));
    }
    if va.int_eq(-1) {
        let even = vb.digit(0) & 1 == 0;
        return Ok(w_long_new(BigInt::from(if even { 1 } else { -1 })));
    }
    // longobject.py: `descr_pow` keeps a `W_IntObject` exponent
    // unwrapped (`exp_bigint` stays None) and calls `rbigint.int_pow`; only a
    // long exponent reaches `rbigint.pow`.
    if is_int_like(b()) {
        return Ok(w_long_new(bigint_int_pow_nomod(&va, int_value(b()))?));
    }
    Ok(w_long_new(bigint_pow_nomod(&va, &vb)?))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // intobject.py `_lshift`: use the machine-int result while the
    // shift is in range and `ovfcheck(a << b)` succeeds. Rust's
    // `checked_shl` checks only the count, so verify the arithmetic result by
    // shifting it back. The overflow recovery is the exact
    // `rbigint.lshift_int_int_bigint_result` helper used at
    // intobject.py.
    // RPython's target constant `LONG_BIT` is 64 for pyre's i64 word.
    if vb < 64 {
        let shifted = va << vb;
        if (shifted >> vb) == va {
            return Ok(w_int_new(shifted));
        }
    }
    if va == 0 {
        return Ok(w_int_new(0));
    }
    let big = bigint_lshift_int_int_result(va, vb)?;
    Ok(bigint_result(big))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // intobject.py `_rshift(space, a, b)`:
    //   if r_uint(b) >= LONG_BIT:
    //       if b < 0: raise ValueError("negative shift count")
    //       # b >= LONG_BIT
    //       if a == 0: return wrapint(space, a)
    //       a = -1 if a < 0 else 0
    //   else: a = a >> b
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    if vb >= 64 {
        return Ok(w_int_new(if va < 0 { -1 } else { 0 }));
    }
    Ok(w_int_new(va >> vb))
}

/// `_int_lshift`. Nothing allocates between the guard and the receiver read.
unsafe fn long_int_lshift(a: PyObjectRef, other: i64) -> PyResult {
    if other < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    debug_assert!(is_long(a));
    // `rbigint.lshift` returns `self` for a zero count or a zero receiver.
    // `W_LongObject` wraps that payload.
    if other == 0 || live_long_num(a).get_sign() == 0 {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    let va = live_long_num(a);
    Ok(w_long_new(bigint_lshift_count(&va, other)?))
}

/// `_int_rshift`: `newlong(self.num.rshift(other))`.
unsafe fn long_int_rshift(a: PyObjectRef, other: i64) -> PyResult {
    if other < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    debug_assert!(is_long(a));
    // `rbigint.rshift(0)` returns `self`; `newlong` adds a fresh wrapper.
    if other == 0 {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    let va = live_long_num(a);
    Ok(w_long_new(bigint_rshift(&va, other)))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `_lshift` reads `w_other.asbigint()` before `self`. `descr_lshift`
    // keeps both operands live across that call (`gct_direct_call` /
    // `get_livevars_for_roots`). A machine-int count is `_int_lshift`
    // (`long_int_lshift`) and does not reach here.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    debug_assert!(is_long(b()));
    let vb = live_long_num(b());
    if vb.get_sign() < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py: `toint()` (signed machine int / i64) overflows
    // when the count exceeds i64::MAX → 0 if base is zero, OverflowError
    // otherwise.
    let shift = if jit_bigint_to_i64_fits(&vb) != 0 {
        jit_bigint_to_i64_value(&vb)
    } else {
        let base_is_zero = if is_long(a()) {
            live_long_num(a()).get_sign() == 0
        } else {
            int_value(a()) == 0
        };
        if base_is_zero {
            // `_lshift` returns `self` (a W_LongObject) for a zero base.
            return Ok(if is_long(a()) {
                a()
            } else {
                // A reflected compact-int operand is first coerced to the
                // W_LongObject receiver in `_make_descr_binop`.
                w_long_new(BigInt::zero())
            });
        }
        return Err(PyError::overflow_error("shift count too large"));
    };
    // rbigint.lshift returns `self` for a zero count or zero receiver.
    // `W_LongObject._lshift` then puts that same payload into a fresh long
    // wrapper (except for the huge-count zero case above, which returns the
    // receiver wrapper itself).
    if is_long(a()) && (shift == 0 || live_long_num(a()).get_sign() == 0) {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a()),
        ));
    }
    let va = if is_long(a()) {
        live_long_num(a())
    } else {
        RBigIntGcRoot::new(BigInt::from(int_value(a())))
    };
    let shifted = bigint_lshift_count(&va, shift)?;
    Ok(w_long_new(shifted))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `_rshift` reads `w_other.asbigint()` before `self`. `descr_rshift`
    // keeps both operands live (`gct_direct_call` / `get_livevars_for_roots`).
    // A machine-int count is `_int_rshift` (`long_int_rshift`).
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    debug_assert!(is_long(b()));
    let vb = live_long_num(b());
    if vb.get_sign() < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py: `toint()` overflow (count > i64::MAX) → positive
    // yields 0, negative yields -1 (all bits shifted out).
    let shift = if jit_bigint_to_i64_fits(&vb) != 0 {
        jit_bigint_to_i64_value(&vb)
    } else {
        let negative = if is_long(a()) {
            live_long_num(a()).get_sign() < 0
        } else {
            int_value(a()) < 0
        };
        return Ok(w_int_new(if negative { -1 } else { 0 }));
    };
    // rbigint.rshift(0) returns `self`; `newlong` adds only a fresh wrapper.
    if is_long(a()) && shift == 0 {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a()),
        ));
    }
    // `_rshift`/`_int_rshift` `newlong` the normal-count result, keeping a long.
    let va = if is_long(a()) {
        live_long_num(a())
    } else {
        RBigIntGcRoot::new(BigInt::from(int_value(a())))
    };
    Ok(w_long_new(bigint_rshift(&va, shift)))
}

// ── bool-as-int helpers ──────────────────────────────────────────────

/// True when obj is int or bool (bool is a subclass of int in Python).
#[inline]
pub(crate) unsafe fn is_int_like(obj: PyObjectRef) -> bool {
    is_int(obj) || is_bool(obj)
}

/// Extract i64 from an int or bool object.
#[inline]
pub(crate) unsafe fn int_value(obj: PyObjectRef) -> i64 {
    if is_bool(obj) {
        w_bool_get_value(obj) as i64
    } else {
        w_int_get_value(obj)
    }
}

// ── Bitwise operations ───────────────────────────────────────────────

// W_IntObject.descr_and/or/xor — always int; the bool result is produced
// by W_BoolObject's own descr_and/or/xor, not by the int path.
unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) & int_value(b)))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) | int_value(b)))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) ^ int_value(b)))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            live_long_num(a).int_and_(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(live_long_num(a).int_and_(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            live_long_num(b).int_and_(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(live_long_num(b).int_and_(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(live_long_num(a).and_(&live_long_num(b))))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            live_long_num(a).int_or_(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(live_long_num(a).int_or_(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            live_long_num(b).int_or_(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(live_long_num(b).int_or_(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(live_long_num(a).or_(&live_long_num(b))))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            live_long_num(a).int_xor(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(live_long_num(a).int_xor(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            live_long_num(b).int_xor(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(live_long_num(b).int_xor(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(live_long_num(a).xor(&live_long_num(b))))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.

pub(crate) unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `w_str_concat` joins the surrogate-aware WTF-8 views, so concatenating a
    // surrogateescape/surrogatepass-decoded string does not go through
    // `w_str_get_value` (which rejects lone surrogates).
    Ok(pyre_object::unicodeobject::w_str_concat(a, b))
}

/// Extract a non-negative repeat count from an int or long.
unsafe fn repeat_count(n: PyObjectRef) -> Result<usize, PyError> {
    if is_long(n) {
        let big = w_long_get_value(n);
        // The count coerces to a signed machine word (an index-sized integer):
        // a non-negative value exceeding `isize::MAX` overflows rather than
        // wrapping to a huge `usize` count, matching `getindex_w`. Negative
        // counts clamp to 0 only after they fit the index-sized word.
        match big.to_isize() {
            Some(v) => Ok(if v < 0 { 0 } else { v as usize }),
            None => Err(PyError::new(
                PyErrorKind::OverflowError,
                "cannot fit 'int' into an index-sized integer",
            )),
        }
    } else {
        let nv = w_int_get_value(n);
        Ok(if nv < 0 { 0 } else { nv as usize })
    }
}

/// tupleobject.py descr_mul
pub(crate) unsafe fn tuple_repeat(t: PyObjectRef, n: PyObjectRef) -> PyResult {
    let n = repeat_count(n)?;
    // tupleobject.py: `if times == 1 and space.type(self) == space.w_tuple:
    // return self`. Subclasses must still be copied to a base tuple.
    if n == 1 && is_exact_tuple(t) {
        return Ok(t);
    }
    let len = w_tuple_len(t);
    let cap = len
        .checked_mul(n)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "tuple is too large"))?;
    let mut items: Vec<PyObjectRef> = Vec::new();
    items
        .try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    // `tuple_repeat` makes no copies for `input_size == 0`, so an empty
    // receiver never walks the count as a trip count.
    if len != 0 {
        for _ in 0..n {
            for i in 0..len {
                if let Some(item) = w_tuple_getitem(t, i as i64) {
                    items.push(item);
                }
            }
        }
    }
    Ok(w_tuple_new(items))
}

/// The builtin sequences repeat through `sq_repeat`, never `nb_multiply`.
pub(crate) unsafe fn is_repeat_sequence(obj: PyObjectRef) -> bool {
    is_str(obj)
        || is_list(obj)
        || is_tuple(obj)
        || pyre_object::bytesobject::is_bytes_like(obj)
        || crate::module::_collections::is_deque(obj)
}

/// `sequence_repeat` for a receiver [`is_repeat_sequence`] accepted, with the
/// count already reduced through `__index__`.
unsafe fn sequence_repeat(seq: PyObjectRef, count: PyObjectRef) -> PyResult {
    if is_str(seq) {
        str_repeat(seq, count)
    } else if is_list(seq) {
        list_repeat(seq, count)
    } else if is_tuple(seq) {
        tuple_repeat(seq, count)
    } else if pyre_object::bytesobject::is_bytes_like(seq) {
        bytes_repeat(seq, count)
    } else {
        crate::module::_collections::deque_repeat(seq, count)
    }
}

/// unicodeobject.py descr_mul
pub(crate) unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    // Repeat at the WTF-8 byte level — a repetition of valid WTF-8 is valid
    // WTF-8 — so a surrogate-bearing string repeats without going through
    // `w_str_get_value`.
    let bytes = w_str_get_wtf8(s).as_bytes();
    let count = repeat_count(n)?;
    if count == 1 {
        return Ok(crate::type_methods::str_result_unchanged(s));
    }
    // unicode_repeat: overflow is judged on the character count
    // (length * nchars), not the WTF-8 byte length. A non-ASCII character
    // is wider in WTF-8 than its Py_UCS storage, so the byte product may
    // exceed isize::MAX while the character product does not. That case is
    // a MemoryError from the allocation, not an OverflowError.
    let char_len = w_str_len(s);
    if char_len != 0 && count > isize::MAX as usize / char_len {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "repeated string is too long",
        ));
    }
    let Some(total) = bytes.len().checked_mul(count) else {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    };
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    // `unicode_repeat` answers a receiver of no characters before it copies,
    // so an empty one never walks the count as a trip count.
    if !bytes.is_empty() {
        for _ in 0..count {
            out.extend_from_slice(bytes);
        }
    }
    let buf = Wtf8Buf::from_bytes(out).expect("repetition of WTF-8 is WTF-8");
    // Repetition churns fresh dynamic strings; make the result collectable.
    Ok(w_str_from_wtf8_managed(buf))
}

pub(crate) unsafe fn bytes_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `buffer_as_bytes_like` can mint a snapshot, so pin both operands
    // first and copy their payloads off the objects.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let Some(b_src) = crate::typedef::buffer_as_bytes_like(b())? else {
        return Err(PyError::type_error(format!(
            "can't concat {} to {}",
            crate::baseobjspace::object_functionstr_type_name(b()),
            crate::baseobjspace::object_functionstr_type_name(a())
        )));
    };
    let a_data = pyre_object::bytesobject::bytes_like_data(a()).to_vec();
    let b_data = pyre_object::bytesobject::bytes_like_data(b_src).to_vec();
    let mut result = a_data;
    result.extend_from_slice(&b_data);
    Ok(if pyre_object::bytesobject::is_bytes(a()) {
        pyre_object::bytesobject::w_bytes_from_bytes(&result)
    } else {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
    })
}

pub(crate) unsafe fn bytes_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    // `repeat_count` runs `__index__`, so pin the receiver and count
    // together and copy the payload after that conversion.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[s, n]);
    let s = || pyre_object::gc_roots::shadow_stack_get(base);
    let count = repeat_count(pyre_object::gc_roots::shadow_stack_get(base + 1))?;
    let data = pyre_object::bytesobject::bytes_like_data(s()).to_vec();
    // A count of 1 on exact `bytes` (immutable) returns the receiver unchanged;
    // a subclass yields a fresh base `bytes`, and mutable `bytearray` copies.
    if count == 1
        && pyre_object::pyobject::is_exact_type(s(), &pyre_object::bytesobject::BYTES_TYPE)
    {
        return Ok(s());
    }
    let cap = data
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "repeated bytes are too long"))?;
    if cap > isize::MAX as usize {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "repeated bytes are too long",
        ));
    }
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    // `_PyBytes_Repeat` returns on `len_dest == 0`, so an empty receiver never
    // walks the count as a trip count.
    if !data.is_empty() {
        for _ in 0..count {
            buf.extend_from_slice(&data);
        }
    }
    Ok(if pyre_object::bytesobject::is_bytes(s()) {
        pyre_object::bytesobject::w_bytes_from_bytes(&buf)
    } else {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf)
    })
}

pub(crate) unsafe fn list_concat(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    let len_a = w_list_len(a);
    let len_b = w_list_len(b);
    // W_ListObject.descr_add: when the left operand is empty, clone the right
    // operand. A SizeListStrategy clone retains that exact strategy instance.
    if len_a == 0
        && let Some(clone) = pyre_object::with_roots!(a, b => pyre_object::listobject::w_list_clone_if_shared_strategy(b))
    {
        return Ok(clone);
    }
    // `W_ListObject.descr_add`: same-strategy storage append, otherwise
    // `getitems_copy` into an object list. Bytes/Ascii keep the loop below.
    if let Some(sum) = pyre_object::with_roots!(
        a,
        b => pyre_object::listobject::w_list_add_unwrapped(a, b)
    ) {
        return Ok(sum);
    }
    // `w_list_getitem` boxes on the Range/Integer/Float strategies, and
    // `w_list_new` itself allocates, so a copy already made would sit
    // unrooted in a plain `Vec`. Pin both operands and each fetched item
    // on one bracket, then hand the constructor the live slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let ab = pyre_object::gc_roots::pin_roots(&[a, b]);
    let items_base = pyre_object::gc_roots::shadow_stack_len();
    let mut fetched = 0usize;
    for i in 0..len_a {
        if let Some(item) = w_list_getitem(pyre_object::gc_roots::shadow_stack_get(ab), i as i64) {
            let _ = pyre_object::gc_roots::pin_root(item);
            fetched += 1;
        }
    }
    for i in 0..len_b {
        if let Some(item) =
            w_list_getitem(pyre_object::gc_roots::shadow_stack_get(ab + 1), i as i64)
        {
            let _ = pyre_object::gc_roots::pin_root(item);
            fetched += 1;
        }
    }
    let mut items = Vec::with_capacity(fetched);
    for i in 0..fetched {
        items.push(pyre_object::gc_roots::shadow_stack_get(items_base + i));
    }
    Ok(w_list_new(items))
}

pub(crate) unsafe fn tuple_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_tuple_len(a);
    let len_b = w_tuple_len(b);
    let _roots = pyre_object::gc_roots::push_roots();
    let ab = pyre_object::gc_roots::pin_roots(&[a, b]);
    let items_base = pyre_object::gc_roots::shadow_stack_len();
    let mut fetched = 0usize;
    for i in 0..len_a {
        if let Some(item) = w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(ab), i as i64) {
            let _ = pyre_object::gc_roots::pin_root(item);
            fetched += 1;
        }
    }
    for i in 0..len_b {
        if let Some(item) =
            w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(ab + 1), i as i64)
        {
            let _ = pyre_object::gc_roots::pin_root(item);
            fetched += 1;
        }
    }
    let mut items = Vec::with_capacity(fetched);
    for i in 0..fetched {
        items.push(pyre_object::gc_roots::shadow_stack_get(items_base + i));
    }
    Ok(w_tuple_new(items))
}

/// listobject.py descr_mul
pub(crate) unsafe fn list_repeat(mut list: PyObjectRef, n: PyObjectRef) -> PyResult {
    let count = repeat_count(n)?;
    if let Some(clone) =
        pyre_object::with_roots!(list => pyre_object::listobject::w_list_clone_if_size(list))
    {
        return Ok(clone);
    }
    let len = w_list_len(list);
    let cap = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "list is too large"))?;
    // CPython 3.14 `list_resize`: the element count may fit Py_ssize_t while
    // `new_allocated * sizeof(PyObject*)` does not (gh-97616).
    if cap > (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>() {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    }
    // `AbstractUnwrappedStrategy.mul` repeats unwrapped storage. Bytes/Ascii
    // and a failed reservation fall through to the getitem loop below.
    if let Some(repeated) = pyre_object::with_roots!(
        list => pyre_object::listobject::w_list_repeat_unwrapped(list, count)
    ) {
        return Ok(repeated);
    }
    // RPython `ll_mul` allocates the result through `ll_newlist(resultlen)`;
    // the fallible Rust helper is translated as `newlist_hint(cap)`, keeping
    // both the preallocation and its MemoryError edge without exposing
    // `Vec::try_reserve_exact` to source translation.
    let mut items: Vec<PyObjectRef> = crate::builtins::try_pyobject_vec_with_capacity(cap)?;
    // `w_list_getitem` boxes on the Range/Integer/Float strategies, so a copy
    // already made would sit unrooted in a plain `Vec` across the next box.
    // The preallocation above stays for its MemoryError edge; the pins are
    // what `build_list_storage` requires of the elements.
    let list_roots = pyre_object::gc_roots::push_roots();
    let list_slot = list_roots.publish(&[list]);
    list_roots.normalize(list_slot, 1);
    let mut rooted = pyre_object::gc_roots::RootedItems::new();
    // `list_repeat` answers `input_size == 0` before it copies, so an empty
    // receiver never walks the count as a trip count.
    if len != 0 {
        for _ in 0..count {
            for i in 0..len {
                if let Some(item) = w_list_getitem(list_roots.get(list_slot), i as i64) {
                    rooted.push(item);
                }
            }
        }
    }
    // Keep the rtyper-visible `args[i]`/ArrayRead shape used throughout the
    // port; `Vec::extend` has no source-lifted CallRegistry entry.
    let pinned = rooted.take();
    for i in 0..pinned.len() {
        items.push(pinned[i]);
    }
    Ok(w_list_new(items))
}

/// listobject.py descr_inplace_mul — repeat the list in place; the
/// list object identity is preserved.  Count and overflow handling mirror
/// `list_repeat`, but the extra copies are appended into the existing
/// storage instead of building a fresh list.
pub(crate) unsafe fn list_inplace_repeat(list: PyObjectRef, n: PyObjectRef) -> Result<(), PyError> {
    let count = repeat_count(n)?;
    // BaseRangeListStrategy.inplace_mul materialises before delegating for
    // every multiplier, including 0 and 1.
    let list = pyre_object::listobject::w_list_materialize_range(list);
    if pyre_object::listobject::w_list_sizehint(list).is_some() {
        return Ok(());
    }
    if count == 0 {
        w_list_clear(list);
        return Ok(());
    }
    let len = w_list_len(list);
    if count == 1 || len == 0 {
        return Ok(());
    }
    let cap = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "list is too large"))?;
    if cap > (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>() {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    }
    // Snapshot the original items so the growing list is not re-read while
    // the copies are appended.  Holding the refs across `w_list_append` is
    // the same idiom `list_method_extend` uses for its iterable branch.
    let snapshot = w_list_items_copy_as_vec_mode(list, majit_metainterp::jit::we_are_jitted());
    // The snapshot is a plain `Vec` and roots nothing, and the resize below
    // allocates: publish both the receiver -- a `W_List` header, one of the two
    // kinds a minor collection relocates -- and the snapshot, then address them
    // through their slots.
    let roots = pyre_object::gc_roots::push_roots();
    let list_slot = roots.publish(&[list]);
    let base = roots.publish(&snapshot);
    roots.normalize(list_slot, 1 + snapshot.len());
    // `list_inplace_repeat_lock_held` sizes the whole result with one
    // `list_resize` before it copies, so a count the machine cannot meet is
    // refused here rather than walked as a trip count by the loop below.
    if !pyre_object::listobject::w_list_try_resize(roots.get(list_slot), cap) {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    }
    for _ in 1..count {
        for k in 0..snapshot.len() {
            pyre_object::listobject::w_list_append_preallocated(
                roots.get(list_slot),
                roots.get(base + k),
            );
        }
    }
    Ok(())
}

/// bytearrayobject.py descr_inplace_mul — repeat the bytearray in place,
/// preserving object identity.  A resize while the buffer is exported raises
/// BufferError, mirroring the `__iadd__` export guard; a count of 1 leaves the
/// length unchanged and needs no resize.
pub(crate) unsafe fn bytearray_inplace_repeat(
    ba: PyObjectRef,
    n: PyObjectRef,
) -> Result<(), PyError> {
    let count = repeat_count(n)?;
    let len = pyre_object::bytearrayobject::w_bytearray_len(ba);
    if count == 1 {
        return Ok(());
    }
    let new_size = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "repeated bytes are too long"))?;
    // Only an actual length change touches the buffer size; an exported buffer
    // blocks it.
    let ba = if new_size != len {
        crate::builtins::bytearray_check_exports(ba)?
    } else {
        ba
    };
    if count == 0 {
        pyre_object::bytearrayobject::w_bytearray_vec_mut(ba).clear();
        pyre_object::bytearrayobject::w_bytearray_sync_alloc(ba, len);
        return Ok(());
    }
    if len == 0 {
        return Ok(());
    }
    let snapshot = pyre_object::bytearrayobject::w_bytearray_data(ba).to_vec();
    let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(ba);
    vec.try_reserve_exact(new_size - len)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 1..count {
        vec.extend_from_slice(&snapshot);
    }
    pyre_object::bytearrayobject::w_bytearray_sync_alloc(ba, len);
    Ok(())
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) < int_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) <= int_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) > int_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) >= int_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) == int_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) != int_value(b)))
}

/// pypy/objspace/std/longobject.py `_make_descr_cmp`: compare a W_LongObject's
/// rbigint payload directly with a W_IntObject's machine word.
unsafe fn long_int_compare(long: PyObjectRef, iother: i64, op: CompareOp) -> bool {
    let value = w_long_get_value(long);
    match op {
        CompareOp::Lt => value.int_lt(iother),
        CompareOp::Le => value.int_le(iother),
        CompareOp::Gt => value.int_gt(iother),
        CompareOp::Ge => value.int_ge(iother),
        CompareOp::Eq => value.int_eq(iother),
        CompareOp::Ne => value.int_ne(iother),
    }
}

/// `rstr.py` `AbstractStringRepr.ll_strcmp`. Bytewise prefix, then
/// length. Indexing is `ll_getitem_nonneg` (`as_ptr().add`), not
/// `slice::cmp` / `Vec::index`. No `stroruni.cmp` oopspec: the first
/// argument is a byte slice, not `Ptr(STR)` / `Ptr(UNICODE)`.
///
/// Elidable only for immutable `bytes`/`bytes`. A `bytearray` operand
/// reaches [`ll_bytes_strcmp_mutable`] instead: an in-place store
/// (`b[i] = ...`) changes the compared bytes without changing the slice
/// pointer or length, and `CALL_PURE` would reuse the first result.
#[majit_macros::elidable]
fn ll_bytes_strcmp(left: &[u8], right: &[u8]) -> isize {
    let cmplen = left.len().min(right.len());
    let left_p = left.as_ptr();
    let right_p = right.as_ptr();
    let mut i = 0usize;
    while i < cmplen {
        let diff = unsafe { *left_p.add(i) as isize - *right_p.add(i) as isize };
        if diff != 0 {
            return diff;
        }
        i += 1;
    }
    left.len() as isize - right.len() as isize
}

/// Same bytewise prefix-then-length compare as [`ll_bytes_strcmp`], not
/// elidable. `_handle_stroruni_call` raises `NotSupported` for `BYTEARRAY`;
/// a residual call re-reads the buffer after an in-place store.
fn ll_bytes_strcmp_mutable(left: &[u8], right: &[u8]) -> isize {
    let cmplen = left.len().min(right.len());
    let left_p = left.as_ptr();
    let right_p = right.as_ptr();
    let mut i = 0usize;
    while i < cmplen {
        let diff = unsafe { *left_p.add(i) as isize - *right_p.add(i) as isize };
        if diff != 0 {
            return diff;
        }
        i += 1;
    }
    left.len() as isize - right.len() as isize
}

/// Read a total-order result under `op` — the shape every `_memcmp`-based
/// comparison ends in, where the common prefix and then the lengths have
/// already been folded into one `Ordering`.
#[inline]
pub(crate) fn ordering_satisfies(ordering: std::cmp::Ordering, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => ordering.is_lt(),
        CompareOp::Le => ordering.is_le(),
        CompareOp::Gt => ordering.is_gt(),
        CompareOp::Ge => ordering.is_ge(),
        CompareOp::Eq => ordering.is_eq(),
        CompareOp::Ne => ordering.is_ne(),
    }
}

#[inline]
fn reverse_compare_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
    }
}

#[inline]
fn compare_f64(f1: f64, f2: f64, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => f1 < f2,
        CompareOp::Le => f1 <= f2,
        CompareOp::Gt => f1 > f2,
        CompareOp::Ge => f1 >= f2,
        CompareOp::Eq => f1 == f2,
        CompareOp::Ne => f1 != f2,
    }
}

/// `tupleobject.py _unroll_condition_cmp`.
fn tuple_compare_iff(a: PyObjectRef, b: PyObjectRef, _op: CompareOp) -> bool {
    pyre_object::tupleobject::unroll_condition(a) || pyre_object::tupleobject::unroll_condition(b)
}

/// Pairs whose identity and same-type equality are field reads.
///
/// `framework.py` brackets a call only when that call can reach the
/// collector. [`pin_free_builtin_is`] and [`pin_free_builtin_ord`] are that
/// whole callee: machine `intval`, float bits, `ll_streq`, `w_tuple_len`,
/// and the tuple walk in [`compare_tuples`]. A pair this predicate accepts
/// whose element `eq` can still run Python takes the rooted [`compare_slot`].
#[majit_macros::always_inline]
pub(crate) unsafe fn builtin_pair_needs_no_caller_roots(a: PyObjectRef, b: PyObjectRef) -> bool {
    if is_exact_type(a, &STR_TYPE) && is_exact_type(b, &STR_TYPE) {
        return true;
    }
    if is_exact_type(a, &TUPLE_TYPE) && is_exact_type(b, &TUPLE_TYPE) {
        return true;
    }
    if is_exact_type(a, &FLOAT_TYPE) && is_exact_type(b, &FLOAT_TYPE) {
        return true;
    }
    if is_exact_type(a, &bytesobject::BYTES_TYPE) && is_exact_type(b, &bytesobject::BYTES_TYPE) {
        return true;
    }
    if is_exact_type(a, &BOOL_TYPE) && is_exact_type(b, &BOOL_TYPE) {
        return true;
    }
    is_exact_type(a, &INT_TYPE) && is_exact_type(b, &INT_TYPE) && !is_long(a) && !is_long(b)
}

/// `eq_w` of two exact builtins that does not collect: `is_w`, or the
/// content compare `compare_slot` uses for that layout. `None` means the
/// pair can run user `__eq__` and the caller must take the rooted walk.
unsafe fn pin_free_builtin_eq(a: PyObjectRef, b: PyObjectRef) -> Option<bool> {
    if std::ptr::eq(a, b) {
        return Some(true);
    }
    if is_exact_type(a, &STR_TYPE) && is_exact_type(b, &STR_TYPE) {
        let s1 = pyre_object::unicodeobject::w_str_storage(a);
        let s2 = pyre_object::unicodeobject::w_str_storage(b);
        return Some(pyre_object::lowlevel_string::jit_ll_streq(s1, s2) != 0);
    }
    if is_exact_type(a, &INT_TYPE) && is_exact_type(b, &INT_TYPE) && !is_long(a) && !is_long(b) {
        return Some(w_int_get_value(a) == w_int_get_value(b));
    }
    if is_exact_type(a, &BOOL_TYPE) && is_exact_type(b, &BOOL_TYPE) {
        return Some(w_bool_get_value(a) == w_bool_get_value(b));
    }
    if is_exact_type(a, &FLOAT_TYPE) && is_exact_type(b, &FLOAT_TYPE) {
        // `eq_w` answers `is_w` first, so the same NaN is equal. Distinct
        // NaNs fall through to float `==`, which is false. `+0.0 == -0.0`.
        return Some(w_float_get_value(a) == w_float_get_value(b));
    }
    if is_exact_type(a, &bytesobject::BYTES_TYPE) && is_exact_type(b, &bytesobject::BYTES_TYPE) {
        return Some(
            pyre_object::bytesobject::w_bytes_data(a) == pyre_object::bytesobject::w_bytes_data(b),
        );
    }
    None
}

/// `is_w` for a pair [`builtin_pair_needs_no_caller_roots`] accepted.
///
/// Same answers as `W_IntObject.is_w`, `W_FloatObject.is_w`,
/// `W_AbstractTupleObject.is_w`, `W_AbstractBytesObject.is_w`, and
/// `W_UnicodeObject.is_w` on those exact layouts. Bool stays pointer
/// identity (`W_BoolObject.is_w`). Does not call [`is_w`]: that function
/// also builds an `rbigint` for `i64::MIN`, and `framework.py` would then
/// bracket the caller.
pub(crate) unsafe fn pin_free_builtin_is(a: PyObjectRef, b: PyObjectRef) -> bool {
    if std::ptr::eq(a, b) {
        return true;
    }
    if is_exact_type(a, &INT_TYPE) && is_exact_type(b, &INT_TYPE) && !is_long(a) && !is_long(b) {
        return w_int_get_value(a) == w_int_get_value(b);
    }
    if is_exact_type(a, &FLOAT_TYPE) && is_exact_type(b, &FLOAT_TYPE) {
        let one = w_float_get_value(a);
        let two = w_float_get_value(b);
        // Distinct NaNs keep pointer identity (`Py_Is`). `W_FloatObject.is_w`
        // compares `float2longlong` bits, so two NaNs compare equal there.
        // That method has no `@jit` hint (`_hash_float` is the elidable).
        // `IS_OP` for a pin-free pair must match `ObjSpace.is_w`, which
        // already takes this NaN arm; finite floats still compare bits
        // (`0.0 is -0.0` is false).
        if one.is_nan() || two.is_nan() {
            return false;
        }
        return one.to_bits() == two.to_bits();
    }
    if is_exact_type(a, &TUPLE_TYPE) && is_exact_type(b, &TUPLE_TYPE) {
        return w_tuple_len(a) == 0 && w_tuple_len(b) == 0;
    }
    if is_exact_type(a, &bytesobject::BYTES_TYPE) && is_exact_type(b, &bytesobject::BYTES_TYPE) {
        let len1 = bytesobject::w_bytes_len(a);
        let len2 = bytesobject::w_bytes_len(b);
        if len2 > 1 {
            return bytesobject::w_bytes_block(a) == bytesobject::w_bytes_block(b);
        }
        if len2 == 0 {
            return len1 == 0;
        }
        return len1 == 1
            && bytesobject::w_bytes_getitem(a, 0) == bytesobject::w_bytes_getitem(b, 0);
    }
    if is_exact_type(a, &STR_TYPE) && is_exact_type(b, &STR_TYPE) {
        let s1 = pyre_object::unicodeobject::w_str_storage(a);
        let s2 = pyre_object::unicodeobject::w_str_storage(b);
        if pyre_object::unicodeobject::w_str_len(a) > 1 {
            return std::ptr::eq(s1, s2);
        }
        return pyre_object::lowlevel_string::jit_ll_streq(s1, s2) != 0;
    }
    false
}

/// Same-type ordering for a pair [`builtin_pair_needs_no_caller_roots`]
/// accepted, when every step is a field read. `None` means an element can
/// run Python and the caller must publish `a` and `b` before
/// [`compare_slot`].
unsafe fn pin_free_builtin_ord(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> Option<bool> {
    if (is_exact_type(a, &INT_TYPE) && !is_long(a) || is_exact_type(a, &BOOL_TYPE))
        && (is_exact_type(b, &INT_TYPE) && !is_long(b) || is_exact_type(b, &BOOL_TYPE))
    {
        let ia = int_value(a);
        let ib = int_value(b);
        return Some(match op {
            CompareOp::Lt => ia < ib,
            CompareOp::Le => ia <= ib,
            CompareOp::Gt => ia > ib,
            CompareOp::Ge => ia >= ib,
            CompareOp::Eq => ia == ib,
            CompareOp::Ne => ia != ib,
        });
    }
    if is_exact_type(a, &FLOAT_TYPE) && is_exact_type(b, &FLOAT_TYPE) {
        return Some(compare_f64(w_float_get_value(a), w_float_get_value(b), op));
    }
    if is_exact_type(a, &STR_TYPE) && is_exact_type(b, &STR_TYPE) {
        let s1 = pyre_object::unicodeobject::w_str_storage(a);
        let s2 = pyre_object::unicodeobject::w_str_storage(b);
        return Some(match op {
            CompareOp::Lt => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) < 0,
            CompareOp::Le => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) <= 0,
            CompareOp::Gt => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) > 0,
            CompareOp::Ge => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) >= 0,
            CompareOp::Eq => pyre_object::lowlevel_string::jit_ll_streq(s1, s2) != 0,
            CompareOp::Ne => pyre_object::lowlevel_string::jit_ll_streq(s1, s2) == 0,
        });
    }
    if is_exact_type(a, &bytesobject::BYTES_TYPE) && is_exact_type(b, &bytesobject::BYTES_TYPE) {
        let diff = ll_bytes_strcmp(bytesobject::w_bytes_data(a), bytesobject::w_bytes_data(b));
        return Some(match op {
            CompareOp::Lt => diff < 0,
            CompareOp::Le => diff <= 0,
            CompareOp::Gt => diff > 0,
            CompareOp::Ge => diff >= 0,
            CompareOp::Eq => diff == 0,
            CompareOp::Ne => diff != 0,
        });
    }
    if is_exact_type(a, &TUPLE_TYPE) && is_exact_type(b, &TUPLE_TYPE) {
        return compare_tuples(a, b, op);
    }
    None
}

/// `tupleobject.py _compare_tuples` /
/// `W_TupleObject._descr_eq` — `@jit.look_inside_iff(_unroll_condition_cmp)`.
///
/// Only the walk that cannot collect. `None` asks [`compare_tuples_general`]
/// to publish `a` and `b` and run element `eq_w`. A helper with this loop
/// would be a residual (`loop-without-unroll_safe`); the walk stays here,
/// where `look_inside_iff` marks the body `unroll_safe`.
#[majit_macros::look_inside_iff(tuple_compare_iff)]
fn compare_tuples(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> Option<bool> {
    // `_descr_eq` returns as soon as one `eq_w` fails. When every item is
    // an exact builtin whose `eq_w` cannot collect, that walk publishes
    // nothing: `framework.py` would not bracket it. `_ii` / `_ff` read raw
    // payload words. Anything else — an `_oo` element, a user `__eq__` —
    // is `None`, and [`compare_tuples_general`] publishes `a` and `b`.
    // This body calls none of those: a caller that has not opened a root
    // scope still reaches the field reads.
    if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return None;
    }
    let equal = unsafe {
        if is_specialised_tuple_ii(a) && is_specialised_tuple_ii(b) {
            Some(
                w_specialised_tuple_ii_getvalue(a, 0) == w_specialised_tuple_ii_getvalue(b, 0)
                    && w_specialised_tuple_ii_getvalue(a, 1)
                        == w_specialised_tuple_ii_getvalue(b, 1),
            )
        } else if is_specialised_tuple_ff(a) && is_specialised_tuple_ff(b) {
            let a0 = w_specialised_tuple_ff_getvalue(a, 0);
            let b0 = w_specialised_tuple_ff_getvalue(b, 0);
            let a1 = w_specialised_tuple_ff_getvalue(a, 1);
            let b1 = w_specialised_tuple_ff_getvalue(b, 1);
            // Two NaNs compare unequal as doubles, but a tuple checks element
            // identity first, so the same NaN in both slots must still be
            // equal — `float2longlong` upstream, the raw bits here. `+0.0`
            // and `-0.0` differ in bits and are caught by the value compare.
            Some(
                (a0 == b0 || a0.to_bits() == b0.to_bits())
                    && (a1 == b1 || a1.to_bits() == b1.to_bits()),
            )
        } else if is_specialised_tuple_ii(a)
            || is_specialised_tuple_ff(a)
            || is_specialised_tuple_ii(b)
            || is_specialised_tuple_ff(b)
            || !is_tuple(a)
            || !is_tuple(b)
        {
            None
        } else {
            let la = w_tuple_len(a);
            let lb = w_tuple_len(b);
            let n = la.min(lb);
            let mut matched = true;
            let mut pin_free = true;
            // tupleobject.py `_compare_tuples`: `p = 0` then `while p < min_len`.
            let mut i = 0usize;
            while i < n {
                let (Some(ea), Some(eb)) =
                    (w_tuple_getitem(a, i as i64), w_tuple_getitem(b, i as i64))
                else {
                    pin_free = false;
                    break;
                };
                match pin_free_builtin_eq(ea, eb) {
                    Some(true) => {}
                    Some(false) => {
                        matched = false;
                        break;
                    }
                    None => {
                        pin_free = false;
                        break;
                    }
                }
                i += 1;
            }
            if pin_free {
                Some(matched && la == lb)
            } else {
                None
            }
        }
    };
    equal.map(|equal| {
        if matches!(op, CompareOp::Ne) {
            !equal
        } else {
            equal
        }
    })
}

/// Element `eq_w` for two tuples, after [`compare_tuples`] declined.
///
/// `a` and `b` stay published for the whole walk. `w_tuple_getitem`
/// boxes an `_ii` / `_ff` payload, so each element is published in its
/// own scope before the other item is read, and the pair is published
/// again before `eq_w` runs their `__eq__`. The forwarded word is the
/// one `roots.get` returns; the local passed into `pin_roots` is not
/// read again.
fn compare_tuples_general(
    a: PyObjectRef,
    b: PyObjectRef,
    op: CompareOp,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[a, b]);
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        if let Some(equal) =
            unsafe { specialised_tuple_oo_eq(roots.get(base), roots.get(base + 1))? }
        {
            let equal = if matches!(op, CompareOp::Ne) {
                !equal
            } else {
                equal
            };
            return Ok(w_bool_from(equal));
        }
    }
    let la = unsafe { w_tuple_len(roots.get(base)) };
    let lb = unsafe { w_tuple_len(roots.get(base + 1)) };
    let min_len = la.min(lb);
    for i in 0..min_len {
        let ea = unsafe { w_tuple_getitem(roots.get(base), i as i64) }.unwrap_or(PY_NULL);
        let ea_scope = pyre_object::gc_roots::push_roots();
        let ea_at = ea_scope.pin_roots(&[ea]);
        let eb = unsafe { w_tuple_getitem(roots.get(base + 1), i as i64) }.unwrap_or(PY_NULL);
        let pair_scope = pyre_object::gc_roots::push_roots();
        let pair_at = pair_scope.pin_roots(&[ea_scope.get(ea_at), eb]);
        // `_compare_tuples`: `if not space.eq_w(items1[p], items2[p]):
        //     return getattr(space, name)(items1[p], items2[p])`
        if !crate::baseobjspace::eq_w(pair_scope.get(pair_at), pair_scope.get(pair_at + 1))? {
            return compare(pair_scope.get(pair_at), pair_scope.get(pair_at + 1), op);
        }
    }
    Ok(w_bool_from(match op {
        CompareOp::Lt => la < lb,
        CompareOp::Le => la <= lb,
        CompareOp::Gt => la > lb,
        CompareOp::Ge => la >= lb,
        CompareOp::Eq => la == lb,
        CompareOp::Ne => la != lb,
    }))
}

/// `specialisedtupleobject.py descr_eq` for two `W_SpecialisedTupleObject_oo`.
///
/// `_ii` / `_ff` compare raw slots in [`compare_tuples`] and never reach
/// here. `None` means the pair is not that class — a mixed pair still
/// walks elementwise, which is what upstream does too.
///
/// # Safety
/// `a` and `b` must point to valid tuple objects.
unsafe fn specialised_tuple_oo_eq(a: PyObjectRef, b: PyObjectRef) -> Result<Option<bool>, PyError> {
    if is_specialised_tuple_oo(a) && is_specialised_tuple_oo(b) {
        // `eq_w` runs the elements' `__eq__` and is a collection point, while
        // `a` and `b` are native locals no root walker updates: the second
        // iteration would read its values out of two tuples a minor collection
        // has already moved.  Publish the pair and address it through the
        // slots.
        let roots = pyre_object::gc_roots::push_roots();
        let pair = roots.publish(&[a, b]);
        roots.normalize(pair, 2);
        for i in 0..2 {
            // `getvalue` itself does not allocate, but `eq_w` does, and a
            // specialised `_oo` payload can hold a young box.  Pin each
            // value before the sibling read and the comparison, the same
            // first-then-second shape as `list_eq`.
            let _val_roots = pyre_object::gc_roots::push_roots();
            let val_base = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_specialised_tuple_oo_getvalue(
                roots.get(pair),
                i,
            ));
            let _ = pyre_object::gc_roots::pin_root(w_specialised_tuple_oo_getvalue(
                roots.get(pair + 1),
                i,
            ));
            if !crate::baseobjspace::eq_w(
                pyre_object::gc_roots::shadow_stack_get(val_base),
                pyre_object::gc_roots::shadow_stack_get(val_base + 1),
            )? {
                return Ok(Some(false));
            }
        }
        return Ok(Some(true));
    }
    Ok(None)
}

/// floatobject.py `do_compare_bigint` — compare a float against a
/// bigint without converting the bigint to a double, which would round it.
fn do_compare_bigint(f1: f64, b2: &BigInt, op: CompareOp) -> bool {
    // `_fromfloat_finite` collects.
    let b2 = live_rbigint(b2);
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        let ne = matches!(op, CompareOp::Ne);
        // A non-finite or fractional float is never equal to an integer.
        if !f1.is_finite() || f1.floor() != f1 {
            return ne;
        }
        return BigInt::_fromfloat_finite(f1).eq(&b2) != ne;
    }
    if !f1.is_finite() {
        return compare_f64(f1, 0.0, op);
    }
    let f1 = if matches!(op, CompareOp::Gt | CompareOp::Le) {
        // 'float > long'  <==> 'ceil(float) > long'
        // 'float <= long' <==> 'ceil(float) <= long'
        f1.ceil()
    } else {
        // 'float < long'  <==> 'floor(float) < long'
        // 'float >= long' <==> 'floor(float) >= long'
        f1.floor()
    };
    let b1 = BigInt::_fromfloat_finite(f1);
    match op {
        CompareOp::Lt => b1.lt(&b2),
        CompareOp::Le => b1.le(&b2),
        CompareOp::Gt => b1.gt(&b2),
        CompareOp::Ge => b1.ge(&b2),
        CompareOp::Eq | CompareOp::Ne => unreachable!("handled above"),
    }
}

/// floatobject.py `_compare` after `_to_float` for two floats:
/// `space.newbool(self.floatval ? w_other.floatval)`.
#[inline(never)]
pub(crate) fn _float_lt(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x < y))
}

#[inline(never)]
pub(crate) fn _float_le(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x <= y))
}

#[inline(never)]
pub(crate) fn _float_gt(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x > y))
}

#[inline(never)]
pub(crate) fn _float_ge(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x >= y))
}

#[inline(never)]
pub(crate) fn _float_eq(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x == y))
}

#[inline(never)]
pub(crate) fn _float_ne(x: f64, y: f64) -> PyResult {
    Ok(w_bool_from(x != y))
}

/// floatobject.py `_compare` — the float side of a numeric
/// comparison.  `w_float` is a float; `w_other` is a float, an int, a bool
/// or a long.  The relation is evaluated exactly: only an int small enough
/// that a double represents it losslessly takes the plain f64 path.
unsafe fn float_compare(w_float: PyObjectRef, w_other: PyObjectRef, op: CompareOp) -> bool {
    let f1 = w_float_get_value(w_float);
    if is_float(w_other) {
        return compare_f64(f1, w_float_get_value(w_other), op);
    }
    if is_long(w_other) {
        return do_compare_bigint(f1, w_long_get_value(w_other), op);
    }
    let i2 = if is_bool(w_other) {
        w_bool_get_value(w_other) as i64
    } else {
        w_int_get_value(w_other)
    };
    // (double-)floats have always at least 48 bits of precision, so an int
    // whose bit 48 and above are a plain sign extension converts exactly.
    // `int_between(-1, i2 >> 48, 1)` (rarithmetic.py) is `-1 <= i2 >> 48 < 1`.
    let top = i2 >> 48;
    if !(-1..1).contains(&top) {
        return do_compare_bigint(f1, &BigInt::from(i2), op);
    }
    compare_f64(f1, i2 as f64, op)
}

// ── Complex arithmetic operations ────────────────────────────────────

/// `complexobject.c _PyHASH_IMAG` — the imaginary-part hash multiplier.
const HASH_IMAG: i64 = 1_000_003;

/// `(real, imag)` for any numeric operand (`complex` / `int` / `long` /
/// `float` / `bool`), else `None`.
pub(crate) unsafe fn complex_val(obj: PyObjectRef) -> Option<(f64, f64)> {
    if is_complex(obj) {
        Some((w_complex_get_real(obj), w_complex_get_imag(obj)))
    } else if is_bool(obj) {
        Some((w_bool_get_value(obj) as i64 as f64, 0.0))
    } else if is_int(obj) || is_long(obj) || is_float(obj) {
        Some((as_float(obj), 0.0))
    } else {
        None
    }
}

/// True if both operands are numeric and at least one is `complex`: the
/// operands [`complex_val`] accepts, tested by type alone.  Converting a long
/// to test it would collect under the caller's operands.
unsafe fn is_complex_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    (is_complex(a) || is_complex(b)) && is_complex_operand(a) && is_complex_operand(b)
}

/// An operand [`complex_val`] converts.
unsafe fn is_complex_operand(obj: PyObjectRef) -> bool {
    is_complex(obj) || is_bool(obj) || is_int(obj) || is_long(obj) || is_float(obj)
}

unsafe fn complex_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (ar, ai) = complex_val(a()).unwrap();
    let (br, bi) = complex_val(b()).unwrap();
    reject_float_coercion_overflow(a(), ar)?;
    reject_float_coercion_overflow(b(), br)?;
    // CPython 3.14 complexobject.c COMPLEX_BINOP(add, sum): mixed real /
    // complex addition uses _Py_cr_sum / _Py_rc_sum and leaves the complex
    // imaginary lane untouched.  Besides matching C11 Annex G mixed-mode
    // arithmetic, this preserves the sign of an imaginary zero.
    if is_complex(a()) && is_complex(b()) {
        Ok(w_complex_new(ar + br, ai + bi))
    } else if is_complex(a()) {
        Ok(w_complex_new(ar + br, ai))
    } else {
        Ok(w_complex_new(ar + br, bi))
    }
}

unsafe fn complex_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (ar, ai) = complex_val(a()).unwrap();
    let (br, bi) = complex_val(b()).unwrap();
    reject_float_coercion_overflow(a(), ar)?;
    reject_float_coercion_overflow(b(), br)?;
    // CPython 3.14 _Py_c_diff / _Py_cr_diff / _Py_rc_diff.  In the mixed
    // cases only the real lane is combined; real-complex negates the complex
    // imaginary lane directly instead of subtracting it from +0.0.
    if is_complex(a()) && is_complex(b()) {
        Ok(w_complex_new(ar - br, ai - bi))
    } else if is_complex(a()) {
        Ok(w_complex_new(ar - br, ai))
    } else {
        Ok(w_complex_new(ar - br, -bi))
    }
}

unsafe fn complex_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (ar, ai) = complex_val(a()).unwrap();
    let (br, bi) = complex_val(b()).unwrap();
    reject_float_coercion_overflow(a(), ar)?;
    reject_float_coercion_overflow(b(), br)?;
    if is_complex(a()) && is_complex(b()) {
        Ok(complex_prod(ar, ai, br, bi))
    } else if is_complex(a()) {
        // CPython 3.14 _Py_cr_prod: multiply each existing complex lane by
        // the real operand, without manufacturing a zero imaginary lane.
        Ok(w_complex_new(ar * br, ai * br))
    } else {
        // _Py_rc_prod(a, b) delegates to _Py_cr_prod(b, a).
        Ok(w_complex_new(br * ar, bi * ar))
    }
}

/// CPython 3.14 `complexobject.c _Py_c_prod`, including the C11 Annex G.5.1
/// recovery for infinities that first produce `nan+nanj`.
unsafe fn complex_prod(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> PyObjectRef {
    let ac = a * c;
    let bd = b * d;
    let ad = a * d;
    let bc = b * c;
    let mut real = ac - bd;
    let mut imag = ad + bc;
    if real.is_nan() && imag.is_nan() {
        let mut recalc = false;
        if a.is_infinite() || b.is_infinite() {
            a = float_copysign(if a.is_infinite() { 1.0 } else { 0.0 }, a);
            b = float_copysign(if b.is_infinite() { 1.0 } else { 0.0 }, b);
            if c.is_nan() {
                c = float_copysign(0.0, c);
            }
            if d.is_nan() {
                d = float_copysign(0.0, d);
            }
            recalc = true;
        }
        if c.is_infinite() || d.is_infinite() {
            c = float_copysign(if c.is_infinite() { 1.0 } else { 0.0 }, c);
            d = float_copysign(if d.is_infinite() { 1.0 } else { 0.0 }, d);
            if a.is_nan() {
                a = float_copysign(0.0, a);
            }
            if b.is_nan() {
                b = float_copysign(0.0, b);
            }
            recalc = true;
        }
        if !recalc && (ac.is_infinite() || bd.is_infinite() || ad.is_infinite() || bc.is_infinite())
        {
            if a.is_nan() {
                a = float_copysign(0.0, a);
            }
            if b.is_nan() {
                b = float_copysign(0.0, b);
            }
            if c.is_nan() {
                c = float_copysign(0.0, c);
            }
            if d.is_nan() {
                d = float_copysign(0.0, d);
            }
            recalc = true;
        }
        if recalc {
            real = f64::INFINITY * (a * c - b * d);
            imag = f64::INFINITY * (a * d + b * c);
        }
    }
    w_complex_new(real, imag)
}

/// CPython 3.14 `complexobject.c _Py_c_quot`: Smith's stable division plus
/// the C11 Annex G.5.2 recovery for infinite numerators and denominators.
unsafe fn complex_quot(ar: f64, ai: f64, br: f64, bi: f64) -> PyObjectRef {
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    let mut real: f64;
    let mut imag: f64;
    if abs_br >= abs_bi {
        if abs_br == 0.0 {
            // `_Py_c_quot` writes 0+0j and signals EDOM through errno.  The
            // caller performs that side-channel check before using the pair.
            return w_complex_new(0.0, 0.0);
        }
        let ratio = bi / br;
        let denom = br + bi * ratio;
        real = (ar + ai * ratio) / denom;
        imag = (ai - ar * ratio) / denom;
    } else if abs_bi >= abs_br {
        let ratio = br / bi;
        let denom = br * ratio + bi;
        real = (ar * ratio + ai) / denom;
        imag = (ai * ratio - ar) / denom;
    } else {
        real = f64::NAN;
        imag = f64::NAN;
    }
    if real.is_nan() && imag.is_nan() {
        if (ar.is_infinite() || ai.is_infinite()) && br.is_finite() && bi.is_finite() {
            let x = float_copysign(if ar.is_infinite() { 1.0 } else { 0.0 }, ar);
            let y = float_copysign(if ai.is_infinite() { 1.0 } else { 0.0 }, ai);
            real = f64::INFINITY * (x * br + y * bi);
            imag = f64::INFINITY * (y * br - x * bi);
        } else if (abs_br.is_infinite() || abs_bi.is_infinite()) && ar.is_finite() && ai.is_finite()
        {
            let x = float_copysign(if br.is_infinite() { 1.0 } else { 0.0 }, br);
            let y = float_copysign(if bi.is_infinite() { 1.0 } else { 0.0 }, bi);
            real = 0.0 * (ar * x + ai * y);
            imag = 0.0 * (ai * x - ar * y);
        }
    }
    w_complex_new(real, imag)
}

/// CPython 3.14 `_Py_rc_quot`: real divided by complex, with its distinct
/// signed-zero recovery when the denominator contains an infinity.
unsafe fn real_complex_quot(a: f64, br: f64, bi: f64) -> PyObjectRef {
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    let mut real: f64;
    let mut imag: f64;
    if abs_br >= abs_bi {
        if abs_br == 0.0 {
            return w_complex_new(0.0, 0.0);
        }
        let ratio = bi / br;
        let denom = br + bi * ratio;
        real = a / denom;
        imag = (-a * ratio) / denom;
    } else if abs_bi >= abs_br {
        let ratio = br / bi;
        let denom = br * ratio + bi;
        real = (a * ratio) / denom;
        imag = (-a) / denom;
    } else {
        real = f64::NAN;
        imag = f64::NAN;
    }
    if real.is_nan()
        && imag.is_nan()
        && a.is_finite()
        && (abs_br.is_infinite() || abs_bi.is_infinite())
    {
        let x = float_copysign(if br.is_infinite() { 1.0 } else { 0.0 }, br);
        let y = float_copysign(if bi.is_infinite() { 1.0 } else { 0.0 }, bi);
        real = 0.0 * (a * x);
        imag = 0.0 * (-a * y);
    }
    w_complex_new(real, imag)
}

unsafe fn complex_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (ar, ai) = complex_val(a()).unwrap();
    let (br, bi) = complex_val(b()).unwrap();
    reject_float_coercion_overflow(a(), ar)?;
    reject_float_coercion_overflow(b(), br)?;
    if br == 0.0 && bi == 0.0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    if is_complex(a()) && is_complex(b()) {
        Ok(complex_quot(ar, ai, br, bi))
    } else if is_complex(a()) {
        // CPython 3.14 _Py_cr_quot.
        Ok(w_complex_new(ar / br, ai / br))
    } else {
        Ok(real_complex_quot(ar, br, bi))
    }
}

unsafe fn complex_powi(a: PyObjectRef, exponent: i64) -> PyResult {
    let (mut base_real, mut base_imag) = complex_val(a).unwrap();
    let mut result_real = 1.0;
    let mut result_imag = 0.0;
    let mut n = exponent.unsigned_abs();
    while n != 0 {
        if n & 1 != 0 {
            // CPython 3.14 c_powu calls _Py_c_prod for every multiply, so
            // the Annex G infinity recovery is shared with ordinary `*`.
            let product = complex_prod(result_real, result_imag, base_real, base_imag);
            result_real = w_complex_get_real(product);
            result_imag = w_complex_get_imag(product);
        }
        n >>= 1;
        if n != 0 {
            let square = complex_prod(base_real, base_imag, base_real, base_imag);
            base_real = w_complex_get_real(square);
            base_imag = w_complex_get_imag(square);
        }
    }
    let result = if exponent < 0 {
        complex_truediv(
            w_complex_new(1.0, 0.0),
            w_complex_new(result_real, result_imag),
        )?
    } else {
        w_complex_new(result_real, result_imag)
    };
    // complexobject.c complex_pow: `_Py_ADJUST_ERANGE2` forces ERANGE when
    // either component is infinite; the numeric slot reports that as
    // `OverflowError("complex exponentiation")`.
    if w_complex_get_real(result).is_infinite() || w_complex_get_imag(result).is_infinite() {
        Err(PyError::overflow_error("complex exponentiation"))
    } else {
        Ok(result)
    }
}

/// `complexobject.c _Py_c_pow`.
unsafe fn complex_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[a, b]);
    let a = || pyre_object::gc_roots::shadow_stack_get(base);
    let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (ar, ai) = complex_val(a()).unwrap();
    let (br, bi) = complex_val(b()).unwrap();
    let (real, imag) = if br == 0.0 && bi == 0.0 {
        (1.0, 0.0)
    } else if ar == 0.0 && ai == 0.0 {
        if bi != 0.0 || br < 0.0 {
            return Err(PyError::zero_division(
                "zero to a negative or complex power",
            ));
        }
        (0.0, 0.0)
    } else if bi == 0.0 && (-100.0..=100.0).contains(&br) && br == br.trunc() {
        return complex_powi(a(), br as i64);
    } else {
        let vabs = ar.hypot(ai);
        let mut len = vabs.powf(br);
        let at = ai.atan2(ar);
        let mut phase = at * br;
        if bi != 0.0 {
            len /= (at * bi).exp();
            phase += bi * vabs.ln();
        }
        (len * phase.cos(), len * phase.sin())
    };
    if real.is_infinite() || imag.is_infinite() {
        Err(PyError::overflow_error("complex exponentiation"))
    } else {
        Ok(w_complex_new(real, imag))
    }
}

unsafe fn complex_neg(a: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    Ok(w_complex_new(-ar, -ai))
}

/// `abs(complex)` → the float magnitude `hypot(real, imag)`.
pub(crate) unsafe fn complex_abs(a: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let result = ar.hypot(ai);
    if result.is_infinite() && ar.is_finite() && ai.is_finite() {
        return Err(PyError::overflow_error("absolute value too large"));
    }
    // Loop-free looked-inside caller of the unboxed float math leaves.
    // `interp_math::{sqrt,sin,cos}` live under opaque `crate::module`,
    // the same class as `_float_lt` before [`compare_slot`].
    _float_math1(result, FLOAT_MATH1_ABS)
}

pub const FLOAT_MATH1_ABS: i64 = 0;
pub const FLOAT_MATH1_SQRT: i64 = 1;
pub const FLOAT_MATH1_SIN: i64 = 2;
pub const FLOAT_MATH1_COS: i64 = 3;
pub const FLOAT_MATH1_TAN: i64 = 4;
pub const FLOAT_MATH1_ATAN: i64 = 5;
pub const FLOAT_MATH1_EXP: i64 = 6;
pub const FLOAT_MATH1_LOG1P: i64 = 7;
pub const FLOAT_MATH1_ASIN: i64 = 8;
pub const FLOAT_MATH1_ACOS: i64 = 9;
pub const FLOAT_MATH1_SINH: i64 = 10;
pub const FLOAT_MATH1_COSH: i64 = 11;
pub const FLOAT_MATH1_TANH: i64 = 12;
pub const FLOAT_MATH1_ASINH: i64 = 13;
pub const FLOAT_MATH1_ACOSH: i64 = 14;
pub const FLOAT_MATH1_ATANH: i64 = 15;
pub const FLOAT_MATH1_CBRT: i64 = 16;
pub const FLOAT_MATH1_EXP2: i64 = 17;
pub const FLOAT_MATH1_EXPM1: i64 = 18;
pub const FLOAT_MATH1_ERF: i64 = 19;
pub const FLOAT_MATH1_ERFC: i64 = 20;
pub const FLOAT_MATH1_GAMMA: i64 = 21;
pub const FLOAT_MATH1_LGAMMA: i64 = 22;
pub const FLOAT_MATH1_ULP: i64 = 23;
pub const FLOAT_MATH1_DEGREES: i64 = 24;
pub const FLOAT_MATH1_RADIANS: i64 = 25;
pub const FLOAT_MATH1_LOG: i64 = 26;
/// Discovery only: `kind` is a hub parameter, so these arms stay in the graph.
pub const FLOAT_MATH1_MATH2: i64 = 27;
pub const FLOAT_MATH1_INT_FROM_FLOAT: i64 = 28;
pub const FLOAT_MATH1_FREXP: i64 = 29;
pub const FLOAT_MATH1_FREXP_EXP: i64 = 30;
pub const FLOAT_MATH1_LDEXP: i64 = 31;
pub const FLOAT_MATH1_ISQRT: i64 = 32;
pub const FLOAT_MATH1_ISCLOSE: i64 = 33;

/// Hub so `_float_{sqrt,sin,cos,tan}` are jitcodes (`_float_lt` / [`compare_slot`]).
/// `inline(never)` keeps every arm in the graph when a caller passes a constant.
#[inline(never)]
pub fn _float_math1(x: f64, kind: i64) -> PyResult {
    match kind {
        FLOAT_MATH1_SQRT => _float_sqrt(x),
        FLOAT_MATH1_SIN => _float_sin(x),
        FLOAT_MATH1_COS => _float_cos(x),
        FLOAT_MATH1_TAN => _float_tan(x),
        FLOAT_MATH1_ATAN => _float_atan(x),
        FLOAT_MATH1_EXP => _float_exp(x),
        FLOAT_MATH1_LOG1P => _float_log1p(x),
        FLOAT_MATH1_ASIN => _float_asin(x),
        FLOAT_MATH1_ACOS => _float_acos(x),
        FLOAT_MATH1_SINH => _float_sinh(x),
        FLOAT_MATH1_COSH => _float_cosh(x),
        FLOAT_MATH1_TANH => _float_tanh(x),
        FLOAT_MATH1_ASINH => _float_asinh(x),
        FLOAT_MATH1_ACOSH => _float_acosh(x),
        FLOAT_MATH1_ATANH => _float_atanh(x),
        FLOAT_MATH1_CBRT => _float_cbrt(x),
        FLOAT_MATH1_EXP2 => _float_exp2(x),
        FLOAT_MATH1_EXPM1 => _float_expm1(x),
        FLOAT_MATH1_ERF => _float_erf(x),
        FLOAT_MATH1_ERFC => _float_erfc(x),
        FLOAT_MATH1_GAMMA => _float_gamma(x),
        FLOAT_MATH1_LGAMMA => _float_lgamma(x),
        FLOAT_MATH1_ULP => _float_ulp(x),
        FLOAT_MATH1_DEGREES => _float_degrees(x),
        FLOAT_MATH1_RADIANS => _float_radians(x),
        FLOAT_MATH1_LOG => _float_log(x),
        FLOAT_MATH1_MATH2 => _float_math2(x, x, 0),
        FLOAT_MATH1_INT_FROM_FLOAT => _int_from_float(x, 0),
        FLOAT_MATH1_FREXP => _float_frexp_mantissa(x),
        FLOAT_MATH1_FREXP_EXP => _int_frexp_exponent(x),
        FLOAT_MATH1_LDEXP => _float_ldexp(x, 0),
        // `_int_abs` is only named here. `abs` boxes inline so the hot leaf
        // stays virtual; this arm is what emits the `_int_abs` jitcode.
        FLOAT_MATH1_ISQRT => {
            let _ = _int_abs(0)?;
            _int_isqrt(0)
        }
        FLOAT_MATH1_ISCLOSE => _float_isclose(x, x),
        _ => _float_abs(x),
    }
}

pub const FLOAT_MATH2_POW: i64 = 0;
pub const FLOAT_MATH2_FMOD: i64 = 1;
pub const FLOAT_MATH2_COPYSIGN: i64 = 2;
pub const FLOAT_MATH2_REMAINDER: i64 = 3;
pub const FLOAT_MATH2_ATAN2: i64 = 4;

/// Hub so the two-arg unboxed leaves are jitcodes.
#[inline(never)]
pub fn _float_math2(x: f64, y: f64, kind: i64) -> PyResult {
    match kind {
        FLOAT_MATH2_FMOD => _float_fmod(x, y),
        FLOAT_MATH2_COPYSIGN => _float_copysign(x, y),
        FLOAT_MATH2_REMAINDER => _float_remainder(x, y),
        FLOAT_MATH2_ATAN2 => _float_atan2(x, y),
        _ => _float_pow(x, y),
    }
}

pub const INT_FROM_FLOAT_FLOOR: i64 = 0;
pub const INT_FROM_FLOAT_CEIL: i64 = 1;
pub const INT_FROM_FLOAT_TRUNC: i64 = 2;

/// Hub so the `math.floor`/`ceil`/`trunc` int leaves are jitcodes.
#[inline(never)]
pub fn _int_from_float(x: f64, kind: i64) -> PyResult {
    match kind {
        INT_FROM_FLOAT_CEIL => _int_from_ceil(x),
        INT_FROM_FLOAT_TRUNC => _int_from_trunc(x),
        _ => _int_from_floor(x),
    }
}

/// Complex equality: `==`/`!=` only (no ordering).  Mixed numeric
/// operands compare equal when the imaginary part is zero.
unsafe fn complex_richcompare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return Ok(w_not_implemented());
    }
    let equal = if is_complex(a) && is_complex(b) {
        w_complex_get_real(a) == w_complex_get_real(b)
            && w_complex_get_imag(a) == w_complex_get_imag(b)
    } else {
        // `complexobject.py descr_eq`: compare a real-only complex through
        // float/int equality.  Do not round an arbitrary-size integer to f64;
        // convert the integral float lane to BigInt and compare exactly.
        let (z, mut other) = if is_complex(a) { (a, b) } else { (b, a) };
        let real = w_complex_get_real(z);
        if w_complex_get_imag(z) != 0.0 {
            false
        } else if is_float(other) {
            real == w_float_get_value(other)
        } else if is_int(other) || is_long(other) || is_bool(other) {
            real.is_finite()
                && real.fract() == 0.0
                && pyre_object::with_roots!(other => BigInt::from_f64(real)).is_some_and(|value| {
                    if is_long(other) {
                        value.eq(w_long_get_value(other))
                    } else {
                        value.int_eq(int_value(other))
                    }
                })
        } else {
            return Ok(w_not_implemented());
        }
    };
    Ok(w_bool_from(if matches!(op, CompareOp::Eq) {
        equal
    } else {
        !equal
    }))
}

/// `complexobject.c complex_hash` — `hash(real) + _PyHASH_IMAG * hash(imag)`.
pub(crate) fn complex_hash(obj: PyObjectRef, real: f64, imag: f64) -> i64 {
    // `complexobject.py descr_hash`: each NaN lane uses the containing
    // complex object's identity hash, not the float HASH_NAN sentinel.
    let identity = || crate::typedef::default_identity_hash_value(obj);
    let hr = if real.is_nan() {
        identity()
    } else {
        crate::builtins::_hash_float(real)
    };
    let hi = if imag.is_nan() {
        identity()
    } else {
        crate::builtins::_hash_float(imag)
    };
    let combined = hr.wrapping_add(hi.wrapping_mul(HASH_IMAG));
    if combined == -1 { -2 } else { combined }
}

// ── Public dispatch API ───────────────────────────────────────────────

/// Check if w_type is a subtype of cls — delegates to the single MRO
/// membership scan in `pyre_object::w_type_issubtype`.
unsafe fn issubtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    pyre_object::w_type_issubtype(w_type, cls)
}

/// Comparison special-method dispatch.
///
/// `dont_look_inside`: `policy.py look_inside_graph` rejects a graph
/// that contains a loop. This arm walks the MRO (`lookup_in_type_where`);
/// keeping that walk residual lets [`compare`] stay loop-free so a
/// traced `int < int` can look inside `comparison_impl` and reach
/// [`compare_slot`]. Exact builtin pairs never call this
/// (`pair_can_override` is false on a promoted decision).
///
/// PyPy: `descroperation.py:_make_comparison_impl`.  Generic instances use
/// the complete MRO lookup, including inherited `object.__eq__` /
/// `object.__ne__`; the latter deliberately calls the receiver's live
/// `__eq__`.  Builtin layouts retain the override-only lookup because their
/// inherited slots delegate straight back into [`compare`] and would recurse
/// forever.  The reflected method of a proper subclass has priority.  PyPy
/// 3.11 suppresses the second call for equal user-defined types, but CPython
/// 3.14 `Objects/object.c:do_richcompare` tries `tp_richcompare` in both
/// operand directions when the first call returns `NotImplemented`; pyre's
/// Python 3.14 compatibility rule takes precedence here.
#[majit_macros::dont_look_inside]
unsafe fn try_compare_override(
    a: PyObjectRef,
    b: PyObjectRef,
    op: CompareOp,
) -> Result<Option<PyObjectRef>, PyError> {
    let dunder = match op {
        CompareOp::Lt => "__lt__",
        CompareOp::Le => "__le__",
        CompareOp::Gt => "__gt__",
        CompareOp::Ge => "__ge__",
        CompareOp::Eq => "__eq__",
        CompareOp::Ne => "__ne__",
    };
    let rdunder = reverse_dunder(dunder).unwrap_or(dunder);
    let comparison_method = |obj: PyObjectRef, name: &str| {
        if is_instance(obj) {
            let w_type = crate::typedef::r#type(obj)?;
            let method = lookup_in_type_where(w_type.as_ptr(), name)?;
            Some((method, w_type.as_ptr()))
        } else {
            crate::baseobjspace::subclass_special_override(obj, name)
        }
    };
    let a_type = crate::typedef::r#type(a);
    let b_type = crate::typedef::r#type(b);
    let a_ov = comparison_method(a, dunder);
    let b_ov = comparison_method(b, rdunder);
    if a_ov.is_none() && b_ov.is_none() {
        return Ok(None);
    }
    // Python's "subclass reflected op takes priority": if `b`'s type is a
    // proper subclass of `a`'s type and `b` overrides the reflected op, run it
    // first.
    let b_first = b_ov.is_some()
        && match (a_type, b_type) {
            (Some(at), Some(bt)) => at != bt && issubtype_cached(bt.as_ptr(), at.as_ptr()),
            _ => false,
        };
    let order = if b_first {
        [(b_ov, b, a), (a_ov, a, b)]
    } else {
        [(a_ov, a, b), (b_ov, b, a)]
    };
    for (ov, recv, other) in order {
        if let Some((method, w_type)) = ov
            && let Some(result) = unsafe { invoke_comparison(method, recv, w_type, other) }?
        {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// PyPy `descroperation.py:_invoke_comparison`.
///
/// A plain function is called directly, so every exception from its body is
/// observable. Other descriptors bind through `space.get`; only an
/// `AttributeError` raised by that binding step means that this comparison
/// implementation is absent (notably `__eq__ = property(...)`).
unsafe fn invoke_comparison(
    mut w_descr: PyObjectRef,
    w_obj: PyObjectRef,
    w_type: PyObjectRef,
    mut w_other: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let direct_function = std::ptr::eq(
        unsafe { (*w_descr).ob_type },
        &crate::function::FUNCTION_TYPE as *const _,
    ) || std::ptr::eq(
        unsafe { (*w_descr).ob_type },
        &crate::function::METHOD_DESCRIPTOR_TYPE as *const _,
    );
    let result = if direct_function {
        crate::call::call_function_impl_result(w_descr, &[w_obj, w_other])?
    } else {
        let w_impl = match pyre_object::with_roots!(w_descr, w_other => unsafe {
            crate::baseobjspace::get(w_descr, w_obj, w_type)
        }) {
            Ok(Some(w_impl)) => w_impl,
            Ok(None) => w_descr,
            Err(err) if err.kind == PyErrorKind::AttributeError => return Ok(None),
            Err(err) => return Err(err),
        };
        crate::call::call_function_impl_result(w_impl, &[w_other])?
    };
    if is_not_implemented(result) {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// Map forward dunder to reverse dunder.
/// PyPy: descroperation.py `_make_binop_impl` generates both directions.
fn reverse_dunder(dunder: &str) -> Option<&'static str> {
    Some(match dunder {
        // Arithmetic — PyPy: descroperation.py _make_binop_impl
        "__add__" => "__radd__",
        "__sub__" => "__rsub__",
        "__mul__" => "__rmul__",
        "__truediv__" => "__rtruediv__",
        "__floordiv__" => "__rfloordiv__",
        "__mod__" => "__rmod__",
        "__matmul__" => "__rmatmul__",
        "__pow__" => "__rpow__",
        "__lshift__" => "__rlshift__",
        "__rshift__" => "__rrshift__",
        "__and__" => "__rand__",
        "__or__" => "__ror__",
        "__xor__" => "__rxor__",
        // Comparison reflected — PyPy: descroperation.py _cmp_dispatch
        "__lt__" => "__gt__",
        "__le__" => "__ge__",
        "__gt__" => "__lt__",
        "__ge__" => "__le__",
        "__eq__" => "__eq__",
        "__ne__" => "__ne__",
        _ => return None,
    })
}

/// Try to call a unary dunder resolved off the receiver's type.
///
/// `descroperation.py _make_unaryop_impl`:
/// `ObjSpace.call_function(space.lookup(w_obj, dunder), w_obj)`
/// The Python-level OperationError must propagate to the caller; use the
/// Result-returning call path so PENDING_CALL_ERROR is consumed.
///
/// `lookup` is the whole gate, as it is upstream.  Screening the receiver for
/// `is_instance` first answered `-C` for a class `C` whose METAclass defines
/// `__neg__` with `bad operand type for unary -`, because a class object is
/// not an instance: `lookup` reads the dunder off `type(C)`, which is the
/// metaclass, and that is the resolution `-C` owes.
unsafe fn try_lookup_unaryop(a: PyObjectRef, dunder: &str) -> Result<Option<PyObjectRef>, PyError> {
    let Some(method) = lookup(a, dunder) else {
        return Ok(None);
    };
    let Some(w_type) = crate::typedef::r#type(a) else {
        return Ok(None);
    };
    Ok(Some(crate::baseobjspace::get_and_call_function(
        method,
        a,
        w_type.as_ptr(),
        &[],
    )?))
}

/// True when `obj`'s type defines `dunder` in a class other than the
/// builtin type object `tp` — i.e. a subclass overrides it.  Builtin
/// `str`/`list`/`tuple` install `__add__`/`__radd__` on their own type;
/// an inherited (non-overridden) lookup resolves back to `tp`.
unsafe fn dunder_overridden(obj: PyObjectRef, dunder: &str, tp: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj)
        .and_then(|t| lookup_where_with_method_cache(t.as_ptr(), dunder))
    {
        Some((src, _)) => !std::ptr::eq(src, tp),
        None => false,
    }
}

/// Special-method pair passed to the residual override gates below.  A
/// residual call gives every argument one machine word, so a `&str` — a fat
/// pointer — cannot be published through `jit_fnaddr`, and a traced caller
/// is left holding an unbound symbolic residual where a bound call belongs.
/// A fieldless enum is one word, so the gate stays bindable; the names are
/// recovered inside the residual, where the lookups they feed already live.
#[derive(Clone, Copy)]
pub(crate) enum BinopDunder {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    TrueDiv,
    Pow,
    DivMod,
    LShift,
    RShift,
    And,
    Or,
    Xor,
}

impl BinopDunder {
    /// The forward name and its reflected counterpart, in that order.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Add => ("__add__", "__radd__"),
            Self::Sub => ("__sub__", "__rsub__"),
            Self::Mul => ("__mul__", "__rmul__"),
            Self::FloorDiv => ("__floordiv__", "__rfloordiv__"),
            Self::Mod => ("__mod__", "__rmod__"),
            Self::TrueDiv => ("__truediv__", "__rtruediv__"),
            Self::Pow => ("__pow__", "__rpow__"),
            Self::DivMod => ("__divmod__", "__rdivmod__"),
            Self::LShift => ("__lshift__", "__rlshift__"),
            Self::RShift => ("__rshift__", "__rrshift__"),
            Self::And => ("__and__", "__rand__"),
            Self::Or => ("__or__", "__ror__"),
            Self::Xor => ("__xor__", "__rxor__"),
        }
    }
}

/// Unary special-method name, carried as a word for the same reason as
/// [`BinopDunder`].
#[derive(Clone, Copy)]
pub(crate) enum UnaryDunder {
    Pos,
    Neg,
    Invert,
}

impl UnaryDunder {
    fn name(self) -> &'static str {
        match self {
            Self::Pos => "__pos__",
            Self::Neg => "__neg__",
            Self::Invert => "__invert__",
        }
    }
}

/// The set of special-method names a repeat gate tests, carried as a word for
/// the same reason as [`BinopDunder`] — a `&[&str]` is a slice, so it fills two
/// argument slots and leaves the gate unpublishable.
#[derive(Clone, Copy)]
pub(crate) enum RepeatDunder {
    /// `__mul__` and its reflected counterpart, the pair a binary repeat tests.
    MulPair,
    /// `__imul__` alone; an in-place repeat has no reflected form.
    IMul,
}

impl RepeatDunder {
    fn names(self) -> &'static [&'static str] {
        match self {
            Self::MulPair => &["__mul__", "__rmul__"],
            Self::IMul => &["__imul__"],
        }
    }
}

/// Builtin sequence base selected by [`needs_seq_binop_dispatch`].  The
/// caller passes this discriminant instead of a `&STR_TYPE`/… static so
/// the type-static load stays inside the residual helper, off the traced
/// `add` graph.
#[derive(Clone, Copy)]
pub(crate) enum SeqBase {
    Str,
    List,
    Tuple,
}

/// descroperation.py `binop_impl` shortcut — the builtin sequence
/// fast path (`str`/`list`/`tuple` concat) bypasses `__op__`/`__rop__`
/// dispatch unless one operand is a subclass that actually overrides the
/// forward or reflected special method (descroperation.py "unicode +
/// string subclass").  Returns `false` when no override exists so the
/// caller concatenates directly — this also avoids re-entering the
/// builtin `__add__` slot, which would recurse back into `add`.
///
/// `dont_look_inside`: the builtin-base type static is loaded here, so a
/// traced caller emits a residual call instead of carrying an
/// unresolvable `LoadStatic` into its JitCode.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn needs_seq_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    base: SeqBase,
    op: BinopDunder,
) -> bool {
    let (fwd, rev) = op.names();
    let tp: *const pyre_object::PyType = match base {
        SeqBase::Str => &pyre_object::STR_TYPE,
        SeqBase::List => &pyre_object::LIST_TYPE,
        SeqBase::Tuple => &pyre_object::TUPLE_TYPE,
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(a, fwd, t)
        || dunder_overridden(a, rev, t)
        || dunder_overridden(b, fwd, t)
        || dunder_overridden(b, rev, t)
}

/// [`needs_seq_binop_dispatch`] behind the promoted exact-builtin test of
/// both operands; see [`needs_numeric_binop_dispatch_unless_exact`].
#[inline]
unsafe fn needs_seq_binop_dispatch_unless_exact(
    a: PyObjectRef,
    b: PyObjectRef,
    base: SeqBase,
    op: BinopDunder,
) -> bool {
    !both_exact_builtin_instances_promoted(a, b) && needs_seq_binop_dispatch(a, b, base, op)
}

/// `bytes`/`bytearray` analog of `needs_seq_binop_dispatch`.  The two
/// builtin types share one `+` branch, so each operand is judged against
/// its own builtin base (`bytes` vs `bytearray`).  `dont_look_inside`
/// keeps the base type-static loads in the residual call, off the traced
/// `add` graph.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn needs_bytes_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinopDunder,
) -> bool {
    let (fwd, rev) = op.names();
    bytes_operand_overrides(a, fwd, rev) || bytes_operand_overrides(b, fwd, rev)
}

/// [`needs_bytes_binop_dispatch`] behind the promoted exact-builtin test of
/// both operands; see [`needs_numeric_binop_dispatch_unless_exact`].
#[inline]
unsafe fn needs_bytes_binop_dispatch_unless_exact(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinopDunder,
) -> bool {
    !both_exact_builtin_instances_promoted(a, b) && needs_bytes_binop_dispatch(a, b, op)
}

/// True when `obj`'s type overrides `fwd`/`rev` relative to its builtin
/// base (`bytes` or `bytearray`).  Only reached from the residual
/// `needs_bytes_binop_dispatch`, so the type-static loads never enter a
/// traced graph.
unsafe fn bytes_operand_overrides(obj: PyObjectRef, fwd: &str, rev: &str) -> bool {
    let tp: *const pyre_object::PyType = if pyre_object::bytesobject::is_bytes(obj) {
        &pyre_object::bytesobject::BYTES_TYPE
    } else {
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(obj, fwd, t) || dunder_overridden(obj, rev, t)
}

/// `update_one_slot`'s duplicated-slotdef rule.  `slotdefs` names `__add__`
/// twice — once at `nb_add`, once at `sq_concat` — and `__mul__` twice, at
/// `nb_multiply` and `sq_repeat`.  A type whose entry for the name is the
/// sequence wrapper is left with the numeric half NULL, and a subclass that
/// writes no method of its own inherits that NULL, so `binary_op1` never
/// offers the sequence as a numeric operand and the other one answers alone:
/// `SS('a') + IntR(1)` is the `int` subclass's `__radd__` and `SS('a') *
/// IntR(2)` its `__rmul__`, not a concat refusal and not a repetition.
///
/// Only a subclass that writes the dunder in Python installs the generic slot
/// and takes part, which is what the `dunder_overridden` test below asks.
///
/// `dont_look_inside` for the reason its neighbours carry it: the builtin
/// base type statics are loaded here, and a traced caller would otherwise
/// carry an unresolvable `LoadStatic`.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn sequence_numeric_slot_is_null(obj: PyObjectRef, op: BinopDunder) -> bool {
    let tp: *const pyre_object::PyType = if is_str(obj) {
        &pyre_object::STR_TYPE
    } else if is_list(obj) {
        &pyre_object::LIST_TYPE
    } else if is_tuple(obj) {
        &pyre_object::TUPLE_TYPE
    } else if pyre_object::bytesobject::is_bytes(obj) {
        &pyre_object::bytesobject::BYTES_TYPE
    } else if pyre_object::bytearrayobject::is_bytearray(obj) {
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE
    } else {
        return false;
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    !dunder_overridden(obj, op.names().0, t.as_ptr())
}

/// `needs_seq_binop_dispatch` for the `sq_repeat` branches of [`mul`], where
/// only one operand is the sequence: a subclass that overrides the multiply
/// specials relative to its own builtin base has to run that override instead
/// of repeating, exactly as the concat branches of [`add`] are gated.
///
/// Judged one operand at a time rather than as a pair, because a repeat's other
/// operand is the multiplier — an `int`, whose own `__mul__` lives on `int` and
/// would read as an override against any sequence base.  A *non*-overriding
/// subclass keeps the repeat path, which is what makes `L([1]) * 2` still a
/// plain list repetition.
///
/// `dont_look_inside`: the builtin-base type statics are loaded here, so a
/// traced caller emits a residual call rather than an unresolvable
/// `LoadStatic`.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn seq_repeat_override(obj: PyObjectRef, dunders: RepeatDunder) -> bool {
    if pyre_object::is_exact_builtin_instance(obj) {
        return false;
    }
    let t = if is_str(obj) {
        let Some(t) = crate::typedef::gettypefor(&pyre_object::STR_TYPE) else {
            return false;
        };
        t.as_ptr()
    } else if is_list(obj) {
        let Some(t) = crate::typedef::gettypefor(&pyre_object::LIST_TYPE) else {
            return false;
        };
        t.as_ptr()
    } else if is_tuple(obj) {
        let Some(t) = crate::typedef::gettypefor(&pyre_object::TUPLE_TYPE) else {
            return false;
        };
        t.as_ptr()
    } else if pyre_object::bytesobject::is_bytes_like(obj) {
        let tp = if pyre_object::bytesobject::is_bytes(obj) {
            &pyre_object::bytesobject::BYTES_TYPE
        } else {
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE
        };
        let Some(t) = crate::typedef::gettypefor(tp) else {
            return false;
        };
        t.as_ptr()
    } else if crate::module::_collections::is_deque(obj) {
        crate::module::_collections::type_object()
    } else {
        return false;
    };
    dunders
        .names()
        .iter()
        .any(|dunder| dunder_overridden(obj, dunder, t))
}

/// True when `obj` is an exact builtin numeric instance
/// (`int`/`long`/`float`/`complex`/`bool`, not a subclass).  These types
/// define no in-place special method (`__iadd__` etc.), so
/// [`try_inplace_special`] can skip the in-place lookup for them.  A
/// subclass ([`is_exact_builtin_instance`] false) may override the slot, so
/// it is excluded.
unsafe fn is_exact_numeric_builtin(obj: PyObjectRef) -> bool {
    pyre_object::is_exact_builtin_instance(obj)
        && (is_int(obj) || is_long(obj) || is_float(obj) || is_complex(obj))
}

/// The builtin numeric base type backing `obj`'s storage — `int` for
/// int/long, `float` for float, `complex` for complex, `None` for a
/// non-numeric operand.
unsafe fn numeric_base_type(obj: PyObjectRef) -> Option<*const pyre_object::PyType> {
    if is_int(obj) || is_long(obj) {
        Some(&pyre_object::INT_TYPE)
    } else if is_float(obj) {
        Some(&pyre_object::FLOAT_TYPE)
    } else if is_complex(obj) {
        Some(&pyre_object::COMPLEX_TYPE)
    } else {
        None
    }
}

/// The builtin numeric base type object of `obj`, returned only when `obj`
/// is a subclass instance that may override a special method.  An exact
/// builtin numeric instance ([`is_exact_builtin_instance`]) cannot override
/// any special method, so it yields `None`, skipping the string-keyed
/// method-cache lookups in [`dunder_overridden`].  Those lookups (one
/// UTF-8-keyed cache probe per dunder) are the dominant per-iteration cost
/// of the interpreter numeric fast path.  This mirrors the exact-builtin
/// gate already opening `subclass_special_override` / `is_true`.
unsafe fn numeric_base_type_of_overriding_subclass(
    obj: PyObjectRef,
) -> Option<std::ptr::NonNull<pyre_object::PyObject>> {
    if pyre_object::is_exact_builtin_instance(obj) {
        return None;
    }
    let base = numeric_base_type(obj)?;
    crate::typedef::gettypefor(base)
}

/// True when numeric operand `obj` (int/long/float storage) has a Python
/// class that overrides `dunder` relative to its builtin base — int/long
/// against `int`, float against `float`, complex against `complex`.  Mirrors [`dunder_overridden`]
/// with the numeric base selected per operand.  Only an *overriding*
/// subclass routes through dispatch: a non-overriding subclass keeps the
/// Rust fast path, which both matches the builtin result and avoids
/// re-entering the inherited slot (it would recurse back into this op).
unsafe fn numeric_operand_overrides(obj: PyObjectRef, dunder: &str, rdunder: &str) -> bool {
    let Some(t) = numeric_base_type_of_overriding_subclass(obj) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(obj, dunder, t) || dunder_overridden(obj, rdunder, t)
}

/// descroperation.py `binop_impl` shortcut — the builtin numeric
/// (int/long/float) fast path bypasses `__op__`/`__rop__` dispatch unless
/// an operand is a subclass that actually overrides the forward or
/// reflected special method.  The seq/bytes analogs are
/// [`needs_seq_binop_dispatch`]/[`needs_bytes_binop_dispatch`].
///
/// `dont_look_inside`: the type-static + typeobject-registry loads stay in
/// this residual helper, off the traced numeric graph; the hot int path is
/// specialized separately via `guard_class` in the JIT.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn needs_numeric_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinopDunder,
) -> bool {
    let (fwd, rev) = op.names();
    numeric_operand_overrides(a, fwd, rev) || numeric_operand_overrides(b, fwd, rev)
}

/// [`needs_numeric_binop_dispatch`] behind the promoted exact-builtin test
/// of both operands.  Two exact builtins override nothing, and that answer
/// is decided on the traced graph (see
/// [`is_exact_builtin_instance_promoted`]), so a traced binary operator on
/// two builtins records neither this probe nor the override arm it gates.
#[inline]
unsafe fn needs_numeric_binop_dispatch_unless_exact(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinopDunder,
) -> bool {
    !both_exact_builtin_instances_promoted(a, b) && needs_numeric_binop_dispatch(a, b, op)
}

/// Set/frozenset analogue of the numeric override gate.  The storage fast
/// paths below are valid for exact builtins, but a heap subclass must enter
/// `_call_binop_impl` so its forward/reflected override and the reflected-
/// subclass priority are observed before inherited set semantics.
///
/// `dont_look_inside`: `is_exact_type` loads the builtin `SET_TYPE`/
/// `FROZENSET_TYPE` statics here, so keep that load in this residual helper
/// off the traced graph — mirrors `needs_seq`/`needs_bytes`/`needs_numeric`.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn needs_set_binop_dispatch(a: PyObjectRef, b: PyObjectRef) -> bool {
    let is_exact_setlike = |obj| {
        pyre_object::is_exact_type(obj, &pyre_object::setobject::SET_TYPE)
            || pyre_object::is_exact_type(obj, &pyre_object::setobject::FROZENSET_TYPE)
    };
    (pyre_object::is_set_or_frozenset(a) && !is_exact_setlike(a))
        || (pyre_object::is_set_or_frozenset(b) && !is_exact_setlike(b))
}

/// [`needs_set_binop_dispatch`] behind the promoted exact-builtin test of
/// both operands; see [`needs_numeric_binop_dispatch_unless_exact`].
#[inline]
unsafe fn needs_set_binop_dispatch_unless_exact(a: PyObjectRef, b: PyObjectRef) -> bool {
    !both_exact_builtin_instances_promoted(a, b) && needs_set_binop_dispatch(a, b)
}

/// [`pyre_object::is_exact_builtin_instance`] with the class word promoted:
/// `jit.promote(w_type)` as `W_TypeObject.lookup` spells it, so the trace
/// pins `w_class` with a `guard_value` and the test against the payload's
/// canonical class folds.  The test itself is the shared
/// [`pyre_object::class_word_is_exact_builtin`] tail, reached only when the
/// typeptr alone ([`pyre_object::typeptr_is_exact_builtin`]) did not decide.
#[inline]
unsafe fn is_exact_builtin_instance_promoted(a: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(a) {
        return true;
    }
    if a.is_null() {
        return false;
    }
    if pyre_object::typeptr_is_exact_builtin(a) {
        return true;
    }
    let w_class = majit_metainterp::jit::promote((*a).w_class);
    pyre_object::class_word_is_exact_builtin(a, w_class)
}

/// Both operands of a binary operator are exact builtin instances, each
/// decided by [`is_exact_builtin_instance_promoted`].
#[inline]
pub(crate) unsafe fn both_exact_builtin_instances_promoted(a: PyObjectRef, b: PyObjectRef) -> bool {
    is_exact_builtin_instance_promoted(a) && is_exact_builtin_instance_promoted(b)
}

/// Unary analog: true when numeric operand `a` overrides the unary
/// special named by `op` relative to its builtin base.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn needs_numeric_unaryop_dispatch(a: PyObjectRef, op: UnaryDunder) -> bool {
    let Some(t) = numeric_base_type_of_overriding_subclass(a) else {
        return false;
    };
    dunder_overridden(a, op.name(), t.as_ptr())
}

/// Call the overriding unary special on a numeric subclass operand before
/// the Rust fast path.  Returns `None` when `a` is an exact builtin
/// numeric or does not override `op`, so the caller falls through.
///
/// The exact-builtin answer is decided on the traced graph, ahead of the
/// `dont_look_inside` probe: `space.lookup(w_obj, name)` promotes the
/// receiver's type, and the promoted class is what makes the lookup fold at
/// trace time.  pyre's Python-level class is `w_class`, so that is the word
/// promoted; the payload class comes from `ob_type`, which the codewriter
/// already turns into `guard_class`.  With both pinned, the comparison is a
/// constant and an exact builtin operand never records the probe at all.
unsafe fn try_numeric_unaryop_override(
    a: PyObjectRef,
    op: UnaryDunder,
) -> Result<Option<PyObjectRef>, PyError> {
    if is_exact_builtin_instance_promoted(a) {
        return Ok(None);
    }
    if !needs_numeric_unaryop_dispatch(a, op) {
        return Ok(None);
    }
    let Some(t) = crate::typedef::r#type(a) else {
        return Ok(None);
    };
    let Some(method) = lookup_in_type_where(t.as_ptr(), op.name()) else {
        return Ok(None);
    };
    Ok(Some(crate::call::call_function_impl_result(method, &[a])?))
}

/// Invoke only the right operand's reflected special method.
///
/// PyPy `_call_binop_impl` normally reaches this after the left implementation
/// returns `NotImplemented`.  CPython 3.14 `PyUnicode_Concat`, however, raises
/// the observable sequence-concatenation error when called as `str.__add__`.
/// The `+` operator must therefore skip that descriptor while retaining
/// PyPy's reflected-method call and its GC-safe operand lifetime.
#[inline(never)]
#[majit_macros::dont_look_inside]
unsafe fn try_reflected_binary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let operands = roots.publish(&[lhs, rhs]);
    let result = if let Some(method) = lookup_type_special(roots.get(operands + 1), rdunder) {
        let method_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(method);
        try_call_special(
            roots.get(method_slot),
            &[roots.get(operands + 1), roots.get(operands)],
        )?
    } else {
        None
    };
    Ok(result)
}

pub(crate) unsafe fn str_concat_type_error(rhs: PyObjectRef) -> PyError {
    let name = unsafe { concat_operand_name(rhs) };
    PyError::type_error(crate::display::wtf8_format!(
        "can only concatenate str (not \"",
        name,
        "\") to str"
    ))
}

/// The operand name every `sq_concat` refusal below spells.
unsafe fn concat_operand_name(obj: PyObjectRef) -> rustpython_wtf8::Wtf8Buf {
    crate::typedef::r#type(obj).map_or_else(
        || rustpython_wtf8::Wtf8Buf::from("object"),
        |w_type| crate::baseobjspace::type_fully_qualified_name(w_type.as_ptr()),
    )
}

/// `list_concat`'s refusal.  It names the base rather than the receiver's own
/// type, so a `list` subclass still reports `list` on the left.
unsafe fn list_concat_type_error(rhs: PyObjectRef) -> PyError {
    let name = unsafe { concat_operand_name(rhs) };
    PyError::type_error(crate::display::wtf8_format!(
        "can only concatenate list (not \"",
        name,
        "\") to list"
    ))
}

/// `tuple_concat`'s refusal, the same shape as `list_concat`'s.
unsafe fn tuple_concat_type_error(rhs: PyObjectRef) -> PyError {
    let name = unsafe { concat_operand_name(rhs) };
    PyError::type_error(crate::display::wtf8_format!(
        "can only concatenate tuple (not \"",
        name,
        "\") to tuple"
    ))
}

/// `bytes_concat`'s refusal, which it raises when either operand fails
/// `PyObject_GetBuffer`.  Unlike the two above it names both operands by
/// their own `tp_name`, so a `bytes` subclass reports itself on the right of
/// `to`.
unsafe fn bytes_concat_type_error(lhs: PyObjectRef, rhs: PyObjectRef) -> PyError {
    let lhs_name = unsafe { concat_operand_name(lhs) };
    let rhs_name = unsafe { concat_operand_name(rhs) };
    PyError::type_error(format!("can't concat {rhs_name} to {lhs_name}"))
}

/// `W_Root.user_overridden_class` (`baseobjspace.py`).  An exact builtin
/// payload keeps the flag false; a user subclass of that payload is true.
#[majit_macros::always_inline]
unsafe fn user_overridden_class(obj: PyObjectRef) -> bool {
    !is_exact_builtin_instance(obj)
}

/// RPython `type(w_obj)` for the `type(w1) is type(w2)` test in
/// `descroperation.py _make_binop_impl`: the interp-level class, not the
/// app-level `w_class`.  Tagged immediates are `W_IntObject`.
#[majit_macros::always_inline]
unsafe fn rpy_type_of(obj: PyObjectRef) -> *const PyType {
    if tagged_int::CAN_BE_TAGGED && tagged_int::is_tagged_int(obj) {
        return &INT_TYPE;
    }
    if obj.is_null() {
        return std::ptr::null();
    }
    (*obj).ob_type
}

#[majit_macros::always_inline]
unsafe fn same_rpy_type(a: PyObjectRef, b: PyObjectRef) -> bool {
    std::ptr::eq(rpy_type_of(a), rpy_type_of(b))
}

/// `_make_binop_impl` / `_make_comparison_impl` first-arm gate:
/// `type(w1) is type(w2) and not w1.user_overridden_class`.
///
/// A user subclass of a builtin without a `_getusercls` class keeps the
/// builtin `ob_type` and only retags `w_class` (`tag_subclass_instance`), so
/// `rpy_type_of` cannot tell the two apart; checking both operands restores
/// the observable, and the shortcut fires only for a pair of exact builtins.
/// An `int` subclass instance already differs by typeptr (`INT_USER_TYPE`).
#[majit_macros::always_inline]
unsafe fn same_unoverridden_rpy_type(a: PyObjectRef, b: PyObjectRef) -> bool {
    same_rpy_type(a, b) && !user_overridden_class(a) && !user_overridden_class(b)
}

/// `typedef.py use_special_method_shortcut('__add__')` as installed on
/// `W_IntObject` / `W_FloatObject`: the type-specific descr, not the
/// generic lookup.  `@try_inline` (`objectmodel.py`) so the backend
/// inliner can fold this into `add` / `add_impl`'s first arm.
#[majit_macros::always_inline_try]
unsafe fn shortcut_add(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_add(a, b)?));
    }
    if is_float(a) {
        return Ok(Some(float_add(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_sub(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_sub(a, b)?));
    }
    if is_float(a) {
        return Ok(Some(float_sub(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_mul(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_mul(a, b)?));
    }
    if is_float(a) {
        return Ok(Some(float_mul(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_floordiv(
    a: PyObjectRef,
    b: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_floordiv(a, b)?));
    }
    if is_float(a) {
        return Ok(Some(float_floordiv(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_mod(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_mod(a, b)?));
    }
    if is_float(a) {
        return Ok(Some(float_mod(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_lshift(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_lshift(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_rshift(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_int(a) {
        return Ok(Some(int_rshift(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_and(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    // `is_int` accepts BOOL_TYPE; W_BoolObject.descr_and returns the
    // bool singleton, W_IntObject.descr_and always boxes an int.
    if is_bool(a) {
        return Ok(Some(bool_descr_and(a, b)));
    }
    if is_int(a) {
        return Ok(Some(int_bitand(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_or(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_bool(a) {
        return Ok(Some(bool_descr_or(a, b)));
    }
    if is_int(a) {
        return Ok(Some(int_bitor(a, b)?));
    }
    Ok(None)
}

#[majit_macros::always_inline_try]
unsafe fn shortcut_xor(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    if is_bool(a) {
        return Ok(Some(bool_descr_xor(a, b)));
    }
    if is_int(a) {
        return Ok(Some(int_bitxor(a, b)?));
    }
    Ok(None)
}

pub fn add(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    // `_make_binop_impl`: `type(w1) is type(w2) and not user_overridden_class`
    // then the `use_special_method_shortcut` body.
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_add(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    add_impl(a, b, "+")
}

/// `+` and `+=` — one body, see [`binop_type_error`].
pub(crate) fn add_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Add);
        // A sequence left operand has no `nb_add` to offer, so the numeric
        // override runs the right operand's alone; the concat branches below
        // are `PyNumber_Add`'s `sq_concat` fall-through and already reach
        // `__radd__` before refusing.
        if numeric_override
            && !sequence_numeric_slot_is_null(a, BinopDunder::Add)
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_add(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_add(a, b);
            }
            if is_float_pair(a, b) {
                return float_add(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_add(a, b);
            }
        }
        if is_str(a) && is_str(b) {
            // descroperation.py:664 "unicode + string subclass" — a str
            // subclass overriding `__add__`/`__radd__` must reach the
            // reflected dispatch; otherwise concat directly.
            if needs_seq_binop_dispatch_unless_exact(a, b, SeqBase::Str, BinopDunder::Add)
                && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
            {
                return Ok(result);
            }
            return str_concat(a, b);
        }
        if is_str(a) {
            let str_type = crate::typedef::gettypefor(&pyre_object::STR_TYPE);
            let uses_builtin_add = str_type
                .is_some_and(|str_type| !dunder_overridden(a, "__add__", str_type.as_ptr()));
            if uses_builtin_add {
                if let Some(result) = pyre_object::with_roots!(a, b => try_reflected_binary_special(a, b, "__radd__"))?
                {
                    return Ok(result);
                }
                // [3.14-spec] PyPy `W_UnicodeObject.descr_add` returns
                // NotImplemented here and `_make_binop_impl` would emit its
                // generic operator error.  CPython `PyUnicode_Concat` instead
                // exposes the fully-qualified rhs type in this concat error.
                return Err(str_concat_type_error(b));
            }
        }
        if is_list(a) && is_list(b) {
            if needs_seq_binop_dispatch_unless_exact(a, b, SeqBase::List, BinopDunder::Add)
                && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
            {
                return Ok(result);
            }
            return list_concat(a, b);
        }
        if is_list(a) {
            // [3.14-spec] `PyNumber_Add` reaches the left operand's
            // `sq_concat` once `nb_add` has declined both ways, and
            // `list_concat` refuses a non-list by name there.  PyPy
            // `W_ListObject.descr_add` returns NotImplemented instead and
            // `_make_binop_impl` emits its generic operator error.
            //
            // A subclass that writes its own `__add__` keeps the generic
            // wording either way: its `slot_nb_add` stands in for the base's
            // `sq_concat`, so nothing reaches this refusal.
            let list_type = crate::typedef::gettypefor(&pyre_object::LIST_TYPE);
            let uses_builtin_add = list_type
                .is_some_and(|list_type| !dunder_overridden(a, "__add__", list_type.as_ptr()));
            if uses_builtin_add {
                if let Some(result) = pyre_object::with_roots!(a, b => try_reflected_binary_special(a, b, "__radd__"))?
                {
                    return Ok(result);
                }
                return Err(list_concat_type_error(b));
            }
        }
        if is_tuple(a) && is_tuple(b) {
            if needs_seq_binop_dispatch_unless_exact(a, b, SeqBase::Tuple, BinopDunder::Add)
                && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
            {
                return Ok(result);
            }
            return tuple_concat(a, b);
        }
        if is_tuple(a) {
            // `tuple_concat`, the same shape as the list branch above.
            let tuple_type = crate::typedef::gettypefor(&pyre_object::TUPLE_TYPE);
            let uses_builtin_add = tuple_type
                .is_some_and(|tuple_type| !dunder_overridden(a, "__add__", tuple_type.as_ptr()));
            if uses_builtin_add {
                if let Some(result) = pyre_object::with_roots!(a, b => try_reflected_binary_special(a, b, "__radd__"))?
                {
                    return Ok(result);
                }
                return Err(tuple_concat_type_error(b));
            }
        }
        // `bytes`/`bytearray` `__add__` accepts any buffer on the right (a
        // memoryview included), and the result type follows the left operand:
        // `bytes + <buffer>` is bytes, `bytearray + <buffer>` is bytearray.
        if pyre_object::bytesobject::is_bytes_like(a) {
            if let Some(mut b_src) = crate::typedef::buffer_as_bytes_like(b)? {
                // Only a real bytes-like rhs can carry a subclass `__radd__`;
                // a memoryview cannot, so dispatch only when both are bytes-like.
                if pyre_object::bytesobject::is_bytes_like(b)
                    && needs_bytes_binop_dispatch_unless_exact(a, b, BinopDunder::Add)
                    && let Some(result) = pyre_object::with_roots!(b_src, a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
                {
                    return Ok(result);
                }
                return bytes_concat(a, b_src);
            }
            if let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
            {
                return Ok(result);
            }
            // [3.14-spec] `bytes_concat` reaches its own refusal when
            // `PyObject_GetBuffer` fails on the right operand.  PyPy
            // `W_BytesObject.descr_add` returns NotImplemented and
            // `_make_binop_impl` emits the generic operator error instead.
            return Err(bytes_concat_type_error(a, b));
        }
        // Forward `__add__` + reflected `__radd__` per
        // `descroperation.py:_make_binop_impl` — try_dispatch_binary_special
        // already implements the reflected-first reordering rule for
        // subclass operands.
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__add__", "__radd__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

pub fn matmul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    matmul_impl(a, b, "@")
}

/// `@` and `@=` — one body, see [`binop_type_error`].
pub(crate) fn matmul_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        if let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__matmul__", "__rmatmul__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

pub fn sub(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_sub(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    sub_impl(a, b, "-")
}

/// `-` and `-=` — one body, see [`binop_type_error`].
pub(crate) fn sub_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch_unless_exact(a, b);
        if set_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__sub__", "__rsub__"))?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Sub);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__sub__", "__rsub__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_sub(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_sub(a, b);
            }
            if is_float_pair(a, b) {
                return float_sub(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_sub(a, b);
            }
        }
        // set / frozenset difference — PyPy: setobject.py W_BaseSetObject.descr_sub.
        // descr_sub returns NotImplemented for a non-set rhs, so `-` requires
        // both operands to be sets (the `difference` method takes iterables).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_difference(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__sub__", "__rsub__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

pub fn mul(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_mul(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    mul_impl(a, b, "*")
}

/// `*` and `*=` — one body, see [`binop_type_error`].
pub(crate) fn mul_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Mul);
        // As in [`binop_add_impl`]: a sequence operand carries no
        // `nb_multiply`, so only the other operand's runs here.
        if numeric_override
            && !sequence_numeric_slot_is_null(a, BinopDunder::Mul)
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__mul__", "__rmul__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_mul(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_mul(a, b);
            }
            if is_float_pair(a, b) {
                return float_mul(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_mul(a, b);
            }
        }
        // The `sq_repeat` fast paths below are valid for exact builtin
        // sequences.  A sequence subclass that overrides `__mul__`/`__rmul__`
        // must reach its override first — `LM([1]) * 2` is `LM.__mul__`, not a
        // list repetition — the same gate the concat branches of `add` apply.
        if (seq_repeat_override(a, RepeatDunder::MulPair)
            || seq_repeat_override(b, RepeatDunder::MulPair))
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__mul__", "__rmul__"))?
        {
            return Ok(result);
        }
        // `PyNumber_Multiply` reaches `sq_repeat` only once `binary_op1` has
        // declined, and a builtin sequence contributes nothing to that call,
        // so a count that is a numeric subclass writing `__mul__`/`__rmul__`
        // is the only operand offered and answers first.  The repeat runs at
        // the tail of this function when it declines.
        let count_answers_first = numeric_override
            && (sequence_numeric_slot_is_null(a, BinopDunder::Mul)
                || sequence_numeric_slot_is_null(b, BinopDunder::Mul));
        if !count_answers_first {
            if is_str(a) && is_int_or_long(b) {
                return str_repeat(a, b);
            }
            if is_int_or_long(a) && is_str(b) {
                return str_repeat(b, a);
            }
            if is_list(a) && is_int_or_long(b) {
                return list_repeat(a, b);
            }
            if is_int_or_long(a) && is_list(b) {
                return list_repeat(b, a);
            }
            // tupleobject.py descr_mul
            if is_tuple(a) && is_int_or_long(b) {
                return tuple_repeat(a, b);
            }
            if is_int_or_long(a) && is_tuple(b) {
                return tuple_repeat(b, a);
            }
            // bytesobject.py descr_mul / bytearrayobject.py descr_mul
            if pyre_object::bytesobject::is_bytes_like(a) && is_int_or_long(b) {
                return bytes_repeat(a, b);
            }
            if is_int_or_long(a) && pyre_object::bytesobject::is_bytes_like(b) {
                return mul(b, a);
            }
        }
        // `PyNumber_Multiply`: none of the builtin sequences implements
        // `nb_multiply`, so their `__mul__` / `__rmul__` slot wrappers take no
        // part in the operator dispatch — only the other operand can supply a
        // numeric implementation.
        let a_seq = is_repeat_sequence(a);
        let b_seq = is_repeat_sequence(b);
        // Every arm below can run Python — the two reflected invocations
        // directly, `try_dispatch_binary_special` through its own — and by
        // this point either operand may be a list, whose header moves.  Pin
        // the pair so the error path can name the types, and the repeat path
        // can address the sequence, at their post-collection addresses.
        let _roots = pyre_object::gc_roots::push_roots();
        let operands = pyre_object::gc_roots::pin_roots(&[a, b]);
        let dispatched = match (a_seq, b_seq) {
            (true, true) => None,
            (true, false) => match lookup_type_special(b, "__rmul__") {
                Some(method) => try_call_special(method, &[b, a])?,
                None => None,
            },
            (false, true) => match lookup_type_special(a, "__mul__") {
                Some(method) => try_call_special(method, &[a, b])?,
                None => None,
            },
            (false, false) if !numeric_override => {
                pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__mul__", "__rmul__"))?
            }
            (false, false) => None,
        };
        if let Some(result) = dispatched {
            return Ok(result);
        }
        let a = pyre_object::gc_roots::shadow_stack_get(operands);
        let b = pyre_object::gc_roots::shadow_stack_get(operands + 1);
        // `sequence_repeat`: the count goes through `__index__`, and an
        // operand that has none is reported by its own type — the sequence is
        // never the one named.
        if a_seq || b_seq {
            let (seq_slot, other) = if a_seq {
                (operands, b)
            } else {
                (operands + 1, a)
            };
            // Named before the MRO walk below, so the report never reads a
            // word the lookup could have left behind.
            let other_name = crate::baseobjspace::object_functionstr_type_name(other);
            if !(a_seq && b_seq) && crate::baseobjspace::lookup(other, "__index__").is_some() {
                // `__index__` is Python, so the sequence is addressed again
                // only after the count is in hand.
                let count = crate::baseobjspace::getindex_repeat(other)?;
                return sequence_repeat(pyre_object::gc_roots::shadow_stack_get(seq_slot), count);
            }
            return Err(PyError::type_error(format!(
                "can't multiply sequence by non-int of type '{other_name}'"
            )));
        }
        Err(binop_type_error(symbol, a, b))
    }
}

pub fn floordiv(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_floordiv(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    floordiv_impl(a, b, "//")
}

/// `//` and `//=` — one body, see [`binop_type_error`].
pub(crate) fn floordiv_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override =
            needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::FloorDiv);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_floordiv(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                // `_make_descr_binop` hands a `W_IntObject` operand to
                // `_int_floordiv` as a machine word; only a long operand
                // reaches `_floordiv`.
                if is_int_like(b) {
                    return long_int_floordiv(a, int_value(b));
                }
                return long_floordiv(a, b);
            }
            if is_float_pair(a, b) {
                return float_floordiv(a, b);
            }
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Call `dunder` on `obj` with `other` when `obj`'s type overrides it relative
/// to `obj`'s builtin layout base, returning `None` when there is no override
/// or the call answered `NotImplemented`.
///
/// The lookup happens immediately before the call, so no method reference is
/// held across an operation that can collect.  An *inherited* slot is not an
/// override ([`crate::baseobjspace::subclass_special_override`] rejects it):
/// the builtin `str`/`bytes` `__mod__` re-enters [`mod_`], so dispatching it
/// would recurse.
unsafe fn try_subclass_binop_override(
    obj: PyObjectRef,
    other: PyObjectRef,
    dunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    let Some((method, w_type)) = crate::baseobjspace::subclass_special_override(obj, dunder) else {
        return Ok(None);
    };
    let result = crate::baseobjspace::get_and_call_function(method, obj, w_type, &[other])?;
    Ok((!is_not_implemented(result)).then_some(result))
}

pub fn mod_(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_mod(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    mod_impl(a, b, "%")
}

/// `%` and `%=` — one body, see [`binop_type_error`].
pub(crate) fn mod_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Mod);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__mod__", "__rmod__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_mod(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                // `_make_descr_binop(_mod, _int_mod)`.
                if is_int_like(b) {
                    return long_int_mod(a, int_value(b));
                }
                return long_mod(a, b);
            }
            if is_float_pair(a, b) {
                return float_mod(a, b);
            }
        }
        let is_str_lhs = is_str(a);
        let is_bytes_lhs = pyre_object::bytesobject::is_bytes_like(a);
        // str/bytes % args — a subclass override runs before the built-in
        // formatter: reflected first when `b` subclasses `type(a)`, then `a`'s
        // own forward override.  The `is_exact_builtin_instance` test is the
        // early-out `subclass_special_override` performs itself, kept here so
        // an exact-builtin right operand (the usual format argument) skips the
        // subtype probe.
        if is_str_lhs || is_bytes_lhs {
            // Both override probes run Python, and the formatter runs more of
            // it for every conversion.  The operands were popped off the value
            // stack before this dispatch, so publish the pair: the receiver is
            // immobile and only needs to stay reachable, while the operand may
            // be a dict and is read back from its slot.
            let _roots = pyre_object::gc_roots::push_roots();
            let operands = pyre_object::gc_roots::pin_roots(&[a, b]);
            if !pyre_object::is_exact_builtin_instance(b)
                && let (Some(at), Some(bt)) = (crate::typedef::r#type(a), crate::typedef::r#type(b))
                && at != bt
                && issubtype_cached(bt.as_ptr(), at.as_ptr())
                && let Some(result) = try_subclass_binop_override(b, a, "__rmod__")?
            {
                return Ok(result);
            }
            if let Some(result) = try_subclass_binop_override(
                pyre_object::gc_roots::shadow_stack_get(operands),
                pyre_object::gc_roots::shadow_stack_get(operands + 1),
                "__mod__",
            )? {
                return Ok(result);
            }
            let a = pyre_object::gc_roots::shadow_stack_get(operands);
            let b = pyre_object::gc_roots::shadow_stack_get(operands + 1);
            return if is_str_lhs {
                crate::objspace::std::formatting::str_format_percent(a, b)
            } else {
                crate::objspace::std::formatting::bytes_format_percent(a, b)
            };
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__mod__", "__rmod__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// True division (`/` operator) — always produces a float result.
///
/// intobject.py `_truediv` raises "division by zero" for int/int;
/// floatobject.py:519 `_floatdiv` raises "float division by zero" once
/// any operand is a float.
/// longobject.py `_truediv` catches OverflowError from
/// `rbigint.truediv` and reissues it as
/// "integer division result too large for a float".
pub fn truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    truediv_impl(a, b, "/")
}

/// `/` and `/=` — one body, see [`binop_type_error`].
pub(crate) fn truediv_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override =
            needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::TrueDiv);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__"))?
        {
            return Ok(result);
        }
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if !numeric_override && a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            // intobject.py `descr_truediv` / `_truediv` first, then
            // `_make_ovf2long`.  Long operands use longobject.py `_truediv`.
            if is_int_like(a) && is_int_like(b) {
                return int_truediv(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_truediv(a, b);
            }
        }
        if !numeric_override && is_complex_pair(a, b) {
            return complex_truediv(a, b);
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// `descroperation.py pow_binary` — return `None` when neither numeric
/// fast paths nor `__pow__` / `__rpow__` produce a result.
fn pow_binary(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Pow);
        if numeric_override {
            if let Some(result) = try_dispatch_binary_special(a, b, "__pow__", "__rpow__")? {
                return Ok(Some(result));
            }
            return Ok(None);
        }
        if is_int_like(a) && is_int_like(b) {
            return match int_pow(a, b) {
                Ok(result) => Ok(Some(result)),
                Err(err) => Err(err),
            };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return match long_pow(a, b) {
                Ok(result) => Ok(Some(result)),
                Err(err) => Err(err),
            };
        }
        if is_float_pair(a, b) {
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a = || pyre_object::gc_roots::shadow_stack_get(base);
            let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
            reject_pow_operand_overflow(a())?;
            reject_pow_operand_overflow(b())?;
            let fa = as_float(a());
            let fb = as_float(b());
            return match float_pow_impl(fa, fb) {
                Ok(result) => Ok(Some(result)),
                Err(err) => Err(err),
            };
        }
        if is_complex_pair(a, b) {
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a = || pyre_object::gc_roots::shadow_stack_get(base);
            let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
            reject_pow_operand_overflow(a())?;
            reject_pow_operand_overflow(b())?;
            return match complex_pow(a(), b()) {
                Ok(result) => Ok(Some(result)),
                Err(err) => Err(err),
            };
        }
        try_dispatch_binary_special(a, b, "__pow__", "__rpow__")
    }
}

/// Power operation dispatch (`**` operator).
pub fn pow(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    if let Some(result) = pyre_object::with_roots!(a, b => pow_binary(a, b))? {
        return Ok(result);
    }
    // `**` names the two-argument builtin alongside the operator, so it does
    // not take the shared spelling.
    Err(binop_type_error("** or pow()", a, b))
}

/// `descroperation.py inplace_pow` — unlike the generated in-place
/// binary operations, power has its own fallback error spelling.
pub fn inplace_pow(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    if let Some(result) =
        pyre_object::with_roots!(a, b => try_inplace_special(a, b, "__ipow__", None, false))?
    {
        return Ok(result);
    }
    if let Some(result) = pyre_object::with_roots!(a, b => pow_binary(a, b))? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("**=", a, b))
}

// ── Numeric type-slot builtins ────────────────────────────────────────
//
// The `int`/`float` numeric special methods (`int.__add__`,
// `float.__and__`, …) resolve to these concrete computations, not to the
// operator dispatch above.  The operator (`add`, `and_`, …) drives the
// forward+reflected protocol and, when an operand is a numeric subclass
// that overrides the special method, re-dispatches through that operand's
// type slot.  Wiring the slot back to the operator would re-enter it and
// recurse without bound; wiring the slot to the concrete computation
// terminates after computing the result or returning NotImplemented for
// the reflected method to handle.  The `is_int_or_long`/`is_float` macro
// guards in typedef.rs pre-filter the operand kinds, so the trailing
// NotImplemented is reached only defensively.

pub(crate) fn add_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_add(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn sub_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_sub(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mul_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_mul(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn truediv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            if is_int_like(a) && is_int_like(b) {
                return int_truediv(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_truediv(a, b);
            }
        }
        if is_complex_pair(a, b) {
            return complex_truediv(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn floordiv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // `_make_descr_binop` hands a `W_IntObject` operand to
            // `_int_floordiv` as a machine word; only a long operand
            // reaches `_floordiv`.
            if is_int_like(b) {
                return long_int_floordiv(a, int_value(b));
            }
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mod_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // `_make_descr_binop(_mod, _int_mod)`.
            if is_int_like(b) {
                return long_int_mod(a, int_value(b));
            }
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn pow_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) || is_complex_pair(a, b) {
            let is_float_operands = is_float_pair(a, b);
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a = || pyre_object::gc_roots::shadow_stack_get(base);
            let b = || pyre_object::gc_roots::shadow_stack_get(base + 1);
            reject_pow_operand_overflow(a())?;
            reject_pow_operand_overflow(b())?;
            if is_float_operands {
                let fa = as_float(a());
                let fb = as_float(b());
                return float_pow_impl(fa, fb);
            }
            return complex_pow(a(), b());
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn divmod_builtin(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if lhs_num && rhs_num {
            // Python 3.14 reports the builtin operation's uniform spelling for
            // every numeric zero divisor.  This intentionally differs from
            // PyPy 3.11's int/float-specific divmod and modulo messages.
            if !pyre_object::with_roots!(a, b => is_true(b))? {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            if is_float_pair(a, b) {
                let (x, y) = float_operands(a, b)?;
                let (q, r) = float_divmod_w(x, y)?;
                let mut w_q = w_float_new(q);
                let w_r = pyre_object::with_roots!(w_q => w_float_new(r));
                return Ok(pyre_object::wraptuple2(w_q, w_r));
            }
            return integer_divmod_pair(a, b);
        }
    }
    Ok(w_not_implemented())
}

pub(crate) fn lshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // `_make_descr_binop` hands a `W_IntObject` operand to
            // `_int_lshift` as a machine word; only a long operand
            // reaches `_lshift`.
            if is_int_like(b) {
                return long_int_lshift(a, int_value(b));
            }
            return long_lshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn rshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // `_make_descr_binop` hands a `W_IntObject` operand to
            // `_int_rshift` as a machine word; only a long operand
            // reaches `_rshift`.
            if is_int_like(b) {
                return long_int_rshift(a, int_value(b));
            }
            return long_rshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn and_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        // int.__and__ — bool operands are treated as ints; the bool-typed
        // result is produced by bool.__and__ (init_bool_type), not here.
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn or_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn xor_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        Ok(w_not_implemented())
    }
}

// ── descroperation helpers — pypy/objspace/descroperation.py ──────────
//
// These helpers implement the standard "forward + reverse with
// NotImplemented fallback" dispatch that PyPy generates from
// `_make_binop_impl` / `_make_descr_unaryop`. They were originally in
// `baseobjspace` (not in `builtins`) because they are space-level
// semantics shared between the builtin module, the weakproxy wrappers,
// and any future opcode dispatch — every caller needs the same rule
// or NotImplemented from the forward path silently swallows the
// reflected operand.

/// `space.lookup(w_obj, dunder)` — descroperation.py.
pub(crate) unsafe fn lookup_type_special(obj: PyObjectRef, dunder: &str) -> Option<PyObjectRef> {
    crate::typedef::r#type(obj).and_then(|tp| lookup_in_type(tp.as_ptr(), dunder))
}

/// Call a raw type-MRO special-method descriptor and treat NotImplemented as
/// "no result", per descroperation.py `_invoke_binop` and
/// `_check_notimplemented`.  `args[0]` is the receiver; custom descriptors
/// must bind through `space.get_and_call_function` rather than receiving that
/// object as an explicitly injected argument.
pub(crate) fn try_call_special(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, PyError> {
    debug_assert!(!args.is_empty());
    let receiver = args[0];
    let Some(w_type) = crate::typedef::r#type(receiver) else {
        return Ok(None);
    };
    let result = unsafe {
        crate::baseobjspace::get_and_call_function(method, receiver, w_type.as_ptr(), &args[1..])?
    };
    if unsafe { is_not_implemented(result) } {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// descroperation.py `_call_binop_impl` — resolve the forward
/// (`dunder`) and reflected (`rdunder`) special methods through
/// `lookup_where`, decide whether to try the reflected operand first by
/// comparing the two defining classes, then invoke forward-then-reverse.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub(crate) fn try_dispatch_binary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    // descroperation.py `seq_bug_compat = (symbol == '+' or symbol == '*')`.
    let seq_bug_compat = dunder == "__add__" || dunder == "__mul__";
    unsafe {
        let Some(w_typ1) = crate::typedef::r#type(lhs) else {
            return Ok(None);
        };
        let Some(w_typ2) = crate::typedef::r#type(rhs) else {
            return Ok(None);
        };
        // From here on Python can run: `p_abstract_issubclass_w` reaches
        // `__bases__`, and the two invocations below are Python calls
        // outright.  Both operands are pinned for the whole span and read
        // back through `operand`, so this frame never holds a pre-collection
        // address.  The caller roots its own copies across the call.
        let _roots = pyre_object::gc_roots::push_roots();
        let operands = pyre_object::gc_roots::pin_roots(&[lhs, rhs]);
        let w_left = lookup_where_with_method_cache(w_typ1.as_ptr(), dunder);
        let mut w_left_impl = match w_left {
            Some((_, imp)) => Some(imp),
            None => None,
        };
        // Which slot the forward call's first operand is read from.  Upstream
        // swaps the two values; swapping the *index* keeps the pinned copies
        // authoritative across a collection.
        let mut swapped = false;
        let mut w_right_impl: Option<PyObjectRef> = None;
        // descroperation.py:652 — same type means the reflected method is
        // never considered.
        if w_typ1 != w_typ2 {
            let w_right = lookup_where_with_method_cache(w_typ2.as_ptr(), rdunder);
            w_right_impl = match w_right {
                Some((_, imp)) => Some(imp),
                None => None,
            };
            // descroperation.py:662 — both `__op__` and `__rop__` are
            // found, in different MRO classes.
            if let Some((rsrc, _)) = w_right
                && let Some((lsrc, _)) = w_left
                && !std::ptr::eq(lsrc, rsrc)
            {
                // descroperation.py:667-670.
                let prefer_reverse = (seq_bug_compat
                    && crate::baseobjspace::flag_sequence_bug_compat(w_typ1.as_ptr())
                    && !crate::baseobjspace::flag_sequence_bug_compat(w_typ2.as_ptr()))
                    || issubtype_w(w_typ2.as_ptr(), w_typ1.as_ptr());
                // descroperation.py:671-672.
                if prefer_reverse {
                    let has_left_impl = w_left_impl.is_some();
                    let has_right_impl = w_right_impl.is_some();
                    let typ1 = w_typ1.as_ptr();
                    let impl_base = pyre_object::gc_roots::pin_roots(&[
                        lsrc,
                        rsrc,
                        typ1,
                        w_left_impl.unwrap_or(pyre_object::PY_NULL),
                        w_right_impl.unwrap_or(pyre_object::PY_NULL),
                    ]);
                    let derived_is_subclass = p_abstract_issubclass_w(
                        pyre_object::gc_roots::shadow_stack_get(impl_base),
                        pyre_object::gc_roots::shadow_stack_get(impl_base + 1),
                    )?;
                    let keep_order = derived_is_subclass
                        || p_abstract_issubclass_w(
                            pyre_object::gc_roots::shadow_stack_get(impl_base + 2),
                            pyre_object::gc_roots::shadow_stack_get(impl_base + 1),
                        )?;
                    w_left_impl = has_left_impl
                        .then(|| pyre_object::gc_roots::shadow_stack_get(impl_base + 3));
                    w_right_impl = has_right_impl
                        .then(|| pyre_object::gc_roots::shadow_stack_get(impl_base + 4));
                    if !keep_order {
                        swapped = true;
                        std::mem::swap(&mut w_left_impl, &mut w_right_impl);
                    }
                }
            }
        }
        let (first, second) = if swapped { (1, 0) } else { (0, 1) };
        // The reflected implementation has to survive the forward call too.
        let right_slot = match w_right_impl {
            Some(method) => {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(method);
                Some(slot)
            }
            None => None,
        };
        let mut result = None;
        // descroperation.py — _invoke_binop(w_left_impl, w_obj1, w_obj2).
        if let Some(method) = w_left_impl {
            result = try_call_special(
                method,
                &[
                    pyre_object::gc_roots::shadow_stack_get(operands + first),
                    pyre_object::gc_roots::shadow_stack_get(operands + second),
                ],
            )?;
        }
        // descroperation.py — _invoke_binop(w_right_impl, w_obj2, w_obj1).
        if result.is_none()
            && let Some(slot) = right_slot
        {
            let method = pyre_object::gc_roots::shadow_stack_get(slot);
            result = try_call_special(
                method,
                &[
                    pyre_object::gc_roots::shadow_stack_get(operands + second),
                    pyre_object::gc_roots::shadow_stack_get(operands + first),
                ],
            )?;
        }
        Ok(result)
    }
}

/// CPython 3.14 `typeobject.c:slot_nb_power` — the three-argument copy of
/// `SLOT1BINFULL`.  Unlike PyPy 3.11's `space.pow`, Python 3.14 also offers
/// the exponent's reflected method the modulus argument.  A proper subtype's
/// distinct `__rpow__` is tried first, then the base's `__pow__`, then the
/// reflected method when it has not already been tried.
fn try_dispatch_ternary_pow_special(
    base: &mut PyObjectRef,
    exp: &mut PyObjectRef,
    modulus: &mut PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        let Some(w_base_type) = crate::typedef::r#type(*base) else {
            return Ok(None);
        };
        let Some(w_exp_type) = crate::typedef::r#type(*exp) else {
            return Ok(None);
        };
        // Up to three Python invocations follow, and a `list` subclass
        // defining `__pow__` puts a movable header in an operand slot.  Pin
        // all three, read them back through `operand`, and hand them out — the
        // caller spells the error message from them.
        let _roots = pyre_object::gc_roots::push_roots();
        let operands = pyre_object::gc_roots::pin_roots(&[*base, *exp, *modulus]);
        let operand = |i: usize| pyre_object::gc_roots::shadow_stack_get(operands + i);
        let (w_base_src, w_base_impl) =
            match lookup_where_with_method_cache(w_base_type.as_ptr(), "__pow__") {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            };
        let (w_exp_src, w_exp_impl) = if w_base_type != w_exp_type {
            match lookup_where_with_method_cache(w_exp_type.as_ptr(), "__rpow__") {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        // Each implementation also has to survive the invocations that precede
        // it.  A class attribute is an arbitrary object, so a `dict` subclass
        // defining `__call__` puts a movable header in this slot as readily as
        // in an operand's.
        let pin = |method: PyObjectRef| {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(method);
            slot
        };
        let base_impl_slot = w_base_impl.map(pin);
        let mut exp_impl_slot = w_exp_impl.map(pin);

        // slot_nb_power: a proper exponent subtype whose reflected method is
        // distinct from the base type's definition gets the first chance.
        if issubtype_w(w_exp_type.as_ptr(), w_base_type.as_ptr())
            && let (Some(exp_src), Some(base_src), Some(slot)) =
                (w_exp_src, w_base_src, exp_impl_slot)
            && !std::ptr::eq(base_src, exp_src)
        {
            let base_type = w_base_type.as_ptr();
            let src_base = pyre_object::gc_roots::pin_roots(&[base_src, exp_src, base_type]);
            let exp_now = || pyre_object::gc_roots::shadow_stack_get(src_base + 1);
            if !p_abstract_issubclass_w(
                pyre_object::gc_roots::shadow_stack_get(src_base),
                exp_now(),
            )? && !p_abstract_issubclass_w(
                pyre_object::gc_roots::shadow_stack_get(src_base + 2),
                exp_now(),
            )? {
                let exp_impl = pyre_object::gc_roots::shadow_stack_get(slot);
                if let Some(result) =
                    try_call_special(exp_impl, &[operand(1), operand(0), operand(2)])?
                {
                    *base = operand(0);
                    *exp = operand(1);
                    *modulus = operand(2);
                    return Ok(Some(result));
                }
                // The subtype-first call clears the reflected method, including
                // when it returns NotImplemented.
                exp_impl_slot = None;
            }
        }

        let mut result = None;
        if let Some(slot) = base_impl_slot {
            let method = pyre_object::gc_roots::shadow_stack_get(slot);
            result = try_call_special(method, &[operand(0), operand(1), operand(2)])?;
        }
        if result.is_none()
            && let Some(slot) = exp_impl_slot
        {
            let method = pyre_object::gc_roots::shadow_stack_get(slot);
            result = try_call_special(method, &[operand(1), operand(0), operand(2)])?;
        }
        *base = operand(0);
        *exp = operand(1);
        *modulus = operand(2);
        Ok(result)
    }
}

/// descroperation.py `inplace_impl` — try the in-place special
/// (`__iadd__` etc.) on the lhs.  Returns `None` when the type has no
/// such method or the call yields `NotImplemented`, so the caller falls
/// back to the corresponding binary operation.
pub(crate) fn try_inplace_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    idunder: &str,
    rdunder: Option<&str>,
    seq_bug_compat: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    // An exact builtin numeric lhs (int/long/float/complex/bool) defines no
    // in-place special, so the lookup below is always a miss.  Skip it — the
    // string-keyed method-cache probe (a UTF-8 dunder-name intern) is the
    // dominant per-iteration cost of `total += …`.  Only exact numerics are
    // gated: builtin sequences (list/bytearray) define `__iadd__`/`__imul__`,
    // and a subclass may override the in-place slot, so both must fall
    // through to the lookup.
    if unsafe { is_exact_numeric_builtin(lhs) } {
        return Ok(None);
    }
    // descroperation.py:826 — only when the lhs in-place method exists.
    if let Some(method) = unsafe { lookup_type_special(lhs, idunder) } {
        // descroperation.py:831 seq_bug_compat — for `+=` / `*=` where the
        // lhs is a builtin sequence and the rhs is not, try the rhs
        // reflected method before the lhs in-place method.
        let rmethod = if seq_bug_compat
            && let Some(rd) = rdunder
            && let (Some(lhs_type), Some(rhs_type)) =
                (crate::typedef::r#type(lhs), crate::typedef::r#type(rhs))
            && crate::baseobjspace::flag_sequence_bug_compat(lhs_type.as_ptr())
            && !crate::baseobjspace::flag_sequence_bug_compat(rhs_type.as_ptr())
        {
            unsafe { lookup_type_special(rhs, rd) }
        } else {
            None
        };
        // That reflected call runs Python, and the branch guarding it is
        // reached precisely when the lhs is a builtin sequence — a list, whose
        // header moves.  Pin both operands and the in-place method across it
        // and read them back, or the in-place call below is handed a
        // pre-collection address.
        let (lhs, rhs, method) = match rmethod {
            Some(rmethod) => {
                let _roots = pyre_object::gc_roots::push_roots();
                let base = pyre_object::gc_roots::pin_roots(&[lhs, rhs, method]);
                if let Some(result) = try_call_special(rmethod, &[rhs, lhs])? {
                    return Ok(Some(result));
                }
                (
                    pyre_object::gc_roots::shadow_stack_get(base),
                    pyre_object::gc_roots::shadow_stack_get(base + 1),
                    pyre_object::gc_roots::shadow_stack_get(base + 2),
                )
            }
            None => (lhs, rhs, method),
        };
        if let Some(result) = try_call_special(method, &[lhs, rhs])? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// `intobject.py _pow_nomod` — look inside when the exponent is constant.
fn _pow_nomod_iff(_iv: i64, iw: i64) -> bool {
    majit_rlib::jit::isconstant(&iw)
}

#[majit_macros::look_inside_iff(_pow_nomod_iff)]
pub(crate) fn _pow_nomod(iv: i64, mut iw: i64) -> Result<i64, PyError> {
    if iw <= 0 {
        if iw == 0 {
            return Ok(1);
        }
        // `intobject.py _pow_nomod`: a negative exponent always returns a float,
        // so the machine body raises and the caller takes the float path.
        return Err(PyError::value_error(""));
    }
    let mut temp = iv;
    let mut ix = 1_i64;
    loop {
        if iw & 1 != 0 {
            // `ovfcheck(ix * temp)` — `checked_mul` plus a `Some`/`None` match.
            let Some(value) = ix.checked_mul(temp) else {
                return Err(PyError::overflow_error(
                    "signed integer expression did overflow",
                ));
            };
            ix = value;
        }
        iw >>= 1;
        if iw == 0 {
            break;
        }
        let Some(value) = temp.checked_mul(temp) else {
            return Err(PyError::overflow_error(
                "signed integer expression did overflow",
            ));
        };
        temp = value;
    }
    Ok(ix)
}

/// `intobject.py _pow_mod` — look inside when the exponent and the modulus
/// are both constants, so the squaring loop unrolls into `mulmod` calls.
fn _pow_mod_iff(_iv: i64, iw: i64, iz: i64) -> bool {
    majit_rlib::jit::isconstant(&iw) && majit_rlib::jit::isconstant(&iz)
}

/// Floor remainder. `iz` may be negative (`1 % iz` when the exponent is 0).
fn floor_mod_i64(iv: i64, iz: i64) -> i64 {
    let mut remainder = iv % iz;
    if remainder != 0 && (remainder < 0) != (iz < 0) {
        remainder += iz;
    }
    remainder
}

#[majit_macros::look_inside_iff(_pow_mod_iff)]
pub(crate) fn _pow_mod(mut iv: i64, mut iw: i64, mut iz: i64) -> Result<i64, PyError> {
    if iw == 0 {
        return Ok(floor_mod_i64(1, iz));
    }
    let mut iz_negative = false;
    if iz < 0 {
        // `intobject.py _pow_mod`: `iz = ovfcheck(-iz)`. `checked_neg` plus a
        // `Some`/`None` match is that overflow edge; `i64::MIN` raises.
        let Some(negated) = iz.checked_neg() else {
            return Err(PyError::overflow_error(
                "signed integer expression did overflow",
            ));
        };
        iz = negated;
        iz_negative = true;
    }
    if iw <= 0 {
        // A negative exponent takes `invmod` on the long path. Signal that
        // with ValueError so the caller leaves this machine body.
        return Err(PyError::value_error(""));
    }
    let mut temp = iv;
    let mut ix = 1_i64;
    loop {
        if iw & 1 != 0 {
            ix = majit_rlib::rarithmetic::mulmod(ix, temp, iz);
        }
        iw >>= 1;
        if iw == 0 {
            break;
        }
        temp = majit_rlib::rarithmetic::mulmod(temp, temp, iz);
    }
    if iz_negative && ix > 0 {
        ix -= iz;
    }
    Ok(ix)
}

/// `intobject.py _pow`. `iz == 0` is the two-argument form.
pub(crate) fn _pow(iv: i64, iw: i64, iz: i64) -> Result<i64, PyError> {
    if iz == 0 {
        _pow_nomod(iv, iw)
    } else {
        _pow_mod(iv, iw, iz)
    }
}

/// `(int|long) ** (int|long) % (int|long)` fast path used by `space.pow`
/// when a modulus is supplied — longobject.py `int_pow`.
pub(crate) fn try_int_long_pow_with_modulo(
    mut base: PyObjectRef,
    mut exp: PyObjectRef,
    mut modulus: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if !is_int_or_long(base) || !is_int_or_long(exp) || !is_int_or_long(modulus) {
            return Ok(None);
        }

        // `descr_pow` wraps its 3-arg result as `W_LongObject` whenever the
        // receiver is a long; the int path (`W_IntObject.descr_pow`) reaches
        // that same long path via `_pow_ovf2long` when the exponent or modulus
        // is a long. So the result stays a long unless all three operands are
        // machine ints, in which case `space.newint` demotes it.
        let all_int_like = is_int_like(base) && is_int_like(exp) && is_int_like(modulus);

        // `W_IntObject.descr_pow` / `_pow_mod` for three machine ints.
        // `is_int_like` is a subclass check and is also true of a long, so
        // a long stays on the bigint path below. A negative exponent does too.
        if all_int_like && !is_long(base) && !is_long(exp) && !is_long(modulus) {
            let iv = int_value(base);
            let iw = int_value(exp);
            let iz = int_value(modulus);
            if iz == 0 {
                return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
            }
            if iw >= 0 {
                // Overflow (`ovfcheck(-iz)`) and a negative exponent both leave
                // the machine body; the long path below is that continuation.
                // `_pow_mod` is pin-free so `look_inside_iff` can record the
                // squaring loop; `w_int_new` is the collecting seed.
                if let Ok(result) = _pow_mod(iv, iw, iz) {
                    return Ok(Some(
                        pyre_object::with_roots!(base, exp, modulus => w_int_new(result)),
                    ));
                }
            }
        }

        // `fromint` / payload clones collect. Pin the three Python
        // operands and reload after each conversion so a nursery-born
        // int/long is not read from from-space as a negative exponent.
        let _roots = pyre_object::gc_roots::push_roots();
        let slots = pyre_object::gc_roots::pin_roots(&[base, exp, modulus]);
        let load = |i: usize| pyre_object::gc_roots::shadow_stack_get(slots + i);
        let as_bigint = |obj: PyObjectRef| -> RBigIntGcRoot {
            if is_long(obj) {
                live_long_num(obj)
            } else {
                RBigIntGcRoot::new(BigInt::from(int_value(obj)))
            }
        };
        let base = as_bigint(load(0));
        let exp = as_bigint(load(1));
        let modulus = as_bigint(load(2));

        if modulus.get_sign() == 0 {
            return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
        }
        if exp.get_sign() < 0 {
            // 3-arg pow with a negative exponent: raise the modular
            // inverse of `base` to `-exp` (longobject.c long_pow →
            // long_invmod).  The inverse exists only when `base` is
            // coprime to the modulus.
            let negative_modulus = modulus.get_sign() < 0;
            let abs_modulus = if negative_modulus {
                RBigIntGcRoot::new(modulus.neg())
            } else {
                RBigIntGcRoot::new(modulus.translated_alias())
            };
            let inverse = RBigIntGcRoot::new(bigint_mod_inverse(&base, &abs_modulus)?);
            let pos_exp = RBigIntGcRoot::new(exp.neg());
            let mut result = RBigIntGcRoot::new(
                inverse
                    .pow(&pos_exp, Some(&abs_modulus))
                    .map_err(|_| PyError::memory_error("exponent too large"))?,
            );
            if negative_modulus && result.get_sign() > 0 {
                result = RBigIntGcRoot::new(result.sub(&abs_modulus));
            }
            return Ok(Some(pow_mod_result(
                result.translated_alias(),
                all_int_like,
            )));
        }
        if exp.get_sign() == 0 {
            // `x ** 0 % m` is `1 % m` under floor semantics, so a negative
            // modulus yields a negative residue (`pow(2, 0, -13) == -12`).
            return Ok(Some(pow_mod_result(
                BigInt::one()
                    .r#mod(&modulus)
                    .expect("modulus was checked nonzero"),
                all_int_like,
            )));
        }

        let negative_modulus = modulus.get_sign() < 0;
        let abs_modulus = if negative_modulus {
            RBigIntGcRoot::new(modulus.neg())
        } else {
            RBigIntGcRoot::new(modulus.translated_alias())
        };
        let mut result = RBigIntGcRoot::new(
            base.pow(&exp, Some(&abs_modulus))
                .map_err(|_| PyError::memory_error("exponent too large"))?,
        );
        if negative_modulus && result.get_sign() > 0 {
            result = RBigIntGcRoot::new(result.sub(&abs_modulus));
        }
        Ok(Some(pow_mod_result(
            result.translated_alias(),
            all_int_like,
        )))
    }
}

/// Box a 3-arg `pow` result: `space.newint` (demote) when every operand was a
/// machine int, else `W_LongObject` (a long receiver keeps the long
/// representation). The demote arm always fits — `result < |modulus|`.
fn pow_mod_result(value: BigInt, all_int_like: bool) -> PyObjectRef {
    if all_int_like {
        w_int_new(jit_bigint_to_i64_value(&value))
    } else {
        w_long_new(value)
    }
}

/// `%T` — the runtime type name of `obj` (`space.type(obj).name`), falling
/// back to the layout-level type for objects without a Python type.
fn operand_type_name(obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => (*ll_type(obj)).name.to_string(),
        }
    }
}

/// The `unsupported operand type(s) for <symbol>` TypeError raised once both
/// the forward and the reflected special have declined.
///
/// `_make_binop_impl` and `_make_inplace_impl` generate two families from one
/// body: the pair differs only in the `symbol` baked into `errormsg`, `&`
/// against `&=`.  So each operation here is one function taking the symbol,
/// and the augmented assignment reaches the same dispatch under its own
/// spelling instead of borrowing the operator's.
fn binop_type_error(symbol: &str, a: PyObjectRef, b: PyObjectRef) -> PyError {
    let a_name = crate::baseobjspace::object_functionstr_type_name(a);
    let b_name = crate::baseobjspace::object_functionstr_type_name(b);
    PyError::type_error(format!(
        "unsupported operand type(s) for {symbol}: '{a_name}' and '{b_name}'"
    ))
}

pub(crate) fn binary_builtin_type_error(
    opname: &str,
    lhs: PyObjectRef,
    rhs: PyObjectRef,
) -> PyError {
    let lhs_name = operand_type_name(lhs);
    let rhs_name = operand_type_name(rhs);
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{lhs_name}' and '{rhs_name}'"
    ))
}

/// The three-operand form of [`binary_builtin_type_error`] for `pow(a, b, c)`
/// — descroperation.py `unsupported operand type(s) for pow(): T, T, T`.
pub(crate) fn ternary_builtin_type_error(
    opname: &str,
    a: PyObjectRef,
    b: PyObjectRef,
    c: PyObjectRef,
) -> PyError {
    let a_name = operand_type_name(a);
    let b_name = operand_type_name(b);
    let c_name = operand_type_name(c);
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{a_name}', '{b_name}', '{c_name}'"
    ))
}

/// 3-arg `pow(a, b, c)` dispatch. PyPy 3.11's
/// `pypy/objspace/descroperation.py` tries only the base's `__pow__`;
/// Python 3.14 changed this to offer the exponent's `__rpow__` the modulus
/// too (`Objects/typeobject.c:slot_nb_power`).
pub fn pow3(mut base: PyObjectRef, mut exp: PyObjectRef, mut modulus: PyObjectRef) -> PyResult {
    if unsafe { is_none(modulus) } {
        return pow(base, exp);
    }
    // `W_IntObject.descr_pow` / `_pow_mod` for three exact machine ints.
    // `try_dispatch_ternary_pow_special` publishes the operands with
    // `pin_roots`, which is `dont_look_inside`. Taking that first would
    // residualize the modular-power loop. A subclass that overrides
    // `__pow__` is not an exact `int` and falls through to the lookup.
    // Unbox and run `_pow_mod` pin-free, then root only the boxing.
    if unsafe {
        is_exact_type(base, &INT_TYPE)
            && is_exact_type(exp, &INT_TYPE)
            && is_exact_type(modulus, &INT_TYPE)
            && !is_long(base)
            && !is_long(exp)
            && !is_long(modulus)
    } {
        let iv = unsafe { int_value(base) };
        let iw = unsafe { int_value(exp) };
        let iz = unsafe { int_value(modulus) };
        if iz == 0 {
            return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
        }
        if iw >= 0 {
            if let Ok(result) = _pow_mod(iv, iw, iz) {
                return Ok(pyre_object::with_roots!(base, exp, modulus => w_int_new(result)));
            }
        }
    }
    if let Some(result) = try_dispatch_ternary_pow_special(&mut base, &mut exp, &mut modulus)? {
        return Ok(result);
    }
    Err(ternary_builtin_type_error(
        "** or pow()",
        base,
        exp,
        modulus,
    ))
}

/// `divmod(a, b)` dispatch — pypy/interpreter/baseobjspace.py
/// `('divmod', 'divmod', 2, ['__divmod__', '__rdivmod__'])`. Numeric
/// fast path then forward + reverse special-method dispatch with the
/// standard NotImplemented fallback.
pub fn divmod(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    let numeric_override;
    unsafe {
        numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::DivMod);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__"))?
        {
            return Ok(result);
        }
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if !numeric_override && lhs_num && rhs_num {
            // Python 3.14 target-version spelling; see `divmod_builtin` above
            // for the PyPy 3.11 difference.
            if !pyre_object::with_roots!(a, b => is_true(b))? {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            if is_float_pair(a, b) {
                let (x, y) = float_operands(a, b)?;
                let (q, r) = float_divmod_w(x, y)?;
                let mut w_q = w_float_new(q);
                let w_r = pyre_object::with_roots!(w_q => w_float_new(r));
                return Ok(pyre_object::wraptuple2(w_q, w_r));
            }
            return integer_divmod_pair(a, b);
        }
    }
    if !numeric_override
        && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__"))?
    {
        return Ok(result);
    }
    Err(binary_builtin_type_error("divmod()", a, b))
}

/// floatobject.py `PowDomainError` — sentinel for a negative base raised to
/// a fractional power, which `descr_pow` promotes to a complex result.
enum FloatPowError {
    Domain,
    ZeroDivision,
    Overflow,
}

/// Float scalar helper for the rtyper residual surface. Mirrors RPython's
/// lltype-level math helper pattern (`rpython/rtyper/lltypesystem/module/
/// ll_math.py`) while keeping Rust's `f64::abs` intrinsic ABI out of the
/// two-phase rtyper.
#[majit_macros::dont_look_inside]
pub fn jit_float_abs(v: f64) -> f64 {
    v.abs()
}

/// `rpython.rlib.rfloat.rfloat` / `math.fmod` scalar residual.  PyPy's float
/// modulo, divmod, and power code call `math_fmod` directly; Rust's `%`
/// spelling would instead expose a `mod(SomeFloat, SomeFloat)` operation that
/// RPython's annotator intentionally does not define.
#[majit_macros::dont_look_inside]
pub fn jit_float_fmod(x: f64, y: f64) -> f64 {
    x % y
}

/// `float_pow`: libm sets `ERANGE` when a finite base produces an
/// out-of-range result.  An infinite base is excluded because `pow(±inf, y)`
/// is answered by the special cases above rather than by libm, so its infinity
/// is the exact result and not a range error.
fn float_pow_range_check(z: f64, base: f64) -> Result<f64, FloatPowError> {
    if z.is_infinite() && !base.is_infinite() {
        return Err(FloatPowError::Overflow);
    }
    Ok(z)
}

/// 3.14 surfaces the `ERANGE` libm sets for `float_pow` through
/// `PyErr_SetFromErrno` as the `(errno, strerror)` pair.
/// `floatobject.py:937-943` instead lets its own `math.pow` OverflowError
/// through as the message `"float power"`.
fn float_pow_overflow_error() -> PyError {
    // 34 on every platform pyre targets; spelled out rather than taken from
    // `libc`, which does not export the errno constants for `wasm32`.
    const ERANGE: i32 = 34;
    PyError::errno_pair(
        crate::PyErrorKind::OverflowError,
        pyre_object::interp_exceptions::ExcKind::OverflowError,
        ERANGE,
    )
}

/// floatobject.py `_pow`.
fn float_pow_inner(x: f64, y: f64) -> Result<f64, FloatPowError> {
    // floatobject.py:800-801
    if y == 2.0 {
        return float_pow_range_check(x * x, x);
    }
    // floatobject.py:803-804
    if y == 0.0 {
        return Ok(1.0);
    }
    // floatobject.py:806-807
    if x.is_nan() {
        return Ok(x);
    }
    // floatobject.py:809-814
    if y.is_nan() {
        return Ok(if x == 1.0 { 1.0 } else { y });
    }
    // floatobject.py:815-827
    if y.is_infinite() {
        let ax = jit_float_abs(x);
        if ax == 1.0 {
            return Ok(1.0);
        }
        return Ok(if (y > 0.0) == (ax > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    // floatobject.py:828-842
    if x.is_infinite() {
        let y_is_odd = jit_float_fmod(jit_float_abs(y), 2.0) == 1.0;
        return Ok(if y > 0.0 {
            if y_is_odd { x } else { jit_float_abs(x) }
        } else if y_is_odd {
            float_copysign(0.0, x)
        } else {
            0.0
        });
    }
    // floatobject.py:844-847
    if x == 0.0 && y < 0.0 {
        return Err(FloatPowError::ZeroDivision);
    }
    // floatobject.py:849-862
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return Err(FloatPowError::Domain);
        }
        bx = -bx;
        negate_result = jit_float_fmod(jit_float_abs(y), 2.0) == 1.0;
    }
    // floatobject.py:864-869
    if bx == 1.0 {
        return Ok(if negate_result { -1.0 } else { 1.0 });
    }
    // floatobject.py:871-877
    let z = float_pow_range_check(bx.powf(y), bx)?;
    // floatobject.py:879-881
    Ok(if negate_result { -z } else { z })
}

/// Raw `x ** y` as an `f64` for the `int`/`long` negative-exponent path. The
/// negative-base fractional case cannot arise with an integral exponent, but is
/// mapped back to the ValueError that `_pow` raises via `PowDomainError`.
pub fn float_pow_raw(x: f64, y: f64) -> Result<f64, PyError> {
    match float_pow_inner(x, y) {
        Ok(z) => Ok(z),
        Err(FloatPowError::Domain) => Err(PyError::value_error(
            "negative number cannot be raised to a fractional power",
        )),
        Err(FloatPowError::ZeroDivision) => Err(PyError::zero_division("zero to a negative power")),
        Err(FloatPowError::Overflow) => Err(float_pow_overflow_error()),
    }
}

/// floatobject.py `W_FloatObject.descr_pow`.
///
/// Descent leaf: the success arm boxes with the same in-graph
/// `malloc_typed_managed` spelling as `_float_truediv`. `w_float_new` is the
/// synthetic constructor the float-binop walk does not lower.
#[inline(never)]
pub(crate) fn float_pow_impl(x: f64, y: f64) -> PyResult {
    match float_pow_inner(x, y) {
        Ok(z) => Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
            ob_header: PyObject {
                ob_type: &FLOAT_TYPE as *const PyType,
                w_class: get_instantiate(&FLOAT_TYPE),
            },
            floatval: z,
        }) as PyObjectRef),
        // Negative numbers raised to fractional powers become complex.
        Err(FloatPowError::Domain) => unsafe {
            complex_pow(w_complex_new(x, 0.0), w_complex_new(y, 0.0))
        },
        Err(FloatPowError::ZeroDivision) => Err(PyError::zero_division("zero to a negative power")),
        Err(FloatPowError::Overflow) => Err(float_pow_overflow_error()),
    }
}

/// Host predicate for the float `**` descent: a raising `_pow` arm is not
/// descended (same rule as a zero divisor on `_float_truediv`).
pub fn float_pow_would_raise(x: f64, y: f64) -> bool {
    float_pow_inner(x, y).is_err()
}

/// Left shift dispatch (`<<` operator).

pub fn lshift(mut a: PyObjectRef, mut b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = pyre_object::with_roots!(a, b => shortcut_lshift(a, b))?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    lshift_impl(a, b, "<<")
}

/// `<<` and `<<=` — one body, see [`binop_type_error`].
pub(crate) fn lshift_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::LShift);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_lshift(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                // `_make_descr_binop` hands a `W_IntObject` operand to
                // `_int_lshift` as a machine word; only a long operand
                // reaches `_lshift`.
                if is_int_like(b) {
                    return long_int_lshift(a, int_value(b));
                }
                return long_lshift(a, b);
            }
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = shortcut_rshift(a, b)?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    rshift_impl(a, b, ">>")
}

/// `>>` and `>>=` — one body, see [`binop_type_error`].
pub(crate) fn rshift_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::RShift);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_rshift(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                // `_make_descr_binop` hands a `W_IntObject` operand to
                // `_int_rshift` as a machine word; only a long operand
                // reaches `_rshift`.
                if is_int_like(b) {
                    return long_int_rshift(a, int_value(b));
                }
                return long_rshift(a, b);
            }
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn and_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = shortcut_and(a, b)?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    and_impl(a, b, "&")
}

/// `&` and `&=` — one body, see [`binop_type_error`].
pub(crate) fn and_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch_unless_exact(a, b);
        if set_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__and__", "__rand__"))?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::And);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__and__", "__rand__"))?
        {
            return Ok(result);
        }
        // boolobject.py W_BoolObject.descr_and — both operands bool
        // → space.newbool(op(a, b)). MRO ensures this runs before the
        // W_IntObject.descr_and fallback in int_bitand.
        if !numeric_override {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_and(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitand(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitand(a, b);
            }
        }
        // set / frozenset intersection — PyPy: setobject.py W_BaseSetObject.descr_and.
        // descr_and returns NotImplemented for a non-set rhs, so `&` requires
        // both operands to be sets (the `intersection` method takes iterables).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_intersection(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__and__", "__rand__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Check if an object can participate in `X | Y` union syntax.
///
/// PyPy equivalent: _unionable() in _pypy_generic_alias.py
#[inline]
pub(crate) fn unionable(obj: PyObjectRef) -> bool {
    unsafe {
        is_none(obj)
            || is_type(obj)
            || pyre_object::is_union(obj)
            || pyre_object::is_generic_alias(obj)
    }
}

/// Bitwise OR dispatch (`|` operator).

pub fn or_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = shortcut_or(a, b)?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    or_impl(a, b, "|")
}

/// `|` and `|=` — one body, see [`binop_type_error`].
pub(crate) fn or_impl(a: PyObjectRef, b: PyObjectRef, symbol: &str) -> PyResult {
    // `pypy/objspace/std/dictproxyobject.py descr_or` /
    // `pypy/objspace/std/dictproxyobject.py descr_ror` —
    // mappingproxy `|` dispatches by copying the proxy's wrapped
    // mapping then `update`-ing with the other operand.  Pre-unwrap
    // each side so the dict-arm below sees plain dicts and produces
    // the same merge result.  The proxy-on-rhs case mirrors
    // `descr_ror` (proxy wraps the rhs operand inside `__or__`).
    let mut a = unsafe {
        if pyre_object::is_dict_proxy(a) {
            pyre_object::w_dict_proxy_get_mapping(a)
        } else {
            a
        }
    };
    let mut b = unsafe {
        if pyre_object::is_dict_proxy(b) {
            pyre_object::w_dict_proxy_get_mapping(b)
        } else {
            b
        }
    };
    unsafe {
        let set_override = needs_set_binop_dispatch_unless_exact(a, b);
        let numeric = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Or);
        if set_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__or__", "__ror__"))?
        {
            return Ok(result);
        }
        if numeric
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__or__", "__ror__"))?
        {
            return Ok(result);
        }
        // boolobject.py W_BoolObject.descr_or — both operands bool
        // → space.newbool(op(a, b)).
        if !numeric {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_or(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitor(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitor(a, b);
            }
        }
        // set / frozenset union — PyPy: setobject.py W_BaseSetObject.descr_or.
        // descr_or returns NotImplemented unless w_other is a set/frozenset,
        // so the binary `|` operator requires both operands to be sets
        // (the `union` method accepts arbitrary iterables, descr_or does not).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_union(&[a, b]);
        }
        // dict | dict — PEP 584 merge. PyPy: dictmultiobject.py descr_or.
        // Returns a new dict built from `a`'s items, then updated with `b`'s.
        if pyre_object::is_dict(a) && pyre_object::is_dict(b) {
            // `dictmultiobject.py descr_or` — `copyself = self.copy()`
            // then `update1(space, copyself, w_other)`.  Storing `a`'s items
            // one at a time instead re-hashes every key through its `__hash__`,
            // which the strategy copy does not do.  Each of those hashes also
            // runs Python, and the merge target, both operands and the item
            // snapshots are bare locals no root walker scans; `dict_update1`
            // already brackets its own operands for that reason.
            let _roots = pyre_object::gc_roots::push_roots();
            let root_base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let src = || pyre_object::gc_roots::shadow_stack_get(root_base);
            let other = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
            let merged = || pyre_object::gc_roots::shadow_stack_get(root_base + 2);
            let _ =
                pyre_object::gc_roots::pin_root(crate::type_methods::dict_method_copy(&[src()])?);
            crate::type_methods::dict_update1(merged(), other())?;
            return Ok(merged());
        }
        // user-class + typedef (dict_view, …) dispatch: forward __or__ then reflected
        // __ror__, exactly once.  Skipped when a gated special above already ran, so a
        // set-/numeric-subclass override is never re-invoked.
        if !set_override
            && !numeric
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__or__", "__ror__"))?
        {
            return Ok(result);
        }
        // type | type — PEP 604 union types (Python 3.10+)
        // PyPy: typeobject.py descr_or → _pypy_generic_alias._create_union,
        // which collapses identical operands (`int | int` is `int`).
        // `None | type` reaches the type's reflected union descriptor, but
        // `None | None` has no type / UnionType / GenericAlias descriptor on
        // either side and must remain an unsupported bitwise operation.
        if unionable(a) && unionable(b) && !(is_none(a) && is_none(b)) {
            return crate::_pypy_generic_alias::create_union(a, b);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn xor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if same_unoverridden_rpy_type(a, b) {
            if let Some(w_res) = shortcut_xor(a, b)?
                && !is_not_implemented(w_res)
            {
                return Ok(w_res);
            }
        }
    }
    xor_impl(a, b, "^")
}

/// `^` and `^=` — one body, see [`binop_type_error`].
pub(crate) fn xor_impl(mut a: PyObjectRef, mut b: PyObjectRef, symbol: &str) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch_unless_exact(a, b);
        if set_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__xor__", "__rxor__"))?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch_unless_exact(a, b, BinopDunder::Xor);
        if numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__xor__", "__rxor__"))?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_xor(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitxor(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitxor(a, b);
            }
        }
        // set / frozenset symmetric difference — `pypy/objspace/std/
        // setobject.py W_BaseSetObject.descr_xor`.  Mirrors `and_`'s
        // intersection arm: walk both sides, keep elements present in
        // exactly one set.  Result type follows the left operand
        // (frozenset stays frozenset).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_symmetric_difference(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = pyre_object::with_roots!(a, b => try_dispatch_binary_special(a, b, "__xor__", "__rxor__"))?
        {
            return Ok(result);
        }
        Err(binop_type_error(symbol, a, b))
    }
}

/// Comparison operation dispatch.
///
/// Loop-free: the MRO override walk lives in residual
/// [`try_compare_override`] (`look_inside_graph` / `contains_loop`).
/// `compare_slot_rest` is the same split for non-int layouts.
pub fn compare(mut a: PyObjectRef, mut b: PyObjectRef, op: CompareOp) -> PyResult {
    // Do not pin at the head: `pin_roots` is `dont_look_inside`, and a
    // residual on the walked `int < int` path blocks `compare_op_descent`
    // and doubles guard-failures. The MayForce C ABI copies are published
    // in `jit_compare_value_from_tag`; the override arm below publishes
    // only when `try_compare_override` can collect.
    // `_make_comparison_impl`: only `__eq__`/`__ne__` have `left == right`,
    // so only they take the same-type shortcut.
    unsafe {
        if matches!(op, CompareOp::Eq | CompareOp::Ne) && same_unoverridden_rpy_type(a, b) {
            // `_check_notimplemented`: a `NotImplemented` answer from the
            // shortcut falls through to the full lookup below.
            // `pin_roots` is `dont_look_inside`. Exact builtins whose
            // ordering is a field read (`builtin_pair_needs_no_caller_roots`)
            // stay on [`pin_free_builtin_ord`], so a traced `int == int`
            // does not record that residual. `compare_slot` itself can
            // collect on another arm, and the fallthrough reads `a` and `b`,
            // so every pair this body does not finish publishes first.
            if builtin_pair_needs_no_caller_roots(a, b)
                && let Some(bit) = pin_free_builtin_ord(a, b, op)
            {
                return Ok(w_bool_from(bit));
            }
            let w_res = pyre_object::with_roots!(a, b => compare_slot(a, b, op))?;
            if !pyre_object::is_not_implemented(w_res) {
                return Ok(w_res);
            }
        }
    }
    // A builtin subclass overriding the comparison dunder dispatches the
    // override first (with reflected-subclass priority); exact builtins and
    // non-overriding subclasses fall through to the by-layout comparison slot,
    // which gives the inherited builtin comparison.
    unsafe {
        // A pair of exact builtins cannot reach an override, so the probe can
        // only answer `None` for it: `is_instance` is false for both, and
        // `subclass_special_override` returns `None` on
        // `is_exact_builtin_instance` before it looks anything up.  Deciding
        // that here spares the pair a reverse-dunder resolution and two MRO
        // lookups, which is the whole of the probe for the commonest
        // comparison there is.  The decision is promoted
        // (`both_exact_builtin_instances_promoted`), so a traced comparison
        // of two exact builtins records neither the probe nor this test.
        //
        // Only the probe is skipped.  The subtype ordering below still runs
        // for such a pair, because `bool` is a proper subclass of `int` and
        // both are exact builtin instances.
        let pair_can_override = !both_exact_builtin_instances_promoted(a, b);
        if pair_can_override {
            // `transform.py insert_ll_stackcheck` puts a stack check on a block
            // of every call-graph cycle.  An override implemented natively
            // re-enters `compare` without pushing a Python frame, so the frame
            // limit never fires: `C.__eq__ = types.MethodType(_operator.eq, c)`
            // closes the cycle through `call_function_impl_result`, which runs
            // no check of its own.  The guard sits on the arm that can reach
            // one rather than at the head, because an exact-builtin pair
            // answers `false` above on a promoted decision, so a traced
            // `int < int` records neither the probe nor this check.  The
            // by-layout container cycle is covered separately, by the check in
            // [`compare_slot_rest`].
            crate::stack_check::stack_check()?;
            // `try_compare_override` is residual and can collect. Publish
            // only on this arm so an exact-builtin pair never records the
            // pin (`both_exact_builtin_instances_promoted` is false here).
            let _override_roots = pyre_object::gc_roots::push_roots();
            let cmp_base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a_live = pyre_object::gc_roots::shadow_stack_get(cmp_base);
            let b_live = pyre_object::gc_roots::shadow_stack_get(cmp_base + 1);
            if let Some(result) = try_compare_override(a_live, b_live, op)? {
                return Ok(result);
            }
            a = pyre_object::gc_roots::shadow_stack_get(cmp_base);
            b = pyre_object::gc_roots::shadow_stack_get(cmp_base + 1);
        }
        // PyPy `descroperation.py:_make_comparison_impl` swaps the operands
        // whenever the right-hand type is a proper subtype, before invoking
        // the inherited comparison implementation.  `try_compare_override`
        // already performs that ordering for generic instances and genuine
        // builtin overrides; this is the corresponding path for a builtin
        // subclass which inherits (rather than overrides) its comparison
        // slot, such as unittest.mock's `_CallList(list)`.
        // Keep generic instances out of this slot-only path: PyPy preserves
        // `w_orig_obj1` / `w_orig_obj2` when subtype-first methods all return
        // NotImplemented, so the final ordering TypeError still names the
        // original operator and operand order.
        if !is_instance(a)
            && !is_instance(b)
            // Two `let` chains, not one tuple pattern: a tuple of two
            // `Option`s is an allocation on the walked path.
            && let Some(a_type) = crate::typedef::r#type(a)
            && let Some(b_type) = crate::typedef::r#type(b)
            // Spelled as a raw-pointer identity, the way this file spells every
            // other one.  `NonNull`'s own `!=` goes through `NonNull::ne`,
            // which is not a call a trace can lower, and this test now sits on
            // the walked path for an exact-builtin pair.
            && !std::ptr::eq(a_type.as_ptr(), b_type.as_ptr())
            && issubtype_cached(b_type.as_ptr(), a_type.as_ptr())
        {
            return compare_slot(b, a, reverse_compare_op(op));
        }
    }
    compare_slot(a, b, op)
}

/// The builtin comparison slot body: rich-comparison dispatch by concrete
/// layout.  Reached from the operator [`compare`] for exact builtins and
/// non-overriding subclasses, and bound by the `cmp_dunder!` slots so a
/// subclass override's `super().__eq__` (etc.) resolves to the inherited
/// builtin comparison instead of re-entering override dispatch (which would
/// recurse).
pub fn compare_slot(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    // The machine-int arm stands alone so that this function has no loop:
    // the codewriter looks inside a loop-free graph only
    // (`policy.py look_inside_graph`), and a traced `int < int` must reach
    // `int_lt` through here.  Exact float/float is the same shape
    // (`_float_lt` after `_to_float`).  Long/long (`rbigint.lt`), mixed
    // long/int (`rbigint.int_lt` via [`long_int_compare`]), and str/str
    // (`W_UnicodeObject.descr_eq` / `descr_lt`: `ll_streq` / `ll_strcmp`
    // over the two `_utf8` payloads) are loop-free too and must live here,
    // or the leaf is only reachable from [`compare_slot_rest`] and never
    // becomes a jitcode.  Every layout that iterates stays in
    // [`compare_slot_rest`].
    // Tuple comparison stays there: the container cycle's stack check is the
    // first thing `compare_slot_rest` does, and a tuple arm ahead of that
    // check would recurse through `compare_tuples` with no guard.  Short
    // tuple equality is folded in the tracer instead.
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return match op {
                CompareOp::Lt => int_lt(a, b),
                CompareOp::Le => int_le(a, b),
                CompareOp::Gt => int_gt(a, b),
                CompareOp::Ge => int_ge(a, b),
                CompareOp::Eq => int_eq(a, b),
                CompareOp::Ne => int_ne(a, b),
            };
        }
        if is_float(a) && is_float(b) {
            let x = w_float_get_value(a);
            let y = w_float_get_value(b);
            return match op {
                CompareOp::Lt => _float_lt(x, y),
                CompareOp::Le => _float_le(x, y),
                CompareOp::Gt => _float_gt(x, y),
                CompareOp::Ge => _float_ge(x, y),
                CompareOp::Eq => _float_eq(x, y),
                CompareOp::Ne => _float_ne(x, y),
            };
        }
        // longobject.py `_make_descr_cmp` and intobject.py
        // `_make_descr_cmp`: both mixed orders call an rbigint.int_* method
        // on the long payload. The int-left order uses the reversed relation.
        // W_BoolObject shares W_IntObject storage upstream; pyre's distinct
        // bool layout requires the paired projections below.
        if is_long(a) && is_bool(b) {
            return Ok(w_bool_from(long_int_compare(
                a,
                w_bool_get_value(b) as i64,
                op,
            )));
        }
        if is_long(a) && is_int(b) {
            return Ok(w_bool_from(long_int_compare(a, w_int_get_value(b), op)));
        }
        if is_bool(a) && is_long(b) {
            return Ok(w_bool_from(long_int_compare(
                b,
                w_bool_get_value(a) as i64,
                reverse_compare_op(op),
            )));
        }
        if is_int(a) && is_long(b) {
            return Ok(w_bool_from(long_int_compare(
                b,
                w_int_get_value(a),
                reverse_compare_op(op),
            )));
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // All machine/mixed integer pairs returned above, so this is the
            // longobject.py `_make_descr_cmp` long/long arm. `asbigint()`
            // returns each W_LongObject's immutable payload upstream; borrow
            // those payloads rather than allocating two translated clones.
            debug_assert!(is_long(a) && is_long(b));
            let va = w_long_get_value(a);
            let vb = w_long_get_value(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => va.lt(vb),
                CompareOp::Le => va.le(vb),
                CompareOp::Gt => va.gt(vb),
                CompareOp::Ge => va.ge(vb),
                CompareOp::Eq => va.eq(vb),
                CompareOp::Ne => va.ne(vb),
            }));
        }
        if is_str(a) && is_str(b) {
            // `W_UnicodeObject.descr_eq` / `descr_lt` and their siblings
            // compare the two `_utf8` payloads, which `rtype_eq` /
            // `rtype_lt` turn into `ll_streq` / `ll_strcmp` calls. WTF-8
            // byte order matches code-point order, lone surrogates included.
            let s1 = pyre_object::unicodeobject::w_str_storage(a);
            let s2 = pyre_object::unicodeobject::w_str_storage(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) < 0,
                CompareOp::Le => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) <= 0,
                CompareOp::Gt => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) > 0,
                CompareOp::Ge => pyre_object::lowlevel_string::jit_ll_strcmp(s1, s2) >= 0,
                CompareOp::Eq => pyre_object::lowlevel_string::jit_ll_streq(s1, s2) != 0,
                CompareOp::Ne => pyre_object::lowlevel_string::jit_ll_streq(s1, s2) == 0,
            }));
        }
    }
    compare_slot_rest(a, b, op)
}

/// [`compare_slot`] for layouts whose comparison iterates (containers) or
/// is not the loop-free long/int/str arm.
#[inline(never)]
fn compare_slot_rest(mut a: PyObjectRef, mut b: PyObjectRef, op: CompareOp) -> PyResult {
    // RPython inserts a stack check on this recursive object-space call.
    // Container comparisons recurse through [`compare`] without pushing a
    // Python frame (for example two distinct self-referential lists), so keep
    // the same guard explicitly in the Rust port and raise RecursionError
    // before exhausting the native stack.  It sits on the recursive arm: a
    // machine-int pair never recurses, and the check would otherwise be one
    // residual call on every traced `int < int`.
    crate::stack_check::stack_check()?;
    unsafe {
        if is_float_pair(a, b) {
            // Exact float/float already returned from [`compare_slot`].
            // Mixed int/long keeps `float_compare` for the mantissa / bigint
            // arm (`int_between(-1, i2 >> 48, 1)`).
            // `_compare` is a method on the float operand; the int-left order
            // reaches it through the reflected comparison, with the relation
            // reversed.
            return Ok(w_bool_from(if is_float(a) {
                float_compare(a, b, op)
            } else {
                float_compare(b, a, reverse_compare_op(op))
            }));
        }
        // complexobject.py only implements equality here.  Its ordering
        // dunders return NotImplemented, which must continue through the
        // reflected comparison and the generic TypeError fallback below.
        if is_complex_pair(a, b) && matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return complex_richcompare(a, b, op);
        }
        // bytesobject.py W_BytesObject.descr_eq / _lt / ... and the
        // bytearray counterparts — lexicographic comparison on the raw
        // bytes.  bytes and bytearray compare by content
        // (b"a" == bytearray(b"a")), so both operands route through
        // bytes_like_data. Immutable `bytes`/`bytes` keeps the elidable
        // `ll_bytes_strcmp`; any `bytearray` operand uses the non-elidable
        // twin, matching `_handle_stroruni_call`'s `BYTEARRAY` reject.
        if pyre_object::bytesobject::is_bytes_like(a) && pyre_object::bytesobject::is_bytes_like(b)
        {
            let da = pyre_object::bytesobject::bytes_like_data(a);
            let db = pyre_object::bytesobject::bytes_like_data(b);
            let diff =
                if pyre_object::bytesobject::is_bytes(a) && pyre_object::bytesobject::is_bytes(b) {
                    ll_bytes_strcmp(da, db)
                } else {
                    ll_bytes_strcmp_mutable(da, db)
                };
            return Ok(w_bool_from(match op {
                CompareOp::Lt => diff < 0,
                CompareOp::Le => diff <= 0,
                CompareOp::Gt => diff > 0,
                CompareOp::Ge => diff >= 0,
                CompareOp::Eq => diff == 0,
                CompareOp::Ne => diff != 0,
            }));
        }
        // Tuple lexicographic comparison — PyPy: tupleobject.py descr_lt / _eq / etc.
        // Kept behind this function's stack check: nested tuples re-enter
        // `compare` → `compare_slot` → here, and that is the container cycle.
        if is_tuple(a) && is_tuple(b) {
            if let Some(bit) = compare_tuples(a, b, op) {
                return Ok(w_bool_from(bit));
            }
            return compare_tuples_general(a, b, op);
        }
        // dict equality — `pypy/objspace/std/dictmultiobject.py
        // W_DictMultiObject.descr_eq` is order-independent: same length
        // AND each key-value pair in `a` exists with equal value in `b`.
        // CPython only defines == / != for dicts (no ordering), so we
        // restrict to those ops; other ops fall through to the dunder
        // dispatch which currently raises TypeError, matching the
        // unimplemented `__lt__` etc. on plain dict.
        if is_exact_type(a, &pyre_object::DICT_TYPE)
            && is_exact_type(b, &pyre_object::DICT_TYPE)
            && matches!(op, CompareOp::Eq | CompareOp::Ne)
        {
            // `w_dict_lookup_checked` hashes `k` through its `__hash__` and
            // `eq_w` runs the values' `__eq__`; both are collection points.
            // `b` and every entry of the `w_dict_items` snapshot are native
            // locals no root walker updates, and the operands were popped off
            // the value stack before this dispatch, so publish the whole set
            // and read each entry back after the comparisons that precede it.
            let _roots = pyre_object::gc_roots::push_roots();
            let root_base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let la = pyre_object::w_dict_len(pyre_object::gc_roots::shadow_stack_get(root_base));
            let lb =
                pyre_object::w_dict_len(pyre_object::gc_roots::shadow_stack_get(root_base + 1));
            let mut equal = la == lb;
            if equal {
                let items =
                    pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(root_base));
                let pairs = items.as_slice();
                let n_items = pairs.len();
                let mut flat = Vec::with_capacity(n_items * 2);
                let mut i = 0usize;
                while i < n_items {
                    flat.push(pairs[i].0);
                    flat.push(pairs[i].1);
                    i += 1;
                }
                let items_base = pyre_object::gc_roots::pin_roots(&flat);
                let mut index = 0usize;
                while index < n_items {
                    let k = pyre_object::gc_roots::shadow_stack_get(items_base + index * 2);
                    let other = pyre_object::dictmultiobject::w_dict_lookup_checked(
                        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
                        k,
                    )
                    .map_err(|_| {
                        crate::baseobjspace::take_pending_dict_key_error(
                            pyre_object::gc_roots::shadow_stack_get(items_base + index * 2),
                        )
                    })?;
                    match other {
                        Some(other_v) => {
                            // dictmultiobject.py:664 `if not space.eq_w(w_val,
                            // w_rightval): return space.w_False`
                            let _val_roots = pyre_object::gc_roots::push_roots();
                            let pair = pyre_object::gc_roots::pin_roots(&[
                                pyre_object::gc_roots::shadow_stack_get(items_base + index * 2 + 1),
                                other_v,
                            ]);
                            if !crate::baseobjspace::eq_w(
                                pyre_object::gc_roots::shadow_stack_get(pair),
                                pyre_object::gc_roots::shadow_stack_get(pair + 1),
                            )? {
                                equal = false;
                                break;
                            }
                        }
                        None => {
                            equal = false;
                            break;
                        }
                    }
                    index += 1;
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        // `dictmultiobject.py _is_set_like` parity — when
        // one side is a set/frozenset and the other is a set-like
        // dict_view (Keys / Items), the comparison reduces to the
        // set-set arm with the dict_view materialised through its
        // snapshot.  Without this arm, `set == d.keys()` would fall
        // through to `object.__eq__`'s identity check and return
        // False even when the contents match.
        if (pyre_object::is_set_or_frozenset(a) || pyre_object::is_set_or_frozenset(b))
            && (pyre_object::dictmultiobject::is_dict_view(a)
                || pyre_object::dictmultiobject::is_dict_view(b))
        {
            let view_set_like = |obj: PyObjectRef| -> bool {
                if pyre_object::is_set_or_frozenset(obj) {
                    return true;
                }
                if pyre_object::dictmultiobject::is_dict_view(obj) {
                    let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj);
                    return matches!(
                        kind,
                        pyre_object::dictmultiobject::DictViewKind::Keys
                            | pyre_object::dictmultiobject::DictViewKind::Items
                    );
                }
                false
            };
            if view_set_like(a) && view_set_like(b) {
                // `SetLikeDictView` owns every one of these comparisons
                // upstream: a set answers `NotImplemented` for a view, and the
                // reflected call on the view side is what decides.  Reaching
                // that method is what this arm stands in for, so it hands the
                // pair to the same walk with the operator reflected when the
                // view is the right-hand operand — rather than reducing both
                // sides to sets, which answers from a snapshot taken before
                // the first user `__eq__` and so cannot see that dict change
                // size under the comparison.
                use crate::typedef::DictViewCmp;
                // At least one operand is a set here, so a view on the left is
                // the whole test for which side the reflected call names as
                // `self`.
                let view_on_left = pyre_object::dictmultiobject::is_dict_view(a);
                let op = match (op, view_on_left) {
                    (CompareOp::Eq, _) => DictViewCmp::Eq,
                    (CompareOp::Ne, _) => DictViewCmp::Ne,
                    (CompareOp::Lt, true) | (CompareOp::Gt, false) => DictViewCmp::Lt,
                    (CompareOp::Le, true) | (CompareOp::Ge, false) => DictViewCmp::Le,
                    (CompareOp::Gt, true) | (CompareOp::Lt, false) => DictViewCmp::Gt,
                    (CompareOp::Ge, true) | (CompareOp::Le, false) => DictViewCmp::Ge,
                };
                let (view, other) = if view_on_left { (a, b) } else { (b, a) };
                return crate::typedef::dict_view_compare(view, other, op);
            }
        }
        // set / frozenset comparison — subset / superset / equality.
        // PyPy: setobject.py W_BaseSetObject.descr_eq, descr_le, descr_lt
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            // The subset walks probe with the digest each element was stored
            // under (`setobject.py _issubset_unwrapped`), so a
            // comparison hashes nothing and an `eq_w` raised from a bucket
            // probe propagates instead of reading as "not a subset".
            // `eq_w` is a collection point; `a`/`b` were popped off the
            // value stack, so pin them for the length reads and the walk.
            let _roots = pyre_object::gc_roots::push_roots();
            let root_base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a = || pyre_object::gc_roots::shadow_stack_get(root_base);
            let b = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
            let la = pyre_object::w_set_len(a());
            let lb = pyre_object::w_set_len(b());
            let a_subset_b = || crate::typedef::set_is_subset_of(a(), b());
            let b_subset_a = || crate::typedef::set_is_subset_of(b(), a());
            return Ok(w_bool_from(match op {
                CompareOp::Eq => la == lb && a_subset_b()?,
                CompareOp::Ne => la != lb || !a_subset_b()?,
                CompareOp::Le => la <= lb && a_subset_b()?,
                CompareOp::Lt => la < lb && a_subset_b()?,
                CompareOp::Ge => la >= lb && b_subset_a()?,
                CompareOp::Gt => la > lb && b_subset_a()?,
            }));
        }
        // List comparison. Unlike tuples, element comparison may mutate either
        // operand, so every loop boundary and the final size comparison read
        // the live lists (CPython 3.14 `list_richcompare_impl`; PyPy
        // `list_eq` / `_compare_unwrappeditems`).
        if is_list(a) && is_list(b) {
            // `eq_w` below runs a user `__eq__`, which is a collection point,
            // and `a`/`b` are native copies no root walker updates — the
            // operands were popped off the value stack before this dispatch, so
            // the frame no longer roots them either.  Publish both and read
            // them back at every loop boundary and length read, which is also
            // what makes the "read the live lists" contract above hold.
            let _roots = pyre_object::gc_roots::push_roots();
            let root_base = pyre_object::gc_roots::pin_roots(&[a, b]);
            let a = || pyre_object::gc_roots::shadow_stack_get(root_base);
            let b = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
            if matches!(op, CompareOp::Eq | CompareOp::Ne)
                && pyre_object::w_list_len(a()) != pyre_object::w_list_len(b())
            {
                return Ok(w_bool_from(matches!(op, CompareOp::Ne)));
            }

            // Two extra slots sit next to the lists so each boxed getitem
            // is stored with `set` rather than `pin_root`: a push can grow
            // the shadow-stack buffer, and an inlined `shadow_stack_get`
            // would then read the old `base`.  `tupleobject.py
            // _compare_tuples` uses the same four-slot shape.
            let elem_roots = pyre_object::gc_roots::push_roots();
            let pair = elem_roots.publish(&[PY_NULL, PY_NULL]);
            elem_roots.normalize(pair, 2);
            let mut i = 0usize;
            while i < pyre_object::w_list_len(a()) && i < pyre_object::w_list_len(b()) {
                // Integer/float/range strategies box through `w_int_new` /
                // `w_float_new`.  Store the first box before the second
                // getitem, which is another collecting malloc
                // (`listobject.py list_eq` / `_compare_unwrappeditems`).
                elem_roots.set(
                    pair,
                    pyre_object::w_list_getitem(a(), i as i64).unwrap_or(PY_NULL),
                );
                elem_roots.set(
                    pair + 1,
                    pyre_object::w_list_getitem(b(), i as i64).unwrap_or(PY_NULL),
                );
                if !crate::baseobjspace::eq_w(elem_roots.get(pair), elem_roots.get(pair + 1))? {
                    break;
                }
                i += 1;
            }
            let la = pyre_object::w_list_len(a());
            let lb = pyre_object::w_list_len(b());
            if i >= la || i >= lb {
                return Ok(w_bool_from(match op {
                    CompareOp::Lt => la < lb,
                    CompareOp::Le => la <= lb,
                    CompareOp::Eq => la == lb,
                    CompareOp::Ne => la != lb,
                    CompareOp::Gt => la > lb,
                    CompareOp::Ge => la >= lb,
                }));
            }
            if matches!(op, CompareOp::Eq | CompareOp::Ne) {
                return Ok(w_bool_from(matches!(op, CompareOp::Ne)));
            }
            // Fetch the live items again: equality may have replaced either
            // element before the ordering comparison.  Same first-then-second
            // `set` as the equality loop: the second getitem can collect.
            elem_roots.set(
                pair,
                pyre_object::w_list_getitem(a(), i as i64).unwrap_or(PY_NULL),
            );
            elem_roots.set(
                pair + 1,
                pyre_object::w_list_getitem(b(), i as i64).unwrap_or(PY_NULL),
            );
            return compare(elem_roots.get(pair), elem_roots.get(pair + 1), op);
        }
        // range value comparison — functional.py W_Range.descr_eq:
        // two ranges are equal iff they generate the same sequence
        // (equal length, and for non-empty ranges equal start and —
        // for length > 1 — equal step).  Only `==` / `!=` are defined;
        // ordering falls through to the dunder dispatch (TypeError).
        if pyre_object::is_w_range(a)
            && pyre_object::is_w_range(b)
            && matches!(op, CompareOp::Eq | CompareOp::Ne)
        {
            let equal = pyre_object::w_range_eq(a, b);
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        let dunder = match op {
            CompareOp::Lt => "__lt__",
            CompareOp::Le => "__le__",
            CompareOp::Gt => "__gt__",
            CompareOp::Ge => "__ge__",
            CompareOp::Eq => "__eq__",
            CompareOp::Ne => "__ne__",
        };
        // `SetLikeDictView` operands expose comparison dunders through the
        // typedef. Instance-shaped operands were already handled by
        // `try_compare_override`, so only non-instance operands dispatch here.
        // Reflected: if RHS is a dict view, try `b.dunder(a)` —
        // PyPy's `_is_set_like(other)` short-circuits the LHS-side
        // descr_eq when the other is set-like, so the reflected call
        // path is the one that succeeds for `set == d.keys()`.
        if !is_instance(a)
            && let Some(a_type) = crate::typedef::r#type(a)
            && let Some(method) = lookup_in_type_where(a_type.as_ptr(), dunder)
        {
            // A raised exception (not NotImplemented) propagates; only
            // NotImplemented falls through to the reflected comparison.
            let result = pyre_object::with_roots!(a, b => crate::call::call_function_impl_result(method, &[a, b]))?;
            if !is_not_implemented(result) {
                return Ok(result);
            }
        }
        if !is_instance(b)
            && let Some(rdunder) = reverse_dunder(dunder)
            && let Some(b_type) = crate::typedef::r#type(b)
            && let Some(method) = lookup_in_type_where(b_type.as_ptr(), rdunder)
        {
            let result = pyre_object::with_roots!(a, b => crate::call::call_function_impl_result(method, &[b, a]))?;
            if !is_not_implemented(result) {
                return Ok(result);
            }
        }
        // Identity comparison fallback for == and !=
        if matches!(op, CompareOp::Eq) {
            return Ok(w_bool_from(std::ptr::eq(a, b)));
        }
        if matches!(op, CompareOp::Ne) {
            return Ok(w_bool_from(!std::ptr::eq(a, b)));
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        let op_symbol = op.symbol();
        Err(PyError::type_error(format!(
            "'{op_symbol}' not supported between instances of '{a_name}' and '{b_name}'"
        )))
    }
}

/// The `CompareOp` a jitcode integer operand names.
///
/// A fieldless enum argument is lowered as its discriminant, so the decode
/// has to read the discriminant rather than a hand-written table that happens
/// to agree with today's declaration order — a variant inserted in the middle
/// would renumber every later one and the table would keep answering the old
/// meaning. Searching [`CompareOp::ALL`] by `op as i64` asks the compiler for
/// the number it actually assigned, so the two sides cannot drift.
fn compare_op_from_discriminant(value: i64) -> Option<CompareOp> {
    CompareOp::ALL
        .iter()
        .copied()
        .find(|op| *op as i64 == value)
}

/// One-word residual-call ABI for [`compare_slot`].
///
/// `compare_slot` itself is `(PyObjectRef, PyObjectRef, CompareOp) -> PyResult`,
/// and neither end of that signature fits the residual-call ABI: `PyError` is
/// far wider than a register so `Result<PyObjectRef, PyError>` returns through
/// an sret pointer, and `CompareOp` is not a `ResidualSlot`. Publishing the
/// raw address would pass one register where the callee reads a hidden return
/// slot. `jit_fnaddr.rs`'s publication helpers reject both by their bounds.
///
/// The trace calls this as `(r, r, i) -> r`:
/// `declared_result_type_for_target` projects a callee's `Result<T, PyError>`
/// through the transparent `Ok` unwrap and the error travels out of band, the
/// same shape `runtime_ops::call_callable_with_args` publishes. An operand
/// outside the enum's discriminants cannot come from a correct lowering; it
/// takes the same route `call_jit.rs` takes for an unknown compare tag — a
/// published `TypeError` and the null sentinel, so a `GUARD_NO_EXCEPTION`
/// fires instead of a silently wrong comparison.
pub extern "C" fn compare_slot_jit_abi(a: PyObjectRef, b: PyObjectRef, op: i64) -> PyObjectRef {
    let Some(op) = compare_op_from_discriminant(op) else {
        return crate::runtime_ops::jit_publish_residual_error_ref(PyError::type_error(format!(
            "unknown compare op discriminant {op}"
        )));
    };
    match compare_slot(a, b, op) {
        Ok(result) => result,
        Err(error) => crate::runtime_ops::jit_publish_residual_error_ref(error),
    }
}

/// Comparison operator enum (mirrors RustPython's ComparisonOperator).
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompareOp {
    /// Every variant, in declaration order.
    ///
    /// `compare_op_from_discriminant` searches this rather than indexing it,
    /// so the array is a set of variants to consider and not a position table:
    /// reordering it cannot change which integer decodes to which operator.
    pub const ALL: [CompareOp; 6] = [
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
        CompareOp::Eq,
        CompareOp::Ne,
    ];

    /// The Python operator symbol (`<`, `<=`, `>`, `>=`, `==`, `!=`) used in
    /// the "not supported between instances of" TypeError message.
    pub fn symbol(self) -> &'static str {
        match self {
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
        }
    }
}

/// Unary positive (`+a`).
///
/// `inline(never)` is load-bearing: rustc otherwise folds this body into its
/// one-call wrapper and the codewriter never mints the graph named by
/// `flatten.rs build_orthodox_inline_call_r_r`.
#[inline(never)]
pub fn pos(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) =
            pyre_object::with_roots!(a => try_numeric_unaryop_override(a, UnaryDunder::Pos))?
        {
            return Ok(result);
        }
        pos_inner(a)
    }
}

/// [`pos`] past its `__pos__` override probe.
///
/// Split out so that a trace can descend this body rather than re-emit it by
/// hand.  The probe is what stops such a descent: its
/// `needs_numeric_unaryop_dispatch` is `dont_look_inside` and is the second
/// operation the body executes.  A caller that has already proven an exact
/// builtin receiver cannot take it, so entering here records the arm that
/// receiver selects and nothing else.  Every other caller reaches this through
/// [`pos`] and is unaffected, except the `int`/`float`/`complex` `__pos__`
/// slots, which name this body directly because an unbound slot must not
/// dispatch.
///
/// Unlike [`invert_inner`] this keeps the `bool` operand: `+True` is a
/// rewrapping to a plain int, not a deprecation warning.  Unlike [`neg_inner`]
/// the exact-int and exact-long arms are identity — they return the operand —
/// matching `_self_unaryop('pos')` / `W_IntObject.int`.
///
/// `inline(never)` is load-bearing: `pos` is a one-call wrapper and rustc
/// otherwise folds this body into `unary_positive_value`, so the codewriter
/// never mints a graph the walker can descend.  Same knob as
/// `w_list_append_inner`.
#[inline(never)]
pub fn pos_inner(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && !is_bool(a) && pyre_object::is_exact_builtin_instance(a) {
            // `_self_unaryop('pos')` delegates to `W_IntObject.int`, which
            // returns `self` for the exact builtin representation.  `bool`
            // takes the rewrapping arm below because its `int` yields a plain
            // int rather than the singleton it was called on.
            return Ok(a);
        }
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(int_value(a)));
        }
        if is_long(a) {
            // intobject.py `_self_unaryop('pos')` delegates to
            // `self.int(space)`, which returns `self` only for the exact
            // builtin representation.  A subclass returns a plain-int copy
            // (long_pos: exact → self, else `_PyLong_Copy`), so leaking the
            // subclass instance would be wrong; an exact long returns `self`
            // without allocating another wrapper or shallow-cloned payload.
            if pyre_object::is_exact_builtin_instance(a) {
                return Ok(a);
            }
            return Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                bigint_clone(w_long_get_value(a)),
            ));
        }
        if is_float(a) {
            return _float_pos(w_float_get_value(a));
        }
        if is_complex(a) {
            let (ar, ai) = complex_val(a).unwrap();
            return Ok(w_complex_new(ar, ai));
        }
        if let Some(result) = pyre_object::with_roots!(a => try_lookup_unaryop(a, "__pos__"))? {
            return Ok(result);
        }
        Err(bad_operand_type("unary +", a))
    }
}

/// [3.14-spec] `UNARY_FUNC`'s refusal, `bad operand type for unary -: '<T>'`,
/// spelled with the operator.  `_make_unaryop_impl` builds its wording from the
/// symbol column of `ObjSpace.MethodTable`, which carries the *name* `neg` and
/// `pos` for those two rows and the symbol only for `invert`, so PyPy answers
/// `unsupported operand type for unary neg`.  Measured against 3.14.2;
/// `abs_bad_operand` already words the fourth arm of the same macro this way.
///
/// A null receiver reaches here from the callers below, which reserve it for
/// `None`.
fn bad_operand_type(descr: &str, a: PyObjectRef) -> PyError {
    let type_name = if a.is_null() {
        "NoneType".to_string()
    } else {
        crate::baseobjspace::object_functionstr_type_name(a)
    };
    PyError::type_error(format!("bad operand type for {descr}: '{type_name}'"))
}

/// intobject.py `descr_neg` after `ovfcheck(-a)`: `wrapint(-a)`.
/// `INT_MIN` stays in [`neg_inner`]'s ovf2long arm so this graph has
/// no overflow diamond.
#[inline(never)]
pub(crate) fn _int_neg(x: i64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        // `0 - x`, not `wrapping_neg`: Charon leaves the rustc intrinsic
        // as `residual_call_ir_i`, while subtraction is `int_sub`.
        intval: 0i64.wrapping_sub(x),
    }) as PyObjectRef)
}

/// floatobject.py `descr_pos`: `W_FloatObject(self.floatval)`.
#[inline(never)]
pub fn _float_pos(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x,
    }) as PyObjectRef)
}

/// ll_math.py `sqrt_nonneg`: `math_sqrt(x)`. The domain check stays in the
/// caller. `@jit.elidable` plus `sqrt_nonneg.oopspec = "math.sqrt_nonneg(x)"`.
#[majit_macros::elidable]
#[majit_macros::oopspec("math.sqrt_nonneg(x)")]
#[inline(never)]
pub(crate) fn sqrt_nonneg(x: f64) -> f64 {
    x.sqrt()
}

/// ll_math.py `ll_math_sqrt` after the domain pin: `W_FloatObject(sqrt_nonneg(x))`.
#[inline(never)]
pub fn _float_sqrt(x: f64) -> PyResult {
    let floatval = sqrt_nonneg(x);
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval,
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_sin` after the finite pin: `W_FloatObject(sin(x))`.
#[inline(never)]
pub fn _float_sin(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.sin(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_cos` after the finite pin: `W_FloatObject(cos(x))`.
#[inline(never)]
pub fn _float_cos(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.cos(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_tan` after the finite pin: `W_FloatObject(tan(x))`.
#[inline(never)]
pub fn _float_tan(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.tan(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_atan`: `W_FloatObject(atan(x))`.
#[inline(never)]
pub fn _float_atan(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.atan(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_exp` after the overflow pin: `W_FloatObject(exp(x))`.
#[inline(never)]
pub fn _float_exp(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.exp(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_log1p` after `x > -1`: `W_FloatObject(log1p(x))`.
#[inline(never)]
pub fn _float_log1p(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.ln_1p(),
    }) as PyObjectRef)
}

/// ll_math.py `ll_math_asin` after the `[-1, 1]` pin: `W_FloatObject(asin(x))`.
#[inline(never)]
pub fn _float_asin(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.asin(),
    }) as PyObjectRef)
}

/// Unboxed `W_FloatObject` leaf. `|x| $compute` names the parameter; it is
/// not a Rust closure (those residualize an extra `new` + call).
macro_rules! float_math1_leaf {
    ($vis:vis $fn:ident, |$x:ident| $compute:expr) => {
        #[inline(never)]
        $vis fn $fn($x: f64) -> PyResult {
            Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
                ob_header: PyObject {
                    ob_type: &FLOAT_TYPE as *const PyType,
                    w_class: get_instantiate(&FLOAT_TYPE),
                },
                floatval: $compute,
            }) as PyObjectRef)
        }
    };
}

float_math1_leaf!(pub _float_acos, |x| x.acos());
float_math1_leaf!(pub _float_tanh, |x| x.tanh());
float_math1_leaf!(pub _float_cbrt, |x| x.cbrt());

/// `ll_math.py new_unary_math_function` C calls for `asinh` / `acosh` /
/// `atanh`, reached only inside the gateway's domain pin where the C
/// function cannot fail.  `f64::asinh` and its siblings are std's own
/// formulas rather than the C library and differ from it in the last place,
/// so these call the same libm routine the slow path does, as one residual.
#[majit_macros::elidable]
pub fn _float_asinh_raw(x: f64) -> f64 {
    pymath::math::asinh(x).unwrap_or(f64::NAN)
}

#[majit_macros::elidable]
pub fn _float_acosh_raw(x: f64) -> f64 {
    pymath::math::acosh(x).unwrap_or(f64::NAN)
}

#[majit_macros::elidable]
pub fn _float_atanh_raw(x: f64) -> f64 {
    pymath::math::atanh(x).unwrap_or(f64::NAN)
}

float_math1_leaf!(pub _float_asinh, |x| _float_asinh_raw(x));
float_math1_leaf!(pub _float_acosh, |x| _float_acosh_raw(x));
float_math1_leaf!(pub _float_atanh, |x| _float_atanh_raw(x));

/// `ll_math.py` C calls (`llexternal(.., elidable_function=True)`) the
/// crates.io pymath port answers; that crate is not in the Charon artefact,
/// so each is one elidable residual.
macro_rules! float_math1_raw {
    ($fn:ident, |$x:ident| $compute:expr) => {
        #[majit_macros::elidable]
        pub fn $fn($x: f64) -> f64 {
            $compute
        }
    };
}

float_math1_raw!(_float_erf_raw, |x| pymath::math::erf(x).unwrap_or(f64::NAN));
float_math1_raw!(_float_erfc_raw, |x| pymath::math::erfc(x)
    .unwrap_or(f64::NAN));
float_math1_raw!(_float_ulp_raw, |x| pymath::math::ulp(x));
float_math1_leaf!(pub _float_erf, |x| _float_erf_raw(x));
float_math1_leaf!(pub _float_erfc, |x| _float_erfc_raw(x));

float_math1_raw!(_float_gamma_raw, |x| pymath::math::gamma(x)
    .unwrap_or(f64::NAN));
float_math1_raw!(_float_lgamma_raw, |x| pymath::math::lgamma(x)
    .unwrap_or(f64::NAN));
float_math1_leaf!(pub _float_sinh, |x| x.sinh());
float_math1_leaf!(pub _float_cosh, |x| x.cosh());
float_math1_leaf!(pub _float_exp2, |x| x.exp2());
float_math1_leaf!(pub _float_expm1, |x| x.exp_m1());
float_math1_leaf!(pub _float_gamma, |x| _float_gamma_raw(x));
float_math1_leaf!(pub _float_lgamma, |x| _float_lgamma_raw(x));
float_math1_leaf!(pub _float_ulp, |x| _float_ulp_raw(x));
// pymath::math::{degrees,radians} is `x * (180/π)` / `x * (π/180)`.
float_math1_leaf!(pub _float_degrees, |x| x * (180.0 / std::f64::consts::PI));
float_math1_leaf!(pub _float_radians, |x| x * (std::f64::consts::PI / 180.0));
float_math1_leaf!(pub _float_log, |x| x.ln());

/// Unboxed two-arg `W_FloatObject` leaf. `|x, y| $compute` names the
/// parameters; it is not a Rust closure.
macro_rules! float_math2_leaf {
    ($fn:ident, |$x:ident, $y:ident| $compute:expr) => {
        #[inline(never)]
        pub fn $fn($x: f64, $y: f64) -> PyResult {
            Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
                ob_header: PyObject {
                    ob_type: &FLOAT_TYPE as *const PyType,
                    w_class: get_instantiate(&FLOAT_TYPE),
                },
                floatval: $compute,
            }) as PyObjectRef)
        }
    };
}

float_math2_leaf!(_float_pow, |x, y| x.powf(y));
float_math2_leaf!(_float_copysign, |x, y| x.copysign(y));

/// `ll_math.py` `math_fmod` (`llexternal(.., elidable_function=True)`); `%`
/// over two floats is not an RPython operation.
#[majit_macros::elidable]
pub fn _float_fmod_raw(x: f64, y: f64) -> f64 {
    x % y
}

#[majit_macros::elidable]
pub fn _float_remainder_raw(x: f64, y: f64) -> f64 {
    pymath::math::remainder(x, y).unwrap_or(f64::NAN)
}

float_math2_leaf!(_float_fmod, |x, y| _float_fmod_raw(x, y));

float_math2_leaf!(_float_remainder, |x, y| _float_remainder_raw(x, y));
float_math2_leaf!(_float_atan2, |x, y| x.atan2(y));

/// `math.floor`/`ceil`/`trunc` after the signed-range pin: `W_IntObject`.
macro_rules! int_from_float_leaf {
    ($fn:ident, |$x:ident| $compute:expr) => {
        #[inline(never)]
        pub fn $fn($x: f64) -> PyResult {
            Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
                ob_header: PyObject {
                    ob_type: &INT_TYPE as *const PyType,
                    w_class: get_instantiate(&INT_TYPE),
                },
                intval: $compute,
            }) as PyObjectRef)
        }
    };
}

// The walker pins the signed range before descent, so the unchecked
// conversion is defined.  `as i64` residualizes as a saturating rustc
// helper (symbolic fnaddr); `to_int_unchecked` is the fptosi the
// translator rewrites to `cast_float_to_int`.
int_from_float_leaf!(_int_from_floor, |x| unsafe {
    x.floor().to_int_unchecked::<i64>()
});
int_from_float_leaf!(_int_from_ceil, |x| unsafe {
    x.ceil().to_int_unchecked::<i64>()
});
int_from_float_leaf!(_int_from_trunc, |x| unsafe { x.to_int_unchecked::<i64>() });

/// ll_math.py `ll_math_frexp` exponent half for a normal finite non-zero,
/// unboxed: the `pow` and `ldexp` gateways read it for their domain pins.
/// The boxing leaves below spell the bit arithmetic in their own bodies so
/// their jitcode records it directly.
macro_rules! frexp_exponent_bits {
    ($x:expr) => {{
        let bits = $x.to_bits() as i64;
        ((bits >> 52) & 0x7ff) - 1022
    }};
}

#[inline(never)]
pub fn _int_frexp_exponent_raw(x: f64) -> i64 {
    frexp_exponent_bits!(x)
}

/// ll_math.py `ll_math_frexp` mantissa half for the same operand.
float_math1_leaf!(pub _float_frexp_mantissa, |x| {
    let bits = x.to_bits() as i64;
    let sign = bits & i64::MIN;
    let fraction = bits & ((1i64 << 52) - 1);
    f64::from_bits((sign | (1022i64 << 52) | fraction) as u64)
});
int_from_float_leaf!(_int_frexp_exponent, |x| frexp_exponent_bits!(x));

/// ll_math.py `ll_math_ldexp` on its exact arm: `-1022 <= exp <= 1023`, so
/// `2**exp` is the normal double built from its exponent bits, and the
/// caller keeps the product normal, so the multiplication cannot round.
/// The boxing leaf below spells the same product in its own body.
macro_rules! ldexp_exact_bits {
    ($x:expr, $exp:expr) => {
        $x * f64::from_bits((($exp + 1023) << 52) as u64)
    };
}

/// Only valid on the exact arm: the caller has already checked
/// `-1022 <= exp <= 1023`. Outside it the exponent field wraps, and the
/// result is not `x * 2**exp`.
#[inline(never)]
pub fn _float_ldexp_raw(x: f64, exp: i64) -> f64 {
    ldexp_exact_bits!(x, exp)
}

/// ll_math.py `ll_math_ldexp` after the finite-x pin.
#[inline(never)]
pub fn _float_ldexp(x: f64, exp: i64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: ldexp_exact_bits!(x, exp),
    }) as PyObjectRef)
}

/// app_math.py `isqrt` on a positive machine int that fits an exact
/// `f64`.  One ulp correction, written as arithmetic so the body
/// stays branch-free for `fuse_boxing_alloc`.
#[inline(never)]
pub fn _int_isqrt(n: i64) -> PyResult {
    let guess = unsafe { (n as f64).sqrt().to_int_unchecked::<i64>() };
    let too_high = i64::from(guess > n / guess);
    let too_low = i64::from(guess < n / (guess + 1));
    let root = guess - too_high + too_low;
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: root,
    }) as PyObjectRef)
}

/// interp_math.py `isclose` with both keyword tolerances defaulted,
/// after the walker pins finite operands.  `|` rather than `||` so
/// the comparison is one expression.
#[inline(never)]
pub fn _float_isclose(a: f64, b: f64) -> PyResult {
    let diff = (b - a).abs();
    let close = (a == b) | (diff <= (1e-9 * b).abs()) | (diff <= (1e-9 * a).abs());
    Ok(w_bool_from(close))
}

/// floatobject.py `descr_abs`: `W_FloatObject(abs(self.floatval))`.
#[inline(never)]
pub fn _float_abs(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: x.abs(),
    }) as PyObjectRef)
}

/// intobject.py `descr_abs` after `ovfcheck(abs(a))`.
/// `0 - x` on the negative arm, matching [`_int_neg`], so the leaf records
/// `int_sub` rather than a residual intrinsic.
#[inline(never)]
pub fn _int_abs(x: i64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: if x < 0 { 0i64.wrapping_sub(x) } else { x },
    }) as PyObjectRef)
}

/// floatobject.py `descr_neg`: `W_FloatObject(-self.floatval)`.
#[inline(never)]
pub(crate) fn _float_neg(x: f64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: -x,
    }) as PyObjectRef)
}

/// Unary negation.
///
/// `inline(never)` is load-bearing: rustc otherwise folds this body into its
/// one-call wrapper and the codewriter never mints the graph named by
/// `flatten.rs build_orthodox_inline_call_r_r`.
#[inline(never)]
pub fn neg(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) =
            pyre_object::with_roots!(a => try_numeric_unaryop_override(a, UnaryDunder::Neg))?
        {
            return Ok(result);
        }
        neg_inner(a)
    }
}

/// [`neg`] past its `__neg__` override probe.
///
/// Split out so that a trace can descend this body rather than re-emit it by
/// hand.  The probe is what stops such a descent: its
/// `needs_numeric_unaryop_dispatch` is `dont_look_inside` and is the second
/// operation the body executes.  A caller that has already proven an exact
/// builtin receiver cannot take it, so entering here records the arm that
/// receiver selects and nothing else.  Every other caller reaches this through
/// [`neg`] and is unaffected, except the `int`/`float`/`complex` `__neg__`
/// slots, which name this body directly because an unbound slot must not
/// dispatch.
///
/// Unlike [`invert_inner`] this keeps the `bool` operand: `neg` has no separate
/// bool slot to leave behind, and the integer arm below already answers `True`
/// and `False` through [`int_value`].
pub fn neg_inner(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) || is_bool(a) {
            let v = int_value(a);
            return match v.checked_neg() {
                Some(_) => _int_neg(v),
                None => Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                    bigint_neg(&BigInt::from(v)),
                )),
            };
        }
        if is_long(a) {
            return Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                bigint_neg(w_long_get_value(a)),
            ));
        }
        if is_float(a) {
            return _float_neg(w_float_get_value(a));
        }
        if is_complex(a) {
            return complex_neg(a);
        }
        // Instance __neg__
        if let Some(result) = pyre_object::with_roots!(a => try_lookup_unaryop(a, "__neg__"))? {
            return Ok(result);
        }
        Err(bad_operand_type("unary -", a))
    }
}

const BOOL_INVERT_DEPRECATION_TEXT: &str = "Bitwise inversion '~' on bool is deprecated and will be removed in \
Python 3.16. This returns the bitwise inversion of the underlying int \
object and is usually not what you expect from negating a bool. \
Use the 'not' operator for boolean negation or ~int(x) if you really want \
the bitwise inversion of the underlying int.";

/// The wrapped form of [`BOOL_INVERT_DEPRECATION_TEXT`]: `~True` in a loop
/// issues the same message on every iteration, and the registry lookup that
/// deduplicates it hashes and compares the message each time.
///
/// `dont_look_inside` because the cell is a `static` of a host type, which
/// the front-end cannot lift: reaching it from a lifted body fails the
/// callee's own lift and, transitively, every caller's — [`invert`] then has
/// no jitcode, the tracer has to residualise the whole `~` opcode, and a
/// virtualizable forced inside that residual aborts the trace
/// (`ABORT_ESCAPE`), so a `~bool` in a hot loop stops the loop compiling at
/// all.  Hiding the static behind an opaque accessor keeps the body out of
/// the lift, exactly as `w_dict_new` does for its host `IndexMap`.
#[majit_macros::dont_look_inside]
pub(crate) fn bool_invert_deprecation_text() -> PyObjectRef {
    static CELL: crate::warn::PrebuiltText = crate::warn::PrebuiltText::new();
    CELL.get(BOOL_INVERT_DEPRECATION_TEXT)
}

/// Unary bitwise inversion.
///
/// `inline(never)` is load-bearing: rustc otherwise folds this body into its
/// one-call wrapper and the codewriter never mints the graph named by
/// `flatten.rs build_orthodox_inline_call_r_r`.
#[inline(never)]
pub fn invert(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) =
            pyre_object::with_roots!(a => try_numeric_unaryop_override(a, UnaryDunder::Invert))?
        {
            return Ok(result);
        }
        if is_bool(a) {
            // CPython 3.14 `Objects/boolobject.c:bool_invert`.  The bundled
            // PyPy source inherits `W_IntObject.descr_invert`; 3.14 inserts
            // this warning-bearing bool slot before the integer inversion.
            pyre_object::with_roots!(a => crate::warn::warn_category_w(
                bool_invert_deprecation_text(),
                "DeprecationWarning",
                2
            ))?;
            return Ok(w_int_new(!int_value(a)));
        }
        invert_inner(a)
    }
}

/// [`invert`] past its `__invert__` override probe and its bool slot.
///
/// Split out so that a trace can descend this body rather than re-emit it by
/// hand.  The two arms left behind are what stop such a descent: the probe's
/// `needs_numeric_unaryop_dispatch` is `dont_look_inside` and is the second
/// operation the body executes, and the bool slot's deprecation warning
/// reaches `lookup_exc_class`.  A caller that has already proven an exact
/// `int` receiver takes neither, so entering here records the integer arm
/// alone.  Every other caller reaches this through [`invert`] and is
/// unaffected, except `int.__invert__`, which names this body directly
/// because an unbound slot must not dispatch.
/// intobject.py `descr_invert`: `wrapint(~self.intval)`.
#[inline(never)]
pub(crate) fn _int_invert(x: i64) -> PyResult {
    Ok(pyre_object::lltype::malloc_typed_managed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: !x,
    }) as PyObjectRef)
}

pub fn invert_inner(mut a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) {
            return _int_invert(int_value(a));
        }
        if is_long(a) {
            return Ok(w_long_new(bigint_invert(w_long_get_value(a))));
        }
        if let Some(result) = pyre_object::with_roots!(a => try_lookup_unaryop(a, "__invert__"))? {
            return Ok(result);
        }
        Err(bad_operand_type("unary ~", a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_pointer_width = "64")]
    fn decoded_bigint_result(value: pyre_object::longobject::JitBigIntResult) -> *mut BigInt {
        value
    }

    #[cfg(target_pointer_width = "32")]
    fn decoded_bigint_result(value: pyre_object::longobject::JitBigIntResult) -> *mut BigInt {
        value as usize as *mut BigInt
    }

    /// Test projection for assertions that accept either compact or long
    /// integer results. Production consumers borrow W_LongObject payloads.
    unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
        if is_bool(obj) {
            BigInt::from(w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }

    fn assert_compare_bool(a: PyObjectRef, b: PyObjectRef, op: CompareOp, expected: bool) {
        let result = compare(a, b, op).unwrap();
        assert_eq!(unsafe { w_bool_get_value(result) }, expected);
    }

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn ll_2_int_floordiv_and_mod_match_truncating_machine_ops() {
        let samples = [
            (7, 3),
            (-7, 3),
            (7, -3),
            (-7, -3),
            (1, 1),
            (-1, 1),
            (0, 1),
            (0, -1),
            (i64::MIN, 1),
            (i64::MIN, -1),
            (i64::MIN, -2),
            (i64::MAX, -1),
            (5, i64::MIN),
        ];
        for (x, y) in samples {
            assert_eq!(
                _ll_2_int_floordiv(x, y),
                x.wrapping_div(y),
                "floordiv {x} {y}"
            );
            assert_eq!(_ll_2_int_mod(x, y), x.wrapping_rem(y), "mod {x} {y}");
        }
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_specialised_tuple_equality_uses_same_class_raw_slots() {
        // The `_oo` case interns its strs, which reaches a str-keyed dict.
        crate::test_hooks::install_hash_hook();
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            w_tuple_new(vec![w_int_new(1), w_int_new(3)]),
            CompareOp::Eq,
            false,
        );
        assert_compare_bool(
            pyre_object::makespecialisedtuple2(w_float_new(1.0), w_float_new(2.0)),
            pyre_object::makespecialisedtuple2(w_float_new(1.0), w_float_new(2.0)),
            CompareOp::Eq,
            true,
        );

        let nan = w_float_new(f64::NAN);
        let nan_pair = pyre_object::makespecialisedtuple2(nan, nan);
        assert_compare_bool(nan_pair, nan_pair, CompareOp::Eq, true);

        assert_compare_bool(
            w_tuple_new(vec![w_str_new("a"), w_str_new("b")]),
            w_tuple_new(vec![w_str_new("a"), w_str_new("b")]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            pyre_object::w_tuple_new_array_backed(vec![w_int_new(1), w_int_new(2)]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![
                w_str_new("i"),
                w_str_new("marker"),
                w_str_new("odd_only"),
            ]),
            w_tuple_new(vec![
                w_str_new("i"),
                w_str_new("marker"),
                w_str_new("odd_only"),
            ]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_str_new("i"), w_str_new("marker")]),
            w_tuple_new(vec![
                w_str_new("i"),
                w_str_new("marker"),
                w_str_new("odd_only"),
            ]),
            CompareOp::Eq,
            false,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_str_new("a")]),
            w_tuple_new(vec![w_str_new("b")]),
            CompareOp::Lt,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_tuple_new(vec![w_int_new(1)])]),
            w_tuple_new(vec![w_tuple_new(vec![w_int_new(1)])]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_tuple_new(vec![w_int_new(1)])]),
            w_tuple_new(vec![w_tuple_new(vec![w_int_new(2)])]),
            CompareOp::Eq,
            false,
        );
        let nan_a = w_float_new(f64::NAN);
        let nan_b = w_float_new(f64::NAN);
        assert_compare_bool(
            w_tuple_new(vec![nan_a]),
            w_tuple_new(vec![nan_b]),
            CompareOp::Eq,
            false,
        );
        assert_compare_bool(
            w_tuple_new(vec![nan_a]),
            w_tuple_new(vec![nan_a]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_float_new(0.0)]),
            w_tuple_new(vec![w_float_new(-0.0)]),
            CompareOp::Eq,
            true,
        );
    }

    #[test]
    fn test_long_pos_preserves_exact_object_identity() {
        let value = BigInt::from(1).lshift(80).unwrap();
        let a = w_long_new(value);
        let result = pos(a).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(floordiv(a, b).is_err());
    }

    #[test]
    fn test_int_floordiv_and_mod_bounce_min_overflow_to_rbigint() {
        // intobject.py uses `ovfcheck` for both operations.  On a signed
        // machine word MIN / -1 is their sole non-zero-divisor overflow.
        let a = w_int_new(i64::MIN);
        let b = w_int_new(-1);
        let q = floordiv(a, b).unwrap();
        let r = mod_(a, b).unwrap();
        unsafe {
            assert_eq!(as_bigint(q), BigInt::from(i64::MAX).int_add(1));
            assert_eq!(as_bigint(r), BigInt::zero());
        }
    }

    #[test]
    fn test_long_floordiv_and_mod_keep_python_sign_rules() {
        let magnitude = BigInt::one().lshift(80).unwrap().int_add(13);
        for (lhs, rhs) in [
            (magnitude.clone(), BigInt::fromint(10)),
            (magnitude.neg(), BigInt::fromint(10)),
            (magnitude.clone(), BigInt::fromint(-10)),
            (magnitude.neg(), BigInt::fromint(-10)),
        ] {
            let a = w_long_new(lhs.clone());
            let b = w_long_new(rhs.clone());
            let q = floordiv(a, b).unwrap();
            let r = mod_(a, b).unwrap();
            let q = unsafe { as_bigint(q) };
            let r = unsafe { as_bigint(r) };
            assert!(q.mul(&rhs).add(&r).eq(&lhs));
            assert!(r.get_sign() == 0 || r.get_sign() == rhs.get_sign());
        }
    }

    #[test]
    fn test_truthiness() {
        assert!(is_true(w_int_new(1)).unwrap());
        assert!(!is_true(w_int_new(0)).unwrap());
        assert!(!is_true(w_none()).unwrap());
        assert!(is_true(w_bool_from(true)).unwrap());
        assert!(!is_true(w_bool_from(false)).unwrap());
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_sub_overflow() {
        let a = w_int_new(i64::MIN);
        let b = w_int_new(1);
        let result = sub(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MIN) - BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_mul_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(2);
        let result = mul(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) * BigInt::from(2)
            );
        }
    }

    #[test]
    fn test_long_add() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(100);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(101)
            );
        }
    }

    #[test]
    fn test_long_add_keeps_long_when_fits() {
        // long + int whose sum fits back in i64 stays a W_LongObject: `newlong`
        // never demotes (withsmalllong=False), so a shrunk long keeps the long
        // representation.
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(-1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_mixed_long_int_operations_match_rbigint_int_specializations() {
        // pypy/objspace/std/longobject.py:_make_generic_descr_binop routes
        // mixed W_LongObject/W_IntObject operations through rbigint.int_*.
        // Use a multi-digit value and both signs so this covers paths where
        // materializing the small operand as a second rbigint would select a
        // different internal algorithm.
        let magnitude = BigInt::one()
            .lshift(190)
            .unwrap()
            .int_add(0x0123_4567_89ab_cdef);
        for value in [magnitude.clone(), magnitude.neg()] {
            for machine in [-17_i64, -1, 0, 1, 37, i64::MIN, i64::MAX] {
                let expected_add = value.int_add(machine);
                let expected_mul = value.int_mul(machine);
                let expected_and = value.int_and_(machine);
                let expected_or = value.int_or_(machine);
                let expected_xor = value.int_xor(machine);
                let expected_sub = value.int_sub(machine);
                let expected_rsub = BigInt::fromint(machine).sub(&value);

                for result in [
                    add(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    add(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_add));
                }
                for result in [
                    mul(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    mul(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_mul));
                }
                for result in [
                    and_(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    and_(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_and));
                }
                for result in [
                    or_(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    or_(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_or));
                }
                for result in [
                    xor(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    xor(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_xor));
                }

                let result = sub(w_long_new(value.clone()), w_int_new(machine)).unwrap();
                assert!(unsafe { as_bigint(result) }.eq(&expected_sub));
                let reflected = sub(w_int_new(machine), w_long_new(value.clone())).unwrap();
                assert!(unsafe { as_bigint(reflected) }.eq(&expected_rsub));
            }
        }

        // W_BoolObject subclasses W_IntObject upstream, so bool operands use
        // the same machine-int specialization.
        let expected = magnitude.int_add(1);
        for result in [
            add(w_long_new(magnitude.clone()), w_bool_from(true)).unwrap(),
            add(w_bool_from(true), w_long_new(magnitude.clone())).unwrap(),
        ] {
            assert!(unsafe { as_bigint(result) }.eq(&expected));
        }
    }

    #[test]
    fn test_jit_w_long_floordiv_mod_raw() {
        // Both operands out of i64 range → long // long / long % long fast path
        // payload helpers return a bare `*mut BigInt` of the quotient/remainder.
        let x = BigInt::from(i64::MAX) * BigInt::from(1000) + BigInt::from(7);
        let y = BigInt::from(i64::MAX) + BigInt::from(3);
        let a = w_long_new(x.clone());
        let b = w_long_new(y.clone());
        unsafe {
            let d = jit_w_long_floordiv_raw(a, b) as *mut BigInt;
            assert_eq!(*d, x.floordiv(&y).expect("test divisor is nonzero"));
            let m = jit_w_long_mod_raw(a, b) as *mut BigInt;
            assert_eq!(*m, x.r#mod(&y).expect("test divisor is nonzero"));
        }
    }

    /// A shift count wider than a machine word must shift by the count, not by
    /// its low word.  `usize` is 32 bits on wasm32 while `intval` is `i64` on
    /// every target, so `2**33` reaches these residuals as an ordinary
    /// `W_IntObject`; narrowing it first answered `x >> 2**33` with `x`.
    ///
    /// This discriminates only on a 32-bit target — on a 64-bit host the
    /// narrowing that used to be here was lossless, so both spellings pass.
    #[test]
    fn test_rbigint_shift_residuals_take_a_count_wider_than_a_machine_word() {
        let wide = 1i64 << 33;
        let positive = BigInt::one().lshift(200).unwrap().int_add(12345);
        let negative = positive.neg();
        let p = pyre_object::longobject::alloc_bigint_nursery(positive);
        let n = pyre_object::longobject::alloc_bigint_nursery(negative);

        // Arithmetic right shift: to zero, and to -1 for a negative operand.
        assert!(unsafe { &*decoded_bigint_result(jit_bigint_shr(p, wide)) }.get_sign() == 0);
        assert_eq!(
            unsafe { &*decoded_bigint_result(jit_bigint_shr(n, wide)) }.toint(),
            Ok(-1)
        );
        // `jit_bigint_shl` takes the identical route (`lshift(b)`), but a count
        // this wide asks it for a result of hundreds of megabytes, so it is not
        // asserted here.
    }

    #[test]
    fn test_rbigint_residuals_preserve_upstream_reference_fast_paths() {
        let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
        let larger = magnitude.lshift(70).unwrap();
        let one = pyre_object::longobject::alloc_bigint_nursery(BigInt::one());
        let zero = pyre_object::longobject::alloc_bigint_nursery(BigInt::zero());
        let a = pyre_object::longobject::alloc_bigint_nursery(magnitude);
        let b = pyre_object::longobject::alloc_bigint_nursery(larger);

        assert_eq!(decoded_bigint_result(jit_bigint_add(a, zero)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_sub(a, zero)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_add(a, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_sub(a, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_mul(a, 1)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_div_floor(a, 1)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_pow_nomod(a, one)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_pow_nomod(a, 1)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_lshift_count(a, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_shl(a, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_shr(a, 0)), a);

        // `_divrem` returns its literal input remainder when its early
        // magnitude test succeeds. Floored modulo retains it for equal signs
        // and a non-one-digit divisor.
        assert_eq!(decoded_bigint_result(jit_bigint_rem(a, b)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_mod_floor(a, b)), a);

        // `lshift` returns even a non-canonical zero handle unchanged.
        let fresh_zero = majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting(unsafe {
            (&*zero).clone()
        });
        assert_ne!(fresh_zero, zero);
        assert_eq!(
            decoded_bigint_result(jit_bigint_lshift_count(fresh_zero, 37)),
            fresh_zero
        );
    }

    #[test]
    fn test_machine_int_host_seams_use_dedicated_rbigint_legs() {
        let value = BigInt::one().lshift(130).unwrap().int_add(17);
        let quotient = bigint_int_floordiv_nonzero(&value, 3);
        assert!(quotient.eq(&value.int_floordiv(3).unwrap()));

        let power = bigint_int_pow_nomod(&value, 3).expect("small exponent");
        assert!(power.eq(&value.int_pow(3, None).unwrap()));
    }

    #[test]
    fn test_bigint_truediv_exponent() {
        // Regression: the exponent assembly carried a spurious `+ 1` that
        // doubled every quotient (equal operands gave 2.0, not 1.0).
        let big = BigInt::from(10u64).int_pow(40, None).unwrap();
        assert_eq!(bigint_truediv(&big, &big).unwrap(), 1.0);
        let a = BigInt::from(10u64).int_pow(60, None).unwrap();
        let b = BigInt::from(2) * BigInt::from(10u64).int_pow(59, None).unwrap();
        assert_eq!(bigint_truediv(&a, &b).unwrap(), 5.0);
        assert_eq!(bigint_truediv(&a.neg(), &b).unwrap(), -5.0);
        assert_eq!(bigint_truediv(&a, &b.neg()).unwrap(), -5.0);
        assert!(bigint_truediv(&a, &BigInt::from(0)).is_err());
    }

    #[test]
    fn test_machine_int_truediv_uses_rbigint_rounding_past_binary64_mantissa() {
        let result = truediv_builtin(w_int_new(63_050_394_783_186_940), w_int_new(7)).unwrap();
        unsafe {
            assert_eq!(w_float_get_value(result), 9_007_199_254_740_991.0);
        }
    }

    #[test]
    fn test_int_truediv_leaf_matches_truediv_for_exact_machine_ints() {
        // intobject.py `_truediv`: both operands fit the binary64 mantissa,
        // so the leaf is `newfloat(float(x)/float(y))`.
        unsafe {
            let direct = int_truediv(w_int_new(7), w_int_new(2)).unwrap();
            let leaf = _truediv(7, 2).unwrap();
            let via_builtin = truediv_builtin(w_int_new(7), w_int_new(2)).unwrap();
            assert_eq!(w_float_get_value(direct), 3.5);
            assert_eq!(w_float_get_value(leaf), 3.5);
            assert_eq!(w_float_get_value(via_builtin), 3.5);
            assert!(int_truediv(w_int_new(1), w_int_new(0)).is_err());
            assert!(_truediv(1, 0).is_err());
        }
    }

    #[test]
    fn test_bigint_truediv_sticky_rounding() {
        // a ≫ b (shift < 0): low bits of `a` that a right-shift would discard
        // must still steer round-half-to-even. b is odd and > 2^63 so the path
        // exercises the bigint divide, not i64.
        let b = BigInt::from(2u64).int_pow(64, None).unwrap() + BigInt::from(1); // 2^64 + 1, odd
        let two55 = 2.0_f64.powi(55);
        // a_exact/b == 2^55 + 4 exactly: a half-ULP tie between 2^55 and 2^55+8.
        // Round-half-to-even → 2^55 (its low mantissa bit is 0).
        let a_exact = (BigInt::from(2u64).int_pow(53, None).unwrap() + BigInt::from(1))
            * BigInt::from(4)
            * &b;
        assert_eq!(bigint_truediv(&a_exact, &b).unwrap(), two55);
        // +1 makes the true quotient exceed the tie → sticky → round up to 2^55+8.
        assert_eq!(
            bigint_truediv(&a_exact.int_add(1), &b).unwrap(),
            two55 + 8.0
        );
        // -1 drops it just below the tie → round down to 2^55.
        assert_eq!(bigint_truediv(&a_exact.int_sub(1), &b).unwrap(), two55);
    }

    #[test]
    fn test_bigint_truediv_subnormal() {
        // Subnormal-range results must match what `math.ldexp` produces; a lone
        // `2f64.powi` underflows the scale and loses them. Expected bit patterns
        // are CPython's.
        let p = |e: u32| BigInt::from(2u64).int_pow(e as i64, None).unwrap();
        // 1 / 2^1030 == 2^-1030 (exact subnormal)
        assert_eq!(
            bigint_truediv(&BigInt::from(1), &p(1030)).unwrap(),
            f64::from_bits(0x0000_1000_0000_0000)
        );
        // 7 / 2^1074 == 7 * 2^-1074 (seven smallest subnormals)
        assert_eq!(
            bigint_truediv(&BigInt::from(7), &p(1074)).unwrap(),
            f64::from_bits(0x0000_0000_0000_0007)
        );
        // (2^53+1) / (2^1075+7) rounds across the subnormal/normal boundary to 2^-1022
        assert_eq!(
            bigint_truediv(&p(53).int_add(1), &p(1075).int_add(7)).unwrap(),
            f64::from_bits(0x0010_0000_0000_0000)
        );
        // sign preserved
        assert_eq!(
            bigint_truediv(&BigInt::from(-1), &p(1030)).unwrap(),
            -f64::from_bits(0x0000_1000_0000_0000)
        );
    }

    #[test]
    fn test_jit_w_long_shift_truediv_raw() {
        let x = BigInt::from(i64::MAX) * BigInt::from(1000) + BigInt::from(7);
        let a = w_long_new(x.clone());
        let two = w_long_new(BigInt::from(2));
        let y = BigInt::from(i64::MAX) + BigInt::from(3);
        let b = w_long_new(y.clone());
        unsafe {
            let l = jit_w_long_lshift_raw(a, two) as *mut BigInt;
            assert_eq!(*l, bigint_lshift(&x, 2).unwrap());
            let r = jit_w_long_rshift_raw(a, two) as *mut BigInt;
            assert_eq!(*r, bigint_rshift(&x, 2));
            // true-divide returns the f64 quotient directly (CallPureF).
            let f = jit_w_long_truediv_raw(a, b);
            assert_eq!(f, bigint_truediv(&x, &y).unwrap());
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = neg(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_long_neg_and_zero_count_shifts_preserve_rpython_payload_identity() {
        unsafe {
            // rbigint.neg always constructs a fresh handle, including zero.
            let zero = w_long_new(BigInt::zero());
            let negated = neg(zero).unwrap();
            assert_ne!(negated, zero);
            assert_ne!(w_long_get_raw_value(negated), w_long_get_raw_value(zero));
            assert!(w_long_get_value(negated).is_zero());

            let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
            let value = w_long_new(magnitude);

            // rbigint returns `self`; `W_LongObject` / `newlong` wraps that payload.
            // around that exact translated payload.
            let shifted_left = long_int_lshift(value, 0).unwrap();
            assert_ne!(shifted_left, value);
            assert_eq!(
                w_long_get_raw_value(shifted_left),
                w_long_get_raw_value(value)
            );
            let shifted_right = long_int_rshift(value, 0).unwrap();
            assert_ne!(shifted_right, value);
            assert_eq!(
                w_long_get_raw_value(shifted_right),
                w_long_get_raw_value(value)
            );

            let zero_shifted = long_int_lshift(zero, 5).unwrap();
            assert_ne!(zero_shifted, zero);
            assert_eq!(
                w_long_get_raw_value(zero_shifted),
                w_long_get_raw_value(zero)
            );

            // Overflowing count is the W_LongObject-level early return.
            let huge_count = w_long_new(BigInt::one().lshift(100).unwrap());
            assert_eq!(long_lshift(zero, huge_count).unwrap(), zero);
        }
    }

    #[test]
    fn test_long_arithmetic_rewraps_rbigint_operand_fast_paths() {
        unsafe {
            let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
            let value = w_long_new(magnitude.translated_alias());
            let zero = w_long_new(BigInt::zero());
            let payload = w_long_get_raw_value(value);

            for result in [
                long_add(value, w_int_new(0)).unwrap(),
                long_add(w_int_new(0), value).unwrap(),
                long_add(value, zero).unwrap(),
                long_add(zero, value).unwrap(),
                long_sub(value, w_int_new(0)).unwrap(),
                long_sub(value, zero).unwrap(),
                long_mul(value, w_int_new(1)).unwrap(),
                long_mul(w_bool_from(true), value).unwrap(),
                long_int_floordiv(value, 1).unwrap(),
                long_pow(value, w_int_new(1)).unwrap(),
            ] {
                assert_ne!(result, value);
                assert_eq!(w_long_get_raw_value(result), payload);
            }

            // Two nonzero operands: `bigint_add` allocates the sum.
            let other = w_long_new(magnitude.lshift(3).unwrap());
            let other_payload = w_long_get_raw_value(other);
            let sum = long_add(value, other).unwrap();
            let sum_payload = w_long_get_raw_value(sum);
            assert_ne!(sum_payload, payload);
            assert_ne!(sum_payload, other_payload);
            assert!(w_long_get_value(sum).eq(&magnitude.add(&magnitude.lshift(3).unwrap())));

            // `rbigint.int_floordiv(1)` returns `self`, but the long-divisor
            // path goes through `rbigint.divmod` -> `int_divmod`, whose
            // quotient is freshly constructed.
            let long_divisor_result = long_floordiv(value, w_long_new(BigInt::one())).unwrap();
            assert_ne!(w_long_get_raw_value(long_divisor_result), payload);
            assert!(w_long_get_value(long_divisor_result).eq(w_long_get_value(value)));

            // These upstream methods check zero before their alias shortcut
            // and return canonical NULLRBIGINT, not a fresh zero operand.
            let fresh_zero = neg(zero).unwrap();
            assert_ne!(w_long_get_raw_value(fresh_zero), w_long_get_raw_value(zero));
            for result in [
                long_add(fresh_zero, w_int_new(0)).unwrap(),
                long_mul(fresh_zero, w_int_new(1)).unwrap(),
                long_pow(fresh_zero, w_int_new(1)).unwrap(),
            ] {
                assert_eq!(w_long_get_raw_value(result), w_long_get_raw_value(zero));
            }

            let larger = w_long_new(magnitude.lshift(70).unwrap());
            let remainder = long_mod(value, larger).unwrap();
            assert_eq!(w_long_get_raw_value(remainder), payload);
            let pair = integer_divmod_pair(value, larger).unwrap();
            let pair_remainder = w_tuple_getitem(pair, 1).expect("remainder");
            assert_eq!(w_long_get_raw_value(pair_remainder), payload);

            let pair = integer_divmod_pair(value, w_int_new(1)).unwrap();
            let pair_quotient = w_tuple_getitem(pair, 0).expect("quotient");
            assert_ne!(w_long_get_raw_value(pair_quotient), payload);
            assert!(w_long_get_value(pair_quotient).eq(w_long_get_value(value)));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };

        let magnitude = BigInt::one().lshift(190).unwrap();
        let a = w_long_new(magnitude.int_add(1));
        let b = w_long_new(magnitude.int_add(2));
        for (op, expected) in [
            (CompareOp::Lt, true),
            (CompareOp::Le, true),
            (CompareOp::Gt, false),
            (CompareOp::Ge, false),
            (CompareOp::Eq, false),
            (CompareOp::Ne, true),
        ] {
            let result = compare(a, b, op).unwrap();
            assert_eq!(unsafe { w_bool_get_value(result) }, expected);
        }
    }

    #[test]
    fn test_complex_long_comparison_is_exact() {
        let magnitude = BigInt::one().lshift(100).unwrap();
        let complex = w_complex_new(2_f64.powi(100), 0.0);
        let equal = compare(complex, w_long_new(magnitude.clone()), CompareOp::Eq).unwrap();
        assert!(unsafe { w_bool_get_value(equal) });

        let unequal = compare(complex, w_long_new(magnitude.int_add(1)), CompareOp::Eq).unwrap();
        assert!(!unsafe { w_bool_get_value(unequal) });
    }

    #[test]
    fn test_mixed_long_int_comparisons_match_rbigint_int_specializations() {
        let magnitude = BigInt::one().lshift(190).unwrap().int_add(37);
        for value in [magnitude.clone(), magnitude.neg()] {
            for machine in [-1_i64, 0, 1, i64::MIN, i64::MAX] {
                for op in [
                    CompareOp::Lt,
                    CompareOp::Le,
                    CompareOp::Gt,
                    CompareOp::Ge,
                    CompareOp::Eq,
                    CompareOp::Ne,
                ] {
                    let expected = match op {
                        CompareOp::Lt => value.int_lt(machine),
                        CompareOp::Le => value.int_le(machine),
                        CompareOp::Gt => value.int_gt(machine),
                        CompareOp::Ge => value.int_ge(machine),
                        CompareOp::Eq => value.int_eq(machine),
                        CompareOp::Ne => value.int_ne(machine),
                    };
                    let result =
                        compare(w_long_new(value.clone()), w_int_new(machine), op).unwrap();
                    assert_eq!(unsafe { w_bool_get_value(result) }, expected);

                    let expected_reversed = match op {
                        CompareOp::Lt => value.int_gt(machine),
                        CompareOp::Le => value.int_ge(machine),
                        CompareOp::Gt => value.int_lt(machine),
                        CompareOp::Ge => value.int_le(machine),
                        CompareOp::Eq => value.int_eq(machine),
                        CompareOp::Ne => value.int_ne(machine),
                    };
                    let result =
                        compare(w_int_new(machine), w_long_new(value.clone()), op).unwrap();
                    assert_eq!(unsafe { w_bool_get_value(result) }, expected_reversed);
                }
            }
        }

        for (op, expected) in [
            (CompareOp::Gt, true),
            (CompareOp::Ge, true),
            (CompareOp::Eq, false),
            (CompareOp::Ne, true),
        ] {
            let result = compare(w_long_new(magnitude.clone()), w_bool_from(true), op).unwrap();
            assert_eq!(unsafe { w_bool_get_value(result) }, expected);
        }
    }

    #[test]
    fn test_long_truthiness() {
        assert!(is_true(w_long_new(BigInt::from(i64::MAX) + BigInt::from(1))).unwrap());
        assert!(!is_true(w_long_new(BigInt::from(0))).unwrap());
    }

    #[test]
    fn test_numeric_divmod_computes_matching_pair_for_all_integer_shapes() {
        let big_positive = BigInt::one().lshift(190).unwrap().int_add(37);
        let big_negative = big_positive.neg();
        for (lhs, rhs) in [
            (big_positive.clone(), BigInt::from(-257)),
            (big_negative.clone(), BigInt::from(257)),
            (BigInt::from(-257), big_positive.clone()),
            (BigInt::from(i64::MIN), BigInt::from(-1)),
        ] {
            let lhs_obj = if lhs.toint().is_ok() {
                w_int_new(lhs.toint().unwrap())
            } else {
                w_long_new(lhs.clone())
            };
            let rhs_obj = if rhs.toint().is_ok() {
                w_int_new(rhs.toint().unwrap())
            } else {
                w_long_new(rhs.clone())
            };
            let expected = lhs.divmod(&rhs).unwrap();
            let pair = divmod(lhs_obj, rhs_obj).unwrap();
            let q = unsafe { w_tuple_getitem(pair, 0).expect("quotient") };
            let r = unsafe { w_tuple_getitem(pair, 1).expect("remainder") };
            assert!(unsafe { as_bigint(q) }.eq(&expected.0));
            assert!(unsafe { as_bigint(r) }.eq(&expected.1));
        }
    }

    #[test]
    fn test_int_pow() {
        let result = pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(2).int_pow(63, None).unwrap()
            );
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_long_pow_negative_exponent_rejects_float_overflow() {
        let huge = BigInt::one().lshift(2000).unwrap();
        let error = pow(w_long_new(huge), w_int_new(-1)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::OverflowError);
        assert_eq!(error.message_text(), "int too large to convert to float");

        let huge_negative_exponent = BigInt::one().lshift(2000).unwrap().neg();
        let error = pow(w_int_new(1), w_long_new(huge_negative_exponent)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::OverflowError);
        assert_eq!(error.message_text(), "int too large to convert to float");
    }

    #[test]
    fn test_int_lshift() {
        let result = lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
        let min = lshift(w_int_new(-1), w_int_new(63)).unwrap();
        unsafe { assert_eq!(w_int_get_value(min), i64::MIN) };
        let positive = lshift(w_int_new(1), w_int_new(63)).unwrap();
        unsafe {
            assert!(is_long(positive));
            assert_eq!(*w_long_get_value(positive), BigInt::from(1) << 63);
        }
        let error = lshift(w_int_new(1), w_int_new(i64::MAX)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::MemoryError);
    }

    #[test]
    fn test_int_rshift() {
        let result = rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_long_shift_mixed_operands_and_large_counts() {
        unsafe {
            let base = BigInt::one().lshift(70).unwrap();
            let shifted = lshift(w_long_new(base.clone()), w_long_new(BigInt::from(5))).unwrap();
            assert!(is_long(shifted));
            assert_eq!(*w_long_get_value(shifted), base.lshift(5).unwrap());

            let mixed = lshift(w_int_new(3), w_long_new(BigInt::from(65))).unwrap();
            assert!(is_long(mixed));
            assert_eq!(
                *w_long_get_value(mixed),
                BigInt::from(3).lshift(65).unwrap()
            );

            let negative = BigInt::from(-3).lshift(70).unwrap();
            let right = rshift(w_long_new(negative.clone()), w_int_new(69)).unwrap();
            assert!(is_long(right));
            assert_eq!(
                *w_long_get_value(right),
                negative.rshift(69, false).unwrap()
            );

            let enormous_count = BigInt::one().lshift(70).unwrap();
            let saturated =
                rshift(w_long_new(negative), w_long_new(enormous_count.clone())).unwrap();
            assert_eq!(w_int_get_value(saturated), -1);
            let zero = lshift(w_long_new(BigInt::zero()), w_long_new(enormous_count)).unwrap();
            assert!(is_long(zero));
            assert_eq!(w_long_get_value(zero).get_sign(), 0);
        }
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(rshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(lshift(w_long_new(BigInt::from(1)), w_long_new(BigInt::from(-1))).is_err());
        assert!(rshift(w_long_new(BigInt::from(1)), w_long_new(BigInt::from(-1))).is_err());
    }

    #[test]
    fn test_long_repeat_count_bounds() {
        unsafe {
            let huge = BigInt::one().lshift(100).unwrap();
            let negative_error = repeat_count(w_long_new(huge.neg())).unwrap_err();
            assert_eq!(negative_error.kind, PyErrorKind::OverflowError);
            let error = repeat_count(w_long_new(huge)).unwrap_err();
            assert_eq!(error.kind, PyErrorKind::OverflowError);
        }
    }

    #[test]
    fn test_int_bitand() {
        let result = and_(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn same_unoverridden_rpy_type_rejects_user_int_subclass() {
        unsafe {
            let exact = w_int_new(7);
            let user = w_int_subclass_new(3);
            // `get_instantiate` is still null in this unit-test process;
            // retag `w_class` the way `tag_subclass_instance` would.
            (*user).w_class = &FLOAT_TYPE as *const PyType as PyObjectRef;
            // `W_IntObjectUser` is its own interp-level class.
            assert!(!same_rpy_type(exact, user));
            assert!(!same_unoverridden_rpy_type(exact, user));
            assert!(same_unoverridden_rpy_type(exact, w_int_new(3)));
        }
    }

    #[test]
    fn test_bool_bitwise_shortcuts_return_bool_singletons() {
        unsafe {
            let tand = and_(w_bool_from(true), w_bool_from(true)).unwrap();
            let tand_f = and_(w_bool_from(true), w_bool_from(false)).unwrap();
            let tor = or_(w_bool_from(false), w_bool_from(true)).unwrap();
            let txor = xor(w_bool_from(true), w_bool_from(false)).unwrap();
            assert!(std::ptr::eq(tand, w_bool_from(true)));
            assert!(std::ptr::eq(tand_f, w_bool_from(false)));
            assert!(std::ptr::eq(tor, w_bool_from(true)));
            assert!(std::ptr::eq(txor, w_bool_from(true)));
        }
    }

    #[test]
    fn test_int_bitor() {
        let result = or_(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = xor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = and_(a, b).unwrap();
        // long & int keeps a W_LongObject even when the result fits (newlong).
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(0));
        }
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }
}
