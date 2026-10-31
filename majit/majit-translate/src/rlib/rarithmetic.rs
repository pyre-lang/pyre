//! RPython `rpython/rlib/rarithmetic.py` — integer arithmetic helpers.
//!
//! Only the subset consumed by the annotator port lands here. The
//! full RPython rarithmetic module covers ~700 LOC of overflow-safe
//! operators, r_uint / r_longlong / r_singlefloat types, ovfcheck
//! decorators, etc. Those land as their call sites in downstream
//! ports come online.

use crate::annotator::model::KnownType;

/// RPython `compute_restype(self_type, other_type)` (rarithmetic.py).
///
/// ```python
/// def compute_restype(self_type, other_type):
///     if self_type is other_type:
///         if self_type is bool:
///             return int
///         return self_type
///     if other_type in (bool, int, long):
///         if self_type is bool:
///             return int
///         return self_type
///     if self_type in (bool, int, long):
///         return other_type
///     if self_type is float or other_type is float:
///         return float
///     if self_type.SIGNED == other_type.SIGNED:
///         return build_int(None, self_type.SIGNED,
///                          max(self_type.BITS, other_type.BITS))
///     raise AssertionError("Merging these types (%s, %s) is not supported" % (self_type, other_type))
/// ```
///
/// Rust port scopes the domain to [`KnownType`] variants the annotator
/// actually exercises. Upstream `long` is Python 2's arbitrary-precision
/// int, folded into [`KnownType::Int`] for Rust. `r_uint` maps to
/// [`KnownType::Ruint`]. `r_longlong` / `r_ulonglong` map to the
/// corresponding 64-bit [`KnownType`] variants.
pub fn compute_restype(self_type: KnownType, other_type: KnownType) -> KnownType {
    // Upstream: `if self_type is other_type:`.
    if self_type == other_type {
        if self_type == KnownType::Bool {
            return KnownType::Int;
        }
        return self_type;
    }
    // Upstream: `if other_type in (bool, int, long):`.
    if matches!(other_type, KnownType::Bool | KnownType::Int) {
        if self_type == KnownType::Bool {
            return KnownType::Int;
        }
        return self_type;
    }
    // Upstream: `if self_type in (bool, int, long):`.
    if matches!(self_type, KnownType::Bool | KnownType::Int) {
        return other_type;
    }
    // Upstream: `if self_type is float or other_type is float:`.
    if self_type == KnownType::Float || other_type == KnownType::Float {
        return KnownType::Float;
    }
    // Upstream: `if self_type.SIGNED == other_type.SIGNED: build_int(...)`.
    if matches!(
        (self_type, other_type),
        (KnownType::LongLong, KnownType::Int)
            | (KnownType::Int, KnownType::LongLong)
            | (KnownType::LongLong, KnownType::LongLong)
    ) {
        return KnownType::LongLong;
    }
    if matches!(
        (self_type, other_type),
        (KnownType::ULongLong, KnownType::Ruint)
            | (KnownType::Ruint, KnownType::ULongLong)
            | (KnownType::ULongLong, KnownType::ULongLong)
    ) {
        return KnownType::ULongLong;
    }
    panic!(
        "compute_restype: unsupported merge of {:?} and {:?}",
        self_type, other_type
    );
}

/// `r_singlefloat(floatval)` (`rarithmetic.py`): `struct.pack("f", floatval)`.
/// `ieee.float_pack` raises `OverflowError` when a finite value overflows
/// the format (`"float too large to pack in this format"`). Inf/NaN pack.
pub fn r_singlefloat_bits(floatval: f64) -> Result<u32, &'static str> {
    if floatval.is_nan() || floatval.is_infinite() {
        return Ok((floatval as f32).to_bits());
    }
    let packed = floatval as f32;
    if packed.is_infinite() {
        return Err("float too large to pack in this format");
    }
    Ok(packed.to_bits())
}

/// `r_longfloat(floatval)` (`rarithmetic.py`): stores `floatval` as a Python float.
pub fn r_longfloat_bits(floatval: f64) -> u64 {
    floatval.to_bits()
}

/// RPython `signedtype(t)` (rarithmetic.py).
///
/// ```python
/// @specialize.memo()
/// def signedtype(t):
///     if t in (bool, int, long):
///         return True
///     else:
///         return t.SIGNED
/// ```
pub fn signedtype(t: KnownType) -> bool {
    match t {
        // Upstream: `if t in (bool, int, long): return True`.
        KnownType::Bool | KnownType::Int => true,
        // Upstream: `return t.SIGNED`.
        KnownType::LongLong => true,
        KnownType::Ruint | KnownType::ULongLong => false,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_bool_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Bool),
            KnownType::Int
        );
    }

    #[test]
    fn int_int_stays_int() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_ruint_picks_ruint() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Ruint),
            KnownType::Ruint
        );
        assert_eq!(
            compute_restype(KnownType::Ruint, KnownType::Int),
            KnownType::Ruint
        );
    }

    #[test]
    fn bool_int_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_float_widens_to_float() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Float),
            KnownType::Float
        );
    }

    #[test]
    fn r_singlefloat_bits_raises_on_overflow_and_packs_inf() {
        assert!(r_singlefloat_bits(1e300).is_err());
        assert!(r_singlefloat_bits(f64::INFINITY).unwrap() == f32::INFINITY.to_bits());
        assert!(r_singlefloat_bits(2.1).is_ok());
    }

    #[test]
    fn signedtype_follows_rarithmetic_signed_flag() {
        assert!(signedtype(KnownType::Bool));
        assert!(signedtype(KnownType::Int));
        assert!(signedtype(KnownType::LongLong));
        assert!(!signedtype(KnownType::Ruint));
        assert!(!signedtype(KnownType::ULongLong));
    }
}
