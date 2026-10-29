//! Platform float-storage helpers from `rpython/jit/codewriter/longlong.py`.
//!
//! Upstream uses this tiny module as the codewriter's abstraction over
//! "float stored directly" on 64-bit hosts versus "float stored as a
//! signed long long bit pattern" on 32-bit hosts.  Pyre's low-level
//! value carrier already stores `Float` constants as IEEE bits, but the
//! public helper names still belong in `codewriter::longlong` so
//! ports of `jtransform.py` / backend code can refer to the same module
//! and symbol names.

use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

#[allow(non_upper_case_globals)]
pub const is_64_bit: bool = cfg!(target_pointer_width = "64");

#[allow(non_upper_case_globals)]
pub const supports_longlong: bool = !is_64_bit;

#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub type r_float_storage = f64;

#[cfg(not(target_pointer_width = "64"))]
#[allow(non_camel_case_types)]
pub type r_float_storage = i64;

#[cfg(target_pointer_width = "64")]
pub const FLOATSTORAGE: LowLevelType = LowLevelType::Float;

#[cfg(not(target_pointer_width = "64"))]
pub const FLOATSTORAGE: LowLevelType = LowLevelType::SignedLongLong;

#[cfg(target_pointer_width = "64")]
pub const ZEROF: r_float_storage = 0.0;

#[cfg(not(target_pointer_width = "64"))]
pub const ZEROF: r_float_storage = 0;

#[cfg(target_pointer_width = "64")]
pub fn getfloatstorage(x: f64) -> r_float_storage {
    x
}

#[cfg(not(target_pointer_width = "64"))]
pub fn getfloatstorage(x: f64) -> r_float_storage {
    x.to_bits() as i64
}

#[cfg(target_pointer_width = "64")]
pub fn getrealfloat(x: r_float_storage) -> f64 {
    x
}

#[cfg(not(target_pointer_width = "64"))]
pub fn getrealfloat(x: r_float_storage) -> f64 {
    f64::from_bits(x as u64)
}

#[cfg(target_pointer_width = "64")]
pub fn gethash(x: r_float_storage) -> i64 {
    compute_hash_float(x)
}

#[cfg(not(target_pointer_width = "64"))]
pub fn gethash(xll: r_float_storage) -> i64 {
    signed_intmask(xll.wrapping_sub(xll >> 32))
}

#[cfg(target_pointer_width = "64")]
pub fn gethash_fast(x: r_float_storage) -> i64 {
    x.to_bits() as i64
}

#[cfg(not(target_pointer_width = "64"))]
pub fn gethash_fast(x: r_float_storage) -> i64 {
    gethash(x)
}

#[cfg(target_pointer_width = "64")]
pub fn extract_bits(x: r_float_storage) -> i64 {
    x.to_bits() as i64
}

#[cfg(not(target_pointer_width = "64"))]
pub fn extract_bits(x: r_float_storage) -> i64 {
    x
}

pub fn is_longlong(ty: &LowLevelType) -> bool {
    if is_64_bit {
        false
    } else {
        matches!(
            ty,
            LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong
        )
    }
}

pub fn int2singlefloat(x: i64) -> f32 {
    f32::from_bits(x as u32)
}

/// `longlong.singlefloat2int`: `rffi.cast(lltype.Signed, rffi.cast(rffi.UINT, x))`.
/// `word` is the target `Signed` width in bytes. Word 4 sign-extends bit 31;
/// a wider word zero-extends.
pub fn singlefloat2int_for_word(bits: u32, word: usize) -> i64 {
    if word == 4 {
        bits as i32 as i64
    } else {
        bits as i64
    }
}

pub fn singlefloat2int(x: f32) -> i64 {
    // Runtime-on-target: `cfg!(target_pointer_width)` is the Signed width.
    let word = if cfg!(target_pointer_width = "32") {
        4
    } else {
        8
    };
    singlefloat2int_for_word(x.to_bits(), word)
}

#[cfg(target_pointer_width = "64")]
fn compute_hash_float(f: f64) -> i64 {
    if !f.is_finite() {
        if f.is_infinite() {
            if f < 0.0 {
                return -271828;
            }
            return 314159;
        }
        return 0;
    }
    if f == 0.0 {
        return 0;
    }

    let (mut v, expo) = frexp(f);
    const TAKE_NEXT: f64 = 2147483648.0;
    v *= TAKE_NEXT;
    let hipart = v.trunc() as i64;
    v = (v - hipart as f64) * TAKE_NEXT;
    signed_intmask(
        hipart
            .wrapping_add(v.trunc() as i64)
            .wrapping_add((expo as i64) << 15),
    )
}

fn signed_intmask(x: i64) -> i64 {
    #[cfg(target_pointer_width = "64")]
    {
        x
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        x as i32 as i64
    }
}

#[cfg(target_pointer_width = "64")]
fn frexp(f: f64) -> (f64, i32) {
    debug_assert!(f.is_finite());
    debug_assert_ne!(f, 0.0);

    let bits = f.to_bits();
    let sign = bits & (1_u64 << 63);
    let exp_raw = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1_u64 << 52) - 1);

    if exp_raw != 0 {
        let mantissa = f64::from_bits(sign | (1022_u64 << 52) | frac);
        return (mantissa, exp_raw - 1022);
    }

    let leading_bit = 63 - frac.leading_zeros() as i32;
    let normalized_frac = (frac << (52 - leading_bit)) & ((1_u64 << 52) - 1);
    let mantissa = f64::from_bits(sign | (1022_u64 << 52) | normalized_frac);
    (mantissa, leading_bit - 1073)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn platform_constants_match_longlong_py_branch() {
        assert_eq!(FLOATSTORAGE, LowLevelType::Float);
        assert!(!is_longlong(&LowLevelType::SignedLongLong));
        assert_eq!(ZEROF, 0.0);
    }

    #[test]
    #[cfg(not(target_pointer_width = "64"))]
    fn platform_constants_match_longlong_py_branch() {
        assert!(!is_64_bit);
        assert!(supports_longlong);
        assert_eq!(FLOATSTORAGE, LowLevelType::SignedLongLong);
        assert!(is_longlong(&LowLevelType::SignedLongLong));
        assert!(is_longlong(&LowLevelType::UnsignedLongLong));
        assert_eq!(ZEROF, 0);
    }

    #[test]
    fn float_storage_round_trips_bits() {
        let samples = [0.0, -0.0, 1.5, -3.25, f64::MIN_POSITIVE, f64::from_bits(1)];
        for value in samples {
            let storage = getfloatstorage(value);
            assert_eq!(getrealfloat(storage).to_bits(), value.to_bits());
            assert_eq!(extract_bits(storage), value.to_bits() as i64);
        }
    }

    #[test]
    fn gethash_matches_rpython_float_hash_cases() {
        assert_eq!(gethash(getfloatstorage(0.0)), 0);
        assert_eq!(gethash(getfloatstorage(-0.0)), 0);
        assert_eq!(gethash(getfloatstorage(f64::INFINITY)), 314159);
        assert_eq!(gethash(getfloatstorage(f64::NEG_INFINITY)), -271828);
        assert_eq!(gethash(getfloatstorage(f64::NAN)), 0);
        assert_eq!(gethash(getfloatstorage(1.5)), 1_610_645_504);
        assert_eq!(gethash(getfloatstorage(-3.5)), -1_878_982_656);
    }

    #[test]
    fn singlefloat_helpers_reinterpret_uint_bits() {
        let value = int2singlefloat(0x3fc00000);
        assert_eq!(value, 1.5_f32);
        assert_eq!(singlefloat2int(value), 0x3fc00000);
        let high = singlefloat2int(int2singlefloat(0x8000_0000));
        if cfg!(target_pointer_width = "32") {
            assert_eq!(high, -2147483648);
        } else {
            assert_eq!(high, 0x8000_0000);
        }
        let bits = (-1.0f32).to_bits();
        assert_eq!(singlefloat2int_for_word(bits, 4), -1_082_130_432);
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(singlefloat2int(-1.0), 3_212_836_864);
            assert_eq!(singlefloat2int_for_word(bits, 8), 3_212_836_864);
        }
    }
}
