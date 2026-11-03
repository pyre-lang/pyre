//! RPython parity module for `rpython/jit/metainterp/support.py`.
//!
//! The RPython module is a small low-level helper collection. Pyre represents
//! addresses and raw pointers as machine integers at the metainterp boundary,
//! so the cast helpers are explicit Rust conversions over that shape.

/// Rust-side stand-in for RPython's symbolic `AddressAsInt`.
pub type AddressAsInt = i64;

/// Rust-side stand-in for `llmemory.Address` at metainterp boundaries.
pub type Address = usize;

/// support.py `adr2int`.
pub fn adr2int(addr: Address) -> AddressAsInt {
    addr as AddressAsInt
}

/// support.py `int2adr`.
pub fn int2adr(int: AddressAsInt) -> Address {
    int as Address
}

/// support.py `ptr2int`.
pub fn ptr2int<T>(ptr: *const T) -> AddressAsInt {
    adr2int(ptr as Address)
}

/// Mutable-pointer spelling for Rust call sites that do not have an immutable
/// raw pointer without an extra cast.
#[cfg(test)]
pub(crate) fn ptr2int_mut<T>(ptr: *mut T) -> AddressAsInt {
    ptr2int(ptr.cast_const())
}

/// support.py `int_signext`.
pub fn int_signext(value: i64, numbytes: i64) -> i64 {
    // `support.py int_signext` starts by shifting by `numbytes * 8 - 1`.
    // A zero or negative width therefore raises instead of returning the
    // unmodified value.  Widths above the machine word are different: the
    // intermediate `r_uint` conversions discard every high bit, reducing the
    // upstream expression to the identity on a 64-bit Signed.
    assert!(
        numbytes > 0,
        "int_signext: numbytes must be positive, got {numbytes}",
    );
    if numbytes > 8 {
        return value;
    }
    let shift = 64 - numbytes * 8;
    (value << shift) >> shift
}

#[cfg(test)]
mod tests {
    use super::{adr2int, int_signext, int2adr, ptr2int, ptr2int_mut};

    #[test]
    fn address_integer_roundtrip_preserves_bits() {
        let addr = 0x1234usize;
        assert_eq!(int2adr(adr2int(addr)), addr);
    }

    #[test]
    fn ptr2int_uses_raw_pointer_value() {
        let value = 5u8;
        let ptr = &value as *const u8;
        assert_eq!(ptr2int(ptr), ptr as usize as i64);

        let mut value = 7u8;
        let ptr = &mut value as *mut u8;
        assert_eq!(ptr2int_mut(ptr), ptr as usize as i64);
    }

    #[test]
    fn int_signext_matches_byte_widths() {
        assert_eq!(int_signext(0x80, 1), -128);
        assert_eq!(int_signext(0x7f, 1), 127);
        assert_eq!(int_signext(0x8000, 2), -32768);
        assert_eq!(int_signext(0x7fff, 2), 32767);
        assert_eq!(int_signext(0x8000_0000, 4), -2147483648);
        assert_eq!(int_signext(0x7fff_ffff, 4), 2147483647);
        assert_eq!(int_signext(-1, 8), -1);
    }

    #[test]
    fn int_signext_rejects_nonpositive_widths_like_rpython() {
        for numbytes in [0, -1] {
            assert!(
                std::panic::catch_unwind(|| int_signext(0x80, numbytes)).is_err(),
                "int_signext silently accepted numbytes={numbytes}",
            );
        }
    }

    #[test]
    fn int_signext_preserves_widths_larger_than_the_machine_word() {
        assert_eq!(int_signext(0x80, 9), 0x80);
    }
}
