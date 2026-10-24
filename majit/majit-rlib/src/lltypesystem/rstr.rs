//! `rpython/rtyper/lltypesystem/rstr.py` — the low-level STR body a residual
//! `LLHelpers.ll_find_char` call executes.
//!
//! Layout matches `LowlevelStringHeader` / `BUILTIN_STRING_*`: `hash` at
//! word 0, `length` at one word, `chars` immediately after the header.
//! UNICODE uses the same header; `chars` are UCS-4 `UniChar` (`u32`).

/// `rstr.STR` header in front of the character bytes.
#[repr(C)]
pub(crate) struct StrHeader {
    pub(crate) hash: isize,
    pub(crate) length: usize,
}

/// `LowlevelStringHeader.hash` / `BUILTIN_STRING_HASH_OFFSET`.
pub(crate) const STR_HASH_OFFSET: usize = std::mem::offset_of!(StrHeader, hash);
/// `LowlevelStringHeader.length` / `BUILTIN_STRING_LEN_OFFSET`.
pub(crate) const STR_LEN_OFFSET: usize = std::mem::offset_of!(StrHeader, length);
/// `LOWLEVEL_STRING_CHARS_OFFSET` / `BUILTIN_STRING_CHARS_OFFSET`.
pub(crate) const STR_CHARS_OFFSET: usize = std::mem::size_of::<StrHeader>();

/// `rstr.UNICODE` header; same `hash` / `length` words as [`STR_HASH_OFFSET`].
pub(crate) const UNICODE_HASH_OFFSET: usize = STR_HASH_OFFSET;
/// `rstr.UNICODE` length word; same slot as [`STR_LEN_OFFSET`].
pub(crate) const UNICODE_LEN_OFFSET: usize = STR_LEN_OFFSET;
/// Offset of the UCS-4 `chars` payload (`Array(UniChar)` item width 4).
pub(crate) const UNICODE_CHARS_OFFSET: usize = STR_CHARS_OFFSET;

const _: () = {
    assert!(STR_HASH_OFFSET == 0);
    assert!(STR_LEN_OFFSET == std::mem::size_of::<usize>());
    assert!(STR_CHARS_OFFSET == 2 * std::mem::size_of::<usize>());
    assert!(UNICODE_HASH_OFFSET == STR_HASH_OFFSET);
    assert!(UNICODE_LEN_OFFSET == STR_LEN_OFFSET);
    assert!(UNICODE_CHARS_OFFSET == STR_CHARS_OFFSET);
};

/// The `chars` payload of a live `Ptr(STR)`.
///
/// SAFETY: `s` is a residual `Ptr(STR)` whose header and `length` bytes
/// are valid for this layout.
pub(crate) unsafe fn str_as_bytes<'a>(s: *const u8) -> &'a [u8] {
    let len = unsafe { s.add(STR_LEN_OFFSET).cast::<usize>().read_unaligned() };
    unsafe { std::slice::from_raw_parts(s.add(STR_CHARS_OFFSET), len) }
}

/// The `chars` payload of a live `Ptr(UNICODE)`.
///
/// SAFETY: `s` is a residual `Ptr(UNICODE)` whose header and `length`
/// UCS-4 codepoints are valid for this layout.
pub(crate) unsafe fn unicode_as_unichars<'a>(s: *const u8) -> &'a [u32] {
    let len = unsafe { s.add(UNICODE_LEN_OFFSET).cast::<usize>().read_unaligned() };
    unsafe { std::slice::from_raw_parts(s.add(UNICODE_CHARS_OFFSET).cast::<u32>(), len) }
}

/// Shared `LLHelpers.ll_find_char` loop (`rstr.py`). `C` is `STR` `Char`
/// (`u8`) or `UNICODE` `UniChar` (`u32`, UCS-4).
fn ll_find_char_loop<C>(chars: &[C], ch: i64, start: i64, mut end: i64) -> i64
where
    C: Copy,
    i64: From<C>,
{
    let len = chars.len() as i64;
    if end > len {
        end = len;
    }
    let mut i = start;
    while i < end {
        if i64::from(chars[i as usize]) == ch {
            return i;
        }
        i += 1;
    }
    -1
}

/// `LLHelpers.ll_find_char` (`rstr.py`) for `Ptr(STR)`. `#[elidable]`
/// publishes the residual trampoline on `HELPER_FNADDRS`.
#[majit_macros::elidable]
pub fn ll_find_char(s: *const u8, ch: i64, start: i64, end: i64) -> i64 {
    // SAFETY: residual `Ptr(STR)` callee; the caller supplies a live STR.
    let chars = unsafe { str_as_bytes(s) };
    ll_find_char_loop(chars, ch, start, end)
}

/// `LLHelpers.ll_find_char` (`rstr.py`) for `Ptr(UNICODE)`. UCS-4
/// `UniChar` cells; published on `HELPER_FNADDRS` as
/// `ll_find_char_unicode` so the UNICODE specialization cannot alias
/// the STR row.
#[majit_macros::elidable]
pub fn ll_find_char_unicode(s: *const u8, ch: i64, start: i64, end: i64) -> i64 {
    // SAFETY: residual `Ptr(UNICODE)` callee; the caller supplies a live UNICODE.
    let chars = unsafe { unicode_as_unichars(s) };
    ll_find_char_loop(chars, ch, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_buf(bytes: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; STR_CHARS_OFFSET + bytes.len() + 1];
        unsafe {
            buf.as_mut_ptr()
                .add(STR_HASH_OFFSET)
                .cast::<isize>()
                .write_unaligned(0);
            buf.as_mut_ptr()
                .add(STR_LEN_OFFSET)
                .cast::<usize>()
                .write_unaligned(bytes.len());
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_mut_ptr().add(STR_CHARS_OFFSET),
                bytes.len(),
            );
        }
        buf
    }

    #[test]
    fn ll_find_char_finds_clamps_and_misses() {
        let buf = str_buf(b"hello");
        let s = buf.as_ptr();
        assert_eq!(ll_find_char(s, i64::from(b'e'), 0, 5), 1);
        assert_eq!(ll_find_char(s, i64::from(b'z'), 0, 5), -1);
        assert_eq!(ll_find_char(s, i64::from(b'o'), 0, 100), 4);
        assert_eq!(ll_find_char(s, i64::from(b'o'), 0, 3), -1);
    }

    #[test]
    fn ll_find_char_is_published_on_helper_fnaddrs() {
        let mut found = false;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row.path.ends_with("lltypesystem::rstr::ll_find_char") {
                found = true;
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 4);
            }
        });
        assert!(found, "HELPER_FNADDRS must publish ll_find_char");
    }

    fn unicode_buf(chars: &[u32]) -> Vec<u8> {
        let mut buf = vec![0u8; UNICODE_CHARS_OFFSET + chars.len() * 4];
        unsafe {
            buf.as_mut_ptr()
                .add(UNICODE_HASH_OFFSET)
                .cast::<isize>()
                .write_unaligned(0);
            buf.as_mut_ptr()
                .add(UNICODE_LEN_OFFSET)
                .cast::<usize>()
                .write_unaligned(chars.len());
            std::ptr::copy_nonoverlapping(
                chars.as_ptr(),
                buf.as_mut_ptr().add(UNICODE_CHARS_OFFSET).cast::<u32>(),
                chars.len(),
            );
        }
        buf
    }

    #[test]
    fn ll_find_char_unicode_finds_u0100_in_unicode_buffer() {
        let latin_small_a = u32::from(b'a');
        let latin_extended_a = 0x0100u32;
        let buf = unicode_buf(&[latin_small_a, latin_extended_a, latin_small_a]);
        let s = buf.as_ptr();
        assert_eq!(
            ll_find_char_unicode(s, i64::from(latin_extended_a), 0, 3),
            1
        );
        assert_eq!(ll_find_char_unicode(s, i64::from(latin_small_a), 0, 3), 0);
        assert_eq!(ll_find_char_unicode(s, i64::from(b'z'), 0, 3), -1);
        assert_eq!(
            ll_find_char_unicode(s, i64::from(latin_extended_a), 0, 1),
            -1
        );
    }

    #[test]
    fn ll_find_char_unicode_is_published_on_helper_fnaddrs() {
        let mut found_str = false;
        let mut found_unicode = false;
        let mut str_addr = 0usize;
        let mut unicode_addr = 0usize;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row
                .path
                .ends_with("lltypesystem::rstr::ll_find_char_unicode")
            {
                found_unicode = true;
                unicode_addr = row.get();
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 4);
            } else if row.path.ends_with("lltypesystem::rstr::ll_find_char") {
                found_str = true;
                str_addr = row.get();
            }
        });
        assert!(
            found_unicode,
            "HELPER_FNADDRS must publish ll_find_char_unicode"
        );
        assert!(
            found_str,
            "HELPER_FNADDRS must keep the STR ll_find_char row"
        );
        assert_ne!(
            str_addr, unicode_addr,
            "STR and UNICODE ll_find_char natives must be distinct"
        );
    }
}
