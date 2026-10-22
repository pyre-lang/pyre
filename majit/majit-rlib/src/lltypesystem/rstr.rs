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

/// `LLHelpers.ll_find_char` (`rstr.py`). `#[elidable]` publishes the
/// residual trampoline on `HELPER_FNADDRS`.
#[majit_macros::elidable]
pub fn ll_find_char(s: *const u8, ch: i64, start: i64, end: i64) -> i64 {
    let mut i = start;
    let mut end = end;
    // SAFETY: residual `Ptr(STR)` callee; the caller supplies a live STR.
    let len = unsafe { s.add(STR_LEN_OFFSET).cast::<usize>().read_unaligned() } as i64;
    if end > len {
        end = len;
    }
    let chars = unsafe { s.add(STR_CHARS_OFFSET) };
    while i < end {
        let c = unsafe { chars.add(i as usize).read() } as i64;
        if c == ch {
            return i;
        }
        i += 1;
    }
    -1
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
}
