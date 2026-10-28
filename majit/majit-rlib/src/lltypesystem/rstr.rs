//! `rpython/rtyper/lltypesystem/rstr.py` — the low-level STR body a residual
//! `LLHelpers.ll_find_char` call executes.
//!
//! Layout matches `LowlevelStringHeader` / `BUILTIN_STRING_*`: `hash` at
//! word 0, `length` at one word, `chars` immediately after the header.
//! UNICODE uses the same header; `chars` are UCS-4 `UniChar` (`u32`).

/// `rstr.STR`: `GcStruct('rpy_string', ('hash', Signed), ('chars', Array(Char)))`.
/// `length` is the varsize length word of `chars` (`len(s.chars)`).
#[repr(C)]
#[allow(clippy::upper_case_acronyms)]
pub struct STR {
    pub hash: isize,
    pub length: usize,
    pub chars: [u8; 0],
}

/// `rstr.UNICODE`: `GcStruct('rpy_unicode', ('hash', Signed), ('chars', Array(UniChar)))`.
#[repr(C)]
#[allow(clippy::upper_case_acronyms)]
pub struct UNICODE {
    pub hash: isize,
    pub length: usize,
    pub chars: [u32; 0],
}

impl majit_gc::GcType for STR {
    fn type_id() -> u32 {
        majit_gc::lowlevel_str_type_id()
    }
    const SIZE: usize = std::mem::size_of::<STR>();
}

impl majit_gc::GcType for UNICODE {
    fn type_id() -> u32 {
        majit_gc::lowlevel_unicode_type_id()
    }
    const SIZE: usize = std::mem::size_of::<UNICODE>();
}

/// `LowlevelStringHeader.hash` / `BUILTIN_STRING_HASH_OFFSET`.
pub(crate) const STR_HASH_OFFSET: usize = std::mem::offset_of!(STR, hash);
/// `LowlevelStringHeader.length` / `BUILTIN_STRING_LEN_OFFSET`.
pub const STR_LEN_OFFSET: usize = std::mem::offset_of!(STR, length);
/// `LOWLEVEL_STRING_CHARS_OFFSET` / `BUILTIN_STRING_CHARS_OFFSET`.
pub const STR_CHARS_OFFSET: usize = std::mem::offset_of!(STR, chars);

/// `rstr.UNICODE` header; same `hash` / `length` words as [`STR_HASH_OFFSET`].
pub(crate) const UNICODE_HASH_OFFSET: usize = std::mem::offset_of!(UNICODE, hash);
/// `rstr.UNICODE` length word; same slot as [`STR_LEN_OFFSET`].
pub(crate) const UNICODE_LEN_OFFSET: usize = std::mem::offset_of!(UNICODE, length);
/// Offset of the UCS-4 `chars` payload (`Array(UniChar)` item width 4).
pub(crate) const UNICODE_CHARS_OFFSET: usize = std::mem::offset_of!(UNICODE, chars);

const _: () = {
    assert!(STR_HASH_OFFSET == 0);
    assert!(STR_LEN_OFFSET == std::mem::size_of::<usize>());
    assert!(STR_CHARS_OFFSET == 2 * std::mem::size_of::<usize>());
    assert!(UNICODE_HASH_OFFSET == STR_HASH_OFFSET);
    assert!(UNICODE_LEN_OFFSET == STR_LEN_OFFSET);
    assert!(UNICODE_CHARS_OFFSET == STR_CHARS_OFFSET);
};

/// Pre-boot `rstr.STR` (`init_gc_object_immortal`) for tests.
///
/// `rgc.can_move` is false with no collector, so `get_nonmovingbuffer_ll`
/// takes flag `\x04` and must not `rgc.unpin`.
#[cfg(test)]
pub(crate) fn alloc_immortal_str(bytes: &[u8]) -> majit_ir::GcRef {
    let payload = STR_CHARS_OFFSET + bytes.len() + 1;
    let ptr = majit_gc::header::alloc_varsize_with_gc_header_immortal(payload, 0);
    assert!(!ptr.is_null());
    unsafe {
        (ptr as *mut isize).write(0);
        ptr.add(STR_LEN_OFFSET).cast::<usize>().write(bytes.len());
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(STR_CHARS_OFFSET), bytes.len());
        ptr.add(STR_CHARS_OFFSET + bytes.len()).write(0);
    }
    majit_ir::GcRef(ptr as usize)
}

/// Release a block from [`alloc_immortal_str`].
#[cfg(test)]
pub(crate) unsafe fn free_immortal_str(llobj: majit_ir::GcRef, nbytes: usize) {
    let payload = STR_CHARS_OFFSET + nbytes + 1;
    unsafe { majit_gc::header::dealloc_varsize_with_gc_header(llobj.0 as *mut u8, payload) };
}

/// `rstr.py` `mallocstr` for the host-side `dont_look_inside` rffi helpers.
/// Unlike `malloc_varsize` it does not collect: the same convention as the
/// host allocators described in `COLLECTING_SEEDS` (`try_gc_alloc` routes
/// to `alloc_nursery_typed`, which falls back to old-gen rather than
/// collect). A majit-rlib body the collect analysis cannot see therefore
/// never runs a collection. The result may be a nursery object (movable,
/// pinnable, shrinkable) or an old-gen spill (`can_move` false).
///
/// Size is `STR_CHARS_OFFSET + length + 1` (trailing NUL). `hash` is stored
/// as 0 and the NUL at `chars[length]` is written; `chars` are not cleared
/// (`malloc_zero_filled` is false). Overflow / a refused allocation is
/// [`crate::rffi::MemoryError`].
///
/// When `majit_gc::lowlevel_str_type_id` is published and a nursery
/// allocator is installed (`gc_allocator_installed` /
/// `ACTIVE_ALLOC_NURSERY_TYPED`), the block is `alloc_nursery_typed`.
/// No hook or `tid == 0` uses an immortal `init_gc_object_immortal` block.
/// A hook that answers null is [`crate::rffi::MemoryError`].
pub fn mallocstr(length: usize) -> Result<majit_ir::GcRef, crate::rffi::MemoryError> {
    let size = STR_CHARS_OFFSET
        .checked_add(length)
        .and_then(|n| n.checked_add(1))
        .ok_or(crate::rffi::MemoryError)?;
    let tid = majit_gc::lowlevel_str_type_id();
    let gcref = if tid == 0 || !majit_gc::gc_allocator_installed() {
        let ptr = majit_gc::header::alloc_varsize_with_gc_header_immortal(size, tid);
        if ptr.is_null() {
            return Err(crate::rffi::MemoryError);
        }
        majit_ir::GcRef(ptr as usize)
    } else {
        let gcref = majit_gc::alloc_nursery_typed(tid, size);
        if gcref.is_null() {
            return Err(crate::rffi::MemoryError);
        }
        gcref
    };
    let ptr = gcref.0 as *mut u8;
    unsafe {
        (ptr as *mut isize).write(0);
        ptr.add(STR_LEN_OFFSET).cast::<usize>().write(length);
        ptr.add(STR_CHARS_OFFSET + length).write(0);
    }
    Ok(gcref)
}

/// The `chars` payload of a live `Ptr(STR)`.
///
/// SAFETY: `s` is a residual `Ptr(STR)` whose header and `length` bytes
/// are valid for this layout.
pub unsafe fn str_as_bytes<'a>(s: *const STR) -> &'a [u8] {
    let len = unsafe { (*s).length };
    unsafe { std::slice::from_raw_parts((&raw const (*s).chars).cast::<u8>(), len) }
}

/// The `chars` payload of a live `Ptr(UNICODE)`.
///
/// SAFETY: `s` is a residual `Ptr(UNICODE)` whose header and `length`
/// UCS-4 codepoints are valid for this layout.
pub(crate) unsafe fn unicode_as_unichars<'a>(s: *const UNICODE) -> &'a [u32] {
    let len = unsafe { (*s).length };
    unsafe { std::slice::from_raw_parts((&raw const (*s).chars).cast::<u32>(), len) }
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
pub fn ll_find_char(s: *const STR, ch: i64, start: i64, end: i64) -> i64 {
    // SAFETY: residual `Ptr(STR)` callee; the caller supplies a live STR.
    let chars = unsafe { str_as_bytes(s) };
    ll_find_char_loop(chars, ch, start, end)
}

/// `LLHelpers.ll_find_char` (`rstr.py`) for `Ptr(UNICODE)`. UCS-4
/// `UniChar` cells; published on `HELPER_FNADDRS` as
/// `ll_find_char_unicode` so the UNICODE specialization cannot alias
/// the STR row.
#[majit_macros::elidable]
pub fn ll_find_char_unicode(s: *const UNICODE, ch: i64, start: i64, end: i64) -> i64 {
    // SAFETY: residual `Ptr(UNICODE)` callee; the caller supplies a live UNICODE.
    let chars = unsafe { unicode_as_unichars(s) };
    ll_find_char_loop(chars, ch, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_buf(bytes: &[u8]) -> Vec<usize> {
        let nbytes = STR_CHARS_OFFSET + bytes.len() + 1;
        let nwords = (nbytes + std::mem::size_of::<usize>() - 1) / std::mem::size_of::<usize>();
        let mut buf = vec![0usize; nwords];
        let p = buf.as_mut_ptr().cast::<u8>();
        unsafe {
            (*p.cast::<STR>()).hash = 0;
            (*p.cast::<STR>()).length = bytes.len();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(STR_CHARS_OFFSET), bytes.len());
        }
        buf
    }

    #[test]
    fn ll_find_char_finds_clamps_and_misses() {
        let buf = str_buf(b"hello");
        let s = buf.as_ptr().cast::<STR>();
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

    fn unicode_buf(chars: &[u32]) -> Vec<usize> {
        let nbytes = UNICODE_CHARS_OFFSET + chars.len() * 4;
        let nwords = (nbytes + std::mem::size_of::<usize>() - 1) / std::mem::size_of::<usize>();
        let mut buf = vec![0usize; nwords];
        let p = buf.as_mut_ptr().cast::<u8>();
        unsafe {
            (*p.cast::<UNICODE>()).hash = 0;
            (*p.cast::<UNICODE>()).length = chars.len();
            std::ptr::copy_nonoverlapping(
                chars.as_ptr(),
                p.add(UNICODE_CHARS_OFFSET).cast::<u32>(),
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
        let s = buf.as_ptr().cast::<UNICODE>();
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
