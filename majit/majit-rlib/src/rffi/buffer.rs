//! String and buffer helpers from `make_string_mappings` and the wchar/utf8
//! conversions next to it. "str" is a byte string (`&[u8]` / `Vec<u8>`),
//! except `get_nonmovingbuffer_ll` / `scoped_nonmovingbuffer` /
//! `scoped_view_charp` / `alloc_buffer` / `str_from_buffer` /
//! `keep_buffer_alive_until_here` / `scoped_alloc_buffer`, which use a
//! `Ptr(STR)` (`rstr.py` `STR`).

use super::{CCHARP, CONST_CCHARP, CWCHARP, Wchar, cast};
use crate::lltypesystem::rstr::STR_CHARS_OFFSET;

/// `get_nonmovingbuffer_ll` flag: the pointer is inside a nonmovable object.
/// Nothing to free.
const FLAG_INSIDE_NONMOVABLE: u8 = 0x04;
/// `llobj` was pinned. Release unpins it.
const FLAG_PINNED: u8 = 0x05;
/// Pinning failed. The pointer is a raw malloc; release frees it.
const FLAG_RAW_COPY: u8 = 0x06;

/// `alloc_buffer` case: the pointer is inside an unpinned, non-moving GC buffer.
const CASE_INSIDE: isize = 0;
/// `alloc_buffer` case: the GC buffer was pinned (`rgc.pin`).
const CASE_PINNED: isize = 1;
/// `alloc_buffer` case: raw `malloc` of `count` bytes; [`keep_buffer_alive_until_here`] frees it.
const CASE_RAW: isize = 2;

/// RPython `MemoryError` raised by `mallocfn` / `lltype.malloc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryError;

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MemoryError")
    }
}

impl std::error::Error for MemoryError {}

/// Code point `wcharpsize2utf8` / `wcharp2utf8` refuse (`rutf8.OutOfRange`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfRange {
    pub code: u32,
}

/// `str2charp`. Copies `s` into a fresh NUL-terminated `malloc` block.
///
/// `track_allocation` is `lltype.malloc(..., track_allocation=...)`. Both arms
/// allocate with `malloc`; the flag only selects the debug tracker upstream.
/// `lltype.malloc(..., flavor='raw')` raises `MemoryError`.
pub fn str2charp(s: &[u8], track_allocation: bool) -> Result<CCHARP, MemoryError> {
    let _ = track_allocation;
    let bytes = s.len().checked_add(1).ok_or(MemoryError)?;
    unsafe {
        let array = raw_malloc_or_memory_error(bytes)?.cast::<super::CHAR>();
        copy_bytes(s.as_ptr(), array.cast(), s.len());
        *array.add(s.len()) = 0;
        Ok(array)
    }
}

/// `free_charp`. `free` of a block from [`str2charp`].
///
/// # Safety
/// `cp` is null or a pointer returned by [`str2charp`] / [`str2constcharp`]
/// that has not been freed.
pub unsafe fn free_charp(cp: CCHARP, track_allocation: bool) {
    let _ = track_allocation;
    unsafe { raw_free(cp.cast()) }
}

/// `str2chararray`. Copies `min(len(s), maxsize)` bytes into `array` and
/// returns that count. Does not write a NUL.
///
/// # Safety
/// `array` has room for `min(s.len(), maxsize)` bytes.
pub unsafe fn str2chararray(s: &[u8], array: CCHARP, maxsize: usize) -> usize {
    let length = s.len().min(maxsize);
    unsafe { copy_bytes(s.as_ptr(), array.cast(), length) };
    length
}

/// `str2rawmem`. Copies `s[start..start+length]` into `array`, NULs included.
///
/// # Safety
/// `array` has room for `length` bytes.
pub unsafe fn str2rawmem(s: &[u8], array: CCHARP, start: usize, length: usize) {
    let src = &s[start..start + length];
    unsafe { copy_bytes(src.as_ptr(), array.cast(), length) };
}

/// `charp2str`. Reads up to the first NUL. Does not free `cp`.
/// `charpsize2str` / `mallocfn` raises [`MemoryError`].
///
/// # Safety
/// `cp` points at a NUL-terminated byte sequence.
pub unsafe fn charp2str(cp: CCHARP) -> Result<Vec<u8>, MemoryError> {
    let mut size = 0;
    unsafe {
        while *cp.add(size) != 0 {
            size += 1;
        }
        charpsize2str(cp, size)
    }
}

/// `charp2strn`. Like [`charp2str`], but stops after `maxlen` bytes.
///
/// # Safety
/// `cp` is readable for `maxlen` bytes or up to a NUL, whichever comes first.
pub unsafe fn charp2strn(cp: CCHARP, maxlen: usize) -> Result<Vec<u8>, MemoryError> {
    let mut size = 0;
    unsafe {
        while size < maxlen && *cp.add(size) != 0 {
            size += 1;
        }
        charpsize2str(cp, size)
    }
}

/// `charpsize2str`. Copies `size` bytes, NULs included. Does not free `cp`.
/// `mallocfn` raises [`MemoryError`].
///
/// # Safety
/// `cp` is readable for `size` bytes.
pub unsafe fn charpsize2str(cp: CCHARP, size: usize) -> Result<Vec<u8>, MemoryError> {
    let mut out = mallocfn(size)?;
    unsafe { copy_bytes(cp.cast(), out.as_mut_ptr(), size) };
    Ok(out)
}

/// `constcharp2str`. [`charp2str`] for a `CONST_CCHARP`.
///
/// # Safety
/// `cp` points at a NUL-terminated byte sequence.
pub unsafe fn constcharp2str(cp: CONST_CCHARP) -> Result<Vec<u8>, MemoryError> {
    unsafe { charp2str(cast::<CCHARP>(cp)) }
}

/// `constcharpsize2str`. [`charpsize2str`] for a `CONST_CCHARP`.
///
/// # Safety
/// `cp` is readable for `size` bytes.
pub unsafe fn constcharpsize2str(cp: CONST_CCHARP, size: usize) -> Result<Vec<u8>, MemoryError> {
    unsafe { charpsize2str(cast::<CCHARP>(cp), size) }
}

/// `str2constcharp`. [`str2charp`] cast to `CONST_CCHARP`.
pub fn str2constcharp(s: &[u8], track_allocation: bool) -> Result<CONST_CCHARP, MemoryError> {
    Ok(cast::<CONST_CCHARP>(str2charp(s, track_allocation)?))
}

/// `rffi.get_nonmovingbuffer_ll` (`jit.dont_look_inside`).
///
/// `llobj` is `llstr(data)` (`Ptr(STR)`). Returns `(char*, llobj, flag)`:
/// * `\x04` — no pin; the pointer is inside a nonmovable `llobj` (nothing to free)
/// * `\x05` — `llobj` was pinned (unpin on release)
/// * `\x06` — pinning failed; the pointer is a raw malloc (free on release)
///
/// `rgc.must_split_gc_address_space` is False. `we_are_translated_to_c()` is
/// this compiled body: `rgc.can_move` / `rgc.pin` are `majit_gc::{can_move,pin}`.
/// With no collector `can_move` is false, so the flag is `\x04` and `unpin`
/// is never reached (an immortal pre-boot block must not be unpinned).
#[majit_macros::dont_look_inside]
pub fn get_nonmovingbuffer_ll(
    llobj: majit_ir::GcRef,
) -> Result<(CCHARP, majit_ir::GcRef, u8), MemoryError> {
    let count = unsafe {
        crate::lltypesystem::rstr::str_as_bytes(llobj.0 as *const crate::lltypesystem::rstr::STR)
            .len()
    };
    let flag = if !majit_gc::can_move(llobj) {
        FLAG_INSIDE_NONMOVABLE
    } else if majit_gc::pin(llobj) {
        FLAG_PINNED
    } else {
        FLAG_RAW_COPY
    };
    if flag == FLAG_RAW_COPY {
        // `count + (TYPEP is CCHARP)`: a byte string gets one extra byte.
        let bytes = count.checked_add(1).ok_or(MemoryError)?;
        unsafe {
            let buf = raw_malloc_or_memory_error(bytes)?.cast::<super::CHAR>();
            copy_bytes(
                (llobj.as_usize() as *const u8).add(STR_CHARS_OFFSET),
                buf.cast(),
                count,
            );
            Ok((buf, llobj, FLAG_RAW_COPY))
        }
    } else {
        let buf = (llobj.as_usize() as *mut u8).wrapping_add(STR_CHARS_OFFSET);
        Ok((buf, llobj, flag))
    }
}

/// `rffi.get_nonmovingbuffer_ll_final_null` (`jit.dont_look_inside`). Writes
/// the NUL at `len(data)`. A `\x04`/`\x05` STR already has that trailing NUL;
/// the write is the same store upstream performs unconditionally.
#[majit_macros::dont_look_inside]
pub fn get_nonmovingbuffer_ll_final_null(
    llobj: majit_ir::GcRef,
) -> Result<(CCHARP, majit_ir::GcRef, u8), MemoryError> {
    let (buf, llobj, flag) = get_nonmovingbuffer_ll(llobj)?;
    let count = unsafe {
        crate::lltypesystem::rstr::str_as_bytes(llobj.0 as *const crate::lltypesystem::rstr::STR)
            .len()
    };
    unsafe { *buf.add(count) = 0 };
    Ok((buf, llobj, flag))
}

/// `rffi.free_nonmovingbuffer_ll` (`jit.dont_look_inside`).
///
/// `\x05` → `rgc.unpin(llobj)`. `\x06` → `lltype.free(buf)`. `\x04` is a
/// no-op. Then `keepalive_until_here(llobj)`.
///
/// # Safety
/// `(buf, llobj, flag)` came from [`get_nonmovingbuffer_ll`] or
/// [`get_nonmovingbuffer_ll_final_null`] and has not been released.
#[majit_macros::dont_look_inside]
pub unsafe fn free_nonmovingbuffer_ll(buf: CCHARP, llobj: majit_ir::GcRef, flag: u8) {
    if flag == FLAG_PINNED {
        majit_gc::unpin(llobj);
    }
    if flag == FLAG_RAW_COPY {
        unsafe { raw_free(buf.cast()) };
    }
    let _keepalive = llobj;
}

fn mallocfn(count: usize) -> Result<Vec<u8>, MemoryError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u8>(count).map_err(|_| MemoryError)?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(MemoryError);
    }
    // SAFETY: the global allocator, this exact layout, and every byte initialised by `alloc_zeroed`.
    Ok(unsafe { Vec::from_raw_parts(ptr, count, count) })
}

/// `rffi.alloc_buffer` (`jit.dont_look_inside`).
///
/// Returns `(raw_buffer, gc_buffer, case_num)`:
/// * `0` — the pointer is inside an unpinned, non-moving GC `STR`
/// * `1` — the GC `STR` was pinned (`rgc.pin`)
/// * `2` — raw `malloc` of `count` bytes; [`keep_buffer_alive_until_here`] frees it
///
/// `mallocfn` is `rstr.mallocstr`. `rgc.must_split_gc_address_space` is False.
/// `_alloc_buffer_nonlarge_max` (case 3) is unported: no in-tree user.
///
/// The caller must root `gc_buffer` before the next collecting call: `rgc.pin`
/// is not a root, and a young non-moving / nursery STR that is unreachable is
/// freed by the next minor collection.
#[majit_macros::dont_look_inside]
pub fn alloc_buffer(count: usize) -> Result<(CCHARP, majit_ir::GcRef, isize), MemoryError> {
    let new_buf = crate::lltypesystem::rstr::mallocstr(count)?;
    let mut pinned: isize = 0;
    let mut fallback = false;
    // `rgc.must_split_gc_address_space()` is False.
    if majit_gc::can_move(new_buf) {
        if majit_gc::pin(new_buf) {
            pinned = CASE_PINNED;
        } else {
            fallback = true;
        }
    }
    if fallback {
        let raw_buf = raw_malloc_or_memory_error(count)?.cast::<super::CHAR>();
        return Ok((raw_buf, new_buf, CASE_RAW));
    }
    let raw_buf = (new_buf.as_usize() as *mut u8).wrapping_add(STR_CHARS_OFFSET);
    Ok((raw_buf, new_buf, pinned))
}

/// `rffi.str_from_buffer` (`jit.dont_look_inside`). Converts a pair from
/// [`alloc_buffer`] to a high-level string of `needed_size` bytes: the GC
/// `STR` itself (`hlstr(gc_buf)`).
///
/// A size mismatch tries `llop.shrink_array` (`gc_shrink_array`) first. That
/// write is the length word only, so a successful shrink stores the trailing
/// NUL at `chars[needed_size]` here. On `false`, `rstr.mallocstr(needed_size)`
/// and case 2; case 2 copies `needed_size` bytes from `raw_buf`.
///
/// `mallocstr` does not collect. The caller must still root `gc_buf` across
/// this call: a syscall run with the GIL released (`call_external_function`)
/// lets another thread collect, and a pinned or nursery STR must stay
/// reachable. The returned STR is unrooted: the caller must root it (or copy
/// out of it) before its next allocation.
///
/// # Safety
/// `raw_buf` is readable for `needed_size` bytes. When it points into `gc_buf`,
/// that STR stays alive until after the copy.
#[majit_macros::dont_look_inside]
pub unsafe fn str_from_buffer(
    raw_buf: CCHARP,
    mut gc_buf: majit_ir::GcRef,
    mut case_num: isize,
    allocated_size: usize,
    needed_size: usize,
) -> Result<majit_ir::GcRef, MemoryError> {
    assert!(allocated_size >= needed_size);
    if allocated_size != needed_size {
        if majit_gc::gc_shrink_array(gc_buf.0, needed_size) {
            unsafe {
                (gc_buf.0 as *mut u8)
                    .add(STR_CHARS_OFFSET + needed_size)
                    .write(0);
            }
        } else {
            gc_buf = crate::lltypesystem::rstr::mallocstr(needed_size)?;
            case_num = CASE_RAW;
        }
    }
    if case_num == CASE_RAW {
        unsafe {
            copy_bytes(
                raw_buf.cast(),
                (gc_buf.0 as *mut u8).add(STR_CHARS_OFFSET),
                needed_size,
            );
        }
    }
    Ok(gc_buf)
}

/// `rffi.keep_buffer_alive_until_here` (`jit.dont_look_inside`).
///
/// `keepalive_until_here(gc_buf)`; case 1 → `rgc.unpin(gc_buf)`; case 2 →
/// `lltype.free(raw_buf, flavor='raw')`. Case 3 (`_alloc_buffer_nonlarge_max`)
/// is unported.
///
/// # Safety
/// `(raw_buf, gc_buf, case_num)` came from [`alloc_buffer`] and has not been
/// released. `gc_buf` is the original STR, still rooted by the caller.
#[majit_macros::dont_look_inside]
pub unsafe fn keep_buffer_alive_until_here(
    raw_buf: CCHARP,
    gc_buf: majit_ir::GcRef,
    case_num: isize,
) {
    let _keepalive = gc_buf;
    if case_num == CASE_PINNED {
        majit_gc::unpin(gc_buf);
    }
    if case_num == CASE_RAW {
        unsafe { raw_free(raw_buf.cast()) };
    }
}

/// `liststr2charpp`. NULL-terminated `char**`; each entry is a [`str2charp`].
pub fn liststr2charpp(list: &[&[u8]]) -> Result<super::CCHARPP, MemoryError> {
    let n = list.len();
    let n1 = n.checked_add(1).ok_or(MemoryError)?;
    // `ovfcheck(itemsize * n)` in `transform.py _ll_compute_size` is MemoryError.
    let bytes = n1
        .checked_mul(core::mem::size_of::<CCHARP>())
        .ok_or(MemoryError)?;
    unsafe {
        let array = raw_malloc_or_memory_error(bytes)?.cast::<CCHARP>();
        for (i, item) in list.iter().enumerate() {
            // `rffi.py liststr2charpp`: a raise from `str2charp` leaks earlier elements.
            *array.add(i) = str2charp(item, true)?;
        }
        *array.add(n) = core::ptr::null_mut();
        Ok(array)
    }
}

/// `free_charpp`. Frees each entry, then the pointer array.
///
/// # Safety
/// `charpp` came from [`liststr2charpp`] and has not been freed.
pub unsafe fn free_charpp(charpp: super::CCHARPP) {
    unsafe {
        let mut i = 0;
        while !(*charpp.add(i)).is_null() {
            free_charp(*charpp.add(i), true);
            i += 1;
        }
        raw_free(charpp.cast());
    }
}

/// `charpp2liststr`. Does not free `charpp`.
///
/// # Safety
/// `charpp` is a NULL-terminated array of NUL-terminated `char*`.
pub unsafe fn charpp2liststr(charpp: super::CCHARPP) -> Result<Vec<Vec<u8>>, MemoryError> {
    let mut result = Vec::new();
    unsafe {
        let mut i = 0;
        while !(*charpp.add(i)).is_null() {
            result.push(charp2str(*charpp.add(i))?);
            i += 1;
        }
    }
    Ok(result)
}

/// `wcharpsize2utf8`. Encodes `size` code units, NULs included.
///
/// `rutf8.unichr_as_utf8_append(..., allow_surrogates=True)`. A code above
/// `0x10FFFF` is [`OutOfRange`].
///
/// # Safety
/// `w` is readable for `size` `wchar_t`s.
pub unsafe fn wcharpsize2utf8(w: CWCHARP, size: usize) -> Result<Vec<u8>, OutOfRange> {
    let mut out = Vec::new();
    for i in 0..size {
        push_utf8(&mut out, unsafe { wchar_ord(w, i) })?;
    }
    Ok(out)
}

/// `wcharp2utf8`. Stops at the first zero code unit.
///
/// Returns the UTF-8 bytes and the number of `wchar_t`s read, not counting
/// the terminator.
///
/// # Safety
/// `w` is NUL-terminated.
pub unsafe fn wcharp2utf8(w: CWCHARP) -> Result<(Vec<u8>, usize), OutOfRange> {
    let mut out = Vec::new();
    let mut i = 0;
    unsafe {
        while wchar_ord(w, i) != 0 {
            push_utf8(&mut out, wchar_ord(w, i))?;
            i += 1;
        }
    }
    Ok((out, i))
}

/// `wcharp2utf8n`. Like [`wcharp2utf8`], but stops after `maxlen` units.
///
/// # Safety
/// `w` is readable for `maxlen` units or up to a zero unit, whichever is first.
pub unsafe fn wcharp2utf8n(w: CWCHARP, maxlen: usize) -> Result<(Vec<u8>, usize), OutOfRange> {
    let mut out = Vec::new();
    let mut i = 0;
    unsafe {
        while i < maxlen && wchar_ord(w, i) != 0 {
            push_utf8(&mut out, wchar_ord(w, i))?;
            i += 1;
        }
    }
    Ok((out, i))
}

/// `utf82wcharp`. One `wchar_t` per code point, then a zero unit.
///
/// Allocates `utf8len + 2` units (`utf82wcharp`: one for the terminator, one
/// for a trailing incomplete sequence). `utf8len` is the code-point count, not
/// the byte length; upstream asserts it against `codepoints_in_utf8` before
/// walking the whole string. The count check is debug-only, as upstream checks
/// it only `if not we_are_translated()`; callers pass the `W_UnicodeObject`'s
/// stored length. `track_allocation` is the same debug-tracker flag as
/// [`str2charp`].
pub fn utf82wcharp(
    utf8: &[u8],
    utf8len: usize,
    track_allocation: bool,
) -> Result<CWCHARP, MemoryError> {
    let _ = track_allocation;
    debug_assert_eq!(utf8len, codepoint_len(utf8));
    let chars = utf8len.checked_add(2).ok_or(MemoryError)?;
    // `ovfcheck(itemsize * n)` in `transform.py _ll_compute_size` is MemoryError.
    let bytes = chars
        .checked_mul(core::mem::size_of::<Wchar>())
        .ok_or(MemoryError)?;
    unsafe {
        let w = raw_malloc_or_memory_error(bytes)?.cast::<Wchar>();
        let mut index = 0;
        let mut pos = 0;
        while let Some(ch) = next_codepoint(utf8, &mut pos) {
            *w.add(index) = ch as Wchar;
            index += 1;
        }
        *w.add(index) = 0;
        Ok(w)
    }
}

/// `utf82wcharp_ex`. Like [`utf82wcharp`], but a code point above `0xffff`
/// becomes a surrogate pair, the layout a 2-byte `wchar_t` needs. Surrogates
/// in the input pass through. `unilen` is accepted and unused, and the block
/// holds `wlen + 3` units, where `wlen` counts the units written before the
/// zero terminator.
pub fn utf82wcharp_ex(
    utf8: &[u8],
    unilen: usize,
    track_allocation: bool,
) -> Result<CWCHARP, MemoryError> {
    let _ = (unilen, track_allocation);
    let mut wlen: usize = 0;
    let mut pos = 0;
    while let Some(ch) = next_codepoint(utf8, &mut pos) {
        if ch > 0xffff {
            wlen += 1;
        }
        wlen += 1;
    }
    let n = wlen.checked_add(3).ok_or(MemoryError)?;
    // `ovfcheck(itemsize * n)` in `transform.py _ll_compute_size` is MemoryError.
    let bytes = n
        .checked_mul(core::mem::size_of::<Wchar>())
        .ok_or(MemoryError)?;
    unsafe {
        let w = raw_malloc_or_memory_error(bytes)?.cast::<Wchar>();
        let mut index = 0;
        let mut pos = 0;
        while let Some(ch) = next_codepoint(utf8, &mut pos) {
            if ch > 0xffff {
                *w.add(index) = (0xD800 | ((ch - 0x10000) >> 10)) as Wchar;
                index += 1;
                *w.add(index) = (0xDC00 | ((ch - 0x10000) & 0x3FF)) as Wchar;
            } else {
                *w.add(index) = ch as Wchar;
            }
            index += 1;
        }
        *w.add(index) = 0;
        assert_eq!(wlen, index);
        Ok(w)
    }
}

/// `free_wcharp`. `free` of a block from [`utf82wcharp`].
///
/// # Safety
/// `cp` is null or a pointer returned by [`utf82wcharp`] or [`utf82wcharp_ex`]
/// that has not been freed.
pub unsafe fn free_wcharp(cp: CWCHARP, track_allocation: bool) {
    let _ = track_allocation;
    unsafe { raw_free(cp.cast()) }
}

/// `scoped_str2charp`. `buf` is what `__enter__` yields. `None` yields NULL.
/// [`Drop`] is `__exit__` and frees a non-null buffer.
#[derive(Debug)]
pub struct scoped_str2charp {
    pub buf: CCHARP,
}

impl scoped_str2charp {
    pub fn new(value: Option<&[u8]>) -> Result<Self, MemoryError> {
        let buf = match value {
            Some(s) => str2charp(s, true)?,
            None => core::ptr::null_mut(),
        };
        Ok(Self { buf })
    }
}

impl Drop for scoped_str2charp {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe { free_charp(self.buf, true) };
            self.buf = core::ptr::null_mut();
        }
    }
}

/// `scoped_nonmovingbuffer`. [`Drop`] calls [`free_nonmovingbuffer_ll`].
///
/// `llobj` is `llstr(data)`. While the scope lives the STR must stay
/// reachable: the caller keeps the owning object rooted (`gc_roots`) or
/// the STR is old-gen.
#[derive(Debug)]
pub struct scoped_nonmovingbuffer {
    pub buf: CCHARP,
    pub llobj: majit_ir::GcRef,
    pub flag: u8,
}

impl scoped_nonmovingbuffer {
    pub fn new(llobj: majit_ir::GcRef) -> Result<Self, MemoryError> {
        let (buf, llobj, flag) = get_nonmovingbuffer_ll(llobj)?;
        Ok(Self { buf, llobj, flag })
    }
}

impl Drop for scoped_nonmovingbuffer {
    fn drop(&mut self) {
        unsafe { free_nonmovingbuffer_ll(self.buf, self.llobj, self.flag) };
        self.flag = FLAG_INSIDE_NONMOVABLE;
        self.llobj = majit_ir::GcRef::NULL;
        self.buf = core::ptr::null_mut();
    }
}

/// `scoped_view_charp`. Same as [`scoped_nonmovingbuffer`], but the buffer is
/// NUL-terminated (`get_nonmovingbuffer_ll_final_null`).
///
/// `llobj` is `llstr(data)`. While the scope lives the STR must stay
/// reachable: the caller keeps the owning object rooted (`gc_roots`) or
/// the STR is old-gen.
#[derive(Debug)]
pub struct scoped_view_charp {
    pub buf: CCHARP,
    pub llobj: majit_ir::GcRef,
    pub flag: u8,
}

impl scoped_view_charp {
    pub fn new(llobj: majit_ir::GcRef) -> Result<Self, MemoryError> {
        let (buf, llobj, flag) = get_nonmovingbuffer_ll_final_null(llobj)?;
        Ok(Self { buf, llobj, flag })
    }
}

impl Drop for scoped_view_charp {
    fn drop(&mut self) {
        unsafe { free_nonmovingbuffer_ll(self.buf, self.llobj, self.flag) };
        self.flag = FLAG_INSIDE_NONMOVABLE;
        self.llobj = majit_ir::GcRef::NULL;
        self.buf = core::ptr::null_mut();
    }
}

/// `scoped_alloc_buffer`. `__enter__` is [`Self::new`]; [`Self::str`] is
/// `str_from_buffer` of the original `raw` / `gc_buf` / `case_num`; [`Drop`]
/// is `__exit__` (`keep_buffer_alive_until_here` of those originals).
///
/// Host-side, by-value guard whose `gc_buf` STR lives in a fixed owner-root
/// slot for the guard's lifetime.
///
/// `rgc.pin` is not a root, and a young non-moving / nursery STR that is
/// unreachable is freed by the next minor collection. `mallocstr` does not
/// collect; the owner-root slot stays required because a syscall run under
/// `call_external_function` lets another thread collect, and a pinned or
/// nursery STR must stay reachable.
///
/// RPython's GC transform gives a local that is live across several
/// collecting calls a *fixed* frame slot (`gc_save_root` /
/// `gc_restore_root` into the same index). The LIFO shadow stack is only
/// the `push_roots` / `pop_roots` bump around one call
/// (`framework.py`); `CurrentFrameGuard` and the JIT `FrameRoot` truncate
/// that stack with `pop_to`. A long-lived RAII handle therefore cannot
/// share it: a nested frame that pops, or `release` compacting a hole,
/// would hand the index to a later push and a later read would reload that
/// later value as `gc_buf`.
///
/// [`majit_gc::shadow_stack::acquire_owner_root`] is that fixed-slot vector.
/// Reads go through [`majit_gc::shadow_stack::get_owner_root`]: the GcRef is
/// never cached across a collecting call.
///
/// [`Self::str`] returns an unrooted STR: the caller must root it (or copy
/// out of it) before its next allocation.
#[derive(Debug)]
pub struct scoped_alloc_buffer {
    pub raw: CCHARP,
    slot: usize,
    pub case_num: isize,
    pub size: usize,
    /// `slot` indexes the acquiring thread's owner-root vector, the way a
    /// shadow-stack slot belongs to the frame's own thread. Without this
    /// marker the guard is auto-`Send`, and dropping it on another thread
    /// would release that thread's slot.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl scoped_alloc_buffer {
    pub fn new(size: usize) -> Result<Self, MemoryError> {
        let (raw, gc_buf, case_num) = alloc_buffer(size)?;
        let slot = majit_gc::shadow_stack::acquire_owner_root(gc_buf);
        Ok(Self {
            raw,
            slot,
            case_num,
            size,
            _not_send: std::marker::PhantomData,
        })
    }

    /// `scoped_alloc_buffer.str`.
    ///
    /// Calls [`str_from_buffer`] with the slot's current GcRef. [`Drop`] then
    /// runs [`keep_buffer_alive_until_here`] with the original `raw` / `gc_buf`
    /// / `case_num` and releases the slot.
    ///
    /// The returned STR is unrooted: the caller must root it (or copy out of
    /// it) before its next allocation.
    pub fn str(self, length: usize) -> Result<majit_ir::GcRef, MemoryError> {
        let gc_buf = majit_gc::shadow_stack::get_owner_root(self.slot);
        unsafe { str_from_buffer(self.raw, gc_buf, self.case_num, self.size, length) }
    }
}

impl Drop for scoped_alloc_buffer {
    fn drop(&mut self) {
        let gc_buf = majit_gc::shadow_stack::get_owner_root(self.slot);
        unsafe { keep_buffer_alive_until_here(self.raw, gc_buf, self.case_num) };
        majit_gc::shadow_stack::release_owner_root(self.slot);
        self.case_num = CASE_INSIDE;
        self.raw = core::ptr::null_mut();
    }
}

/// `scoped_utf82wcharp`. `unicode_len < 0` counts code points in `value`.
/// `None` yields NULL. [`Drop`] frees a non-null buffer with [`free_wcharp`].
#[derive(Debug)]
pub struct scoped_utf82wcharp {
    pub buf: CWCHARP,
}

impl scoped_utf82wcharp {
    pub fn new(value: Option<&[u8]>, unicode_len: isize) -> Result<Self, MemoryError> {
        let buf = match value {
            None => core::ptr::null_mut(),
            Some(utf8) => {
                let n = if unicode_len < 0 {
                    codepoint_len(utf8)
                } else {
                    unicode_len as usize
                };
                utf82wcharp(utf8, n, true)?
            }
        };
        Ok(Self { buf })
    }
}

impl Drop for scoped_utf82wcharp {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe { free_wcharp(self.buf, true) };
            self.buf = core::ptr::null_mut();
        }
    }
}

fn codepoint_len(utf8: &[u8]) -> usize {
    let mut pos = 0;
    let mut n = 0;
    while next_codepoint(utf8, &mut pos).is_some() {
        n += 1;
    }
    n
}

/// `Utf8StringIterator.next` for a byte slice. A truncated tail yields the
/// lead byte instead of reading off the end of the slice.
fn next_codepoint(utf8: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos >= utf8.len() {
        return None;
    }
    let b1 = utf8[*pos] as u32;
    if b1 <= 0x7F {
        *pos += 1;
        return Some(b1);
    }
    if *pos + 1 >= utf8.len() {
        *pos += 1;
        return Some(b1);
    }
    let b2 = utf8[*pos + 1] as u32;
    if b1 <= 0xDF {
        *pos += 2;
        return Some((b1 << 6) + b2 - ((0xC0 << 6) + 0x80));
    }
    if *pos + 2 >= utf8.len() {
        *pos += 1;
        return Some(b1);
    }
    let b3 = utf8[*pos + 2] as u32;
    if b1 <= 0xEF {
        *pos += 3;
        return Some((b1 << 12) + (b2 << 6) + b3 - ((0xE0 << 12) + (0x80 << 6) + 0x80));
    }
    if *pos + 3 >= utf8.len() {
        *pos += 1;
        return Some(b1);
    }
    let b4 = utf8[*pos + 3] as u32;
    *pos += 4;
    Some(
        (b1 << 18) + (b2 << 12) + (b3 << 6) + b4
            - ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80),
    )
}

/// `unichr_as_utf8_append(..., allow_surrogates=True)`.
fn push_utf8(out: &mut Vec<u8>, code: u32) -> Result<(), OutOfRange> {
    if code <= 0x7F {
        out.push(code as u8);
    } else if code <= 0x07FF {
        out.push((0xC0 | (code >> 6)) as u8);
        out.push((0x80 | (code & 0x3F)) as u8);
    } else if code <= 0xFFFF {
        out.push((0xE0 | (code >> 12)) as u8);
        out.push((0x80 | ((code >> 6) & 0x3F)) as u8);
        out.push((0x80 | (code & 0x3F)) as u8);
    } else if code <= 0x10FFFF {
        out.push((0xF0 | (code >> 18)) as u8);
        out.push((0x80 | ((code >> 12) & 0x3F)) as u8);
        out.push((0x80 | ((code >> 6) & 0x3F)) as u8);
        out.push((0x80 | (code & 0x3F)) as u8);
    } else {
        return Err(OutOfRange { code });
    }
    Ok(())
}

unsafe fn wchar_ord(w: CWCHARP, index: usize) -> u32 {
    unsafe { *w.add(index) as u32 }
}

/// `llmemory.raw_memcopy` (`llmemory.py ann_raw_memcopy`): a void memcpy.
/// The body is `ptr::copy_nonoverlapping`; residualizing the helper keeps
/// that std leaf as the `ExtRegistryEntry` already registered for it,
/// matching `ll_raw_free`'s `dont_look_inside_cannot_raise` around
/// `raw_free`.
#[majit_macros::dont_look_inside_cannot_raise]
unsafe fn copy_bytes(src: *const u8, dst: *mut u8, n: usize) {
    if n > 0 {
        unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
    }
}

/// `lltype.malloc(..., flavor='raw')` for helpers called from ordinary Rust.
/// NULL from [`raw_malloc`] is [`MemoryError`]. Does not call
/// `majit_gc::raise_memory_error` (`raw_malloc_memory_error` is the
/// `extern "C"` `ll_raw_malloc_*` path).
fn raw_malloc_or_memory_error(size: usize) -> Result<*mut u8, MemoryError> {
    let p = unsafe { raw_malloc(size) };
    if p.is_null() { Err(MemoryError) } else { Ok(p) }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) unsafe fn raw_malloc(size: usize) -> *mut u8 {
    let request = size.max(1);
    unsafe { libc::malloc(request) }.cast()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) unsafe fn raw_free(ptr: *mut u8) {
    unsafe { libc::free(ptr.cast()) }
}

/// Header bytes in front of the payload, and the alignment of that payload.
///
/// C `malloc` aligns for `max_align_t` (16). `GCTransformer` registers
/// `ll_raw_malloc_fixedsize` as one `Signed`; `llmemory.raw_malloc` asks
/// the C allocator for that many bytes. wasm32 has no libc, and a `usize`
/// header there is 4 bytes, so the block uses this 16-byte header instead.
#[cfg(any(target_arch = "wasm32", test))]
const RAW_MALLOC_ALIGN: usize = 16;

/// 16-byte header plus `size` payload bytes. `Layout::from_size_align`
/// rejects a size that is not a multiple of the alignment, so the total is
/// rounded up. [`raw_free`] rebuilds this same layout from the stored size.
#[cfg(any(target_arch = "wasm32", test))]
fn try_raw_malloc_layout(size: usize) -> Option<std::alloc::Layout> {
    let payload = size.max(1);
    let total = RAW_MALLOC_ALIGN
        .checked_add(payload)
        .and_then(|n| n.checked_next_multiple_of(RAW_MALLOC_ALIGN))?;
    std::alloc::Layout::from_size_align(total, RAW_MALLOC_ALIGN).ok()
}

#[cfg(any(target_arch = "wasm32", test))]
fn raw_malloc_layout(size: usize) -> std::alloc::Layout {
    try_raw_malloc_layout(size).expect("raw malloc size overflow")
}

/// wasm32-unknown-unknown does not link `libc`. The requested size sits in
/// the header so [`raw_free`] can rebuild the [`raw_malloc_layout`].
#[cfg(target_arch = "wasm32")]
pub(crate) unsafe fn raw_malloc(size: usize) -> *mut u8 {
    let Some(layout) = try_raw_malloc_layout(size) else {
        return core::ptr::null_mut();
    };
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        base.cast::<usize>().write(size);
        base.add(RAW_MALLOC_ALIGN)
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) unsafe fn raw_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = unsafe { ptr.sub(RAW_MALLOC_ALIGN) };
    let size = unsafe { base.cast::<usize>().read() };
    let layout = raw_malloc_layout(size);
    unsafe { std::alloc::dealloc(base, layout) }
}

/// `framework.py raw_malloc_memory_pressure`: charge `sizehint`, `adr=NULL`.
fn raw_malloc_fixedsize_memory_pressure(size: i64) {
    majit_gc::add_memory_pressure(size as isize, majit_ir::GcRef::NULL);
}

/// Fixed-size `lltype.malloc(STRUCT, flavor='raw')`.
///
/// `support.py build_raw_malloc_fixedsize_builder` closes `zero` /
/// `add_memory_pressure` / `track_allocation` into one helper per flag
/// combination. The size is a constant argument; alignment stays inside
/// `raw_malloc`. `GCTransformer` publishes the helper as one `Signed`.
///
/// `size` and the address are `i64` words (`extern "C"`). A `usize` parameter
/// is i32 on wasm32, and `call_indirect` requires this signature.
fn raw_malloc_fixedsize_core(size: i64, zero: bool, add_memory_pressure: bool) -> i64 {
    if add_memory_pressure {
        raw_malloc_fixedsize_memory_pressure(size);
    }
    let Ok(bytes) = usize::try_from(size) else {
        return raw_malloc_memory_error();
    };
    let ptr = unsafe { raw_malloc(bytes) };
    if ptr.is_null() {
        return raw_malloc_memory_error();
    }
    if zero && bytes > 0 {
        unsafe { ptr.write_bytes(0, bytes) };
    }
    ptr as usize as i64
}

#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize(size: i64) -> i64 {
    raw_malloc_fixedsize_core(size, false, false)
}

/// `zero=True` form of [`ll_raw_malloc_fixedsize`]. The block is cleared
/// for `size` bytes. A zero-size request still returns a distinct block
/// and writes nothing. Same word ABI as [`ll_raw_malloc_fixedsize`].
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_zero(size: i64) -> i64 {
    raw_malloc_fixedsize_core(size, true, false)
}

/// `_rewrite_raw_malloc` `name += '_add_memory_pressure'`.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_add_memory_pressure(size: i64) -> i64 {
    raw_malloc_fixedsize_core(size, false, true)
}

/// `_rewrite_raw_malloc` `_zero` + `_add_memory_pressure`.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_zero_add_memory_pressure(size: i64) -> i64 {
    raw_malloc_fixedsize_core(size, true, true)
}

/// `_rewrite_raw_malloc` `name += '_no_track_allocation'`. The port does
/// no leak tracking; the allocation is the same as the base helper.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_no_track_allocation(size: i64) -> i64 {
    ll_raw_malloc_fixedsize(size)
}

#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_zero_no_track_allocation(size: i64) -> i64 {
    ll_raw_malloc_fixedsize_zero(size)
}

#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation(
    size: i64,
) -> i64 {
    ll_raw_malloc_fixedsize_add_memory_pressure(size)
}

#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation(
    size: i64,
) -> i64 {
    ll_raw_malloc_fixedsize_zero_add_memory_pressure(size)
}

/// `transform.py _ll_compute_size`: `ovfcheck(itemsize * n)` then
/// `ovfcheck(basesize + varsize)`. Overflow is MemoryError. A result
/// that does not fit a signed word (`tot_size < 0` or
/// `tot_size > sys.maxint`) is MemoryError too.
fn ll_compute_size(n: i64, basesize: i64, itemsize: i64) -> Option<usize> {
    let varsize = itemsize.checked_mul(n)?;
    let tot_size = basesize.checked_add(varsize)?;
    let bytes = usize::try_from(tot_size).ok()?;
    if bytes > isize::MAX as usize {
        return None;
    }
    Some(bytes)
}

/// MemoryError result of a raw malloc helper: raise the prebuilt
/// singleton (`llmodel.py pos_exception` / `pos_exc_value`) and return
/// NULL, the same false `allocate` result `_ll_malloc_fixedsize` raises
/// MemoryError on.
fn raw_malloc_memory_error() -> i64 {
    majit_gc::raise_memory_error();
    0
}

/// `framework.py raw_malloc_memory_pressure_varsize`: charge
/// `length * itemsize` when that product is positive, `adr=NULL`.
fn raw_malloc_varsize_memory_pressure(n: i64, itemsize: i64) {
    let Some(totalmem) = n.checked_mul(itemsize) else {
        return;
    };
    if totalmem > 0 {
        majit_gc::add_memory_pressure(totalmem as isize, majit_ir::GcRef::NULL);
    }
}

/// `transform.py ll_malloc_varsize_no_length` / `ll_malloc_varsize`.
/// `lengthoffset` is `Some` when the ARRAY is not `nolength`; the
/// helper then stores `n` at that offset (`llop.raw_store`).
fn raw_malloc_varsize_core(
    n: i64,
    basesize: i64,
    itemsize: i64,
    lengthoffset: Option<i64>,
    zero: bool,
    add_memory_pressure: bool,
) -> i64 {
    if add_memory_pressure {
        raw_malloc_varsize_memory_pressure(n, itemsize);
    }
    let Some(bytes) = ll_compute_size(n, basesize, itemsize) else {
        return raw_malloc_memory_error();
    };
    let ptr = unsafe { raw_malloc(bytes) };
    if ptr.is_null() {
        return raw_malloc_memory_error();
    }
    if zero && bytes > 0 {
        unsafe { ptr.write_bytes(0, bytes) };
    }
    if let Some(offset) = lengthoffset {
        let off = usize::try_from(offset).expect("lengthoffset");
        unsafe { ptr.add(off).cast::<isize>().write(n as isize) };
    }
    ptr as usize as i64
}

macro_rules! publish_raw_varsize {
    ($static:ident, $name:ident, $arity:literal) => {
        #[cfg(not(target_arch = "wasm32"))]
        #[majit_ir::linkme::distributed_slice(majit_ir::helper_fnaddr::HELPER_FNADDRS)]
        #[linkme(crate = majit_ir::linkme)]
        #[allow(non_upper_case_globals)]
        static $static: majit_ir::helper_fnaddr::HelperFnAddr =
            majit_ir::helper_fnaddr::HelperFnAddr::new(
                concat!("majit_rlib::rffi::", stringify!($name)),
                $name as *const (),
                $arity,
            );
    };
}

/// `transform.py ll_malloc_varsize_no_length`.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_varsize_no_length(n: i64, basesize: i64, itemsize: i64) -> i64 {
    raw_malloc_varsize_core(n, basesize, itemsize, None, false, false)
}

/// `transform.py ll_malloc_varsize_no_length_zero`.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_varsize_no_length_zero(
    n: i64,
    basesize: i64,
    itemsize: i64,
) -> i64 {
    raw_malloc_varsize_core(n, basesize, itemsize, None, true, false)
}

/// Variable-size `lltype.malloc(ARRAY, n, flavor='raw')` with no length
/// header. Kept as the 3-arg ABI `blackhole.rs` / `jit_fnaddr.rs` bind.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_varsize(n: i64, basesize: i64, itemsize: i64) -> i64 {
    ll_raw_malloc_varsize_no_length(n, basesize, itemsize)
}

/// `zero=True` form of [`ll_raw_malloc_varsize`].
#[inline(never)]
pub extern "C" fn ll_raw_malloc_varsize_zero(n: i64, basesize: i64, itemsize: i64) -> i64 {
    ll_raw_malloc_varsize_no_length_zero(n, basesize, itemsize)
}

/// `transform.py ll_malloc_varsize` — stores `n` at `lengthoffset`.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_varsize_with_length(
    n: i64,
    basesize: i64,
    itemsize: i64,
    lengthoffset: i64,
) -> i64 {
    raw_malloc_varsize_core(n, basesize, itemsize, Some(lengthoffset), false, false)
}

macro_rules! raw_varsize_flag_wrappers {
    () => {
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_add_memory_pressure(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            raw_malloc_varsize_core(n, basesize, itemsize, None, false, true)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_zero_add_memory_pressure(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            raw_malloc_varsize_core(n, basesize, itemsize, None, true, true)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_no_length(n, basesize, itemsize)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_zero_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_no_length_zero(n, basesize, itemsize)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_add_memory_pressure_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_no_length_add_memory_pressure(n, basesize, itemsize)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_no_length_zero_add_memory_pressure_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_no_length_zero_add_memory_pressure(n, basesize, itemsize)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_with_length_add_memory_pressure(
            n: i64,
            basesize: i64,
            itemsize: i64,
            lengthoffset: i64,
        ) -> i64 {
            raw_malloc_varsize_core(n, basesize, itemsize, Some(lengthoffset), false, true)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_with_length_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
            lengthoffset: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_with_length(n, basesize, itemsize, lengthoffset)
        }
        #[inline(never)]
        pub extern "C" fn ll_raw_malloc_varsize_with_length_add_memory_pressure_no_track_allocation(
            n: i64,
            basesize: i64,
            itemsize: i64,
            lengthoffset: i64,
        ) -> i64 {
            ll_raw_malloc_varsize_with_length_add_memory_pressure(
                n,
                basesize,
                itemsize,
                lengthoffset,
            )
        }
    };
}
raw_varsize_flag_wrappers!();

publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize,
    ll_raw_malloc_varsize,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_zero,
    ll_raw_malloc_varsize_zero,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length,
    ll_raw_malloc_varsize_no_length,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_zero,
    ll_raw_malloc_varsize_no_length_zero,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_add_memory_pressure,
    ll_raw_malloc_varsize_no_length_add_memory_pressure,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_zero_add_memory_pressure,
    ll_raw_malloc_varsize_no_length_zero_add_memory_pressure,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_no_track_allocation,
    ll_raw_malloc_varsize_no_length_no_track_allocation,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_zero_no_track_allocation,
    ll_raw_malloc_varsize_no_length_zero_no_track_allocation,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_add_memory_pressure_no_track_allocation,
    ll_raw_malloc_varsize_no_length_add_memory_pressure_no_track_allocation,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_no_length_zero_add_memory_pressure_no_track_allocation,
    ll_raw_malloc_varsize_no_length_zero_add_memory_pressure_no_track_allocation,
    3
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_with_length,
    ll_raw_malloc_varsize_with_length,
    4
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_with_length_add_memory_pressure,
    ll_raw_malloc_varsize_with_length_add_memory_pressure,
    4
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_with_length_no_track_allocation,
    ll_raw_malloc_varsize_with_length_no_track_allocation,
    4
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_varsize_with_length_add_memory_pressure_no_track_allocation,
    ll_raw_malloc_varsize_with_length_add_memory_pressure_no_track_allocation,
    4
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize,
    ll_raw_malloc_fixedsize,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_zero,
    ll_raw_malloc_fixedsize_zero,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_add_memory_pressure,
    ll_raw_malloc_fixedsize_add_memory_pressure,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_zero_add_memory_pressure,
    ll_raw_malloc_fixedsize_zero_add_memory_pressure,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_no_track_allocation,
    ll_raw_malloc_fixedsize_no_track_allocation,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_zero_no_track_allocation,
    ll_raw_malloc_fixedsize_zero_no_track_allocation,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation,
    ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation,
    1
);
publish_raw_varsize!(
    __MAJIT_HELPER_FNADDR_ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation,
    ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation,
    1
);

#[cfg(target_arch = "wasm32")]
#[ctor::ctor(unsafe)]
fn __majit_register_helper_fnaddr_ll_raw_malloc_varsize() {
    fn add(path: &'static str, addr: *const (), arity: u8) {
        majit_ir::helper_fnaddr::register(path, addr, arity);
    }
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize",
        ll_raw_malloc_varsize as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_zero",
        ll_raw_malloc_varsize_zero as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length",
        ll_raw_malloc_varsize_no_length as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_zero",
        ll_raw_malloc_varsize_no_length_zero as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_add_memory_pressure",
        ll_raw_malloc_varsize_no_length_add_memory_pressure as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_zero_add_memory_pressure",
        ll_raw_malloc_varsize_no_length_zero_add_memory_pressure as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_no_track_allocation",
        ll_raw_malloc_varsize_no_length_no_track_allocation as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_zero_no_track_allocation",
        ll_raw_malloc_varsize_no_length_zero_no_track_allocation as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_add_memory_pressure_no_track_allocation",
        ll_raw_malloc_varsize_no_length_add_memory_pressure_no_track_allocation as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_no_length_zero_add_memory_pressure_no_track_allocation",
        ll_raw_malloc_varsize_no_length_zero_add_memory_pressure_no_track_allocation as *const (),
        3,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_with_length",
        ll_raw_malloc_varsize_with_length as *const (),
        4,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_with_length_add_memory_pressure",
        ll_raw_malloc_varsize_with_length_add_memory_pressure as *const (),
        4,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_with_length_no_track_allocation",
        ll_raw_malloc_varsize_with_length_no_track_allocation as *const (),
        4,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_varsize_with_length_add_memory_pressure_no_track_allocation",
        ll_raw_malloc_varsize_with_length_add_memory_pressure_no_track_allocation as *const (),
        4,
    );
}

#[cfg(target_arch = "wasm32")]
#[ctor::ctor(unsafe)]
fn __majit_register_helper_fnaddr_ll_raw_malloc_fixedsize() {
    fn add(path: &'static str, addr: *const (), arity: u8) {
        majit_ir::helper_fnaddr::register(path, addr, arity);
    }
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize",
        ll_raw_malloc_fixedsize as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_zero",
        ll_raw_malloc_fixedsize_zero as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_add_memory_pressure",
        ll_raw_malloc_fixedsize_add_memory_pressure as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_zero_add_memory_pressure",
        ll_raw_malloc_fixedsize_zero_add_memory_pressure as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_no_track_allocation",
        ll_raw_malloc_fixedsize_no_track_allocation as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_zero_no_track_allocation",
        ll_raw_malloc_fixedsize_zero_no_track_allocation as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation",
        ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation as *const (),
        1,
    );
    add(
        "majit_rlib::rffi::ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation",
        ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation as *const (),
        1,
    );
}

/// `lltype.free(ptr, flavor='raw')`. `jtransform.py rewrite_op_free`
/// residualizes this as `raw_free`; the call cannot raise.
/// `ptr` is an `i64` word; the body casts it to `usize`.
#[inline(never)]
#[majit_macros::oopspec("raw_free(ptr)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn ll_raw_free(ptr: i64) {
    unsafe { raw_free(ptr as usize as *mut u8) }
}

#[cfg(test)]
mod tests {
    use super::{
        ll_raw_free, ll_raw_malloc_fixedsize, ll_raw_malloc_fixedsize_add_memory_pressure,
        ll_raw_malloc_fixedsize_zero, ll_raw_malloc_fixedsize_zero_add_memory_pressure,
        ll_raw_malloc_varsize, ll_raw_malloc_varsize_no_length,
        ll_raw_malloc_varsize_no_length_zero, ll_raw_malloc_varsize_with_length,
        ll_raw_malloc_varsize_zero,
    };

    static RAISE_HOOK_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn fixedsize_roundtrip_zero_clears_and_free() {
        let ptr = ll_raw_malloc_fixedsize(8);
        assert_ne!(ptr, 0);
        assert_eq!((ptr as usize) & 15, 0);
        unsafe {
            let p = ptr as *mut u8;
            p.write(0x5A);
            assert_eq!(p.read(), 0x5A);
        }
        ll_raw_free(ptr);

        let zeroed = ll_raw_malloc_fixedsize_zero(8);
        assert_ne!(zeroed, 0);
        assert_eq!((zeroed as usize) & 15, 0);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, 8);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        ll_raw_free(zeroed);

        // Not a multiple of 16. `malloc` (and the wasm32 16-byte header)
        // still returns a 16-byte-aligned payload.
        let odd = ll_raw_malloc_fixedsize(1);
        assert_ne!(odd, 0);
        assert_eq!((odd as usize) & 15, 0);
        unsafe {
            let p = odd as *mut u8;
            p.write(0x11);
            assert_eq!(p.read(), 0x11);
        }
        ll_raw_free(odd);
        ll_raw_free(0);
    }

    /// `framework.py raw_malloc_memory_pressure(sizehint, adr)` charges
    /// the allocation size. The `_zero` form still clears the block.
    #[test]
    fn raw_malloc_fixedsize_add_memory_pressure_charges_size() {
        use std::sync::atomic::{AtomicIsize, Ordering};
        static CHARGED: AtomicIsize = AtomicIsize::new(0);
        fn hook(size: isize, _object: majit_ir::GcRef) {
            CHARGED.fetch_add(size, Ordering::SeqCst);
        }
        let _guard = crate::MEMORY_PRESSURE_HOOK_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        CHARGED.store(0, Ordering::SeqCst);
        majit_gc::set_active_add_memory_pressure(Some(hook));
        let ptr = ll_raw_malloc_fixedsize_add_memory_pressure(8);
        majit_gc::set_active_add_memory_pressure(None);
        assert_ne!(ptr, 0);
        assert_eq!(CHARGED.load(Ordering::SeqCst), 8);
        ll_raw_free(ptr);

        CHARGED.store(0, Ordering::SeqCst);
        majit_gc::set_active_add_memory_pressure(Some(hook));
        let zeroed = ll_raw_malloc_fixedsize_zero_add_memory_pressure(8);
        majit_gc::set_active_add_memory_pressure(None);
        assert_ne!(zeroed, 0);
        assert_eq!(CHARGED.load(Ordering::SeqCst), 8);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, 8);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        ll_raw_free(zeroed);

        CHARGED.store(0, Ordering::SeqCst);
        majit_gc::set_active_add_memory_pressure(Some(hook));
        let plain = ll_raw_malloc_fixedsize(8);
        majit_gc::set_active_add_memory_pressure(None);
        assert_ne!(plain, 0);
        assert_eq!(CHARGED.load(Ordering::SeqCst), 0);
        ll_raw_free(plain);
    }

    /// `transform.py _ll_compute_size`: payload is `basesize + n * itemsize`.
    /// The `_zero` helper clears that many bytes. Overflow of the signed
    /// product or sum is MemoryError (`NULL`, the false `allocate` result
    /// `_ll_malloc_fixedsize` raises on).
    #[test]
    fn raw_malloc_varsize_basesize_plus_n_itemsize_zero_and_overflow() {
        let ptr = ll_raw_malloc_varsize(3, 8, 4);
        assert_ne!(ptr, 0);
        assert_eq!((ptr as usize) & 15, 0);
        unsafe {
            let p = ptr as *mut u8;
            for i in 0..20 {
                p.add(i).write(0x5A);
                assert_eq!(p.add(i).read(), 0x5A);
            }
        }
        ll_raw_free(ptr);

        let zeroed = ll_raw_malloc_varsize_zero(3, 8, 4);
        assert_ne!(zeroed, 0);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, 20);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        ll_raw_free(zeroed);

        let empty = ll_raw_malloc_varsize(0, 0, 8);
        assert_ne!(empty, 0);
        ll_raw_free(empty);

        let _guard = RAISE_HOOK_TEST.lock().unwrap_or_else(|e| e.into_inner());
        fn hook() {}
        majit_gc::set_active_raise_memory_error(Some(hook));
        assert_eq!(ll_raw_malloc_varsize(i64::MAX, 0, 2), 0);
        assert_eq!(ll_raw_malloc_varsize_zero(i64::MAX, 8, 8), 0);
        // n=-1, basesize=0, itemsize=8 → tot_size=-8, MemoryError.
        assert_eq!(ll_raw_malloc_varsize(-1, 0, 8), 0);
        assert_eq!(ll_raw_malloc_varsize_no_length(i64::MAX, 0, 2), 0);
        assert_eq!(ll_raw_malloc_varsize_with_length(i64::MAX, 8, 8, 0), 0);
        majit_gc::set_active_raise_memory_error(None);
    }

    /// Overflowing `ll_raw_malloc_varsize*` raises MemoryError through
    /// the backend hook (`set_active_raise_memory_error`) and returns 0.
    #[test]
    fn raw_malloc_varsize_overflow_invokes_memory_error_hook() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HITS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            HITS.fetch_add(1, Ordering::SeqCst);
        }
        let _guard = RAISE_HOOK_TEST.lock().unwrap_or_else(|e| e.into_inner());
        HITS.store(0, Ordering::SeqCst);
        majit_gc::set_active_raise_memory_error(Some(hook));
        let result = ll_raw_malloc_varsize(i64::MAX, 0, 2);
        majit_gc::set_active_raise_memory_error(None);
        assert_eq!(result, 0);
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
    }

    /// `transform.py ll_malloc_varsize` stores `length` at `lengthoffset`.
    /// The `_zero` helper clears `basesize + n * itemsize` bytes.
    #[test]
    fn raw_malloc_varsize_stores_length_and_zero_clears() {
        let ptr = ll_raw_malloc_varsize_with_length(3, 8, 4, 0);
        assert_ne!(ptr, 0);
        unsafe {
            assert_eq!((ptr as *const isize).read(), 3);
        }
        ll_raw_free(ptr);

        let zeroed = ll_raw_malloc_varsize_no_length_zero(3, 8, 4);
        assert_ne!(zeroed, 0);
        unsafe {
            let bytes = std::slice::from_raw_parts(zeroed as *const u8, 20);
            assert!(bytes.iter().all(|b| *b == 0));
        }
        ll_raw_free(zeroed);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn raw_malloc_varsize_helpers_are_published() {
        let mut published = Vec::new();
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|desc| published.push(desc.path));
        for leaf in [
            "ll_raw_malloc_fixedsize",
            "ll_raw_malloc_fixedsize_zero",
            "ll_raw_malloc_fixedsize_add_memory_pressure",
            "ll_raw_malloc_fixedsize_zero_add_memory_pressure",
            "ll_raw_malloc_fixedsize_no_track_allocation",
            "ll_raw_malloc_fixedsize_zero_no_track_allocation",
            "ll_raw_malloc_fixedsize_add_memory_pressure_no_track_allocation",
            "ll_raw_malloc_fixedsize_zero_add_memory_pressure_no_track_allocation",
            "ll_raw_malloc_varsize",
            "ll_raw_malloc_varsize_zero",
            "ll_raw_malloc_varsize_no_length",
            "ll_raw_malloc_varsize_with_length",
        ] {
            let path = format!("majit_rlib::rffi::{leaf}");
            assert!(
                published.contains(&path.as_str()),
                "{path} is not published"
            );
        }
    }

    #[test]
    fn wasm_raw_block_is_16_aligned_for_any_size() {
        for size in [0_usize, 1, 7, 8, 15, 16, 17, 24, 32, 48] {
            let layout = super::raw_malloc_layout(size);
            assert_eq!(layout.align(), super::RAW_MALLOC_ALIGN);
            assert_eq!(layout.size() % super::RAW_MALLOC_ALIGN, 0);
            assert!(layout.size() >= super::RAW_MALLOC_ALIGN + size.max(1));
        }
    }

    /// Ordinary-Rust raw malloc returns `Err(MemoryError)` and does not
    /// publish a pending exception (`raw_malloc_or_memory_error`).
    #[test]
    fn raw_malloc_or_memory_error_null_does_not_set_pending_exception() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HITS: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            HITS.fetch_add(1, Ordering::SeqCst);
        }
        let _guard = RAISE_HOOK_TEST.lock().unwrap_or_else(|e| e.into_inner());
        HITS.store(0, Ordering::SeqCst);
        majit_gc::set_active_raise_memory_error(Some(hook));
        let result = super::raw_malloc_or_memory_error(usize::MAX);
        majit_gc::set_active_raise_memory_error(None);
        assert_eq!(result, Err(super::MemoryError));
        assert_eq!(HITS.load(Ordering::SeqCst), 0);
    }

    /// `rffi.py liststr2charpp`: a failed element `str2charp` is MemoryError
    /// and does not return a non-null outer array. The allocator cannot be
    /// forced to fail for a live element here; the overflow of
    /// `ovfcheck(itemsize * n)` on the outer array is the same error.
    #[test]
    fn liststr2charpp_size_overflow_is_memory_error() {
        // `(n+1) * size_of::<CCHARP>()` overflows when n is this large.
        // A real `&[&[u8]]` of that length cannot be built, so the
        // `checked_mul` path is the one this test names.
        let n = usize::MAX / core::mem::size_of::<super::CCHARP>();
        assert!(
            n.checked_add(1)
                .and_then(|n1| n1.checked_mul(core::mem::size_of::<super::CCHARP>()))
                .is_none()
        );
        // Empty and small lists still allocate.
        let empty = super::liststr2charpp(&[]).unwrap();
        unsafe { super::free_charpp(empty) };
        let pp = super::liststr2charpp(&[b"one", b"two"]).unwrap();
        let got = unsafe { super::charpp2liststr(pp) }.unwrap();
        unsafe { super::free_charpp(pp) };
        assert_eq!(got, vec![b"one".to_vec(), b"two".to_vec()]);
    }
}
