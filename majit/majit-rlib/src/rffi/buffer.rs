//! String and buffer helpers from `make_string_mappings` and the wchar/utf8
//! conversions next to it. "str" is a byte string (`&[u8]` / `Vec<u8>`).

use super::{CCHARP, CONST_CCHARP, CWCHARP, Wchar, cast};

/// `get_nonmovingbuffer_ll` flag: the pointer is inside a nonmovable object.
/// Nothing to free.
const FLAG_INSIDE_NONMOVABLE: u8 = 0x04;
/// `llobj` was pinned. Release unpins it.
const FLAG_PINNED: u8 = 0x05;
/// Pinning failed. The pointer is a raw malloc; release frees it.
const FLAG_RAW_COPY: u8 = 0x06;

/// `alloc_buffer` case: the pointer is inside an unpinned, non-moving GC buffer.
const CASE_INSIDE: isize = 0;

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
pub fn str2charp(s: &[u8], track_allocation: bool) -> CCHARP {
    let _ = track_allocation;
    let bytes = s.len().checked_add(1).expect("str2charp length overflow");
    unsafe {
        let array = raw_malloc(bytes).cast::<super::CHAR>();
        copy_bytes(s.as_ptr(), array.cast(), s.len());
        *array.add(s.len()) = 0;
        array
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
///
/// # Safety
/// `cp` points at a NUL-terminated byte sequence.
pub unsafe fn charp2str(cp: CCHARP) -> Vec<u8> {
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
pub unsafe fn charp2strn(cp: CCHARP, maxlen: usize) -> Vec<u8> {
    let mut size = 0;
    unsafe {
        while size < maxlen && *cp.add(size) != 0 {
            size += 1;
        }
        charpsize2str(cp, size)
    }
}

/// `charpsize2str`. Copies `size` bytes, NULs included. Does not free `cp`.
///
/// # Safety
/// `cp` is readable for `size` bytes.
pub unsafe fn charpsize2str(cp: CCHARP, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size];
    unsafe { copy_bytes(cp.cast(), out.as_mut_ptr(), size) };
    out
}

/// `constcharp2str`. [`charp2str`] for a `CONST_CCHARP`.
///
/// # Safety
/// `cp` points at a NUL-terminated byte sequence.
pub unsafe fn constcharp2str(cp: CONST_CCHARP) -> Vec<u8> {
    unsafe { charp2str(cast::<CCHARP>(cp)) }
}

/// `constcharpsize2str`. [`charpsize2str`] for a `CONST_CCHARP`.
///
/// # Safety
/// `cp` is readable for `size` bytes.
pub unsafe fn constcharpsize2str(cp: CONST_CCHARP, size: usize) -> Vec<u8> {
    unsafe { charpsize2str(cast::<CCHARP>(cp), size) }
}

/// `str2constcharp`. [`str2charp`] cast to `CONST_CCHARP`.
pub fn str2constcharp(s: &[u8], track_allocation: bool) -> CONST_CCHARP {
    cast::<CONST_CCHARP>(str2charp(s, track_allocation))
}

/// `rffi.get_nonmovingbuffer_ll` (`jit.dont_look_inside`).
///
/// Upstream returns `(char*, llobj, flag)`:
/// * `\x04` — no pin; the pointer is inside a nonmovable `llobj` (nothing to free)
/// * `\x05` — `llobj` was pinned (unpin on release)
/// * `\x06` — pinning failed; the pointer is a raw malloc (free on release)
///
/// Byte storage can move with the nursery, and a borrowed `&[u8]` carries no
/// pin, so this is only the `\x06` arm: `malloc` of `len + 1` bytes. The extra
/// byte is left uninitialized; [`get_nonmovingbuffer_ll_final_null`] writes the
/// NUL. The `llobj` place is `()` because the copy does not retain a GC string.
#[majit_macros::dont_look_inside]
pub fn get_nonmovingbuffer_ll(data: &[u8]) -> (CCHARP, (), u8) {
    let count = data.len();
    // `count + (TYPEP is CCHARP)`: a byte string gets one extra byte.
    let bytes = count
        .checked_add(1)
        .expect("get_nonmovingbuffer_ll length overflow");
    unsafe {
        let buf = raw_malloc(bytes).cast::<super::CHAR>();
        copy_bytes(data.as_ptr(), buf.cast(), count);
        (buf, (), FLAG_RAW_COPY)
    }
}

/// `rffi.get_nonmovingbuffer_ll_final_null` (`jit.dont_look_inside`). Writes
/// the NUL at `len(data)`.
#[majit_macros::dont_look_inside]
pub fn get_nonmovingbuffer_ll_final_null(data: &[u8]) -> (CCHARP, (), u8) {
    let (buf, llobj, flag) = get_nonmovingbuffer_ll(data);
    unsafe { *buf.add(data.len()) = 0 };
    (buf, llobj, flag)
}

/// `rffi.free_nonmovingbuffer_ll` (`jit.dont_look_inside`).
///
/// `\x05` would unpin `llobj`. `\x06` frees the raw copy. `\x04` does nothing.
/// Nothing is ever pinned here, so only `\x06` releases a block.
///
/// # Safety
/// `(buf, llobj, flag)` came from [`get_nonmovingbuffer_ll`] or
/// [`get_nonmovingbuffer_ll_final_null`] and has not been released.
#[majit_macros::dont_look_inside]
pub unsafe fn free_nonmovingbuffer_ll(buf: CCHARP, llobj: (), flag: u8) {
    let _keepalive = llobj;
    let _inside = flag == FLAG_INSIDE_NONMOVABLE;
    if flag == FLAG_PINNED {
        // `rgc.unpin(llobj)`. This port never pins.
    }
    if flag == FLAG_RAW_COPY {
        unsafe { raw_free(buf.cast()) };
    }
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
/// Upstream returns `(raw_buffer, gc_buffer, case_num)`:
/// * `0` — the pointer is inside an unpinned, non-moving GC buffer
/// * `1` — the GC buffer was pinned
/// * `2` — raw `malloc` of `count` bytes; [`keep_buffer_alive_until_here`] frees it
/// * `3` — large-object skip: raw `malloc` and no GC buffer
///
/// `_alloc_buffer_nonlarge_max()` is 0: there is no GC large-object path, so
/// case 3 never runs. The high-level string is a `Vec<u8>` whose heap block
/// cannot move (`rgc.can_move` is false), which is case 0.
#[majit_macros::dont_look_inside]
pub fn alloc_buffer(count: usize) -> Result<(CCHARP, Vec<u8>, isize), MemoryError> {
    let mut new_buf = mallocfn(count)?;
    let raw_buf = new_buf.as_mut_ptr();
    Ok((raw_buf, new_buf, CASE_INSIDE))
}

/// `rffi.str_from_buffer` (`jit.dont_look_inside`). Converts a pair from
/// [`alloc_buffer`] to a high-level string of `needed_size` bytes.
///
/// A size mismatch does not shrink the original block: `MiniMarkGC.shrink_array`
/// returns False for an object not in the nursery, and [`alloc_buffer`] case 0
/// is a non-moving `alloc_zeroed` Vec. The else is `mallocfn(needed_size)`
/// (fallible `MemoryError`), case 2, copy `needed_size` bytes from `raw_buf`.
///
/// # Safety
/// `raw_buf` is readable for `needed_size` bytes. When it points into `gc_buf`,
/// that Vec stays alive until after the copy.
#[majit_macros::dont_look_inside]
pub unsafe fn str_from_buffer(
    raw_buf: CCHARP,
    mut gc_buf: Vec<u8>,
    mut case_num: isize,
    allocated_size: usize,
    needed_size: usize,
) -> Result<Vec<u8>, MemoryError> {
    assert!(allocated_size >= needed_size);
    // `raw_buf` may point into `gc_buf`; keep the original block alive until
    // after the case-2 copy.
    let _original = if allocated_size != needed_size {
        let new_buf = mallocfn(needed_size)?;
        case_num = 2;
        Some(std::mem::replace(&mut gc_buf, new_buf))
    } else {
        None
    };
    if case_num == 2 {
        unsafe { copy_bytes(raw_buf.cast(), gc_buf.as_mut_ptr(), needed_size) };
    }
    drop(_original);
    Ok(gc_buf)
}

/// `rffi.keep_buffer_alive_until_here` (`jit.dont_look_inside`).
///
/// Case 1 would unpin (none here). Cases 2/3 would free the raw block (none
/// here: case 0). The Vec drops by itself.
///
/// # Safety
/// `(_raw_buf, gc_buf, _case_num)` came from [`alloc_buffer`] and has not been
/// released.
#[majit_macros::dont_look_inside]
pub unsafe fn keep_buffer_alive_until_here(_raw_buf: CCHARP, gc_buf: Vec<u8>, _case_num: isize) {
    let _keepalive = gc_buf;
}

/// `liststr2charpp`. NULL-terminated `char**`; each entry is a [`str2charp`].
pub fn liststr2charpp(list: &[&[u8]]) -> super::CCHARPP {
    let n = list.len();
    let bytes = n
        .checked_add(1)
        .and_then(|n| n.checked_mul(core::mem::size_of::<CCHARP>()))
        .expect("liststr2charpp length overflow");
    unsafe {
        let array = raw_malloc(bytes).cast::<CCHARP>();
        for (i, item) in list.iter().enumerate() {
            *array.add(i) = str2charp(item, true);
        }
        *array.add(n) = core::ptr::null_mut();
        array
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
pub unsafe fn charpp2liststr(charpp: super::CCHARPP) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    unsafe {
        let mut i = 0;
        while !(*charpp.add(i)).is_null() {
            result.push(charp2str(*charpp.add(i)));
            i += 1;
        }
    }
    result
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
/// walking the whole string. The check stays on outside tests too, because a
/// short count would make the walk write past the block. `track_allocation`
/// is the same debug-tracker flag as [`str2charp`].
pub fn utf82wcharp(utf8: &[u8], utf8len: usize, track_allocation: bool) -> CWCHARP {
    let _ = track_allocation;
    assert_eq!(utf8len, codepoint_len(utf8));
    let chars = utf8len.checked_add(2).expect("utf82wcharp length overflow");
    let bytes = chars
        .checked_mul(core::mem::size_of::<Wchar>())
        .expect("utf82wcharp size overflow");
    unsafe {
        let w = raw_malloc(bytes).cast::<Wchar>();
        let mut index = 0;
        let mut pos = 0;
        while let Some(ch) = next_codepoint(utf8, &mut pos) {
            *w.add(index) = ch as Wchar;
            index += 1;
        }
        *w.add(index) = 0;
        w
    }
}

/// `utf82wcharp_ex`. Like [`utf82wcharp`], but a code point above `0xffff`
/// becomes a surrogate pair, the layout a 2-byte `wchar_t` needs. Surrogates
/// in the input pass through. `unilen` is accepted and unused, and the block
/// holds `wlen + 3` units, where `wlen` counts the units written before the
/// zero terminator.
pub fn utf82wcharp_ex(utf8: &[u8], unilen: usize, track_allocation: bool) -> CWCHARP {
    let _ = (unilen, track_allocation);
    let mut wlen: usize = 0;
    let mut pos = 0;
    while let Some(ch) = next_codepoint(utf8, &mut pos) {
        if ch > 0xffff {
            wlen += 1;
        }
        wlen += 1;
    }
    let bytes = wlen
        .checked_add(3)
        .and_then(|n| n.checked_mul(core::mem::size_of::<Wchar>()))
        .expect("utf82wcharp_ex size overflow");
    unsafe {
        let w = raw_malloc(bytes).cast::<Wchar>();
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
        w
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
    pub fn new(value: Option<&[u8]>) -> Self {
        let buf = match value {
            Some(s) => str2charp(s, true),
            None => core::ptr::null_mut(),
        };
        Self { buf }
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
#[derive(Debug)]
pub struct scoped_nonmovingbuffer {
    pub buf: CCHARP,
    pub llobj: (),
    pub flag: u8,
}

impl scoped_nonmovingbuffer {
    pub fn new(data: &[u8]) -> Self {
        let (buf, llobj, flag) = get_nonmovingbuffer_ll(data);
        Self { buf, llobj, flag }
    }
}

impl Drop for scoped_nonmovingbuffer {
    fn drop(&mut self) {
        unsafe { free_nonmovingbuffer_ll(self.buf, self.llobj, self.flag) };
        self.flag = FLAG_INSIDE_NONMOVABLE;
        self.buf = core::ptr::null_mut();
    }
}

/// `scoped_view_charp`. Same as [`scoped_nonmovingbuffer`], but the buffer is
/// NUL-terminated (`get_nonmovingbuffer_ll_final_null`).
#[derive(Debug)]
pub struct scoped_view_charp {
    pub buf: CCHARP,
    pub llobj: (),
    pub flag: u8,
}

impl scoped_view_charp {
    pub fn new(data: &[u8]) -> Self {
        let (buf, llobj, flag) = get_nonmovingbuffer_ll_final_null(data);
        Self { buf, llobj, flag }
    }
}

impl Drop for scoped_view_charp {
    fn drop(&mut self) {
        unsafe { free_nonmovingbuffer_ll(self.buf, self.llobj, self.flag) };
        self.flag = FLAG_INSIDE_NONMOVABLE;
        self.buf = core::ptr::null_mut();
    }
}

/// `scoped_alloc_buffer`. `__enter__` returns the guard (`raw`, `size`,
/// [`Self::str`]). [`Drop`] runs [`keep_buffer_alive_until_here`].
#[derive(Debug)]
pub struct scoped_alloc_buffer {
    pub raw: CCHARP,
    pub gc_buf: Vec<u8>,
    pub case_num: isize,
    pub size: usize,
}

impl scoped_alloc_buffer {
    pub fn new(size: usize) -> Result<Self, MemoryError> {
        let (raw, gc_buf, case_num) = alloc_buffer(size)?;
        Ok(Self {
            raw,
            gc_buf,
            case_num,
            size,
        })
    }

    /// `scoped_alloc_buffer.str`.
    ///
    /// `str_from_buffer` consumes `gc_buf`; a `Vec` has one owner, so `str`
    /// ends the scope and `raw` cannot outlive the original storage.
    pub fn str(mut self, length: usize) -> Result<Vec<u8>, MemoryError> {
        let gc_buf = std::mem::take(&mut self.gc_buf);
        unsafe { str_from_buffer(self.raw, gc_buf, self.case_num, self.size, length) }
    }
}

impl Drop for scoped_alloc_buffer {
    fn drop(&mut self) {
        unsafe {
            keep_buffer_alive_until_here(self.raw, std::mem::take(&mut self.gc_buf), self.case_num)
        };
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
    pub fn new(value: Option<&[u8]>, unicode_len: isize) -> Self {
        let buf = match value {
            None => core::ptr::null_mut(),
            Some(utf8) => {
                let n = if unicode_len < 0 {
                    codepoint_len(utf8)
                } else {
                    unicode_len as usize
                };
                utf82wcharp(utf8, n, true)
            }
        };
        Self { buf }
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

/// `lltype.malloc(..., flavor='raw')` / `lltype.free(..., flavor='raw')`:
/// `malloc` / `free`. A zero-byte request still returns a unique block;
/// `malloc(0)` is implementation-defined and a later `offset` on NULL is not.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn raw_malloc(size: usize) -> *mut u8 {
    let request = size.max(1);
    let p = unsafe { libc::malloc(request) };
    if p.is_null() {
        std::alloc::handle_alloc_error(std::alloc::Layout::from_size_align(request, 1).unwrap());
    }
    p.cast()
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn raw_free(ptr: *mut u8) {
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
fn raw_malloc_layout(size: usize) -> std::alloc::Layout {
    let payload = size.max(1);
    let total = RAW_MALLOC_ALIGN
        .checked_add(payload)
        .and_then(|n| n.checked_next_multiple_of(RAW_MALLOC_ALIGN))
        .expect("raw malloc size overflow");
    std::alloc::Layout::from_size_align(total, RAW_MALLOC_ALIGN).unwrap()
}

/// wasm32-unknown-unknown does not link `libc`. The requested size sits in
/// the header so [`raw_free`] can rebuild the [`raw_malloc_layout`].
#[cfg(target_arch = "wasm32")]
unsafe fn raw_malloc(size: usize) -> *mut u8 {
    let layout = raw_malloc_layout(size);
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    unsafe {
        base.cast::<usize>().write(size);
        base.add(RAW_MALLOC_ALIGN)
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn raw_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = unsafe { ptr.sub(RAW_MALLOC_ALIGN) };
    let size = unsafe { base.cast::<usize>().read() };
    let layout = raw_malloc_layout(size);
    unsafe { std::alloc::dealloc(base, layout) }
}

/// Fixed-size `lltype.malloc(STRUCT, flavor='raw')`.
///
/// `support.py build_ll_0_raw_malloc_fixedsize` bakes the STRUCT into a
/// zero-argument `_ll_0_raw_malloc_fixedsize` (and `_zero` when `zero=True`).
/// The size here is a constant argument; alignment stays inside `raw_malloc`.
/// `GCTransformer` publishes the helper as one `Signed`.
///
/// `size` and the address are `i64` words (`extern "C"`). A `usize` parameter
/// is i32 on wasm32, and `call_indirect` requires this signature.
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize(size: i64) -> i64 {
    unsafe { raw_malloc(size as usize) as usize as i64 }
}

/// `zero=True` form of [`ll_raw_malloc_fixedsize`]. The block is cleared
/// for `size` bytes. A zero-size request still returns a distinct block
/// and writes nothing. Same word ABI as [`ll_raw_malloc_fixedsize`].
#[inline(never)]
pub extern "C" fn ll_raw_malloc_fixedsize_zero(size: i64) -> i64 {
    let bytes = size as usize;
    let ptr = unsafe { raw_malloc(bytes) };
    if bytes > 0 {
        unsafe { ptr.write_bytes(0, bytes) };
    }
    ptr as usize as i64
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
    use super::{ll_raw_free, ll_raw_malloc_fixedsize, ll_raw_malloc_fixedsize_zero};

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

    #[test]
    fn wasm_raw_block_is_16_aligned_for_any_size() {
        for size in [0_usize, 1, 7, 8, 15, 16, 17, 24, 32, 48] {
            let layout = super::raw_malloc_layout(size);
            assert_eq!(layout.align(), super::RAW_MALLOC_ALIGN);
            assert_eq!(layout.size() % super::RAW_MALLOC_ALIGN, 0);
            assert!(layout.size() >= super::RAW_MALLOC_ALIGN + size.max(1));
        }
    }
}
