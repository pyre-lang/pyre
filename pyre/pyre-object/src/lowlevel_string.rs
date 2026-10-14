//! GC-managed low-level strings — pyre's `rstr.STR` / `rstr.UNICODE` equivalent.
//!
//! In a running interpreter a low-level string is a varsize GC leaf, matching
//! RPython's `GcStruct('rpy_string', ...)`.  Before the GC registry is installed
//! (principally unit tests), allocation falls back to an identically laid-out
//! raw block stamped `init_gc_object_immortal`.  Both forms are laid out
//! as `{ hash: Signed @0, len: usize @8, chars: [item; len] @16.. }` after the
//! `GcHeader`. The `len` word is the RPython varsize length and, for the raw
//! fallback only, also the payload size used to reconstruct its freeing
//! `Layout` (header size plus payload).
//!
//! These are the leaf primitives behind the `newstr`/`newunicode`/`strsetitem`
//! blackhole ops and the StringBuilder buffer (`current_buf`, STRINGPIECE `buf`).
//! They live here (not in `pyre-jit`) so the JIT fnaddr registry can reference
//! [`jit_ll_shrink_array`] as a residual-call target.

/// rstr `STR` / `UNICODE` header: `hash` then the varsize `len` word.
/// `LLHelpers.ll_strlen` reads that word as `len(s.chars)`.
#[repr(C)]
struct LowlevelStringHeader {
    hash: isize,
    length: usize,
}

/// `len` field offset — one word past the `hash` field.
pub const LOWLEVEL_STRING_LEN_OFFSET: usize = std::mem::offset_of!(LowlevelStringHeader, length);
/// `chars` array offset — two words past the start (`hash`, then `len`).
pub const LOWLEVEL_STRING_CHARS_OFFSET: usize = std::mem::size_of::<LowlevelStringHeader>();
/// STR base size includes a trailing null byte after the chars array.
pub const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;
/// UNICODE base size has no trailing null.
pub const LOWLEVEL_UNICODE_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET;

static LOWLEVEL_STR_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LOWLEVEL_UNICODE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

pub fn set_lowlevel_str_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    LOWLEVEL_STR_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    majit_gc::set_lowlevel_str_type_id(id);
}

/// Undo [`set_lowlevel_str_gc_type_id`] so a unit test can force the managed
/// string path and then put the process back on the immortal fallback.
pub fn clear_lowlevel_str_gc_type_id() {
    LOWLEVEL_STR_GC_TYPE_ID.store(0, std::sync::atomic::Ordering::Release);
    majit_gc::clear_lowlevel_str_type_id();
}

pub fn set_lowlevel_unicode_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    LOWLEVEL_UNICODE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
    majit_gc::set_lowlevel_unicode_type_id(id);
}

pub fn lowlevel_str_gc_type_id() -> u32 {
    // A unit test publishes a probe id only for its own thread. Other
    // threads keep the unpublished sentinel and stay on the immortal path.
    // Absent from a normal or release build.
    #[cfg(any(test, feature = "test-hooks"))]
    if !crate::gc_hook::hook_test_effects_visible() {
        return 0;
    }
    LOWLEVEL_STR_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

pub fn lowlevel_unicode_gc_type_id() -> u32 {
    LOWLEVEL_UNICODE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

fn lowlevel_string_gc_type_id(base_size: usize, item_size: usize) -> u32 {
    match (base_size, item_size) {
        (LOWLEVEL_STR_BASE_SIZE, 1) => lowlevel_str_gc_type_id(),
        (LOWLEVEL_UNICODE_BASE_SIZE, 4) => lowlevel_unicode_gc_type_id(),
        _ => 0,
    }
}

/// Allocate a zero-filled low-level string of `length` items, storing `length`
/// in the varsize `len` word. Returns 0 on overflow / allocation failure.
pub fn bh_alloc_lowlevel_string(length: usize, base_size: usize, item_size: usize) -> i64 {
    alloc_lowlevel_string(length, base_size, item_size, true)
}

/// `rstr.mallocstr` when `incminimark.IncrementalMiniMarkGC.malloc_zero_filled`
/// is false. The caller fills `chars`. `hash` is stored as 0, and the STR
/// trailing NUL (the extra byte in [`LOWLEVEL_STR_BASE_SIZE`]) is written.
/// Returns 0 on overflow / allocation failure.
pub fn bh_alloc_str_nofill(length: usize) -> i64 {
    alloc_lowlevel_string(length, LOWLEVEL_STR_BASE_SIZE, 1, false)
}

fn alloc_lowlevel_string(
    length: usize,
    base_size: usize,
    item_size: usize,
    zero_fill: bool,
) -> i64 {
    let Some(items_size) = length.checked_mul(item_size) else {
        return 0;
    };
    let Some(total_size) = base_size.checked_add(items_size) else {
        return 0;
    };
    let tid = lowlevel_string_gc_type_id(base_size, item_size);
    let gc_ptr = if tid != 0 {
        // `rstr.malloc` / `malloc_varsize` is a nursery bump
        // (`incminimark.py malloc_varsize`). A request that cannot bump
        // (larger than `nonlarge_max`, or the no-collect bump returns
        // null) is `external_malloc(..., alloc_young=True)`: born young
        // so the next minor can free it, and non-moving so a caller that
        // still holds the operands as i64 can copy them after the call.
        // A raw `alloc_zeroed` fallback is never reclaimed.
        let young = crate::gc_hook::try_gc_alloc_nursery_raw(tid, total_size);
        if young.is_null() {
            crate::gc_hook::try_gc_alloc_young_nonmoving_raw(tid, total_size)
        } else {
            young
        }
    } else {
        std::ptr::null_mut()
    };
    let ptr = if !gc_ptr.is_null() {
        // Nursery GC allocation is not specified to clear the payload.
        // `malloc_zero_filled` is false, so only the zero-fill callers pay it.
        // The payload is a byte block; GCREFOpaque has size 0, so the
        // write is through `*mut u8`.
        let ptr = gc_ptr as *mut u8;
        if zero_fill {
            unsafe { std::ptr::write_bytes(ptr, 0, total_size) };
        }
        ptr
    } else {
        // Off-heap `rstr.STR` / `UNICODE` is still a `GcStruct` (`rstr.py`):
        // a prebuilt `rpy_string` carries `init_gc_object_immortal`. The
        // pointer handed out is the address after the header.
        let ptr = if zero_fill {
            majit_gc::header::alloc_varsize_with_gc_header_immortal_zeroed(total_size, tid)
        } else {
            majit_gc::header::alloc_varsize_with_gc_header_immortal(total_size, tid)
        };
        if ptr.is_null() {
            return 0;
        }
        ptr
    };
    unsafe {
        if !zero_fill {
            // `rstr.mallocstr` stores `r.hash = 0` when the allocator did not
            // clear the block. The extra base byte is the trailing NUL.
            (ptr as *mut usize).write(0);
            if base_size == LOWLEVEL_STR_BASE_SIZE && item_size == 1 {
                ptr.add(LOWLEVEL_STRING_CHARS_OFFSET + length).write(0);
            }
        }
        (ptr.add(LOWLEVEL_STRING_LEN_OFFSET) as *mut usize).write(length);
    }
    ptr as i64
}

/// `LLHelpers.ll_strlen` — the `len` word of an rstr `STR` / `UNICODE`.
pub fn bh_lowlevel_string_len(string: i64) -> usize {
    if string == 0 {
        return 0;
    }
    unsafe { (*(string as *const LowlevelStringHeader)).length }
}

/// Release a raw-fallback low-level string allocated by
/// [`bh_alloc_lowlevel_string`]. GC-managed strings are owned by the collector
/// and deliberately left untouched.
///
/// Reconstructs the exact `Layout` from the capacity word stored at
/// `LOWLEVEL_STRING_LEN_OFFSET` (the allocation stores its `length` argument
/// there, and the StringBuilder passes the buffer *capacity* as that argument).
/// `base_size`/`item_size` must match the allocation call.
pub fn bh_free_lowlevel_string(string: i64, base_size: usize, item_size: usize) {
    if string == 0 {
        return;
    }
    if majit_gc::gc_owns_object(string as usize) {
        return;
    }
    let capacity = bh_lowlevel_string_len(string);
    let Some(items_size) = capacity.checked_mul(item_size) else {
        return;
    };
    let Some(payload_size) = base_size.checked_add(items_size) else {
        return;
    };
    unsafe { majit_gc::header::dealloc_varsize_with_gc_header(string as *mut u8, payload_size) };
}

/// STR / UNICODE width `(base_size, item_size)` of the low-level string at
/// `buf`, read from its GC type id.
///
/// A buffer carries the width in its header tid — the collector's own record
/// of the element size — exactly as `rgc.ll_shrink_array` recovers the element
/// type from the array it is handed. Off-heap raw STR / UNICODE payloads now
/// stamp `init_gc_object_immortal` too, so the tid is readable whether or not
/// the collector owns the object. An unpublished tid (0) defaults to STR.
fn shrink_array_width(buf: i64) -> (usize, usize) {
    if buf != 0 {
        let tid = unsafe { (*majit_gc::header::header_of(buf as usize)).type_id() };
        if tid != 0 && tid == lowlevel_unicode_gc_type_id() {
            return (LOWLEVEL_UNICODE_BASE_SIZE, 4);
        }
    }
    (LOWLEVEL_STR_BASE_SIZE, 1)
}

/// Width-parametric `rgc.ll_shrink_array(buf, new_len)` core.
///
/// `rgc.py` asks the GC first — `if llop.shrink_array(Bool, p,
/// smallerlength): return p` — and only allocates when the answer is no. The
/// GC says yes for a nursery object, where the shrink is the length word alone
/// and the buffer keeps its address.
///
/// The fallback allocates a fresh `new_len` buffer, copies `new_len` items,
/// frees the old buffer, and returns the new one (whose `len` word is now
/// exactly `new_len`, so a caller returning it directly reports the right
/// length). It is what the raw-fallback buffer takes, and there the shrink
/// *cannot* truncate in place: the `len` word doubles as the capacity
/// [`bh_free_lowlevel_string`] reconstructs the `Layout` from. A GC-owned
/// buffer has no such reader — that function returns early for one, and the
/// collector sizes it from the same length word the shrink just wrote.
///
/// `base_size`/`item_size` select STR (`LOWLEVEL_STR_BASE_SIZE`, 1) vs
/// UNICODE (`LOWLEVEL_UNICODE_BASE_SIZE`, 4); the `chars` array offset is the
/// same two words in both widths.
fn shrink_lowlevel_array(buf: i64, new_len: i64, base_size: usize, item_size: usize) -> i64 {
    if buf == 0 {
        return 0;
    }
    // `rgc.ll_shrink_array` is bracketed by the translated graph's GC
    // transform.  Until pyre's transform inserts that bracket, publish the
    // residual argument explicitly: the native ABI copy can otherwise still
    // name a nursery forwarding stub after a collection at call entry.
    let roots = crate::gc_roots::push_roots();
    let buf_slot = roots.base();
    let buf = roots.pin_root(buf as crate::PyObjectRef) as i64;
    let new_len = if new_len < 0 { 0 } else { new_len as usize };
    if majit_gc::gc_shrink_array(buf as usize, new_len) {
        return roots.get(buf_slot) as i64;
    }
    let new_buf = bh_alloc_lowlevel_string(new_len, base_size, item_size);
    if new_buf == 0 {
        return 0;
    }
    // The allocation is the collecting half of `rgc.ll_shrink_array`; reload
    // the old buffer from its transformed livevar slot before copying it.
    let buf = roots.get(buf_slot) as i64;
    // SAFETY: `buf` holds at least `new_len` valid items (the builder shrinks to
    // its own `current_pos`), and `new_buf` was allocated for exactly `new_len`.
    unsafe {
        // Copy the fixed `hash` field (@0) before the variable item array, as
        // `ll_shrink_array` copies the fixed field first; a fresh allocation
        // zeroes it, so a non-zero cached hash would otherwise be lost.
        (new_buf as *mut usize).write(*(buf as *const usize));
        let src = (buf as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let dst = (new_buf as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, new_len * item_size);
    }
    bh_free_lowlevel_string(buf, base_size, item_size);
    new_buf
}

/// `rgc.ll_shrink_array(buf, new_len)` — the non-virtual residual target of the
/// StringBuilder / UnicodeBuilder `build` tree's `ll_shrink_final`.
///
/// Selects the STR / UNICODE width from `buf`'s GC type id
/// ([`shrink_array_width`]) so a single residual target serves both builder
/// widths; the jtransform retarget names this one symbol for either. Only the
/// non-virtual buffer reaches here — a virtual buffer is folded by
/// `opt_call_shrink_array`.
///
/// `extern "C"` so the JIT residual call reaches it through the fnaddr registry.
/// `rstr.py LLHelpers.ll_strconcat` — join two rstr `STR` payloads.
///
/// `@jit.elidable` + `@jit.oopspec('stroruni.concat')`.  The result is a
/// fresh `STR` (`{ hash, len, chars }`), not a `W_UnicodeObject`.
/// `descr_add` wraps that payload afterwards (`space.newutf8`).
#[majit_macros::elidable]
pub extern "C" fn jit_ll_strconcat(
    s1: *mut crate::unicodeobject::Utf8Str,
    s2: *mut crate::unicodeobject::Utf8Str,
) -> *mut crate::unicodeobject::Utf8Str {
    let s1 = s1 as i64;
    let s2 = s2 as i64;
    if s1 == 0 || s2 == 0 {
        return std::ptr::null_mut();
    }
    let n1 = bh_lowlevel_string_len(s1);
    let n2 = bh_lowlevel_string_len(s2);
    // "a single '+' like this is allowed to overflow: it gets a negative
    // result, and the gc will complain" -- MemoryError upstream.  A null
    // payload handed back instead would be wrapped into a `str` whose `len`
    // disagrees with its storage; abort loudly as `rbuilder` does until
    // MemoryError propagation is ported.
    let total = n1
        .checked_add(n2)
        .expect("ll_strconcat length overflow; MemoryError propagation is not ported yet");
    // `malloc_varsize` / `external_malloc` can collect. Pin the operands
    // (`push_roots` / `pin_roots`) and reload them after the allocation,
    // matching the GC transform around `LLHelpers.ll_strconcat`.
    let roots = crate::gc_roots::push_roots();
    let base = roots.pin_roots(&[s1 as crate::PyObjectRef, s2 as crate::PyObjectRef]);
    let out = bh_alloc_lowlevel_string(total, LOWLEVEL_STR_BASE_SIZE, 1);
    assert!(
        out != 0,
        "ll_strconcat failed to allocate {total} bytes; MemoryError propagation is not ported yet"
    );
    let s1 = roots.get(base) as i64;
    let s2 = roots.get(base + 1) as i64;
    unsafe {
        let dst = (out as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping((s1 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), dst, n1);
        std::ptr::copy_nonoverlapping(
            (s2 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
            dst.add(n1),
            n2,
        );
    }
    out as *mut crate::unicodeobject::Utf8Str
}

/// `rstr.py LLHelpers.ll_streq` — compare two rstr `STR` payloads.
///
/// `@jit.elidable`.  Upstream also tags it `@jit.oopspec('stroruni.equal(s1,
/// s2)')`.  The tag stays off here because the direct caller,
/// `descroperation::compare_slot`, is a legacy-typed graph: its operands are
/// `OBJECTPTR`, not `Ptr(STR)`, and `_handle_stroruni_call` asserts on that.
/// A string `==` / `!=` the rtyper types as `Ptr(STR)` still reaches this
/// helper as `OS_STR_EQUAL` through `jtransform`'s string-equality rewrite.
/// Answers 1 / 0 so the result fills a full word.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_streq(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    // also if both are NULLs
    if s1 == s2 {
        return 1;
    }
    if s1.is_null() || s2.is_null() {
        return 0;
    }
    let len1 = bh_lowlevel_string_len(s1 as i64);
    let len2 = bh_lowlevel_string_len(s2 as i64);
    if len1 != len2 {
        return 0;
    }
    let (chars1, chars2) = unsafe {
        (
            std::slice::from_raw_parts((s1 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), len1),
            std::slice::from_raw_parts((s2 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), len2),
        )
    };
    (chars1 == chars2) as i64
}

/// The `chars` array of a live rstr `STR` payload.
fn str_chars<'a>(s: *const crate::unicodeobject::UnicodeValueStorage) -> &'a [u8] {
    let len = bh_lowlevel_string_len(s as i64);
    unsafe { std::slice::from_raw_parts((s as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), len) }
}

/// `support.py _ll_4_str_eq_slice_checknull` — `str1[start : start + length]
/// == str2`.  The `OS_STREQ_SLICE_CHECKNULL` helper `jtransform.py
/// _handle_stroruni_call` registers beside `stroruni.equal`.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_slice_checknull(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    start: i64,
    length: i64,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    if s2.is_null() {
        return 0;
    }
    jit_ll_str_eq_slice_nonnull(s1, start, length, s2)
}

/// `support.py _ll_4_str_eq_slice_nonnull` — `str1[start : start + length]
/// == str2`, assuming `str2 != NULL`.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_slice_nonnull(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    start: i64,
    length: i64,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    let chars2 = str_chars(s2);
    if chars2.len() as i64 != length {
        return 0;
    }
    let start = start as usize;
    (str_chars(s1)[start..start + chars2.len()] == *chars2) as i64
}

/// `support.py _ll_4_str_eq_slice_char` — `str1[start : start + length] ==
/// c2`.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_slice_char(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    start: i64,
    length: i64,
    c2: i64,
) -> i64 {
    if length != 1 {
        return 0;
    }
    (str_chars(s1)[start as usize] as i64 == c2) as i64
}

/// `support.py _ll_2_str_eq_nonnull` — `str1 == str2`, both non-null.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_nonnull(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    (str_chars(s1) == str_chars(s2)) as i64
}

/// `support.py _ll_2_str_eq_nonnull_char` — `str1 == c2`, `str1` non-null.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_nonnull_char(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    c2: i64,
) -> i64 {
    matches!(str_chars(s1), [c] if *c as i64 == c2) as i64
}

/// `support.py _ll_2_str_eq_checknull_char` — `str1 == c2`.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_checknull_char(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    c2: i64,
) -> i64 {
    if s1.is_null() {
        return 0;
    }
    jit_ll_str_eq_nonnull_char(s1, c2)
}

/// `support.py _ll_2_str_eq_lengthok` — `str1 == str2` for two non-null
/// strings already known to have the same length.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_eq_lengthok(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    let chars1 = str_chars(s1);
    (*chars1 == str_chars(s2)[..chars1.len()]) as i64
}

/// `rstr.py LLHelpers.ll_strcmp` — order two rstr `STR` payloads.
///
/// `@jit.elidable` and `@jit.oopspec('stroruni.cmp(s1, s2)')`: the operands
/// are `STR` payloads, which is what vstring's `opt_call_stroruni_STR_CMP`
/// reads with `strlen` / `strgetitem`.  `rtype_lt` and its siblings compare
/// the result against zero.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_strcmp(
    s1: *const crate::unicodeobject::UnicodeValueStorage,
    s2: *const crate::unicodeobject::UnicodeValueStorage,
) -> i64 {
    if s1.is_null() && s2.is_null() {
        return 1;
    }
    if s1.is_null() || s2.is_null() {
        return 0;
    }
    let len1 = bh_lowlevel_string_len(s1 as i64);
    let len2 = bh_lowlevel_string_len(s2 as i64);
    let cmplen = if len1 < len2 { len1 } else { len2 };
    let chars1 = unsafe { (s1 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET) };
    let chars2 = unsafe { (s2 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET) };
    let mut i = 0;
    while i < cmplen {
        let diff = unsafe { *chars1.add(i) as i64 - *chars2.add(i) as i64 };
        if diff != 0 {
            return diff;
        }
        i += 1;
    }
    len1 as i64 - len2 as i64
}

/// `rstr.py LLHelpers.ll_str_mul` — `@jit.elidable` on the STR payload.
///
/// `times < 0` clamps to 0 (`'0' * -1 == ''`).  `ovfcheck(len * times)` is
/// MemoryError upstream; abort until that propagation is ported.  The
/// doubling copy is `ll_str_mul`'s `copy_contents` loop.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_str_mul(
    s: *mut crate::unicodeobject::Utf8Str,
    mut times: i64,
) -> *mut crate::unicodeobject::Utf8Str {
    let s = s as i64;
    if s == 0 {
        return std::ptr::null_mut();
    }
    if times < 0 {
        times = 0;
    }
    let n = bh_lowlevel_string_len(s);
    let size = n
        .checked_mul(times as usize)
        .expect("ll_str_mul length overflow; MemoryError propagation is not ported yet");
    let roots = crate::gc_roots::push_roots();
    let s_slot = roots.base();
    let _ = roots.pin_root(s as crate::PyObjectRef);
    let out = bh_alloc_lowlevel_string(size, LOWLEVEL_STR_BASE_SIZE, 1);
    assert!(
        out != 0,
        "ll_str_mul failed to allocate; MemoryError propagation is not ported yet"
    );
    let s = roots.get(s_slot) as i64;
    if size == 0 || n == 0 {
        return out as *mut crate::unicodeobject::Utf8Str;
    }
    unsafe {
        let src = (s as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let dst = (out as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, n);
        let mut i = n;
        while i < size {
            let j = if i <= size - i { i } else { size - i };
            std::ptr::copy_nonoverlapping(dst, dst.add(i), j);
            i += j;
        }
    }
    out as *mut crate::unicodeobject::Utf8Str
}

/// `rstr.py LLHelpers._ll_stringslice` — `@jit.elidable` +
/// `@jit.oopspec('stroruni.slice(s1, start, stop)')`.
///
/// The first argument is an rstr `STR` (`Utf8Str` / `rpy_string`), not a
/// `W_UnicodeObject`.  `_getitem_result` slices `self._utf8[start:end]`
/// through [`ll_stringslice_startstop`] and wraps the piece afterwards.
#[majit_macros::oopspec("stroruni.slice(s1, start, stop)")]
#[majit_macros::elidable]
pub fn _ll_stringslice(
    s1: *mut crate::unicodeobject::Utf8Str,
    start: i64,
    stop: i64,
) -> *mut crate::unicodeobject::Utf8Str {
    if s1.is_null() {
        return std::ptr::null_mut();
    }
    let lgt = stop - start;
    debug_assert!(start >= 0);
    let roots = crate::gc_roots::push_roots();
    let s1_slot = roots.base();
    let _ = roots.pin_root(s1 as crate::PyObjectRef);
    if lgt < 0 {
        let empty = bh_alloc_lowlevel_string(0, LOWLEVEL_STR_BASE_SIZE, 1);
        return empty as *mut crate::unicodeobject::Utf8Str;
    }
    let lgt = lgt as usize;
    let out = bh_alloc_lowlevel_string(lgt, LOWLEVEL_STR_BASE_SIZE, 1);
    if out == 0 {
        return std::ptr::null_mut();
    }
    let s1 = roots.get(s1_slot) as i64;
    unsafe {
        let src = (s1 as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET + start as usize);
        let dst = (out as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        std::ptr::copy_nonoverlapping(src, dst, lgt);
    }
    out as *mut crate::unicodeobject::Utf8Str
}

/// `rstr.py LLHelpers.ll_stringslice_startstop`.
///
/// Interpreter: `start == 0 and stop >= len` hands the source `STR` back
/// unchanged.  Jitted: clamp `stop` and always go through
/// [`_ll_stringslice`] (`we_are_jitted()` is folded to true in the
/// generated body).
pub fn ll_stringslice_startstop(
    s1: *mut crate::unicodeobject::Utf8Str,
    start: i64,
    stop: i64,
) -> *mut crate::unicodeobject::Utf8Str {
    if s1.is_null() {
        return std::ptr::null_mut();
    }
    let n = bh_lowlevel_string_len(s1 as i64) as i64;
    let stop = if majit_rlib::jit::we_are_jitted() {
        if stop > n { n } else { stop }
    } else if stop >= n {
        if start == 0 {
            return s1;
        }
        n
    } else {
        stop
    };
    _ll_stringslice(s1, start, stop)
}

/// `ll_str.py ll_int2dec` — `@jit.elidable` decimal render of a Signed
/// into a fresh rstr `STR`, returned as the `Ptr(STR)` it is.
/// `descr_repr` (intobject.py) wraps the result with
/// `space.newutf8(res, len(res))`.
#[majit_macros::elidable]
pub extern "C" fn jit_ll_int2dec(val: i64) -> *mut crate::unicodeobject::Utf8Str {
    let sign = usize::from(val < 0);
    let mut uval = if val < 0 {
        (val as u64).wrapping_neg()
    } else {
        val as u64
    };
    let is_zero = uval == 0;
    let mut len = 0usize;
    let mut i = uval;
    while i != 0 {
        len += 1;
        i /= 10;
    }
    let total_len = sign + len + usize::from(is_zero);
    let out = bh_alloc_lowlevel_string(total_len, LOWLEVEL_STR_BASE_SIZE, 1);
    // `descr_str` wraps the payload unchecked, as `ll_strconcat`'s callers do.
    assert!(
        out != 0,
        "ll_int2dec failed to allocate; MemoryError propagation is not ported yet"
    );
    unsafe {
        let dst = (out as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET);
        let first = if sign != 0 {
            Some(b'-')
        } else if is_zero {
            Some(b'0')
        } else {
            None
        };
        if let Some(ch) = first {
            *dst = ch;
        }
        let mut j = 0usize;
        while j < len {
            *dst.add(total_len - j - 1) = b'0' + (uval % 10) as u8;
            uval /= 10;
            j += 1;
        }
    }
    out as *mut crate::unicodeobject::Utf8Str
}

pub extern "C" fn jit_ll_shrink_array(buf: crate::PyObjectRef, new_len: i64) -> crate::PyObjectRef {
    // Width discovery reads the GC type header, so it needs the same entry
    // livevar normalization as the shrink itself rather than dereferencing the
    // raw residual-call word first.
    let roots = crate::gc_roots::push_roots();
    let buf = roots.pin_root(buf) as i64;
    let (base_size, item_size) = shrink_array_width(buf);
    shrink_lowlevel_array(buf, new_len, base_size, item_size) as crate::PyObjectRef
}

pub fn bh_lowlevel_chars_offset(item_size: usize) -> usize {
    if item_size == 1 {
        LOWLEVEL_STR_BASE_SIZE - 1
    } else {
        LOWLEVEL_UNICODE_BASE_SIZE
    }
}

pub fn bh_read_lowlevel_string(string: i64, item_size: usize) -> Vec<i64> {
    let len = bh_lowlevel_string_len(string);
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    let mut chars = Vec::with_capacity(len);
    for index in 0..len {
        let addr = unsafe { (string as *const u8).add(chars_offset + index * item_size) };
        let value = unsafe {
            match item_size {
                1 => *addr as i64,
                4 => *(addr as *const u32) as i64,
                _ => *(addr as *const i64),
            }
        };
        chars.push(value);
    }
    chars
}

pub fn bh_write_lowlevel_char(string: i64, index: usize, char: i64, item_size: usize) {
    if string == 0 {
        return;
    }
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    unsafe {
        let addr = (string as *mut u8).add(chars_offset + index * item_size);
        match item_size {
            1 => addr.write(char as u8),
            4 => (addr as *mut u32).write(char as u32),
            _ => (addr as *mut i64).write(char),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mallocstr_nofill_sets_hash_len_and_trailing_nul() {
        // `rstr.mallocstr` with `malloc_zero_filled = False`.
        let buf = bh_alloc_str_nofill(4);
        assert_ne!(buf, 0);
        unsafe {
            let hdr = majit_gc::header::header_of(buf as usize);
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_NO_HEAP_PTRS));
            assert!((*hdr).has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
            assert_eq!(*(buf as *const usize), 0);
            assert_eq!(
                *((buf as *const u8).add(LOWLEVEL_STRING_LEN_OFFSET) as *const usize),
                4
            );
            assert_eq!(
                *((buf as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET + 4)),
                0
            );
        }
        for index in 0..4 {
            bh_write_lowlevel_char(buf, index, (b'w' + index as u8) as i64, 1);
        }
        assert_eq!(
            bh_read_lowlevel_string(buf, 1),
            vec![b'w' as i64, b'x' as i64, b'y' as i64, b'z' as i64]
        );
        bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }

    #[test]
    fn ll_stringslice_startstop_identity_and_cut() {
        let buf = bh_alloc_lowlevel_string(5, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(buf, 0);
        for index in 0..5 {
            bh_write_lowlevel_char(buf, index, (b'a' + index as u8) as i64, 1);
        }
        let s1 = buf as *mut crate::unicodeobject::Utf8Str;
        assert_eq!(ll_stringslice_startstop(s1, 0, 5), s1);
        let cut = ll_stringslice_startstop(s1, 1, 3);
        assert!(!cut.is_null());
        assert_ne!(cut, s1);
        assert_eq!(bh_lowlevel_string_len(cut as i64), 2);
        assert_eq!(bh_read_lowlevel_string(cut as i64, 1), vec![98, 99]);
        let empty = _ll_stringslice(s1, 3, 1);
        assert!(!empty.is_null());
        assert_eq!(bh_lowlevel_string_len(empty as i64), 0);
        bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, 1);
        bh_free_lowlevel_string(cut as i64, LOWLEVEL_STR_BASE_SIZE, 1);
        bh_free_lowlevel_string(empty as i64, LOWLEVEL_STR_BASE_SIZE, 1);
    }

    #[test]
    fn jit_ll_int2dec_renders_signed_decimal() {
        for val in [0i64, 1, -1, 10, -10, i64::MIN, i64::MAX] {
            let buf = jit_ll_int2dec(val);
            assert!(!buf.is_null());
            let buf = buf as i64;
            let expected = val.to_string();
            assert_eq!(bh_lowlevel_string_len(buf), expected.len());
            let chars: String = bh_read_lowlevel_string(buf, 1)
                .into_iter()
                .map(|c| c as u8 as char)
                .collect();
            assert_eq!(chars, expected);
            bh_free_lowlevel_string(buf, LOWLEVEL_STR_BASE_SIZE, 1);
        }
    }

    #[test]
    fn jit_ll_str_mul_repeats_and_clamps_negative() {
        let one = jit_ll_int2dec(0);
        // `0` renders as `"0"` — one char, used as the fill.
        assert_eq!(bh_lowlevel_string_len(one as i64), 1);
        let four = jit_ll_str_mul(one, 4);
        assert_eq!(bh_lowlevel_string_len(four as i64), 4);
        assert_eq!(
            bh_read_lowlevel_string(four as i64, 1),
            vec![b'0' as i64; 4]
        );
        let empty = jit_ll_str_mul(one, 0);
        assert_eq!(bh_lowlevel_string_len(empty as i64), 0);
        let also_empty = jit_ll_str_mul(one, -1);
        assert_eq!(bh_lowlevel_string_len(also_empty as i64), 0);
        bh_free_lowlevel_string(one as i64, LOWLEVEL_STR_BASE_SIZE, 1);
        bh_free_lowlevel_string(four as i64, LOWLEVEL_STR_BASE_SIZE, 1);
        bh_free_lowlevel_string(empty as i64, LOWLEVEL_STR_BASE_SIZE, 1);
        bh_free_lowlevel_string(also_empty as i64, LOWLEVEL_STR_BASE_SIZE, 1);
    }

    #[test]
    fn jit_ll_streq_compares_payload_chars() {
        use crate::unicodeobject::UnicodeValueStorage;
        let p = |s: i64| s as *const UnicodeValueStorage;
        let seven = jit_ll_int2dec(7) as i64;
        let seven_again = jit_ll_int2dec(7) as i64;
        let eight = jit_ll_int2dec(8) as i64;
        let seventy = jit_ll_int2dec(70) as i64;
        assert_ne!(seven, seven_again);
        assert_eq!(jit_ll_streq(p(seven), p(seven)), 1);
        assert_eq!(jit_ll_streq(p(seven), p(seven_again)), 1);
        assert_eq!(jit_ll_streq(p(seven), p(eight)), 0);
        assert_eq!(jit_ll_streq(p(seven), p(seventy)), 0);
        assert_eq!(jit_ll_streq(p(seven), p(0)), 0);
        assert_eq!(jit_ll_streq(p(0), p(0)), 1);
        for s in [seven, seven_again, eight, seventy] {
            bh_free_lowlevel_string(s, LOWLEVEL_STR_BASE_SIZE, 1);
        }
    }

    #[test]
    fn jit_ll_strcmp_orders_payload_chars_then_length() {
        use crate::unicodeobject::UnicodeValueStorage;
        let p = |s: i64| s as *const UnicodeValueStorage;
        let seven = jit_ll_int2dec(7) as i64;
        let seven_again = jit_ll_int2dec(7) as i64;
        let eight = jit_ll_int2dec(8) as i64;
        let seventy = jit_ll_int2dec(70) as i64;
        assert_eq!(jit_ll_strcmp(p(seven), p(seven_again)), 0);
        assert_eq!(jit_ll_strcmp(p(seven), p(eight)), -1);
        assert_eq!(jit_ll_strcmp(p(eight), p(seven)), 1);
        assert_eq!(jit_ll_strcmp(p(seven), p(seventy)), -1);
        assert_eq!(jit_ll_strcmp(p(seventy), p(eight)), -1);
        for s in [seven, seven_again, eight, seventy] {
            bh_free_lowlevel_string(s, LOWLEVEL_STR_BASE_SIZE, 1);
        }
    }

    #[test]
    fn jit_ll_str_eq_helpers_match_support_py() {
        use crate::unicodeobject::UnicodeValueStorage;
        let p = |s: i64| s as *const UnicodeValueStorage;
        let null = std::ptr::null::<UnicodeValueStorage>();
        let whole = jit_ll_int2dec(12345) as i64;
        let mid = jit_ll_int2dec(234) as i64;
        let mid_again = jit_ll_int2dec(234) as i64;
        let other = jit_ll_int2dec(235) as i64;
        let three = jit_ll_int2dec(3) as i64;
        assert_eq!(jit_ll_str_eq_slice_nonnull(p(whole), 1, 3, p(mid)), 1);
        assert_eq!(jit_ll_str_eq_slice_nonnull(p(whole), 1, 3, p(other)), 0);
        assert_eq!(jit_ll_str_eq_slice_nonnull(p(whole), 1, 2, p(mid)), 0);
        assert_eq!(jit_ll_str_eq_slice_checknull(p(whole), 1, 3, null), 0);
        assert_eq!(jit_ll_str_eq_slice_checknull(p(whole), 1, 3, p(mid)), 1);
        assert_eq!(jit_ll_str_eq_slice_char(p(whole), 2, 1, b'3' as i64), 1);
        assert_eq!(jit_ll_str_eq_slice_char(p(whole), 2, 2, b'3' as i64), 0);
        assert_eq!(jit_ll_str_eq_nonnull(p(mid), p(mid_again)), 1);
        assert_eq!(jit_ll_str_eq_nonnull(p(mid), p(whole)), 0);
        assert_eq!(jit_ll_str_eq_nonnull_char(p(three), b'3' as i64), 1);
        assert_eq!(jit_ll_str_eq_nonnull_char(p(mid), b'2' as i64), 0);
        assert_eq!(jit_ll_str_eq_checknull_char(null, b'3' as i64), 0);
        assert_eq!(jit_ll_str_eq_checknull_char(p(three), b'3' as i64), 1);
        assert_eq!(jit_ll_str_eq_lengthok(p(mid), p(mid_again)), 1);
        assert_eq!(jit_ll_str_eq_lengthok(p(mid), p(other)), 0);
        for s in [whole, mid, mid_again, other, three] {
            bh_free_lowlevel_string(s, LOWLEVEL_STR_BASE_SIZE, 1);
        }
    }

    #[test]
    fn shrink_lowlevel_array_str_truncates_and_preserves_hash() {
        let buf = bh_alloc_lowlevel_string(8, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(buf, 0);
        // A non-zero cached hash must survive the realloc-shrink.
        unsafe { (buf as *mut usize).write(0xdead_beef) };
        for index in 0..8 {
            bh_write_lowlevel_char(buf, index, (b'a' + index as u8) as i64, 1);
        }
        let new_buf = shrink_lowlevel_array(buf, 5, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(new_buf, 0);
        assert_eq!(bh_lowlevel_string_len(new_buf), 5);
        assert_eq!(unsafe { *(new_buf as *const usize) }, 0xdead_beef);
        assert_eq!(
            bh_read_lowlevel_string(new_buf, 1),
            vec![97, 98, 99, 100, 101]
        );
        bh_free_lowlevel_string(new_buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }

    #[test]
    fn shrink_lowlevel_array_unicode_copies_width_4_items() {
        let buf = bh_alloc_lowlevel_string(4, LOWLEVEL_UNICODE_BASE_SIZE, 4);
        assert_ne!(buf, 0);
        for index in 0..4 {
            bh_write_lowlevel_char(buf, index, 0x1_0000 + index as i64, 4);
        }
        let new_buf = shrink_lowlevel_array(buf, 2, LOWLEVEL_UNICODE_BASE_SIZE, 4);
        assert_ne!(new_buf, 0);
        assert_eq!(bh_lowlevel_string_len(new_buf), 2);
        assert_eq!(
            bh_read_lowlevel_string(new_buf, 4),
            vec![0x1_0000, 0x1_0001]
        );
        bh_free_lowlevel_string(new_buf, LOWLEVEL_UNICODE_BASE_SIZE, 4);
    }

    #[test]
    fn jit_ll_shrink_array_defaults_to_str_for_a_raw_buffer() {
        // A raw-fallback buffer is not GC-owned, so the width defaults to STR.
        let buf = bh_alloc_lowlevel_string(6, LOWLEVEL_STR_BASE_SIZE, 1);
        assert_ne!(buf, 0);
        for index in 0..6 {
            bh_write_lowlevel_char(buf, index, (b'A' + index as u8) as i64, 1);
        }
        let new_buf = jit_ll_shrink_array(buf as crate::PyObjectRef, 3);
        assert!(!new_buf.is_null());
        let new_buf = new_buf as i64;
        assert_eq!(bh_lowlevel_string_len(new_buf), 3);
        assert_eq!(bh_read_lowlevel_string(new_buf, 1), vec![65, 66, 67]);
        bh_free_lowlevel_string(new_buf, LOWLEVEL_STR_BASE_SIZE, 1);
    }
}
