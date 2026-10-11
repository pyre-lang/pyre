//! Object header layout for GC-managed objects.
//!
//! Each GC object starts with one logical RPython word that stores both the
//! type ID (lower half) and GC flags (upper half), matching incminimark's
//! `HDR.tid` layout where `first_gcflag = 1 << (LONG_BIT//2)`.
//!
//! The Rust storage remains `u64` on wasm32 because GC payloads include
//! Rust values with eight-byte alignment.  Shrinking the physical prefix to
//! four bytes would misalign those payloads.  Only the low native-word-sized
//! part is semantically occupied there: 16 type-id bits followed by 16 flag
//! bits, just like a translated 32-bit PyPy.  The upper four bytes are ABI
//! padding, not an extension of either field.

use crate::GcFlags;

/// Number of bits reserved for the type ID in the lower half.
pub const TYPE_ID_BITS: u32 = usize::BITS / 2;
pub const TYPE_ID_MASK: u64 = (1u64 << TYPE_ID_BITS) - 1;

/// Shift applied to `GcFlags` to position it in the upper half of the
/// logical header word. `GcFlags` declares each flag at position 0..N.
pub const FLAG_SHIFT: u32 = TYPE_ID_BITS;

/// `llarena.round_up_for_allocation` / `memory_alignment()`: objects align
/// to the wider of the physical header and a machine word.
pub const MEMORY_ALIGNMENT: usize = if GcHeader::ALIGN > std::mem::size_of::<usize>() {
    GcHeader::ALIGN
} else {
    std::mem::size_of::<usize>()
};

/// The sentinel value written into a forwarded nursery object's header.
/// Equivalent to incminimark's `tid = -42` in a native Signed word.
pub const FORWARDED_MARKER: u64 = (usize::MAX - 41) as u64;

/// Mimic-header word of an off-GC host jitframe (`alloc_off_gc_jitframe`).
///
/// `TypeRegistry::register` assigns `id = entries.len()`, so the first type
/// is 0. A nursery object is `GcHeader::new(type_id)` (`init_gc_object`
/// flags=0), so a type-0 JITFRAME's header word is also 0. Host blocks
/// used to zero that word, and `jitframe_is_off_gc_host` then handed the
/// GC replacement to `free_host_jitframe`.
///
/// Flag bit 16 of the header flag half: `_GCFLAG_FIRST_UNUSED` is bit 13,
/// and this GC already claimed 13/14/15 (`FINALIZER_REGISTERED`,
/// `FINALIZER_RUN`, `YOUNG_RAWMALLOC`). No `GcHeader::new` / `with_flags`
/// allocation sets it. `GCFLAG_NO_HEAP_PTRS` marks a prebuilt that still
/// holds no heap pointer (`init_gc_object_immortal`), not an off-heap
/// malloc, and pairing it with `GCFLAG_TRACK_YOUNG_PTRS` would enter the
/// write-barrier fast path. `GCFLAG_TRACK_YOUNG_PTRS` stays clear here so
/// that path still skips. On wasm32 the 16-bit flag half is full, so the
/// bit occupies the first padding bit of the physical `u64`, which a live
/// GC object leaves zero.
pub const OFF_GC_HOST_MARKER: u64 = 1u64 << (16 + FLAG_SHIFT);

const _: () = assert!(OFF_GC_HOST_MARKER != 0);
const _: () = assert!(OFF_GC_HOST_MARKER != FORWARDED_MARKER);
const _: () = assert!(
    OFF_GC_HOST_MARKER & (crate::GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits() << FLAG_SHIFT) == 0
);

/// GC object header, placed immediately before the object payload.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GcHeader {
    /// Low native-word half: type ID. Next half: GC flags (shifted).
    /// On wasm32 the upper 32 bits are physical ABI padding: zero on a live
    /// object, and `FORWARDSTUB.forw` on a forwarding stub
    /// ([`GcHeader::set_forwarding_address`]).
    pub tid_and_flags: u64,
}

impl GcHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = std::mem::size_of::<GcHeader>();
    /// Alignment required by the fixed-width header word.
    ///
    /// This is wider than a native pointer on wasm32. Old-generation size
    /// classes must therefore use this alignment rather than pointer width.
    pub const ALIGN: usize = std::mem::align_of::<GcHeader>();

    /// Minimum object size in the nursery: header + one pointer (for forwarding).
    pub const MIN_NURSERY_OBJ_SIZE: usize = Self::SIZE + std::mem::size_of::<usize>();

    /// Create a new header with the given type ID and no flags set.
    pub fn new(type_id: u32) -> Self {
        assert!(
            u64::from(type_id) <= TYPE_ID_MASK,
            "GC type id {type_id} does not fit the {TYPE_ID_BITS}-bit header field"
        );
        GcHeader {
            tid_and_flags: type_id as u64,
        }
    }

    /// Create a new header with the given type ID and initial flags.
    pub fn with_flags(type_id: u32, raw_flags: GcFlags) -> Self {
        assert!(
            u64::from(type_id) <= TYPE_ID_MASK,
            "GC type id {type_id} does not fit the {TYPE_ID_BITS}-bit header field"
        );
        assert_eq!(
            raw_flags.bits() & !(TYPE_ID_MASK),
            0,
            "GC flags do not fit the {TYPE_ID_BITS}-bit header field"
        );
        GcHeader {
            tid_and_flags: (raw_flags.bits() << FLAG_SHIFT) | (type_id as u64),
        }
    }

    /// Extract the type ID from the lower bits.
    #[inline]
    pub fn type_id(self) -> u32 {
        (self.tid_and_flags & TYPE_ID_MASK) as u32
    }

    /// Extract the flags (shifted back to positions 0..N).
    ///
    /// A forwarded header sets every bit, so the read retains bits no
    /// constant names rather than dropping them.
    #[inline]
    pub fn flags(self) -> GcFlags {
        GcFlags::from_bits_retain(self.tid_and_flags >> FLAG_SHIFT)
    }

    /// Set a flag bit.
    #[inline]
    pub fn set_flag(&mut self, flag: GcFlags) {
        self.tid_and_flags |= flag.bits() << FLAG_SHIFT;
    }

    /// Clear a flag bit.
    #[inline]
    pub fn clear_flag(&mut self, flag: GcFlags) {
        self.tid_and_flags &= !(flag.bits() << FLAG_SHIFT);
    }

    /// Test whether a flag bit is set.
    #[inline]
    pub fn has_flag(self, flag: GcFlags) -> bool {
        self.tid_and_flags & (flag.bits() << FLAG_SHIFT) != 0
    }

    /// Check if this header indicates a forwarded object.
    ///
    /// Compare the logical native word (incminimark's one-`Signed` `tid`),
    /// not the physical `u64`. On wasm32 the upper 32 bits of a stub hold
    /// the forwarding address (`set_forwarding_address`), so a full `u64`
    /// compare against `FORWARDED_MARKER` would never match there.
    #[inline]
    pub fn is_forwarded(self) -> bool {
        (self.tid_and_flags as usize) == (FORWARDED_MARKER as usize)
    }

    /// Mark the header at `hdr` as forwarded and store the forwarding address
    /// in the native word immediately after the logical tid word.
    ///
    /// incminimark writes `tid = -42` into `HDR` (one `Signed`) and
    /// `FORWARDSTUB.forw` at `obj`. The logical tid here is one `usize`;
    /// on wasm32 that is the low half of the physical `u64` header, so
    /// `forw` occupies the padding half rather than the first payload
    /// word. A type-9 `ItemsBlock` keeps its `capacity` at `obj+0`.
    /// On 64-bit the logical word fills the header and `forw` is still
    /// the first payload word, matching a translated 64-bit PyPy.
    ///
    /// # Safety
    /// `hdr` must point to a valid `GcHeader`. On 64-bit, at least
    /// `size_of::<usize>()` bytes of payload must be writable. No other
    /// reference into either range may be alive for the duration of the call.
    #[inline]
    pub unsafe fn set_forwarding_address(hdr: *mut GcHeader, new_addr: usize) {
        unsafe {
            let words = hdr as *mut usize;
            words.write(FORWARDED_MARKER as usize);
            words.add(1).write(new_addr);
        }
    }

    /// Read the forwarding address from a forwarded object.
    ///
    /// # Safety
    /// `hdr` must point to a valid forwarded `GcHeader`
    /// (`(*hdr).is_forwarded()` must hold).
    #[inline]
    pub unsafe fn forwarding_address(hdr: *const GcHeader) -> usize {
        unsafe { (hdr as *const usize).add(1).read() }
    }
}

/// Raw pointer to the `GcHeader` preceding `obj_addr`.
///
/// Returns a raw pointer rather than a reference: upgrading this to `&mut`
/// at the call site creates a Unique retag, and any subsequent call through
/// `header_of` to the same address would invalidate that tag under stacked
/// borrows. Keeping the return raw lets callers re-derive a reference per
/// access (or skip the reference entirely) without producing overlapping
/// exclusive tags.
///
/// # Safety
/// `obj_addr` must point to valid memory with a `GcHeader` immediately
/// before it.
#[inline]
pub unsafe fn header_of(obj_addr: usize) -> *mut GcHeader {
    (obj_addr - GcHeader::SIZE) as *mut GcHeader
}

/// Per-type GC metadata, mirroring the compile-time constants that
/// `gct_fv_gc_malloc` closes over:
///
/// ```python
/// type_id = self.get_type_id(TYPE)
/// c_type_id = rmodel.inputconst(TYPE_ID, type_id)
/// info = self.layoutbuilder.get_info(type_id)
/// c_size = rmodel.inputconst(lltype.Signed, info.fixedsize)
/// ```
///
/// In RPython these are inputconsts woven into the `direct_call` to
/// the malloc helper. Here they are associated items on the payload
/// type, surfaced through `malloc_typed` so the allocator can read
/// them without a runtime dispatch.
///
/// `TYPE_ID` must match the id returned by `gc.register_type` during
/// JitDriver init; a `debug_assert_eq!` there guards against drift.
pub trait GcType {
    /// Backend-registered GC type id, equal to `c_type_id` closed over
    /// by `gct_fv_gc_malloc`. Read at runtime so the value can be
    /// assigned by the JIT driver after `gc.register_type` returns —
    /// auto-id mode delivers the result through this accessor. Explicit
    /// `type_id = N` cells return `N` unchanged.
    fn type_id() -> u32;
    /// Fixed payload size in bytes, equal to `info.fixedsize` from
    /// `layoutbuilder.get_info`.
    const SIZE: usize;
}

/// Pointee of [`GCREF`]. `lltype.GcOpaqueType('GCREF')` (`llmemory.py`).
///
/// Never malloc'd and never passed to `gc.register_type`. `GcType` exists
/// so `DeclaredGcFacts` seeds this StructId as Gc; `history.py` `getkind`
/// of `Ptr(GcOpaqueType)` is then `'ref'`.
#[repr(C)]
pub struct GCREFOpaque {
    _private: [u8; 0],
}

impl GcType for GCREFOpaque {
    fn type_id() -> u32 {
        // No register_type row: the private field makes the type unconstructible, so no
        // malloc_typed call can name it.
        u32::MAX
    }
    const SIZE: usize = 0;
}

/// `llmemory.GCREF = Ptr(GcOpaqueType('GCREF'))`.
///
/// A thin pointer alias with the same machine ABI as `*mut u8`.
#[allow(non_camel_case_types)]
pub type GCREF = *mut GCREFOpaque;

/// Allocate a value of type `T` behind a leading [`GcHeader`] and return the
/// payload pointer (`block + SIZE`).
///
/// `malloc_fixedsize` analog: reserve `size_gc_header + size` bytes,
/// `init_gc_object(result, typeid, flags=0)` at the block start, and return
/// `obj = result + size_gc_header`. A freshly allocated object carries no
/// flags, so the header word is exactly `type_id` (`GcHeader::new`). Callers
/// pass the object's GC type id (`malloc_typed` threads `GcType::type_id()`;
/// the untyped bridge and frames pass the object-root id 0).
///
/// The header's flags are all clear, so a write barrier reading `*(obj - SIZE)`
/// sees `TRACK_YOUNG_PTRS = 0` and skips the object. These objects are not yet
/// routed through the managed nursery allocator; they are leaked host
/// allocations, and the header is what lets the barrier read `*(obj - SIZE)`
/// without dereferencing foreign memory.
///
/// These are NOT incminimark's prebuilt family and must not be stamped
/// `NO_HEAP_PTRS | TRACK_YOUNG_PTRS` (`init_gc_object_immortal`). Upstream's
/// prebuilt objects are translation-time constants whose fields name other
/// prebuilt objects, which is what makes "an untouched prebuilt object is a
/// leaf" (incminimark.py) true. These blocks are allocated at
/// runtime and their reference fields are written directly at construction,
/// with no write barrier — so an untouched one DOES hold heap pointers, and
/// the targets are ordinary collectable objects that nothing else roots.
/// Claiming the flag makes the first barrier on any *other* field enter the
/// object into `prebuilt_root_objects`, after which a major traces every
/// `gc_ptr_offsets` slot, including the never-barriered ones whose targets
/// were already reclaimed. Measured on `synth/pickle_ctor_args`: the walk
/// reaches a freed `W_ListObject` through
/// `prebuilt+0x30 -> W_ListObject -> ItemsBlock[0]`, and under
/// `MAJIT_GC_NURSERY_POISON=1` that list reads back `length = 0xAAAA_AAAA_AAAA_AAAA`.
/// pyre's real prebuilt lifecycle is the interpreter-side dirty bit
/// (`mark_prebuilt_roots_dirty` / `PYRE_GC_PREBUILT_REMEMBER`) with explicit
/// per-structure walkers that resolve ownership before forwarding.
///
/// # Safety
/// - The result points past the header; it MUST NOT be passed to
///   `Box::from_raw` or freed at the payload pointer.
/// - `align_of::<T>() <= GcHeader::SIZE`, enforced at monomorphisation: a
///   wider alignment would move the payload off the fixed `obj - SIZE` offset.
pub fn alloc_with_gc_header<T>(value: T, type_id: u32) -> *mut T {
    alloc_with_gc_header_flags(value, GcHeader::new(type_id))
}

/// [`alloc_with_gc_header`] for a genuine immortal leaf — `init_gc_object_immortal`
/// (`NO_HEAP_PTRS | TRACK_YOUNG_PTRS`), incminimark's prebuilt shape.
///
/// `GCFLAG_NO_HEAP_PTRS` asserts the object holds no heap pointer *yet*: while
/// it is set marking skips the object, and the first pointer written into it is
/// expected to clear the flag and enrol the object in `prebuilt_root_objects`.
/// Upstream does stamp prebuilt objects that have reference fields, because
/// those fields name other prebuilt objects and the GC transform puts a barrier
/// on every later store. Neither condition holds for a box allocated here in
/// general, so a payload may use this only if it is a leaf and stays one (the
/// `None` / `True` / `False` / `NotImplemented` / `Ellipsis` singletons, whose
/// whole body is a `PyObject` header with a null `w_class`), an interned
/// exact-str wrapper (`w_str_from_wtf8_immortal`: off-heap `w_class`, raw
/// `value` / `index_storage`), or if its constructor barriers the whole box
/// before anything can collect and every later reference store is barriered
/// too. An ordinary `#[pyre_class]` box does NOT qualify — its reference
/// fields are written at construction with no write barrier, so once the first
/// barrier on any other field opens the object, a major would follow those
/// never-updated slots into freed memory. See [`alloc_with_gc_header`] for the
/// measured failure that rule comes from.
pub fn alloc_with_gc_header_immortal<T>(value: T, type_id: u32) -> *mut T {
    alloc_with_gc_header_flags(
        value,
        GcHeader::with_flags(
            type_id,
            crate::GcFlags::GCFLAG_NO_HEAP_PTRS | crate::GcFlags::GCFLAG_TRACK_YOUNG_PTRS,
        ),
    )
}

/// Varsize analogue of [`alloc_with_gc_header_immortal`].
///
/// `rstr.STR` is a `GcStruct` (`rstr.py`); a prebuilt `rpy_string` carries a
/// header stamped by `init_gc_object_immortal` (`gc/base.py` /
/// `incminimark.py`). The returned pointer is the address after the header, so
/// `{hash, len, chars}` readers stay unchanged. A payload holds no GC pointer,
/// so it is a true leaf (`GCFLAG_NO_HEAP_PTRS | GCFLAG_TRACK_YOUNG_PTRS`).
/// Returns null on overflow or allocation failure.
pub fn alloc_varsize_with_gc_header_immortal(payload_size: usize, type_id: u32) -> *mut u8 {
    alloc_varsize_with_gc_header_immortal_zeroed_flag(payload_size, type_id, false)
}

/// [`alloc_varsize_with_gc_header_immortal`] with the payload bytes cleared.
pub fn alloc_varsize_with_gc_header_immortal_zeroed(payload_size: usize, type_id: u32) -> *mut u8 {
    alloc_varsize_with_gc_header_immortal_zeroed_flag(payload_size, type_id, true)
}

fn varsize_immortal_layout(payload_size: usize) -> Option<std::alloc::Layout> {
    let total = GcHeader::SIZE.checked_add(payload_size)?;
    std::alloc::Layout::from_size_align(total, GcHeader::SIZE).ok()
}

fn alloc_varsize_with_gc_header_immortal_zeroed_flag(
    payload_size: usize,
    type_id: u32,
    zeroed: bool,
) -> *mut u8 {
    let Some(layout) = varsize_immortal_layout(payload_size) else {
        return std::ptr::null_mut();
    };
    unsafe {
        let raw = if zeroed {
            std::alloc::alloc_zeroed(layout)
        } else {
            std::alloc::alloc(layout)
        };
        if raw.is_null() {
            return std::ptr::null_mut();
        }
        (raw as *mut GcHeader).write(GcHeader::with_flags(
            type_id,
            crate::GcFlags::GCFLAG_NO_HEAP_PTRS | crate::GcFlags::GCFLAG_TRACK_YOUNG_PTRS,
        ));
        raw.add(GcHeader::SIZE)
    }
}

/// Free a payload returned by [`alloc_varsize_with_gc_header_immortal`].
///
/// # Safety
/// `payload` must be a live pointer from that allocator, and `payload_size`
/// must be the size passed at allocation.
pub unsafe fn dealloc_varsize_with_gc_header(payload: *mut u8, payload_size: usize) {
    if payload.is_null() {
        return;
    }
    let layout = varsize_immortal_layout(payload_size).expect("dealloc_varsize_with_gc_header");
    unsafe { std::alloc::dealloc(payload.sub(GcHeader::SIZE), layout) };
}

fn alloc_with_gc_header_flags<T>(value: T, header: GcHeader) -> *mut T {
    const {
        assert!(
            std::mem::align_of::<T>() <= GcHeader::SIZE,
            "alloc_with_gc_header: align_of::<T>() exceeds GcHeader::SIZE; the fixed obj - SIZE header offset would misalign the payload",
        )
    };
    let total = GcHeader::SIZE + std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>().max(GcHeader::SIZE);
    let layout = std::alloc::Layout::from_size_align(total, align)
        .expect("alloc_with_gc_header: invalid layout");
    unsafe {
        let raw = std::alloc::alloc(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        (raw as *mut GcHeader).write(header);
        let ptr = raw.add(GcHeader::SIZE) as *mut T;
        ptr.write(value);
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GcFlags;

    #[test]
    fn test_new_header() {
        let hdr = GcHeader::new(42);
        assert_eq!(hdr.type_id(), 42);
        assert_eq!(hdr.flags(), GcFlags::empty());
    }

    #[test]
    fn off_gc_host_marker_is_not_a_live_gc_header() {
        assert_ne!(GcHeader::new(0).tid_and_flags, OFF_GC_HOST_MARKER);
        let host = GcHeader {
            tid_and_flags: OFF_GC_HOST_MARKER,
        };
        assert!(!host.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(!host.is_forwarded());
    }

    #[test]
    fn test_with_flags() {
        let hdr = GcHeader::with_flags(
            7,
            GcFlags::GCFLAG_TRACK_YOUNG_PTRS | GcFlags::GCFLAG_VISITED,
        );
        assert_eq!(hdr.type_id(), 7);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_VISITED));
        assert!(!hdr.has_flag(GcFlags::GCFLAG_PINNED));
    }

    #[test]
    fn test_set_clear_flag() {
        let mut hdr = GcHeader::new(1);
        assert!(!hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));

        hdr.set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert_eq!(hdr.type_id(), 1);

        hdr.clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        assert!(!hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert_eq!(hdr.type_id(), 1);
    }

    #[test]
    fn test_forwarded_marker() {
        let mut hdr = GcHeader::new(100);
        assert!(!hdr.is_forwarded());

        hdr.tid_and_flags = FORWARDED_MARKER;
        assert!(hdr.is_forwarded());
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn forwarded_marker_ignores_physical_padding() {
        let mut hdr = GcHeader::new(9);
        hdr.tid_and_flags = FORWARDED_MARKER | (0x0107_83d8u64 << 32);
        assert!(
            hdr.is_forwarded(),
            "wasm32 padding in the high half of the physical u64 must not hide tid=-42"
        );
    }

    #[test]
    fn test_header_size() {
        assert_eq!(GcHeader::SIZE, 8);
    }

    #[test]
    fn logical_header_halves_follow_rpython_word_size() {
        assert_eq!(TYPE_ID_BITS, usize::BITS / 2);
        assert_eq!(FLAG_SHIFT, usize::BITS / 2);
        assert_eq!(FORWARDED_MARKER, (usize::MAX - 41) as u64);
    }

    #[test]
    fn has_shadow_lives_in_logical_flag_half() {
        let mut hdr = GcHeader::new(1);
        hdr.set_flag(GcFlags::GCFLAG_HAS_SHADOW);
        let bytes = hdr.tid_and_flags.to_le_bytes();
        let flag_byte = (FLAG_SHIFT / 8) as usize;
        assert_ne!(
            bytes[flag_byte], 0,
            "HAS_SHADOW must occupy the flags half at byte {flag_byte}"
        );
        // Physical header is 8 bytes on every target; wasm32's upper 4 are
        // padding. A store at obj-4 (byte 4) therefore cannot clear HAS_SHADOW.
        if FLAG_SHIFT < 32 {
            assert_eq!(&bytes[4..8], &[0, 0, 0, 0]);
        }
    }

    #[test]
    fn alloc_with_gc_header_prepends_header() {
        // Leaked: the payload pointer points past the header, so it must not
        // be freed.
        let p = alloc_with_gc_header(0xABCD_u64, 0x1357);
        unsafe {
            assert_eq!(*p, 0xABCD);
            let hdr = header_of(p as usize);
            // type id recorded; flags clear (init_gc_object flags=0).
            assert_eq!((*hdr).type_id(), 0x1357);
            assert_eq!((*hdr).flags(), GcFlags::empty());
            assert!(!(*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
            assert!(!(*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS));
        }
    }

    #[test]
    fn alloc_varsize_with_gc_header_immortal_stamps_init_gc_object_immortal() {
        let p = alloc_varsize_with_gc_header_immortal(16, 0x2468);
        assert!(!p.is_null());
        unsafe {
            (p as *mut u64).write(0x1111_2222_3333_4444);
            let hdr = header_of(p as usize);
            assert_eq!((*hdr).type_id(), 0x2468);
            assert!((*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS));
            assert!((*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
            assert_eq!(*(p as *const u64), 0x1111_2222_3333_4444);
            dealloc_varsize_with_gc_header(p, 16);
        }
    }

    #[test]
    fn test_flags_dont_clobber_type_id() {
        let mut hdr = GcHeader::new(0xDEAD);
        hdr.set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        hdr.set_flag(GcFlags::GCFLAG_VISITED);
        hdr.set_flag(GcFlags::GCFLAG_PINNED);
        assert_eq!(hdr.type_id(), 0xDEAD);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_VISITED));
        assert!(hdr.has_flag(GcFlags::GCFLAG_PINNED));
    }

    #[test]
    fn test_multiple_flags() {
        let mut hdr = GcHeader::new(0);
        hdr.set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        hdr.set_flag(GcFlags::GCFLAG_HAS_CARDS);
        hdr.set_flag(GcFlags::GCFLAG_CARDS_SET);

        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_CARDS_SET));
        assert!(!hdr.has_flag(GcFlags::GCFLAG_VISITED));

        hdr.clear_flag(GcFlags::GCFLAG_HAS_CARDS);
        assert!(!hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
    }

    #[test]
    fn test_forwarding_address() {
        // Allocate enough space for header + forwarding pointer
        let mut buf = [0u8; 16];
        let hdr_ptr = buf.as_mut_ptr() as *mut GcHeader;
        unsafe {
            *hdr_ptr = GcHeader::new(42);
        }

        let target_addr: usize = 0xCAFEBABE;
        unsafe {
            GcHeader::set_forwarding_address(hdr_ptr, target_addr);
        }
        assert!(unsafe { (*hdr_ptr).is_forwarded() });
        assert_eq!(
            unsafe { GcHeader::forwarding_address(hdr_ptr) },
            target_addr
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn forwarding_preserves_payload_length_on_32bit() {
        let mut buf = [0u8; 16];
        buf[8..12].copy_from_slice(&42u32.to_le_bytes());
        let hdr_ptr = buf.as_mut_ptr() as *mut GcHeader;
        unsafe {
            *hdr_ptr = GcHeader::new(9);
            GcHeader::set_forwarding_address(hdr_ptr, 0x1000);
        }
        assert!(unsafe { (*hdr_ptr).is_forwarded() });
        assert_eq!(unsafe { GcHeader::forwarding_address(hdr_ptr) }, 0x1000);
        let length = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(
            length, 42,
            "type-9 length at obj+0 must survive a 32-bit forward"
        );
    }
}
