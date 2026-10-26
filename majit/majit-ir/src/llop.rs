//! Translation-visible `llop` carriers (`rtyper/lltypesystem/lloperation.py`).
//!
//! Each function is the runtime meaning of one low-level operation. Its call
//! path survives in LLBC, and `majit-translate` lowers the call to the
//! operation itself instead of walking the body.

/// `llop.gc_load_indexed(T, base, index, scale, base_ofs)`: the `T` stored
/// at byte `base_ofs + index * scale` inside the GC object `base`.
///
/// `scale` and `base_ofs` must be literals at the call site
/// (`jtransform.py rewrite_op_gc_load_indexed`).
///
/// # Safety
/// The addressed bytes must lie inside `base`'s allocation and hold a `T`.
#[inline(always)]
pub unsafe fn gc_load_indexed<T: Copy, B>(
    base: *mut B,
    index: usize,
    scale: usize,
    base_ofs: usize,
) -> T {
    unsafe {
        (base as *const u8)
            .add(base_ofs + index * scale)
            .cast::<T>()
            .read()
    }
}
