//! AbstractLLCPU accessors —
//! `rpython/jit/backend/llsupport/llmodel.py` parity.
//!
//! Two families live here. The jitframe accessors read and write a
//! deadframe's slots; the `*_at_mem` accessors read and write a field
//! of a heap struct at a byte offset.
//!
//! Upstream both live as methods on `AbstractLLCPU`, invoked as
//! `cpu.get_int_value(deadframe, index)` /
//! `cpu.write_int_at_mem(struct, ofs, size, value)`. In majit there is
//! no `AbstractLLCPU`-equivalent trait with `self`-carried state that a
//! backend would override — every backend shares the same
//! JITFRAME-backed deadframe layout and the same raw-memory field
//! layout — so the accessors are free functions keyed on a raw
//! `*const JitFrame` or on a base address.
//!
//! The `AbstractCPU` base class (rpython/jit/backend/model.py)
//! declares the abstract contract for these accessors; all entries
//! below match those signatures.

use majit_gc::shadow_stack::ShadowRoot;
use majit_ir::{FailDescr, GcRef};

use crate::jitframe::{
    FIRST_ITEM_OFFSET, JitFrame, JitframeDescrFacts, malloc_gc_jitframe, malloc_host_jitframe,
    reuse_off_gc_jitframe,
};

/// llmodel.py — get_latest_descr.
///
/// Returns the `jf_descr` field, which holds the descr pointer of
/// the last GUARD or FINISH operation executed.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn get_latest_descr(ptr: *const JitFrame) -> usize {
    unsafe { (*ptr).jf_descr }
}

/// Store the `jf_descr` field directly.
///
/// Upstream writes `jf_descr` through generated assembly or through
/// `compile.py` finish-descr injection; this free-function form exists
/// for host-side test / arena runners that bypass the compiled-code
/// write path.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn set_latest_descr(ptr: *mut JitFrame, descr: usize) {
    unsafe {
        (*ptr).jf_descr = descr;
    }
}

#[inline]
fn decode_rd_loc_slot_from_locs(locs: &[u16], index: usize) -> Option<usize> {
    // Synthetic descrs never receive `write_failure_recovery_description`,
    // so the table stays empty and the fail-arg index *is* the slot
    // (`runner.rs` identity fallback). A stamped table uses `0xFFFF`
    // for a numbering hole (`optimizeopt` `logical_rd_locs`); resume
    // reconstructs those through TAGCONST/TAGVIRTUAL, not the jitframe.
    if locs.is_empty() {
        return Some(index);
    }
    match locs.get(index).copied() {
        None | Some(0xFFFF) => None,
        Some(pos) => Some(pos as usize),
    }
}

/// llmodel.py — `_decode_pos(deadframe, index)`.
///
/// Translate one `rd_locs[index]` entry into the jitframe slot
/// `get_int_value_direct(jf, slot)` consumes.  Returns `None` for
/// 0xFFFF (unmapped — the resume system handles those through the
/// `rd_numb` TAGCONST/TAGVIRTUAL encoding) or for out-of-range indices.
///
/// Upstream `_decode_pos` is a method on `AbstractLLCPU` and fetches the
/// descr itself through `get_latest_descr(deadframe)`; here the descr is
/// passed in, because the deadframe types that hold one are the callers.
#[inline]
pub fn decode_rd_loc_slot(descr: &dyn FailDescr, index: usize) -> Option<usize> {
    decode_rd_loc_slot_from_locs(descr.rd_locs(), index)
}

/// llmodel.py — `get_int_value_direct(deadframe, pos)`.
///
/// Read the `Signed` slot at pre-decoded position `slot` from
/// `jf_frame`.  `slot` is a post-`rd_locs[i]` slot index (i.e.
/// `pos / WORD` in upstream terms), NOT a raw byte offset and NOT
/// the logical fail-arg index.  Upstream's `get_int_value_direct`
/// takes a byte `pos`; majit's `JitFrame::slot_ptr_const` already
/// scales slot * WORD internally, so the pre-WORD-scaled slot is
/// what this accessor expects.
///
/// The logical `get_int_value(deadframe, index)` entry point —
/// which first calls `_decode_pos(deadframe, index)` to translate
/// `index` through `rd_locs[]` — is a method on the deadframe
/// types instead ([`crate::deadframe`], [`crate::libc_deadframe`]),
/// because it needs the descr that `_decode_pos` reaches through
/// `get_latest_descr(deadframe)` and a free function keyed on a
/// bare frame pointer has no way to get one.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_int_value_direct(ptr: *const JitFrame, slot: usize) -> isize {
    unsafe { *JitFrame::slot_ptr_const(ptr, slot) }
}

/// `llmodel.py get_int_value(deadframe, index)` with `rd_locs` already in hand.
///
/// `resume.py ResumeDataDirectReader.decode_int` / `decode_ref` call
/// `cpu.get_int_value(self.deadframe, num)` / `get_ref_value` on the
/// deadframe GCREF. The reader stores that GCREF plus `rd_locs` and
/// hits this instead of matching [`FailArgSource`] per TAGBOX slot.
#[inline(always)]
pub unsafe fn get_int_value_at(deadframe: i64, locs: &[u16], index: usize) -> i64 {
    let ptr = deadframe as *mut JitFrame;
    // `ResumeDataDirectReader::bind_deadframe_root` registers the reader's
    // `deadframe` field, so a collection rewrites it in place; the frame a
    // guard failure returned carries no `jf_forward` link.
    debug_assert_eq!(unsafe { JitFrame::resolve(ptr) }, ptr);
    match decode_rd_loc_slot_from_locs(locs, index) {
        Some(slot) => unsafe { get_int_value_direct(ptr as *const JitFrame, slot) as i64 },
        None => 0,
    }
}

/// llmodel.py `get_int_value(deadframe, index)`.
///
/// `_decode_pos` then `get_int_value_direct`. Values stay in
/// `jf_frame[]`; nothing is copied into a host list.
#[inline]
pub unsafe fn get_int_value(ptr: *const JitFrame, descr: &dyn FailDescr, index: usize) -> i64 {
    let ptr = unsafe { JitFrame::resolve(ptr as *mut JitFrame) };
    // `_decode_pos` is only invoked for a live box. A 0xFFFF hole has
    // no jitframe word — cranelift writes fail args densely and skips
    // `None` holes, so treating the hole as `jf_frame[index]` reads
    // uninitialized memory and hands it to residual calls as a pointer.
    match decode_rd_loc_slot(descr, index) {
        Some(slot) => unsafe { get_int_value_direct(ptr, slot) as i64 },
        None => 0,
    }
}

/// Fail-arg source for `resume.py` TAGBOX decode.
///
/// RPython's decoder calls `cpu.get_int_value` / `get_ref_value` on the
/// deadframe. Tests that already hold a dense fail-arg list keep the
/// slice arm; compiled guard failure uses the jitframe.
///
/// The jitframe arm re-reads the deadframe's own [`ShadowRoot`]
/// (`llmodel.py` `return ll_frame`: the deadframe IS the jitframe). A
/// collection during `blackhole_from_resumedata` updates that slot;
/// `get` re-reads it. A second per-failure acquire is extra.
pub enum FailArgSource<'a> {
    Slice(&'a [i64]),
    JitFrame {
        root: &'a ShadowRoot,
        descr: &'a dyn FailDescr,
        rd_locs: &'a [u16],
        n: usize,
    },
    /// `llmodel.py get_int_value` for a frame `malloc_jitframe` answered
    /// from outside the GC heap. The address does not move, so this arm
    /// takes no [`ShadowRoot`]. Interior Ref slots stay roots through
    /// `walk_live_deadframes` while `LibcJitFrameDeadFrame::owning` is alive.
    LibcJitFrame {
        ptr: *const JitFrame,
        rd_locs: &'a [u16],
        n: usize,
    },
}

impl<'a> FailArgSource<'a> {
    /// Re-use the deadframe's existing shadow-stack slot.
    ///
    /// `compile.py` `ResumeGuardDescr.handle_fail` keeps the deadframe
    /// object; `resume.py` `ResumeDataDirectReader.decode_int` reads
    /// `cpu.get_int_value(self.deadframe, num)` through that object.
    pub fn from_held_root(root: &'a ShadowRoot, descr: &'a dyn FailDescr, n: usize) -> Self {
        Self::from_held_root_locs(root, descr, descr.rd_locs(), n)
    }

    /// `from_held_root` with `rd_locs` already read off the descr.
    ///
    /// `compile.py` `ResumeGuardDescr.handle_fail` is a method on the
    /// concrete descr, so `self.rd_locs` is a field. The trait-object
    /// `rd_locs()` vtable is the extra; pass the slice from a downcast.
    pub fn from_held_root_locs(
        root: &'a ShadowRoot,
        descr: &'a dyn FailDescr,
        rd_locs: &'a [u16],
        n: usize,
    ) -> Self {
        Self::JitFrame {
            root,
            descr,
            rd_locs,
            n,
        }
    }

    /// In-place fail args for an off-heap jitframe. See [`FailArgSource::LibcJitFrame`].
    pub fn from_libc_jitframe(ptr: *const JitFrame, descr: &'a dyn FailDescr, n: usize) -> Self {
        Self::from_libc_jitframe_locs(ptr, descr.rd_locs(), n)
    }

    /// `from_libc_jitframe` with `rd_locs` already read off the descr.
    pub fn from_libc_jitframe_locs(ptr: *const JitFrame, rd_locs: &'a [u16], n: usize) -> Self {
        let ptr = unsafe { JitFrame::resolve(ptr as *mut JitFrame) };
        Self::LibcJitFrame { ptr, rd_locs, n }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Slice(s) => s.len(),
            Self::JitFrame { n, .. } | Self::LibcJitFrame { n, .. } => *n,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> i64 {
        match self {
            Self::Slice(s) => s.get(index).copied().unwrap_or(0),
            Self::JitFrame {
                root, rd_locs, n, ..
            } => {
                debug_assert!(index < *n);
                let ptr = root.get().0 as *const JitFrame;
                match decode_rd_loc_slot_from_locs(rd_locs, index) {
                    Some(slot) => {
                        let ptr = unsafe { JitFrame::resolve(ptr as *mut JitFrame) };
                        unsafe { get_int_value_direct(ptr, slot) as i64 }
                    }
                    None => 0,
                }
            }
            Self::LibcJitFrame { ptr, rd_locs, n } => {
                debug_assert!(index < *n);
                match decode_rd_loc_slot_from_locs(rd_locs, index) {
                    Some(slot) => {
                        let ptr = unsafe { JitFrame::resolve(*ptr as *mut JitFrame) };
                        unsafe { get_int_value_direct(ptr, slot) as i64 }
                    }
                    None => 0,
                }
            }
        }
    }

    #[inline]
    pub fn first(&self) -> Option<i64> {
        (!self.is_empty()).then(|| self.get(0))
    }
}

impl Clone for FailArgSource<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::Slice(s) => Self::Slice(s),
            Self::JitFrame {
                root,
                descr,
                rd_locs,
                n,
            } => Self::JitFrame {
                root: *root,
                descr: *descr,
                rd_locs,
                n: *n,
            },
            Self::LibcJitFrame { ptr, rd_locs, n } => Self::LibcJitFrame {
                ptr: *ptr,
                rd_locs,
                n: *n,
            },
        }
    }
}

impl<'a> From<&'a [i64]> for FailArgSource<'a> {
    fn from(s: &'a [i64]) -> Self {
        Self::Slice(s)
    }
}

impl<'a> From<&'a Vec<i64>> for FailArgSource<'a> {
    fn from(s: &'a Vec<i64>) -> Self {
        Self::Slice(s)
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for FailArgSource<'a> {
    fn from(s: &'a [i64; N]) -> Self {
        Self::Slice(s)
    }
}

impl std::fmt::Debug for FailArgSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vals: Vec<i64> = (0..self.len()).map(|i| self.get(i)).collect();
        match self {
            Self::Slice(_) => f.debug_tuple("Slice").field(&vals).finish(),
            Self::JitFrame { root, n, .. } => f
                .debug_struct("JitFrame")
                .field("ptr", &(root.get().0 as *const JitFrame))
                .field("n", n)
                .field("vals", &vals)
                .finish(),
            Self::LibcJitFrame { ptr, n, .. } => f
                .debug_struct("LibcJitFrame")
                .field("ptr", ptr)
                .field("n", n)
                .field("vals", &vals)
                .finish(),
        }
    }
}

/// Fail args for `resume.py ResumeDataDirectReader.decode_int`.
///
/// A GC jitframe goes through [`FailArgSource::from_held_root`] on the
/// deadframe's own shadow-stack slot. An off-heap frame goes through
/// [`FailArgSource::from_libc_jitframe`] and is read in place
/// (`llmodel.py get_int_value`). `None` is a boxed frame, which the
/// caller copies.
pub fn fail_arg_source_from_frame<'a>(
    frame: &'a crate::DeadFrame,
    descr: &'a dyn FailDescr,
    n: usize,
) -> Option<FailArgSource<'a>> {
    fail_arg_source_from_frame_with_locs(frame, descr, descr.rd_locs(), n)
}

/// `fail_arg_source_from_frame` with `rd_locs` already read.
///
/// `history.py` `AbstractFailDescr.rd_locs` is a field on the descr
/// `handle_fail` is already running as. One field read, not a vtable
/// call per failure.
#[inline]
pub fn fail_arg_source_from_frame_with_locs<'a>(
    frame: &'a crate::DeadFrame,
    descr: &'a dyn FailDescr,
    rd_locs: &'a [u16],
    n: usize,
) -> Option<FailArgSource<'a>> {
    if let Some(jf) = frame.as_jitframe() {
        if let Some(root) = jf.shadow_root() {
            Some(FailArgSource::from_held_root_locs(root, descr, rd_locs, n))
        } else {
            Some(FailArgSource::from_libc_jitframe_locs(
                jf.jf_gcref().0 as *const JitFrame,
                rd_locs,
                n,
            ))
        }
    } else {
        frame.as_libc_jitframe().map(|libc| {
            FailArgSource::from_libc_jitframe_locs(libc.frame_addr() as *const JitFrame, rd_locs, n)
        })
    }
}

/// Symmetric setter for `get_int_value_direct`.
///
/// llsupport/llmodel.py does not expose this: compiled code writes
/// `jf_frame[i]` directly. It is retained here for host-side test /
/// arena runners only.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn set_int_value(ptr: *mut JitFrame, slot: usize, value: isize) {
    unsafe {
        *JitFrame::slot_ptr(ptr, slot) = value;
    }
}

/// llmodel.py — `get_ref_value_direct(deadframe, pos)`.
///
/// Read the slot at pre-decoded position `slot` as a reference
/// (pointer-sized).  See `get_int_value_direct` for the slot/index
/// distinction.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_ref_value_direct(ptr: *const JitFrame, slot: usize) -> usize {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize;
        *base.add(slot)
    }
}

/// llmodel.py — `get_float_value_direct(deadframe, pos)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_float_value_direct(ptr: *const JitFrame, slot: usize) -> u64 {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const u64;
        *base.add(slot)
    }
}

/// llmodel.py — `write_int_at_mem(gcref, ofs, size, newvalue)`.
///
/// Stores the low `size` bytes of `newvalue` at `base + ofs`.
///
/// The width is the field descriptor's, not the value's: an integer
/// field narrower than a word is a real field, and a store that ignores
/// `size` writes over whatever follows it in the struct.
///
/// Upstream walks `unroll_basic_sizes` (symbolic.py — word, char,
/// short, int) and falls through to
/// `raise NotImplementedError("size = %d" % size)` when nothing
/// matches. A size not in that set means the descriptor disagrees with
/// the struct it describes, so this panics rather than widening the
/// store; silently falling back to a word would corrupt the neighbour.
///
/// Float fields are deliberately absent from that table
/// (symbolic.py:78 "does not contain Float ^^^ which must be
/// special-cased") and go through [`write_float_at_mem`].
///
/// # Safety
/// `base + ofs` must be a writable field of at least `size` bytes.
pub unsafe fn write_int_at_mem(base: usize, ofs: usize, size: usize, newvalue: i64) {
    let addr = base.wrapping_add(ofs);
    // Truncation is width-identical for the signed and unsigned member of
    // each `unroll_basic_sizes` pair, so the store needs the size but not
    // the sign — which is why upstream discards it (`_` at llmodel.py:482)
    // while the matching read keeps it.
    unsafe {
        match size {
            1 => (addr as *mut u8).write_unaligned(newvalue as u8),
            2 => (addr as *mut u16).write_unaligned(newvalue as u16),
            4 => (addr as *mut u32).write_unaligned(newvalue as u32),
            8 => (addr as *mut i64).write_unaligned(newvalue),
            _ => panic!(
                "write_int_at_mem: unsupported size {size} \
                 (llmodel.py:488 NotImplementedError)"
            ),
        }
    }
}

/// llmodel.py — `write_ref_at_mem(gcref, ofs, newvalue)`.
///
/// Pointer-width store. Upstream takes no `size` here: pointer fields
/// have one width, which is why `bh_setfield_gc_r` (llmodel.py)
/// unpacks only the offset while `bh_setfield_gc_i` unpacks the size
/// too.
///
/// Upstream's trailing comment reads "the write barrier is implied
/// above" — implied by the `llop.raw_store` that the framework GC
/// transformer rewrites. Nothing rewrites this store, so a caller whose
/// container may be old-generation while `newvalue` is young owes the
/// barrier itself.
///
/// # Safety
/// `base + ofs` must be a writable pointer-width field.
pub unsafe fn write_ref_at_mem(base: usize, ofs: usize, newvalue: usize) {
    unsafe { (base.wrapping_add(ofs) as *mut usize).write_unaligned(newvalue) }
}

/// Pointer-width load at an aligned address.
///
/// `acquire` selects `AtomicPtr::load(Acquire)` for a slot published with
/// `AtomicPtr::store(Release)`. The plain arm is `llmodel.py read_ref_at_mem`.
///
/// # Safety
/// `addr` must be a readable, naturally aligned pointer-width field. When
/// `acquire` is true, every write of that field is an atomic pointer store.
pub unsafe fn read_ref_at_mem(addr: usize, acquire: bool) -> usize {
    unsafe {
        if acquire {
            std::sync::atomic::AtomicPtr::<()>::from_ptr(addr as *mut *mut ())
                .load(std::sync::atomic::Ordering::Acquire) as usize
        } else {
            *(addr as *const usize)
        }
    }
}

/// Integer-width load at `base + ofs`.
///
/// The plain arm is `llmodel.py read_int_at_mem`. `acquire` selects
/// `AtomicU*::from_ptr(..).load(Acquire)` at the field width, then
/// sign-extends. Acquire loads require a naturally aligned address.
///
/// # Safety
/// `base + ofs` must be a readable `size`-byte integer field. When
/// `acquire` is true, the address is naturally aligned and every write
/// of that field is an atomic store of the same width.
pub unsafe fn read_int_at_mem(
    base: usize,
    ofs: usize,
    size: usize,
    sign: bool,
    acquire: bool,
) -> i64 {
    let addr = base.wrapping_add(ofs);
    unsafe {
        if acquire {
            debug_assert!(
                size > 0 && addr % size == 0,
                "acquire integer load requires natural alignment: addr={addr:#x} size={size}"
            );
            use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};
            let raw = match size {
                1 => AtomicU8::from_ptr(addr as *mut u8).load(Ordering::Acquire) as u64,
                2 => AtomicU16::from_ptr(addr as *mut u16).load(Ordering::Acquire) as u64,
                4 => AtomicU32::from_ptr(addr as *mut u32).load(Ordering::Acquire) as u64,
                8 => AtomicU64::from_ptr(addr as *mut u64).load(Ordering::Acquire),
                other => panic!("read_int_at_mem: unsupported acquire size {other}"),
            };
            match (size, sign) {
                (8, _) => raw as i64,
                (4, true) => raw as u32 as i32 as i64,
                (4, false) => raw as u32 as i64,
                (2, true) => raw as u16 as i16 as i64,
                (2, false) => raw as u16 as i64,
                (1, true) => raw as u8 as i8 as i64,
                (1, false) => raw as u8 as i64,
                _ => unreachable!(),
            }
        } else {
            match (size, sign) {
                (1, true) => (addr as *const i8).read_unaligned() as i64,
                (1, false) => (addr as *const u8).read_unaligned() as i64,
                (2, true) => (addr as *const i16).read_unaligned() as i64,
                (2, false) => (addr as *const u16).read_unaligned() as i64,
                (4, true) => (addr as *const i32).read_unaligned() as i64,
                (4, false) => (addr as *const u32).read_unaligned() as i64,
                (8, _) => (addr as *const i64).read_unaligned(),
                other => panic!(
                    "read_int_at_mem: unsupported (size, signed) = {other:?} \
                     (llmodel.py read_int_at_mem NotImplementedError)"
                ),
            }
        }
    }
}

/// llmodel.py — `write_float_at_mem(gcref, ofs, newvalue)`.
///
/// `FLOATSTORAGE`-width store. Like the ref store this takes no `size`:
/// floats are excluded from `unroll_basic_sizes` (symbolic.py) and
/// `bh_setfield_gc_f` (llmodel.py) unpacks only the offset.
///
/// # Safety
/// `base + ofs` must be a writable float-width field.
pub unsafe fn write_float_at_mem(base: usize, ofs: usize, newvalue: f64) {
    unsafe { write_float_at_mem_sized(base, ofs, 8, newvalue) }
}

/// Load a JIT float-bank value. Size 8 is `FLOATSTORAGE`
/// (`llmodel.py read_float_at_mem`). Size 4 is a jit_interp
/// `float(f32)` field: load f32 and widen, matching
/// `VirtualizableInfo::read_field`.
///
/// # Safety
/// `base + ofs` must be a readable `size`-byte float field.
pub unsafe fn read_float_at_mem_sized(base: usize, ofs: usize, size: usize) -> f64 {
    let addr = base.wrapping_add(ofs);
    match size {
        4 => f64::from(unsafe { (addr as *const f32).read_unaligned() }),
        8 => unsafe { (addr as *const f64).read_unaligned() },
        _ => panic!("read_float_at_mem: unsupported size {size}"),
    }
}

/// Store a JIT float-bank value. Size 8 is `FLOATSTORAGE`. Size 4
/// demotes to f32, matching `VirtualizableInfo::write_field`.
///
/// # Safety
/// `base + ofs` must be a writable `size`-byte float field.
pub unsafe fn write_float_at_mem_sized(base: usize, ofs: usize, size: usize, newvalue: f64) {
    let addr = base.wrapping_add(ofs);
    match size {
        4 => unsafe { (addr as *mut f32).write_unaligned(newvalue as f32) },
        8 => unsafe { (addr as *mut f64).write_unaligned(newvalue) },
        _ => panic!("write_float_at_mem: unsupported size {size}"),
    }
}

/// llmodel.py — get_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn get_savedata_ref(ptr: *const JitFrame) -> usize {
    unsafe { (*ptr).jf_savedata }
}

/// llmodel.py — set_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn set_savedata_ref(ptr: *mut JitFrame, value: usize) {
    unsafe {
        (*ptr).jf_savedata = value;
    }
}

/// True when a raw finish entry must take the general `execute_token` path.
///
/// `llmodel.py` `AbstractLLCPU.make_execute_token` always allocates through
/// `self.gc_ll_descr.malloc_jitframe` (`GcLLDescription.malloc_jitframe`).
/// The raw path does that too: a descr with a `JITFRAME` type id bumps a
/// nursery frame (`malloc_gc_jitframe`); `HostHeapGc` keeps the
/// host-block arm, which is what its `malloc_jitframe` is. `diag` is the
/// remaining reason to leave that path — exec logging and dumps walk the
/// typed `execute_token` frame.
#[inline]
pub fn raw_done_entry_use_general(diag: bool) -> bool {
    diag
}

/// Take the token's parked off-GC frame when it fits, else allocate.
///
/// `HostHeapGc.malloc_jitframe` is [`malloc_host_jitframe`]. Reuse is the
/// stand-in for a nursery bump on that descr only.
#[inline]
pub fn take_or_alloc_parked_entry_frame(
    token: &crate::JitCellToken,
    size_bytes: usize,
) -> *mut JitFrame {
    match token.take_entry_frame(size_bytes) {
        Some(p) => {
            unsafe { reuse_off_gc_jitframe(p) };
            p
        }
        None => malloc_host_jitframe(size_bytes),
    }
}

/// Park a single unforwarded host frame on the token, else free each
/// off-GC host link. A GC-owned frame is left for the collector.
///
/// `llmodel.py` `execute_token` does not free: `jitframe_allocate` is
/// `lltype.malloc(JITFRAME)` and the frame becomes garbage after the
/// call. A descr that owns jitframes matches that. `HostHeapGc` still
/// parks, because its malloc is a host block.
#[inline]
pub fn park_or_free_done_entry_frame(
    token: &crate::JitCellToken,
    head: *mut JitFrame,
    tip: *mut JitFrame,
    is_gc_object: bool,
) {
    if is_gc_object {
        return;
    }
    let single = tip == head && unsafe { (*head).jf_forward.is_null() };
    if single && token.park_entry_frame(head) {
        return;
    }
    unsafe { free_off_gc_host_done_entry_chain(head) };
}

/// Same walk as [`crate::libc_deadframe::free_jitframe_chain`]: host
/// links only. `park_or_free_done_entry_frame` is the done-entry caller.
unsafe fn free_off_gc_host_done_entry_chain(head: *mut JitFrame) {
    unsafe { crate::libc_deadframe::free_jitframe_chain(head) };
}

/// Slot 0 of a `DoneWithThisFrameDescrInt` frame. `get_int_value(deadframe, 0)`.
#[inline(always)]
pub fn done_int_slot0(tip: *mut JitFrame) -> i64 {
    unsafe { get_int_value_direct(tip, 0) as i64 }
}

/// Slot 0 of a `DoneWithThisFrameDescrRef` frame. `get_ref_value(deadframe, 0)`.
#[inline(always)]
pub fn done_ref_slot0(tip: *mut JitFrame) -> usize {
    unsafe { get_ref_value_direct(tip, 0) }
}

/// Frame setup for a raw finish. `llmodel.py` `AbstractLLCPU.make_execute_token`.
///
/// A descr that owns jitframes allocates through `gc_ll_descr.malloc_jitframe`
/// (`GcLLDescription.malloc_jitframe` / `jitframe_allocate`) on every call.
/// `facts` is that descr's answers on this CPU (`AbstractLLCPU.gc_ll_descr`).
/// `HostHeapGc` reuses the token's parked off-GC frame when it fits, else
/// allocates a host block — that descr's `malloc_jitframe`. Inits the header
/// and stores `args` at `first_slot`. A GC frame then gets
/// `llop.gc_writebarrier` when the allocation spilled out of the nursery.
/// The caller invokes compiled code under its own calling convention and
/// reads slot 0 of the returned frame.
///
/// # Safety
/// `num_slots` is the `jf_frame` length ([`JitFrame::alloc_size`]).
/// `first_slot + args.len()` must fit in that length.
#[inline(always)]
pub unsafe fn prepare_done_raw_entry_frame(
    token: &crate::JitCellToken,
    args: &[i64],
    first_slot: usize,
    num_slots: usize,
    facts: JitframeDescrFacts,
) -> *mut JitFrame {
    assert!(
        num_slots >= first_slot.saturating_add(args.len()),
        "execute_token: frame depth {num_slots} < input top {} for {} args",
        first_slot + args.len(),
        args.len()
    );
    let clt = unsafe { &*token.compiled_loop_token_ptr() };
    let fi_ptr = clt.frame_info.data_ptr() as *const crate::JitFrameInfo;
    let frame_bytes = JitFrame::alloc_size(num_slots);
    let (jf_ptr, needs_wb) = if facts.is_gc_object {
        malloc_gc_jitframe(facts, frame_bytes)
    } else {
        (take_or_alloc_parked_entry_frame(token, frame_bytes), false)
    };
    unsafe {
        JitFrame::init(jf_ptr, fi_ptr, num_slots);
        for (i, &word) in args.iter().enumerate() {
            set_int_value(jf_ptr, first_slot + i, word as isize);
        }
    }
    if needs_wb {
        majit_gc::gc_write_barrier(GcRef(jf_ptr as usize));
    }
    jf_ptr
}

#[cfg(test)]
mod tests {
    use super::{FailArgSource, decode_rd_loc_slot, get_int_value, set_int_value};
    use crate::jitframe::{JitFrame, alloc_off_gc_jitframe, free_off_gc_jitframe};
    use crate::resume_guard_descr::make_resume_guard_descr_typed;
    use majit_ir::Type;

    #[test]
    fn sized_float_load_store_does_not_touch_neighbor() {
        #[repr(C)]
        struct Pair {
            value: f32,
            neighbor: u32,
        }
        let mut pair = Pair {
            value: 1.25,
            neighbor: 0xa5a5_a5a5,
        };
        let base = (&mut pair as *mut Pair) as usize;
        unsafe {
            assert_eq!(super::read_float_at_mem_sized(base, 0, 4), 1.25);
            super::write_float_at_mem_sized(base, 0, 4, -2.5);
        }
        assert_eq!(pair.value, -2.5);
        assert_eq!(pair.neighbor, 0xa5a5_a5a5);
    }

    #[test]
    fn empty_slice_get_does_not_panic() {
        // A host copy can be empty when resume numbering still asks
        // for TAGBOX 0 (`cpu.get_int_value(deadframe, 0)`). The
        // jitframe arm is the deadframe; a missing host slot reads 0.
        let src = FailArgSource::Slice(&[]);
        assert_eq!(src.get(0), 0);
        assert_eq!(src.len(), 0);
    }

    #[test]
    fn get_int_value_reads_zero_for_ffff_hole() {
        // optimizeopt `logical_rd_locs` stamps 0xFFFF on a None fail-arg.
        // The slot is not written (`emit_guard_exit` skips None). Reading
        // it as `jf_frame[index]` is how cranelift fed residual memmove a
        // poison pointer on exception_reused_object_tb_not_doubled.
        let descr = make_resume_guard_descr_typed(vec![Type::Int, Type::Ref, Type::Int]);
        let fd = descr.as_fail_descr().expect("typed resume guard");
        fd.set_rd_locs(vec![0, 0xFFFF, 2].into());
        assert_eq!(decode_rd_loc_slot(fd, 0), Some(0));
        assert_eq!(decode_rd_loc_slot(fd, 1), None);
        assert_eq!(decode_rd_loc_slot(fd, 2), Some(2));

        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(4));
        assert!(!frame.is_null());
        unsafe {
            for i in 0..4 {
                set_int_value(frame, i, 0x4155_8127);
            }
            set_int_value(frame, 0, 11);
            set_int_value(frame, 2, 22);
            assert_eq!(get_int_value(frame, fd, 0), 11);
            assert_eq!(
                get_int_value(frame, fd, 1),
                0,
                "0xFFFF hole must not surface the unwritten slot"
            );
            assert_eq!(get_int_value(frame, fd, 2), 22);
            free_off_gc_jitframe(frame);
        }
    }

    #[test]
    fn get_int_value_uses_identity_when_rd_locs_is_empty() {
        let descr = make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let fd = descr.as_fail_descr().expect("typed resume guard");
        assert!(fd.rd_locs().is_empty());
        assert_eq!(decode_rd_loc_slot(fd, 0), Some(0));
        assert_eq!(decode_rd_loc_slot(fd, 1), Some(1));

        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(2));
        assert!(!frame.is_null());
        unsafe {
            set_int_value(frame, 0, 7);
            set_int_value(frame, 1, 9);
            assert_eq!(get_int_value(frame, fd, 0), 7);
            assert_eq!(get_int_value(frame, fd, 1), 9);
            free_off_gc_jitframe(frame);
        }
    }

    /// A host entry forwarded (`jf_forward`) to a type-id-0 GC JITFRAME
    /// must free only the host link. `GcHeader::new(0)` is word 0, which
    /// used to look like an off-GC mimic header.
    #[test]
    fn free_off_gc_host_done_entry_chain_leaves_type0_gc_replacement() {
        use crate::jitframe::{
            JitFrameInfo, jitframe_is_off_gc_host, jitframe_type_info, malloc_host_jitframe,
            malloc_jitframe,
        };
        use majit_gc::GcAllocator;

        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let tid = gc.register_type(jitframe_type_info());
        assert_eq!(tid, 0, "this case is the first registered type");
        gc.set_jitframe_type_id(tid);

        let depth = 4;
        let bytes = JitFrame::alloc_size(depth);
        let host = malloc_host_jitframe(bytes);
        let gc_frame = malloc_jitframe(&mut gc, bytes);
        assert!(!host.is_null() && !gc_frame.is_null());
        let info = JitFrameInfo::default();
        unsafe {
            JitFrame::init(host, &info, depth);
            JitFrame::init(gc_frame, &info, depth);
            *JitFrame::slot_ptr(gc_frame, 0) = 0x11C0_FFEE;
            (*host).jf_forward = gc_frame;
            assert!(jitframe_is_off_gc_host(host));
            assert!(
                !jitframe_is_off_gc_host(gc_frame),
                "a type-id-0 nursery JITFRAME must not be classified as a host block"
            );
            majit_gc::shadow_stack::register_libc_jitframe(host as usize);
            super::free_off_gc_host_done_entry_chain(host);
            #[cfg(not(debug_assertions))]
            assert!(
                !majit_gc::shadow_stack::is_libc_jitframe(host as usize),
                "host frame must have been released by free_host_jitframe"
            );
            // Debug `free_host_jitframe` unregisters only when a collector is
            // installed; drop the test registration so a freed address is
            // not left in the set.
            majit_gc::shadow_stack::unregister_libc_jitframe(host as usize);
            assert!(
                gc.is_in_nursery(gc_frame as usize),
                "GC replacement must stay for the collector"
            );
            assert_eq!(
                (*majit_gc::header::header_of(gc_frame as usize)).type_id(),
                0
            );
            assert_eq!(*JitFrame::slot_ptr(gc_frame, 0), 0x11C0_FFEE);
            assert!((*gc_frame).jf_forward.is_null());
            assert!(!jitframe_is_off_gc_host(gc_frame));
        }
    }

    #[test]
    fn orthodox_resume_libc_jitframe_reads_in_place() {
        let descr = make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let fd = descr.as_fail_descr().expect("typed resume guard");
        assert!(fd.rd_locs().is_empty());
        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(2));
        assert!(!frame.is_null());
        unsafe {
            set_int_value(frame, 0, 41);
            set_int_value(frame, 1, 43);
            let src = FailArgSource::from_libc_jitframe(frame, fd, 2);
            assert_eq!(src.get(0), 41);
            assert_eq!(src.get(1), 43);
            assert_eq!(src.len(), 2);
            free_off_gc_jitframe(frame);
        }
    }
}
