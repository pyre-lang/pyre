//! The deadframe of a backend whose jitframes are allocated outside the GC
//! heap.
//!
//! `execute_token` mints the frame at `llmodel.py malloc_jitframe` and
//! hands that same value back at `llmodel.py return ll_frame` — the
//! deadframe IS the jitframe here as well, so nothing is copied out and the
//! accessors read `jf_frame[]` in place. What differs from
//! [`crate::deadframe::JitFrameDeadFrame`] is
//! ownership: these frames come from [`crate::jitframe::alloc_off_gc_jitframe`]
//! instead of the nursery, so they never move, take no root slot, and this
//! value's `Drop` releases each off-GC host link of the `jf_forward`
//! chain it was handed.
use parking_lot::RwLock;
use std::sync::OnceLock;

use majit_ir::DescrRef;
use majit_ir::GcRef;

use crate::deadframe::ExitDescr;
use crate::jitframe::JitFrame;

/// Concrete data stored in DeadFrame by a backend whose frames live off the
/// GC heap.
///
/// **THE DEADFRAME IS THE JITFRAME.** `llmodel.py grab_exc_value` obtains it by
/// `lltype.cast_opaque_ptr(jitframe.JITFRAMEPTR, deadframe)` and reads
/// `jf_guard_exc` / `jf_savedata` straight off it; `llmodel.py:298` allocates
/// exactly one `malloc_jitframe` per `execute_token`, and `:328` returns that
/// same object as the deadframe. So there is nothing to copy out of the frame
/// and no second structure to describe it: this type is the cast, and the
/// accessors below are the reads.
///
/// `fail_descr` carries the metainterp class-distinct Arc identity
/// (ResumeGuardDescr family for guards, DoneWithThisFrame*/
/// ExitFrameWithExceptionDescrRef for FINISH exits).  FailDescr-trait
/// operations on the descr go through `DescrRef::as_fail_descr`.
pub struct LibcJitFrameDeadFrame {
    /// The frame compiled code returned — the tip of `head`'s `jf_forward`
    /// chain whenever `_check_frame_depth` reallocated on the way.
    tip: *mut JitFrame,
    /// Head of the chain this deadframe OWNS and releases on drop, or `None`
    /// when the frames belong to somebody else. `Backend::force` mints a
    /// deadframe over the frame of a compiled run that is still executing
    /// (`llmodel.py force` likewise returns that frame rather than a
    /// copy), and freeing it there would pull the ground out from under the
    /// caller.
    owned_head: Option<*mut JitFrame>,
    /// Number of slots the frame carries. Bounds the index space the
    /// accessors answer for; an index past it reads 0.
    num_slots: usize,
    /// `jf_descr` cast back through [`ExitDescr::from_cell`] /
    /// [`ExitDescr::owned`].
    pub fail_descr: ExitDescr,
    /// Original `jf_descr` object identity when the exit used an attached
    /// metainterp descr (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
}

impl std::fmt::Debug for LibcJitFrameDeadFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibcJitFrameDeadFrame")
            .field("num_values", &self.num_slots)
            .field("owns_frame", &self.owned_head.is_some())
            .field("fail_descr", &self.fail_descr.get().repr())
            .field(
                "latest_descr",
                &self.latest_descr.as_ref().map(|descr| descr.repr()),
            )
            .finish()
    }
}

impl LibcJitFrameDeadFrame {
    /// Take ownership of the jitframe chain `head` and present its tip as the
    /// deadframe.
    ///
    /// # Safety
    /// `head` is a live `jf_forward` chain this value owns. Off-GC host
    /// links are `register_libc_jitframe`-tracked and no one else frees
    /// them; GC links stay for the collector. `tip` is a node of that
    /// chain, and the compiled epilogue has already popped every host
    /// node off the JF shadow stack.
    pub unsafe fn owning(
        head: *mut JitFrame,
        tip: *mut JitFrame,
        num_slots: usize,
        fail_descr: ExitDescr,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        // Interior refs are only roots while a collector can run. With none
        // installed, publishing the address is a set nothing walks.
        if majit_gc::collector_installed() {
            register_live_deadframe(tip as usize);
        }
        LibcJitFrameDeadFrame {
            tip,
            owned_head: Some(head),
            num_slots,
            fail_descr,
            latest_descr,
        }
    }

    /// Present `frame` as a deadframe without taking ownership of it.
    ///
    /// # Safety
    /// `frame` must outlive the returned value.
    pub unsafe fn borrowing(
        frame: *mut JitFrame,
        num_slots: usize,
        fail_descr: ExitDescr,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        LibcJitFrameDeadFrame {
            tip: frame,
            owned_head: None,
            num_slots,
            fail_descr,
            latest_descr,
        }
    }

    /// The frame this deadframe reads through — the tip of the chain the run
    /// returned.
    ///
    /// The counterpart of [`crate::deadframe::JitFrameDeadFrame::jf_gcref`],
    /// but a plain address rather than a re-read of a root slot: these frames
    /// are allocated off the GC heap and never move, so the address the
    /// deadframe was minted with stays correct for its whole lifetime. It is
    /// also the key an owning deadframe is held under in `LIVE_DEADFRAMES`.
    #[inline]
    pub fn frame_addr(&self) -> usize {
        self.tip as usize
    }

    /// `llmodel.py _decode_pos` — the jitframe slot logical failarg
    /// `index` lives in, or `None` when the descr maps it nowhere.
    ///
    /// Indices past `rd_locs` fall through to identity slot indexing, which is
    /// what the synthetic descrs the runner mints rely on. The logical index
    /// is deliberately decoded before the physical frame bound is checked:
    /// sparse resume data can have more failarg positions than this fallback
    /// bound, while a late failarg still maps to a valid live frame slot.
    fn slot_of(&self, index: usize) -> Option<usize> {
        let descr = self.fail_descr.as_fail_descr();
        let locs = descr.rd_locs();
        if let Some(&pos) = locs.get(index) {
            (pos != 0xFFFF).then_some(pos as usize)
        } else if index < self.num_slots {
            Some(index)
        } else {
            None
        }
    }

    pub fn get_int(&self, index: usize) -> i64 {
        match self.slot_of(index) {
            Some(slot) => unsafe { crate::llmodel::get_int_value_direct(self.tip, slot) as i64 },
            None => 0,
        }
    }

    /// `llmodel.py get_value_direct` — the raw frame word at a
    /// deadframe SLOT. `get_int` maps its argument through `slot_of`'s
    /// `rd_locs` decode (`llmodel.py _decode_pos`); this does not,
    /// because the slot a GUARD_VALUE counter records is a register or frame
    /// position of the failing trace, not a fail-argument index.
    pub fn get_int_at_slot(&self, slot: usize) -> i64 {
        if slot >= self.num_slots {
            return 0;
        }
        unsafe { crate::llmodel::get_int_value_direct(self.tip, slot) as i64 }
    }

    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    /// `cpu.grab_exc_value(deadframe)` (llmodel.py) — read
    /// `jf_guard_exc` off the frame. The exc=True failure-recovery stub staged
    /// `pos_exc_value` there for must_save_exception guards; other guards
    /// leave it NULL.
    pub fn exc_value(&self) -> GcRef {
        GcRef(unsafe { (*self.tip).jf_guard_exc })
    }

    /// `cpu.get_savedata_ref(deadframe)` (llmodel.py) — read
    /// `jf_savedata` off the frame.
    pub fn get_savedata_ref(&self) -> GcRef {
        GcRef(unsafe { (*self.tip).jf_savedata })
    }

    /// `cpu.set_savedata_ref(deadframe, data)` (llmodel.py) — write
    /// `jf_savedata` on the frame.
    pub fn set_savedata_ref(&mut self, data: GcRef) {
        unsafe { (*self.tip).jf_savedata = data.0 };
    }
}

/// Walk `jf_forward` (`jitframe.py` `jitframe_resolve`) and release each
/// off-GC host block. `llmodel.py` `jitframe_allocate` is
/// `lltype.malloc(JITFRAME)` — a GC object never freed explicitly — so a
/// GC link is left for the collector. `malloc_jitframe_no_collect` may mint
/// that replacement onto a host head (`llmodel.py` `realloc_frame` stores
/// `frame.jf_forward = new_frame`).
///
/// Host vs GC is [`crate::jitframe::jitframe_is_off_gc_host`]: the mimic
/// header word equals [`majit_gc::header::OFF_GC_HOST_MARKER`], not a
/// zero word (`GcHeader::new(0)` is also zero). Host bookkeeping is
/// [`crate::jitframe::release_malloc_host_jitframe`] inside
/// [`crate::jitframe::free_host_jitframe`].
///
/// # Safety
/// `head` is a live `jf_forward` chain. Off-GC host links must not still
/// be a GC root (`gen_footer_shadowstack` has popped them). GC links stay
/// reachable for the collector.
pub unsafe fn free_jitframe_chain(head: *mut JitFrame) {
    let mut cur = head;
    while !cur.is_null() {
        let next = unsafe { (*cur).jf_forward };
        if unsafe { crate::jitframe::jitframe_is_off_gc_host(cur) } {
            unsafe { crate::jitframe::free_host_jitframe(cur) };
        }
        cur = next;
    }
}

impl Drop for LibcJitFrameDeadFrame {
    fn drop(&mut self) {
        let Some(head) = self.owned_head else {
            return;
        };
        unregister_live_deadframe(self.tip as usize);
        unsafe { free_jitframe_chain(head) };
    }
}

// Live deadframes as GC roots.

/// Payload addresses of the jitframes currently held as deadframes.
///
/// Between a compiled run returning and its [`LibcJitFrameDeadFrame`]
/// dropping, the frame is off the JF shadow stack — `gen_footer_shadowstack`
/// pops it in the epilogue — but its interior `Ref` slots are still live,
/// because that is the window in which the frontend reads them. Nothing else
/// in the collector's root phase can see them, and majit has no conservative
/// stack scan to fall back on, so the set is published to `majit-gc` through
/// [`majit_gc::ActiveGcDeadFrameHooks`] and walked there as a root source.
///
/// Process-global rather than thread-local, matching
/// `shadow_stack::register_libc_jitframe`'s own registry: a collection can run
/// on a thread other than the one holding the deadframe, and a thread-local set
/// would be invisible to it.
// Same shape as `shadow_stack`'s libc-jitframe list: a flat vec, and a
// flag so a process that never publishes a deadframe does not take the lock
// on every drop. A collection walks the vec; hashing the address added a
// SipHash on every entry and exit.
static LIVE_DEADFRAMES_USED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static LIVE_DEADFRAMES: OnceLock<RwLock<Vec<usize>>> = OnceLock::new();

fn live_deadframes() -> &'static RwLock<Vec<usize>> {
    LIVE_DEADFRAMES.get_or_init(|| RwLock::new(Vec::new()))
}

fn register_live_deadframe(addr: usize) {
    LIVE_DEADFRAMES_USED.store(true, std::sync::atomic::Ordering::Release);
    let mut set = live_deadframes().write();
    if !set.contains(&addr) {
        set.push(addr);
    }
}

fn unregister_live_deadframe(addr: usize) {
    if !LIVE_DEADFRAMES_USED.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let mut set = live_deadframes().write();
    if let Some(index) = set.iter().position(|&slot| slot == addr) {
        set.swap_remove(index);
    }
}

/// [`majit_gc::LiveDeadFrameWalkerFn`] for the off-GC frame registry.
pub fn walk_live_deadframes(visit: &mut dyn FnMut(usize)) {
    for &addr in live_deadframes().read().iter() {
        visit(addr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    #[test]
    fn sparse_logical_index_is_decoded_before_physical_frame_bound() {
        let descr = crate::make_resume_guard_descr_typed(vec![Type::Int; 45]);
        let fail_descr = descr.as_fail_descr().expect("resume guard descr");
        let mut rd_locs = vec![0xFFFF; 45];
        rd_locs[40] = 15;
        rd_locs[41] = 13;
        rd_locs[42] = 27;
        rd_locs[44] = 28;
        fail_descr.set_rd_locs(rd_locs.into());

        // No frame access is performed: this test isolates `_decode_pos`'s
        // logical-to-physical ordering. The borrowed null owner is therefore
        // safe for the lifetime of the deadframe.
        let frame = unsafe {
            LibcJitFrameDeadFrame::borrowing(
                std::ptr::null_mut(),
                36,
                ExitDescr::owned(descr),
                None,
            )
        };

        assert_eq!(frame.slot_of(40), Some(15));
        assert_eq!(frame.slot_of(41), Some(13));
        assert_eq!(frame.slot_of(42), Some(27));
        assert_eq!(frame.slot_of(43), None);
        assert_eq!(frame.slot_of(44), Some(28));
        assert_eq!(frame.slot_of(45), None);
    }

    /// A host head forwarded (`jf_forward`) to a type-id-0 non-host
    /// JITFRAME must free only the host link. The non-host payload lives
    /// in this test's buffer: `GcHeader::new(0)` is word 0, which used to
    /// look like an off-GC mimic header, and the size-slot poison makes
    /// `free_off_gc_jitframe` panic (`Layout`) or park this buffer as a
    /// host block.
    #[test]
    fn free_jitframe_chain_releases_host_and_leaves_non_host_forward() {
        use crate::jitframe::{JitFrameInfo, jitframe_is_off_gc_host, malloc_host_jitframe};

        let depth = 4;
        let bytes = JitFrame::alloc_size(depth);
        let prefix = 2 * majit_gc::header::GcHeader::SIZE;
        let word_count = prefix.saturating_add(bytes).div_ceil(8);
        let mut buf = vec![0u64; word_count];
        buf[0] = u64::MAX;
        buf[1] = majit_gc::header::GcHeader::new(0).tid_and_flags;

        let gc_frame = unsafe { (buf.as_mut_ptr() as *mut u8).add(prefix) as *mut JitFrame };
        let host = malloc_host_jitframe(bytes);
        let info = JitFrameInfo::default();
        unsafe {
            JitFrame::init(host, &info, depth);
            JitFrame::init(gc_frame, &info, depth);
            *JitFrame::slot_ptr(gc_frame, 0) = 0x11C0_FFEE;
            (*host).jf_forward = gc_frame;
            assert!(jitframe_is_off_gc_host(host));
            assert!(
                !jitframe_is_off_gc_host(gc_frame),
                "a type-id-0 header must not be classified as a host block"
            );
            majit_gc::shadow_stack::register_libc_jitframe(host as usize);
            super::free_jitframe_chain(host);
            // A buggy walk parks this buffer (size-slot `u64::MAX`) as a
            // host block; the next host malloc then returns `gc_frame`.
            let recycled = malloc_host_jitframe(JitFrame::alloc_size(1));
            assert!(
                recycled != gc_frame,
                "non-host forward link was released as a host block"
            );
            crate::jitframe::free_host_jitframe(recycled);
            #[cfg(not(debug_assertions))]
            assert!(
                !majit_gc::shadow_stack::is_libc_jitframe(host as usize),
                "host frame must have been released by free_host_jitframe"
            );
            majit_gc::shadow_stack::unregister_libc_jitframe(host as usize);
            assert_eq!(
                buf[0],
                u64::MAX,
                "non-host size-slot poison must be untouched"
            );
            assert_eq!(
                (*majit_gc::header::header_of(gc_frame as usize)).tid_and_flags,
                0
            );
            assert_eq!(*JitFrame::slot_ptr(gc_frame, 0), 0x11C0_FFEE);
            assert!((*gc_frame).jf_forward.is_null());
            assert!(!jitframe_is_off_gc_host(gc_frame));
        }
    }
}
