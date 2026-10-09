/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    /// BaseAssembler.store_info_on_descr's fail locations: one physical
    /// i64 slot for each live logical resume position, None for a hole.
    /// An empty list is the identity layout used by reserved FINISH descrs
    /// and backend-only synthetic descriptors.
    pub fail_locs: Vec<Option<usize>>,
    pub is_finish: bool,
    /// Byte offset of GUARD_NOT_FORCED(_2)'s force-only spill area. Native
    /// backends carry these coordinates in their fail locations; keeping a
    /// disjoint area is essential because a following FINISH overwrites the
    /// ordinary exit slots and CALL_ASSEMBLER owns the dispatch-key slot.
    pub force_args_offset: u32,
    /// `0` when every force arg is `force_args_offset + loc * 8` (no tail, or
    /// the loc is still the prefix index). Otherwise `force_tail_base` is the
    /// first tail force word. A physical value index at or past
    /// `force_prefix_slots` names
    /// `force_tail_base + (loc - value_tail_index) * 8`: tail value items are
    /// three apart, and so are the force words beside them.
    pub force_tail_base: u32,
    pub force_prefix_slots: u32,
    pub value_tail_index: u32,
    /// Compile-time guard gcmap retained for FINISH after a
    /// GUARD_NOT_FORCED_2, matching `assembler._finish_gcmap`.
    pub force_gcmap_ptr: usize,
    /// Guest address of this guard's bridge-target cell. The same address
    /// is stamped on the metainterp descr's `adr_jump_offset` until
    /// `compile_bridge` patches it and clears that slot. The cell allocation
    /// stays here so a later re-emit bakes the same address.
    pub bridge_cell: u32,
    /// Per-fail-arg induction-advance flags for this guard. A loop-closing
    /// bridge reads them off the descr the guard already owns.
    pub fail_arg_advanced: Vec<bool>,
    /// Ordinary Ref homes of the trace that emitted this guard.
    pub trace_ref_homes: usize,
    /// LABEL-capture homes of the trace that emitted this guard.
    pub trace_label_homes: usize,
    /// This trace's guard epilogue has typed parameter dispatch arms.
    pub param_dispatch: bool,
    /// Table slot last written into `bridge_cell`. Re-emission reads the
    /// cell itself; this remembers the slot when an inline zeros the cell
    /// and a failed install has to put it back.
    pub bridge_slot: std::sync::atomic::AtomicU32,
    /// `history.py:125 id(descr)` parity — when the optimizer
    /// (`store_final_boxes_in_guard` / `make_and_attach_done_descrs`)
    /// stamps a metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
    /// `op.descr`, we keep it here so `get_latest_descr_arc` returns the
    /// canonical metainterp Arc (matching dynasm/cranelift).  `None`
    /// for synthetic backend-only descrs (`compile_bridge` placeholders,
    /// test scaffolds).
    pub meta_descr: Option<DescrRef>,
}

impl WasmFailDescr {
    pub fn frame_slot(&self, index: usize) -> Option<usize> {
        if self.fail_locs.is_empty() {
            return (index < self.fail_arg_types.len()).then_some(index);
        }
        self.fail_locs.get(index).copied().flatten()
    }
}

impl Descr for WasmFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
}

impl FailDescr for WasmFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }
}

/// Where this deadframe reads the jitframe.
///
/// A GC frame is re-read through [`OwnerRootGuard`] so a moving collection
/// cannot leave a raw pointer at the old copy. A host buffer is not a GC
/// object and does not move; its address stays in [`LiveFrame::Fixed`] and
/// [`WasmFrameData::host_frame`] owns the bytes.
enum LiveFrame {
    Rooted(majit_gc::shadow_stack::OwnerRootGuard),
    Fixed(*mut majit_backend::jitframe::JitFrame),
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame::Boxed` after `execute_token` returns. The
/// deadframe is the jitframe (`llmodel.py` `return ll_frame`): accessors
/// read `jf_frame` in place. `DeadFrame::JitFrame` is not used —
/// `FailArgSource::from_jitframe` decodes `rd_locs` as identity slots, while
/// this backend spills compactly.
///
/// One root: the frame. `jitframe_trace` walks `jf_savedata`, `jf_guard_exc`,
/// `jf_forward`, and the fail-arg Ref slots named by the guard's `jf_gcmap`.
/// [`Self::boxed`] is the unit-test snapshot that has no jitframe.
pub struct WasmFrameData {
    /// [`Self::boxed`] only: values with no jitframe. Empty when `frame` is set.
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
    /// [`Self::boxed`] only. A live frame reads `jf_guard_exc`
    /// (`llmodel.py` `grab_exc_value`).
    pub exc_value: i64,
    /// [`Self::boxed`] only. A live frame reads `jf_savedata`.
    pub savedata: i64,
    frame: Option<LiveFrame>,
    /// `force()` snapshot: fail args live at `force_args_offset`, tagged.
    read_force: bool,
    /// Off-GC host-buffer owner. Keeps the entry JitFrame alive after
    /// `execute_token` returns. The wasm32 host-buffer path.
    #[allow(dead_code)]
    host_frame: Option<majit_backend::libc_deadframe::LibcJitFrameDeadFrame>,
}

impl WasmFrameData {
    /// Unit-test snapshot with no jitframe. Production exits use
    /// [`Self::from_live_frame`].
    pub fn boxed(
        raw_values: Vec<i64>,
        fail_descr: Arc<WasmFailDescr>,
        exc_value: i64,
    ) -> Box<Self> {
        Box::new(WasmFrameData {
            raw_values,
            fail_descr,
            exc_value,
            savedata: 0,
            frame: None,
            read_force: false,
            host_frame: None,
        })
    }

    /// The jitframe `execute_token` / `force` returned. Fail args stay in
    /// its `jf_frame` slots. The frame is the only root: `jf_gcmap` names
    /// the exit's Ref slots and `jitframe_trace` names the header fields.
    ///
    /// `gc_root` takes an [`OwnerRootGuard`]. A force snapshot of a frame
    /// the running call already rooted takes one too — dropping it releases
    /// only this handle, and the call's shadow-stack root stays. A host
    /// buffer (`gc_root == false`) does not move; `host_frame` owns it when
    /// this deadframe does.
    pub fn from_live_frame(
        jf: *mut majit_backend::jitframe::JitFrame,
        fail_descr: Arc<WasmFailDescr>,
        read_force: bool,
        gc_root: bool,
        host_frame: Option<majit_backend::libc_deadframe::LibcJitFrameDeadFrame>,
    ) -> Box<Self> {
        let frame = if gc_root {
            LiveFrame::Rooted(majit_gc::shadow_stack::OwnerRootGuard::new(
                majit_ir::GcRef(jf as usize),
            ))
        } else {
            LiveFrame::Fixed(jf)
        };
        Box::new(WasmFrameData {
            raw_values: Vec::new(),
            fail_descr,
            exc_value: 0,
            savedata: 0,
            frame: Some(frame),
            read_force,
            host_frame,
        })
    }

    fn frame_ptr(&self) -> Option<*mut majit_backend::jitframe::JitFrame> {
        match self.frame.as_ref() {
            Some(LiveFrame::Rooted(root)) => {
                Some(root.get().0 as *mut majit_backend::jitframe::JitFrame)
            }
            Some(LiveFrame::Fixed(jf)) => Some(*jf),
            None => None,
        }
    }

    /// Current items base. A GC root is re-read so a moving collection
    /// cannot leave this pointing at the old frame.
    pub fn items_base(&self) -> Option<usize> {
        self.frame_ptr()
            .map(|jf| jf as usize + majit_backend::jitframe::FIRST_ITEM_OFFSET)
    }

    pub fn read_force(&self) -> bool {
        self.read_force
    }

    /// `cpu.set_savedata_ref(deadframe, data)` — write `jf_savedata`.
    /// `jitframe_trace` traces that header field; the frame root is enough.
    pub fn set_savedata(&mut self, data: majit_ir::GcRef) {
        let mut data_slot = data.0 as i64;
        if let Some(jf) = self.frame_ptr() {
            let depth = majit_gc::shadow_stack::resume_ref_roots_depth();
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut data_slot));
            }
            if crate::wasm_gc_owns_object(jf as usize) {
                crate::wasm_active_gc_write_barrier(majit_ir::GcRef(jf as usize));
            }
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
            unsafe { (*jf).jf_savedata = data_slot as usize };
        }
        self.savedata = data_slot;
    }

    #[allow(dead_code)] // wasm32 `execute_token` host-buffer path
    pub(crate) fn take_host_frame(
        &mut self,
        frame: majit_backend::libc_deadframe::LibcJitFrameDeadFrame,
    ) {
        self.host_frame = Some(frame);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use majit_gc::GcAllocator;
    use majit_ir::GcRef;

    use super::{Type, WasmFailDescr, WasmFrameData};

    struct RootCountingGc(Arc<AtomicUsize>);

    impl GcAllocator for RootCountingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_nursery_no_collect(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize(&mut self, _base_size: usize, _item_size: usize, _length: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize_no_collect(
            &mut self,
            _base_size: usize,
            _item_size: usize,
            _length: usize,
        ) -> GcRef {
            GcRef(0)
        }

        fn write_barrier(&mut self, _obj: GcRef) {}

        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}

        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }

        fn collect_nursery(&mut self) {}

        fn collect_full(&mut self) {}

        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }

        fn nursery_free_addr(&self) -> usize {
            0
        }

        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }

        fn nursery_top_addr(&self) -> usize {
            0
        }

        fn max_nursery_object_size(&self) -> usize {
            0
        }

        unsafe fn add_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn remove_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn install_root_counting_gc() -> (Arc<AtomicUsize>, crate::ActiveGcBox) {
        let roots = Arc::new(AtomicUsize::new(0));
        let gc_box = crate::install_gc_box(Box::new(RootCountingGc(Arc::clone(&roots))));
        (roots, gc_box)
    }

    fn fail_descr(fail_arg_types: Vec<Type>) -> Arc<WasmFailDescr> {
        Arc::new(WasmFailDescr {
            fail_index: 0,
            trace_id: 0,
            fail_arg_types,
            fail_locs: Vec::new(),
            is_finish: false,
            force_args_offset: 8,
            force_tail_base: 0,
            force_prefix_slots: 0,
            value_tail_index: 0,
            force_gcmap_ptr: 0,
            bridge_cell: 0,
            fail_arg_advanced: Vec::new(),
            trace_ref_homes: 0,
            trace_label_homes: 0,
            param_dispatch: false,
            bridge_slot: std::sync::atomic::AtomicU32::new(0),
            meta_descr: None,
        })
    }

    /// `compile.py` `PropagateExceptionDescr.handle_fail` through the same
    /// reader the metainterp calls. The wasm exit cell is that descr, not
    /// `FINISH_EXIT_INDEX_EXC`.
    #[test]
    fn propagate_exception_exit_raises_through_the_reader() {
        let descr: majit_ir::DescrRef = Arc::new(majit_backend::PropagateExceptionDescr::new());
        let fd = descr.as_fail_descr().expect("fail descr");
        assert!(!fd.is_finish());
        assert_eq!(fd.fail_index(), u32::MAX);
        assert_eq!(
            majit_backend::propagate_exception_handle_fail(fd, 0x42),
            Some(0x42)
        );
        // Do not install a provider: `memory_error_singleton_ref` is
        // process-global, and a stand-in address is what later tests
        // dereference as an exception object.
        // `handle_fail` asserts the fallback is non-null, so the null-cell
        // arm is checked only once some provider is registered.
        let memory_error = majit_backend::memory_error_singleton_ref();
        if memory_error != 0 {
            assert_eq!(
                majit_backend::propagate_exception_handle_fail(fd, 0),
                Some(memory_error)
            );
        }

        let cpu = crate::WasmBackend::new();
        cpu.exit_cells.attach_propagate(Arc::clone(&descr));
        let cell = super::descr_at(cpu.exit_cells.descr_ptrs().propagate_exception_descr)
            .expect("propagate cell");
        assert!(!cell.is_finish);
        let meta = cell.meta_descr.clone().expect("attached propagate descr");
        assert!(Arc::ptr_eq(&meta, &descr));
        let meta_fd = meta.as_fail_descr().expect("meta fail descr");
        assert_eq!(
            majit_backend::propagate_exception_handle_fail(meta_fd, 0x99),
            Some(0x99)
        );
    }

    #[test]
    fn a_finish_singleton_resolves_to_its_reserved_exit() {
        let _serialized = super::lock_cpu();
        // The emitted FINISH writes the index this returns and the emitted
        // CALL_ASSEMBLER check compares against the same constant, so a
        // singleton that failed to bind would send every clean callee finish
        // back to the host to be decoded and handed straight over.
        let cpu = crate::WasmBackend::new();
        let descr: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        cpu.exit_cells
            .attach_finish(super::FINISH_EXIT_INDEX_REF, Arc::clone(&descr));
        let attached = cpu.exit_cells.descr_ptrs();
        assert_eq!(
            super::attached_finish_exit_index(&attached, &Some(Arc::clone(&descr))),
            Some(super::FINISH_EXIT_INDEX_REF),
        );
        assert_eq!(
            super::done_with_this_frame_exit_index(Type::Ref),
            super::FINISH_EXIT_INDEX_REF,
        );
        // A descr this cpu was never handed has no shared identity, so it keeps
        // its own exit.
        let unattached: majit_ir::DescrRef =
            Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        assert_eq!(
            super::attached_finish_exit_index(&attached, &Some(unattached)),
            None
        );
    }

    #[test]
    fn finish_cells_keep_a_stable_address() {
        // CALL_ASSEMBLER compares `jf_descr` with the address baked at
        // compile time. Rebinding the singleton must not move that address.
        let _serialized = super::lock_cpu();
        let cpu = crate::WasmBackend::new();
        let ptr =
            super::finish_cell_ptr(&cpu.exit_cells.descr_ptrs(), super::FINISH_EXIT_INDEX_REF);
        let again =
            super::finish_cell_ptr(&cpu.exit_cells.descr_ptrs(), super::FINISH_EXIT_INDEX_REF);
        assert_eq!(ptr, again);
        assert_ne!(
            ptr,
            super::finish_cell_ptr(&cpu.exit_cells.descr_ptrs(), super::FINISH_EXIT_INDEX_INT)
        );
        let descr = super::descr_at(ptr).expect("finish cell");
        assert!(descr.is_finish);
        assert_eq!(descr.fail_arg_types, vec![Type::Ref]);
    }

    #[test]
    fn two_backends_attach_distinct_finish_cells() {
        let _serialized = super::lock_cpu();
        let left = crate::WasmBackend::new();
        let right = crate::WasmBackend::new();
        let left_descr: majit_ir::DescrRef =
            Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        let right_descr: majit_ir::DescrRef =
            Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        left.exit_cells
            .attach_finish(super::FINISH_EXIT_INDEX_REF, Arc::clone(&left_descr));
        right
            .exit_cells
            .attach_finish(super::FINISH_EXIT_INDEX_REF, Arc::clone(&right_descr));
        let left_ptr = left.exit_cells.descr_ptrs().done_with_this_frame_descr_ref;
        let right_ptr = right.exit_cells.descr_ptrs().done_with_this_frame_descr_ref;
        assert_ne!(left_ptr, right_ptr);
        let left_cell = super::descr_at(left_ptr).expect("left cell");
        let right_cell = super::descr_at(right_ptr).expect("right cell");
        assert!(Arc::ptr_eq(
            left_cell.meta_descr.as_ref().expect("left descr"),
            &left_descr
        ));
        assert!(Arc::ptr_eq(
            right_cell.meta_descr.as_ref().expect("right descr"),
            &right_descr
        ));
    }

    #[test]
    fn boxed_does_not_root_interior_slots() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(
            vec![0x10, 42, 0, 0x20],
            fail_descr(vec![Type::Ref, Type::Int, Type::Float, Type::Ref]),
            0x30,
        );
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn boxed_without_refs_or_exception_does_not_bracket_roots() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(vec![1, 2], fail_descr(vec![Type::Int, Type::Float]), 0);
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn set_savedata_writes_the_word_without_an_interior_root() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let mut frame = WasmFrameData::boxed(vec![1], fail_descr(vec![Type::Int]), 0);
        frame.set_savedata(GcRef(0x40));
        assert_eq!(frame.savedata, 0x40);
        frame.set_savedata(GcRef(0));
        assert_eq!(frame.savedata, 0);
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn set_savedata_on_a_force_snapshot_writes_the_live_jitframe() {
        let _serialized = super::lock_cpu();
        use majit_backend::jitframe::{JitFrame, alloc_off_gc_jitframe, free_off_gc_jitframe};

        // A leftover MiniMark box would treat `GcRef(0x51)` as a heap
        // pointer and rewrite `jf_savedata` when the snapshot's root is
        // dropped.
        crate::clear_gc_allocator();
        let jf = alloc_off_gc_jitframe(JitFrame::alloc_size(4));
        let mut frame =
            WasmFrameData::from_live_frame(jf, fail_descr(vec![Type::Int]), true, false, None);
        // `NO_CONCRETE` is not a heap object (even, non-8-aligned, non-null).
        // A low dummy (0x51) was chased as a nursery pointer when another
        // test's MiniMark was still the process hook target.
        let saved = GcRef::NO_CONCRETE;
        frame.set_savedata(saved);
        unsafe {
            assert_eq!((*jf).jf_savedata, saved.0);
        }
        drop(frame);
        unsafe {
            assert_eq!((*jf).jf_savedata, saved.0);
            free_off_gc_jitframe(jf);
        }
    }

    fn dummy_label_target(func_handle: u32) -> super::LabelTarget {
        super::LabelTarget {
            func_handle,
            wide_slot: 0,
            key: 0,
            num_args: 0,
            resume_safe: true,
            requires_own_frame: false,
            is_last_label: true,
            frame: crate::codegen::FrameGeometry::fixed(),
            owner_token: 0,
        }
    }

    #[test]
    fn retract_label_target_keeps_a_replacement_handle() {
        let _serialized = super::lock_cpu();
        let descr = majit_ir::make_loop_target_descr(0x7e71, false);
        let mut first = crate::release::LoopAsmResources::default();
        let mut second = crate::release::LoopAsmResources::default();
        super::publish_label_target(&mut first, &descr, dummy_label_target(7));
        super::retract_label_target_if_handle(&descr, 7);
        assert!(super::label_target(&descr).is_none());

        super::publish_label_target(&mut second, &descr, dummy_label_target(9));
        super::retract_label_target_if_handle(&descr, 7);
        assert_eq!(super::label_target(&descr).map(|t| t.func_handle), Some(9));
        super::retract_label_target_if_handle(&descr, 9);
        assert!(super::label_target(&descr).is_none());
    }
}

/// A resumable `LABEL` of a compiled loop. The publishing loop's
/// `LoopAsmResources` owns the `Box`, and `LoopTargetDescr::ll_loop_code`
/// holds its address (`history.py` `TargetToken._ll_loop_code`;
/// `assembler.py` `closing_jump` reads that word off the JUMP's descr).
#[derive(Clone, Copy, Debug)]
pub struct LabelTarget {
    /// Table slot of the owning loop's compiled function.
    pub func_handle: u32,
    /// Table slot of the fixed-arity label-parameter entry, or 0 when absent.
    pub wide_slot: u32,
    /// Resume dispatch key (`label ordinal + 1`) the bridge's JUMP writes.
    pub key: u32,
    /// The label's arg count — the resume loader reads exactly this many
    /// positional frame slots, so the JUMP arity must equal it.
    pub num_args: usize,
    /// Whether every live-in can be reconstructed from LABEL args or frozen
    /// backend capture slots (`codegen::label_resume_info`).
    pub resume_safe: bool,
    /// Backend capture slots are populated by this target loop's own
    /// fall-through path. If true, a bridge may resume only its source loop,
    /// not a sibling specialization that happens to share the geometry.
    pub requires_own_frame: bool,
    /// Whether this is the owning loop's LAST label (the loop header). A
    /// bridge landing here re-runs no segment code before the `loop`, so the
    /// livelock advance-check applies; earlier labels execute the peeled
    /// segment, which advances the state by itself.
    pub is_last_label: bool,
    /// Frozen frame geometry of the target token. A tail-call can only reuse
    /// a frame when its offsets agree exactly, not merely when its allocation
    /// is large enough.
    pub frame: crate::codegen::FrameGeometry,
    /// `JitCellToken.number` that published this row.
    pub owner_token: u64,
}

/// Frozen metadata for entering a compiled loop from a `CALL_ASSEMBLER` arm.
/// The table slot and frame layout are published only after the loop module is
/// installed, so a caller can decline before baking an unresolved target.
#[derive(Clone, Debug)]
pub struct CallAssemblerTarget {
    /// Owning `JitCellToken` number. Used only by the dormant wasm regression
    /// hook to select one target deterministically.
    pub token_number: u64,
    /// Address of this token's `WasmCaDispatchEntry` (`_ll_function_addr`).
    pub dispatch_entry: u32,
    pub func_handle: u32,
    pub input_types: Vec<Type>,
    /// Byte offset at which the target reads its fresh-entry dispatch key.
    /// Redirects may replace a temporary callback with a differently laid-out
    /// real loop, so this travels with the target's other runtime metadata.
    pub dispatch_key_ofs: u64,
    pub callee_frame_bytes: u32,
    pub callee_gcmap_ptr: i64,
    pub compiled_ptr: u64,
    /// Callee Ref-home origin, in item-base bytes. Redirects may replace a
    /// temporary callback with a differently laid-out loop, so this is
    /// loaded with the other runtime snapshot fields.
    pub home_slot_base: u32,
    pub home_slots: u32,
    /// Callee retains `_finish_gcmap` (`GUARD_NOT_FORCED_2`). The caller
    /// pop footer keeps that map while the callee's force token is armed
    /// when this is set, even if the caller module itself has no GNF2.
    pub has_guard_not_forced_2: u32,
    /// Homes `build_home_gcmap` marks: the used ordinary prefix, then the
    /// LABEL-capture tail. Reserved padding between them is unmarked.
    pub marked_ordinary: u32,
    pub marked_labels: u32,
    pub label_ref_slots: u32,
}

// ── CALL_ASSEMBLER dispatch table ──
//
// A trace module imports the guest's linear memory, so a boxed wasm-side
// allocation is addressable by every trace with an ordinary i32.load.  The
// box keeps that address stable while the map grows; the emitted code must
// never bake a table slot because redirects and the pending->real transition
// replace it after the caller module was compiled.
#[repr(C)]
pub struct WasmCaRuntimeTarget {
    /// `__indirect_function_table` slot. Zero means pending/unavailable.
    pub func_handle: u32,
    /// `CompiledWasmLoop` address for the deopt helper, in wasm32 memory.
    pub compiled_ptr: u32,
    /// Current callee JitFrame item-region size.  This is redirectable state:
    /// `CompiledLoopToken.update_frame_info` permits the replacement loop to
    /// need a deeper frame than the temporary callback.
    pub callee_frame_bytes: u32,
    /// Byte offset where this target reads its fresh-entry dispatch key.
    /// Redirects may replace a temporary callback with a differently sized
    /// real loop, so callers load this together with the frame allocation.
    pub dispatch_key_ofs: u32,
    /// Current callee GC map.  It must change together with frame depth when
    /// `redirect_call_assembler` installs the real loop.
    pub callee_gcmap_ptr: i64,
    pub home_slot_base: u32,
    pub home_slots: u32,
    pub has_guard_not_forced_2: u32,
    /// Used ordinary Ref homes the callee's static gcmap marks.
    pub marked_ordinary: u32,
    /// Used LABEL-capture homes the same map marks.
    pub marked_labels: u32,
    /// Reserved LABEL tail. The marked tail starts at
    /// `home_slots - label_ref_slots`.
    pub label_ref_slots: u32,
}

/// Stable cell baked by callers.  A redirect publishes one pointer to an
/// immutable target snapshot, so generated code cannot combine the new loop's
/// function with the temporary callback's frame geometry under concurrent
/// execution.  Old snapshots stay owned by the entry for as long as any caller
/// can still hold the pointer it loaded.
///
/// `has_guard_not_forced_2` is not snapshot state: an out-of-line
/// `GUARD_NOT_FORCED_2` bridge can attach while a CALL_ASSEMBLER is
/// already inside the callee. The in-flight footer still holds the
/// pre-call snapshot, so this flag lives on the cell and only goes
/// 0 → 1.
#[repr(C)]
pub struct WasmCaDispatchEntry {
    pub target_ptr: AtomicU32,
    pub has_guard_not_forced_2: AtomicU32,
    /// Host pointer of the `CompiledWasmLoop` the latest snapshot names.
    /// The guest snapshot keeps the wasm32 address; this is the full word.
    pub host_compiled_ptr: std::sync::atomic::AtomicUsize,
    pub targets: std::sync::Mutex<Vec<Box<WasmCaRuntimeTarget>>>,
}

pub const WASM_CA_DISPATCH_TARGET_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaDispatchEntry, target_ptr) as u64;
pub const WASM_CA_DISPATCH_HAS_GNF2_OFS: u64 =
    std::mem::offset_of!(WasmCaDispatchEntry, has_guard_not_forced_2) as u64;
pub const WASM_CA_TARGET_FUNC_HANDLE_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, func_handle) as u64;
pub const WASM_CA_TARGET_COMPILED_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, compiled_ptr) as u64;
pub const WASM_CA_TARGET_FRAME_BYTES_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, callee_frame_bytes) as u64;
pub const WASM_CA_TARGET_DISPATCH_KEY_OFS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, dispatch_key_ofs) as u64;
pub const WASM_CA_TARGET_GCMAP_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, callee_gcmap_ptr) as u64;
pub const WASM_CA_TARGET_HOME_SLOT_BASE_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, home_slot_base) as u64;
pub const WASM_CA_TARGET_HOME_SLOTS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, home_slots) as u64;
pub const WASM_CA_TARGET_HAS_GNF2_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, has_guard_not_forced_2) as u64;
pub const WASM_CA_TARGET_MARKED_ORDINARY_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, marked_ordinary) as u64;
pub const WASM_CA_TARGET_MARKED_LABELS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, marked_labels) as u64;
pub const WASM_CA_TARGET_LABEL_REF_SLOTS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, label_ref_slots) as u64;

/// `make_and_attach_done_descrs` gives every cpu one `DoneWithThisFrame*` per
/// result kind plus one `ExitFrameWithExceptionDescrRef`, and
/// `compile_done_with_this_frame` / `compile_exit_frame_with_exception` stamp
/// that singleton on the FINISH — so every trace that finishes the same way
/// writes the same `jf_descr`, and `_call_assembler_check_descr` recognises a
/// clean callee finish by comparing against one value.
///
/// The shared identity is the cell address stored in `jf_descr`. The five
/// cells live on the owning `WasmBackend` and never move;
/// `CpuExitCells::attach_finish` replaces the `Arc` inside the cell.
/// `get_latest_descr` reads that `Arc`'s `meta_descr`.
pub const FINISH_EXIT_INDEX_VOID: u32 = 0;
pub const FINISH_EXIT_INDEX_INT: u32 = 1;
pub const FINISH_EXIT_INDEX_REF: u32 = 2;
pub const FINISH_EXIT_INDEX_FLOAT: u32 = 3;
pub const FINISH_EXIT_INDEX_EXC: u32 = 4;

/// Reserved exit index for the `done_with_this_frame_descr_*` of `ty`.
pub fn done_with_this_frame_exit_index(ty: Type) -> u32 {
    match ty {
        Type::Void => FINISH_EXIT_INDEX_VOID,
        Type::Int => FINISH_EXIT_INDEX_INT,
        Type::Ref => FINISH_EXIT_INDEX_REF,
        Type::Float => FINISH_EXIT_INDEX_FLOAT,
    }
}

/// The four `DoneWithThisFrameDescr*` classes carry the one result of their
/// kind; `ExitFrameWithExceptionDescrRef` carries the exception Ref.
fn reserved_fail_arg_types(exit_index: u32) -> Vec<Type> {
    match exit_index {
        FINISH_EXIT_INDEX_VOID => Vec::new(),
        FINISH_EXIT_INDEX_INT => vec![Type::Int],
        FINISH_EXIT_INDEX_FLOAT => vec![Type::Float],
        _ => vec![Type::Ref],
    }
}

fn reserved_finish_descr(exit_index: u32, meta_descr: Option<DescrRef>) -> Arc<WasmFailDescr> {
    Arc::new(WasmFailDescr {
        fail_index: exit_index,
        trace_id: 0,
        fail_arg_types: reserved_fail_arg_types(exit_index),
        fail_locs: Vec::new(),
        is_finish: true,
        force_args_offset: 0,
        force_tail_base: 0,
        force_prefix_slots: 0,
        value_tail_index: 0,
        force_gcmap_ptr: 0,
        bridge_cell: 0,
        fail_arg_advanced: Vec::new(),
        trace_ref_homes: 0,
        trace_label_homes: 0,
        param_dispatch: false,
        bridge_slot: std::sync::atomic::AtomicU32::new(0),
        meta_descr,
    })
}

/// Address baked into `jf_descr` / `jf_force_descr`.
///
/// The cell is the unit `LoopAsmResources` (or the finish singleton array)
/// keeps alive. `get_latest_descr` casts `jf_descr` back to this cell.
pub struct FailDescrCell {
    descr: parking_lot::Mutex<Arc<WasmFailDescr>>,
}

impl FailDescrCell {
    pub fn new(descr: Arc<WasmFailDescr>) -> Self {
        Self {
            descr: parking_lot::Mutex::new(descr),
        }
    }

    pub fn get(&self) -> Arc<WasmFailDescr> {
        Arc::clone(&self.descr.lock())
    }

    pub fn set(&self, descr: Arc<WasmFailDescr>) {
        *self.descr.lock() = descr;
    }
}

/// Read the descr an exit stored in `jf_descr` or `jf_force_descr`.
pub fn descr_at(cell: usize) -> Option<Arc<WasmFailDescr>> {
    if cell == 0 {
        return None;
    }
    Some(unsafe { &*(cell as *const FailDescrCell) }.get())
}

/// Publish the real descr into a cell `build_wasm_module` allocated.
pub fn fill_exit_cell(cell: usize, descr: Arc<WasmFailDescr>) {
    if cell == 0 {
        return;
    }
    unsafe { &*(cell as *const FailDescrCell) }.set(descr);
}

/// Allocate the cell whose address the exit stores in `jf_descr`.
///
/// `sink` is the `LoopAsmResources` the compile will push into
/// `asmmemmgr_blocks`. A null sink (a direct `build_wasm_module` test)
/// leaks the cell the same way `park_gcmap_raw` leaks a map.
pub fn alloc_exit_cell(sink: usize, fail_index: u32) -> usize {
    let descr = Arc::new(WasmFailDescr {
        fail_index,
        trace_id: 0,
        fail_arg_types: Vec::new(),
        fail_locs: Vec::new(),
        is_finish: false,
        force_args_offset: 0,
        force_tail_base: 0,
        force_prefix_slots: 0,
        value_tail_index: 0,
        force_gcmap_ptr: 0,
        bridge_cell: 0,
        fail_arg_advanced: Vec::new(),
        trace_ref_homes: 0,
        trace_label_homes: 0,
        param_dispatch: false,
        bridge_slot: std::sync::atomic::AtomicU32::new(0),
        meta_descr: None,
    });
    if sink == 0 {
        let cell = Box::new(FailDescrCell::new(descr));
        let ptr = &*cell as *const FailDescrCell as usize;
        Box::leak(cell);
        return ptr;
    }
    let resources = unsafe { &mut *(sink as *mut crate::release::LoopAsmResources) };
    resources.alloc_fail_cell(descr)
}

fn thin_descr_ptr(descr: &DescrRef) -> usize {
    Arc::as_ptr(descr) as *const () as usize
}

fn cell_addr(cell: &FailDescrCell) -> usize {
    cell as *const FailDescrCell as usize
}

/// `jf_descr` immediate captured for one reserved finish exit.
pub fn finish_cell_ptr(attached: &majit_backend::AttachedDescrPtrs, index: u32) -> usize {
    match index {
        FINISH_EXIT_INDEX_VOID => attached.done_with_this_frame_descr_void,
        FINISH_EXIT_INDEX_INT => attached.done_with_this_frame_descr_int,
        FINISH_EXIT_INDEX_REF => attached.done_with_this_frame_descr_ref,
        FINISH_EXIT_INDEX_FLOAT => attached.done_with_this_frame_descr_float,
        FINISH_EXIT_INDEX_EXC => attached.exit_frame_with_exception_descr_ref,
        _ => 0,
    }
}

/// `make_and_attach_done_descrs` pointer identity: the reserved exit index for
/// `descr` when it is one of the five singletons this cpu was handed, else
/// `None`.
/// Mirrors `AttachedDescrPtrs::is_done_with_this_frame_descr`, which is how the
/// native backends recognise the same descrs on the FINISH fast path.
///
/// Answered from the cells whose addresses `compile_loop` /
/// `compile_bridge` captured, so an index this returns always names a cell
/// carrying `descr`.
pub fn attached_finish_exit_index(
    attached: &majit_backend::AttachedDescrPtrs,
    descr: &Option<DescrRef>,
) -> Option<u32> {
    let ptr = thin_descr_ptr(descr.as_ref()?);
    let slots = [
        (
            FINISH_EXIT_INDEX_VOID,
            attached.done_with_this_frame_descr_void,
        ),
        (
            FINISH_EXIT_INDEX_INT,
            attached.done_with_this_frame_descr_int,
        ),
        (
            FINISH_EXIT_INDEX_REF,
            attached.done_with_this_frame_descr_ref,
        ),
        (
            FINISH_EXIT_INDEX_FLOAT,
            attached.done_with_this_frame_descr_float,
        ),
        (
            FINISH_EXIT_INDEX_EXC,
            attached.exit_frame_with_exception_descr_ref,
        ),
    ];
    slots.into_iter().find_map(|(index, cell)| {
        descr_at(cell)?
            .meta_descr
            .as_ref()
            .is_some_and(|attached_descr| thin_descr_ptr(attached_descr) == ptr)
            .then_some(index)
    })
}

fn propagate_wasm_descr(meta_descr: Option<DescrRef>) -> Arc<WasmFailDescr> {
    Arc::new(WasmFailDescr {
        fail_index: u32::MAX,
        trace_id: 0,
        fail_arg_types: Vec::new(),
        fail_locs: Vec::new(),
        is_finish: false,
        force_args_offset: 0,
        force_tail_base: 0,
        force_prefix_slots: 0,
        value_tail_index: 0,
        force_gcmap_ptr: 0,
        bridge_cell: 0,
        fail_arg_advanced: Vec::new(),
        trace_ref_homes: 0,
        trace_label_homes: 0,
        param_dispatch: false,
        bridge_slot: std::sync::atomic::AtomicU32::new(0),
        meta_descr,
    })
}

/// The six exit cells of one cpu.
///
/// `compile.py` `make_and_attach_done_descrs` writes
/// `cpu.done_with_this_frame_descr_*` /
/// `cpu.exit_frame_with_exception_descr_ref`, and `pyjitpl.py` sets
/// `cpu.propagate_exception_descr`. Each cell is a `Box` allocated once
/// in [`CpuExitCells::new`], so the address a module bakes as `jf_descr`
/// stays valid when [`CpuExitCells::attach_finish`] rebinds the `Arc`.
/// The owner is an `Arc` on `WasmBackend`: moving the backend does not
/// move the cells.
pub struct CpuExitCells {
    finish: [Box<FailDescrCell>; 5],
    propagate: Box<FailDescrCell>,
}

impl CpuExitCells {
    pub fn new() -> Self {
        Self {
            finish: std::array::from_fn(|index| {
                Box::new(FailDescrCell::new(reserved_finish_descr(
                    index as u32,
                    None,
                )))
            }),
            propagate: Box::new(FailDescrCell::new(propagate_wasm_descr(None))),
        }
    }

    /// Snapshot the six cell addresses for one `compile_loop` /
    /// `compile_bridge`. Finish cells are always published: `_call_assembler`
    /// compares `jf_descr` against the reserved cell even before a singleton
    /// is bound. `propagate_exception_descr` is `0` until
    /// [`Self::attach_propagate`], the unattached answer
    /// `AttachedDescrPtrs` uses on the native cpus.
    pub fn descr_ptrs(&self) -> majit_backend::AttachedDescrPtrs {
        let propagate = cell_addr(&self.propagate);
        majit_backend::AttachedDescrPtrs {
            done_with_this_frame_descr_void: cell_addr(&self.finish[0]),
            done_with_this_frame_descr_int: cell_addr(&self.finish[1]),
            done_with_this_frame_descr_ref: cell_addr(&self.finish[2]),
            done_with_this_frame_descr_float: cell_addr(&self.finish[3]),
            exit_frame_with_exception_descr_ref: cell_addr(&self.finish[4]),
            propagate_exception_descr: descr_at(propagate)
                .is_some_and(|descr| descr.meta_descr.is_some())
                .then_some(propagate)
                .unwrap_or(0),
        }
    }

    /// `make_and_attach_done_descrs`. Rebinds the `Arc` inside the existing
    /// cell, so a module that already baked the address still names this descr.
    pub fn attach_finish(&self, exit_index: u32, descr: DescrRef) {
        let cell = cell_addr(&self.finish[exit_index as usize]);
        fill_exit_cell(cell, reserved_finish_descr(exit_index, Some(descr)));
    }

    /// `pyjitpl.py` `self.cpu.propagate_exception_descr = exc_descr`.
    ///
    /// `_build_propagate_exception_path` writes this cell into `jf_descr`.
    /// `get_latest_descr_arc` recovers the Arc from `WasmFailDescr.meta_descr`.
    /// The metainterp reader runs `PropagateExceptionDescr.handle_fail`; this
    /// cell is not a finish exit.
    pub fn attach_propagate(&self, descr: DescrRef) {
        fill_exit_cell(
            cell_addr(&self.propagate),
            propagate_wasm_descr(Some(descr)),
        );
    }
}

impl WasmCaDispatchEntry {
    pub fn pending() -> Self {
        Self {
            target_ptr: AtomicU32::new(0),
            has_guard_not_forced_2: AtomicU32::new(0),
            host_compiled_ptr: std::sync::atomic::AtomicUsize::new(0),
            targets: std::sync::Mutex::new(Vec::new()),
        }
    }
}

/// The token's `_ll_function_addr` stand-in. `0` is "not allocated".
pub fn ca_entry(addr: usize) -> Option<&'static WasmCaDispatchEntry> {
    if addr == 0 {
        None
    } else {
        Some(unsafe { &*(addr as *const WasmCaDispatchEntry) })
    }
}

/// One indirect cell per looptoken. The `Box` lives in `LoopAsmResources`
/// (`asmmemmgr_blocks`); callers bake its address.
pub fn ensure_ca_cell(token: &majit_backend::JitCellToken) -> &'static WasmCaDispatchEntry {
    if let Some(entry) = ca_entry(token.ll_function_addr()) {
        return entry;
    }
    let entry = Box::new(WasmCaDispatchEntry::pending());
    let addr = &*entry as *const WasmCaDispatchEntry as usize;
    if let Some(clt) = token.compiled_loop_token() {
        let mut blocks = clt.asmmemmgr_blocks.lock();
        let resources = if let Some(existing) = blocks
            .iter_mut()
            .rev()
            .find_map(|block| block.downcast_mut::<crate::release::LoopAsmResources>())
        {
            existing
        } else {
            blocks.push(Box::new(crate::release::LoopAsmResources::default()));
            blocks
                .last_mut()
                .and_then(|block| block.downcast_mut::<crate::release::LoopAsmResources>())
                .expect("just-pushed LoopAsmResources")
        };
        resources.ca_entry = Some(entry);
    } else {
        Box::leak(entry);
    }
    token.set_ll_function_addr(addr);
    unsafe { &*(addr as *const WasmCaDispatchEntry) }
}

/// Publish one immutable snapshot and release-store its address.
pub fn ca_publish(
    entry: &WasmCaDispatchEntry,
    func_handle: u32,
    compiled_ptr: u64,
    callee_frame_bytes: u32,
    dispatch_key_ofs: u32,
    callee_gcmap_ptr: i64,
    home_slot_base: u32,
    home_slots: u32,
    has_guard_not_forced_2: u32,
    marked_ordinary: u32,
    marked_labels: u32,
    label_ref_slots: u32,
) {
    let has_guard_not_forced_2 = if has_guard_not_forced_2 != 0
        || entry.has_guard_not_forced_2.load(Ordering::Acquire) != 0
    {
        1
    } else {
        0
    };
    if has_guard_not_forced_2 != 0 {
        entry.has_guard_not_forced_2.store(1, Ordering::Release);
    }
    let mut targets = entry.targets.lock().unwrap_or_else(|e| e.into_inner());
    if targets.last().is_some_and(|current| {
        current.func_handle == func_handle
            && current.compiled_ptr == compiled_ptr as u32
            && current.callee_frame_bytes == callee_frame_bytes
            && current.dispatch_key_ofs == dispatch_key_ofs
            && current.callee_gcmap_ptr == callee_gcmap_ptr
            && current.home_slot_base == home_slot_base
            && current.home_slots == home_slots
            && current.has_guard_not_forced_2 == has_guard_not_forced_2
            && current.marked_ordinary == marked_ordinary
            && current.marked_labels == marked_labels
            && current.label_ref_slots == label_ref_slots
    }) {
        return;
    }
    let target = Box::new(WasmCaRuntimeTarget {
        func_handle,
        compiled_ptr: compiled_ptr as u32,
        callee_frame_bytes,
        dispatch_key_ofs,
        callee_gcmap_ptr,
        home_slot_base,
        home_slots,
        has_guard_not_forced_2,
        marked_ordinary,
        marked_labels,
        label_ref_slots,
    });
    let target_ptr = (&*target as *const WasmCaRuntimeTarget as usize) as u32;
    targets.push(target);
    entry
        .host_compiled_ptr
        .store(compiled_ptr as usize, Ordering::Release);
    entry.target_ptr.store(target_ptr, Ordering::Release);
}

pub fn ca_mark_entry(entry: &WasmCaDispatchEntry) {
    entry.has_guard_not_forced_2.store(1, Ordering::Release);
}

/// Raise the cell flag on every entry whose snapshots still name `compiled_ptr`.
pub fn mark_cells_holding(cells: &[&WasmCaDispatchEntry], compiled_ptr: u32) {
    if compiled_ptr == 0 {
        return;
    }
    for entry in cells {
        let holds = entry
            .targets
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|target| target.compiled_ptr == compiled_ptr);
        if holds {
            ca_mark_entry(entry);
        }
    }
}

fn token_ca_cells(token: &majit_backend::JitCellToken) -> Vec<&'static WasmCaDispatchEntry> {
    let mut cells = Vec::new();
    if let Some(entry) = ca_entry(token.ll_function_addr()) {
        cells.push(entry);
    }
    let Some(clt) = token.compiled_loop_token() else {
        return cells;
    };
    let chain = clt.looptokens_redirected_to.lock().clone();
    for weak in chain {
        let Some(old) = weak.upgrade() else {
            continue;
        };
        let Some(alias) = old.upgrade_loop_token() else {
            continue;
        };
        if let Some(entry) = ca_entry(alias.ll_function_addr()) {
            cells.push(entry);
        }
    }
    cells
}

/// Raise GNF2 on this token's cell and on redirect aliases that still
/// name its compiled loop. `update_frame_info` records those aliases.
pub fn mark_gnf2_token(token: &majit_backend::JitCellToken) {
    let cells = token_ca_cells(token);
    let compiled_ptr = cells.first().and_then(|entry| {
        entry
            .targets
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .map(|target| target.compiled_ptr)
    });
    if let Some(compiled_ptr) = compiled_ptr.filter(|&ptr| ptr != 0) {
        let matched = cells.iter().any(|entry| {
            entry
                .targets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .any(|target| target.compiled_ptr == compiled_ptr)
        });
        mark_cells_holding(&cells, compiled_ptr);
        if !matched {
            if let Some(entry) = cells.first() {
                ca_mark_entry(entry);
            }
        }
    } else if let Some(entry) = cells.first() {
        ca_mark_entry(entry);
    }
}

pub fn target_from_token(token: &majit_backend::JitCellToken) -> Option<CallAssemblerTarget> {
    let entry = ca_entry(token.ll_function_addr())?;
    let targets = entry.targets.lock().unwrap_or_else(|e| e.into_inner());
    let snap = targets.last()?;
    let host_ptr = entry.host_compiled_ptr.load(Ordering::Acquire);
    let input_types = if host_ptr != 0 {
        unsafe { (host_ptr as *const CompiledWasmLoop).as_ref() }
            .map(|loop_| loop_.input_types.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let live_flag = entry.has_guard_not_forced_2.load(Ordering::Acquire);
    Some(CallAssemblerTarget {
        token_number: token.number,
        dispatch_entry: token.ll_function_addr() as u32,
        func_handle: snap.func_handle,
        input_types,
        dispatch_key_ofs: snap.dispatch_key_ofs as u64,
        callee_frame_bytes: snap.callee_frame_bytes,
        callee_gcmap_ptr: snap.callee_gcmap_ptr,
        compiled_ptr: host_ptr as u64,
        home_slot_base: snap.home_slot_base,
        home_slots: snap.home_slots,
        has_guard_not_forced_2: live_flag.max(snap.has_guard_not_forced_2),
        marked_ordinary: snap.marked_ordinary,
        marked_labels: snap.marked_labels,
        label_ref_slots: snap.label_ref_slots,
    })
}

pub fn publish_token_target(token: &majit_backend::JitCellToken, target: &CallAssemblerTarget) {
    let entry = ensure_ca_cell(token);
    ca_publish(
        entry,
        target.func_handle,
        target.compiled_ptr,
        target.callee_frame_bytes,
        target.dispatch_key_ofs as u32,
        target.callee_gcmap_ptr,
        target.home_slot_base,
        target.home_slots,
        target.has_guard_not_forced_2,
        target.marked_ordinary,
        target.marked_labels,
        target.label_ref_slots,
    );
}

/// Write a bridge table slot into the guard cell `patch_jump_for_descr` names.
pub fn write_guard_cell(cell_addr: u32, slot: u32) {
    #[cfg(target_arch = "wasm32")]
    if cell_addr != 0 {
        unsafe { core::ptr::write(cell_addr as *mut u32, slot) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (cell_addr, slot);
}

/// Serializes tests that mutate cpu-global tables (GC box, label targets).
/// The wasm host never interleaves those; cargo's parallel unit-test runner
/// does. Held by every lib test in this crate.
#[cfg(test)]
pub static FAIL_DESCR_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Acquires [`FAIL_DESCR_TEST_LOCK`], then installs a fresh cpu.
///
/// `BaseBackendTest.setup_method` does `self.cpu = self.get_cpu()` and
/// sets `done_with_this_frame_descr_* = None`. Those descrs live on the
/// `WasmBackend` under test. Drop is `AsmMemoryManager._delete`: the cpu
/// dies with the frame (`cpu.gc_ll_descr` / shadowstack). Struct fields
/// drop in declaration order, so `_cleanup` is declared first and runs
/// while `_lock` is still held.
#[cfg(test)]
pub fn lock_cpu() -> CpuTestGuard {
    let guard = CpuTestGuard {
        _cleanup: CpuTestCleanup,
        _lock: FAIL_DESCR_TEST_LOCK.lock(),
    };
    reset_cpu_for_tests();
    guard
}

/// Uninstall this thread's `gc_ll_descr`. `BaseBackendTest.setup_method` /
/// `get_cpu`. Finish and propagate cells belong to the `WasmBackend`.
#[cfg(test)]
fn reset_cpu_for_tests() {
    crate::gc_box::clear();
    majit_gc::shadow_stack::clear();
    crate::jit_exc_clear();
    crate::set_wasm_jitframe_tid(0);
}

#[cfg(test)]
struct CpuTestCleanup;

#[cfg(test)]
impl Drop for CpuTestCleanup {
    fn drop(&mut self) {
        // `AsmMemoryManager._delete` / cpu teardown, still under the lock.
        reset_cpu_for_tests();
    }
}

#[cfg(test)]
pub struct CpuTestGuard {
    _cleanup: CpuTestCleanup,
    _lock: parking_lot::MutexGuard<'static, ()>,
}

/// `closing_jump` reads `_ll_loop_code` off the JUMP's descr.
///
/// SAFETY: a non-zero word is `&*Box<LabelTarget>` owned by the publishing
/// loop's `LoopAsmResources`. `free_loop_and_bridges` drops that block after
/// `LoopAsmResources::drop` clears the word when it still names this box.
/// `LabelTarget` is `Copy`.
pub fn label_target(descr: &majit_ir::DescrRef) -> Option<LabelTarget> {
    let addr = descr.as_loop_target_descr()?.ll_loop_code();
    if addr == 0 {
        return None;
    }
    Some(unsafe { *(addr as *const LabelTarget) })
}

/// `fixup_target_tokens`: write this label's entry onto its `TargetToken`.
/// The box stays in `resources` for the emission's life. A later compile
/// overwrites `_ll_loop_code` with its own box's address.
pub fn publish_label_target(
    resources: &mut crate::release::LoopAsmResources,
    descr: &majit_ir::DescrRef,
    target: LabelTarget,
) {
    let Some(loop_target) = descr.as_loop_target_descr() else {
        return;
    };
    let boxed = Box::new(target);
    let addr = &*boxed as *const LabelTarget as usize;
    loop_target.set_ll_loop_code(addr);
    resources.label_targets.push((descr.clone(), boxed));
}

/// Retract a published label if it still names `func_handle`.
/// Same handle guard as [`CompiledWasmLoop::drop`]: a later publish that
/// re-stamped the same descr onto a different slot keeps the replacement.
/// The box stays with its `LoopAsmResources` until that emission is freed.
pub fn retract_label_target_if_handle(descr: &majit_ir::DescrRef, func_handle: u32) {
    if func_handle == 0 {
        return;
    }
    let Some(loop_target) = descr.as_loop_target_descr() else {
        return;
    };
    let addr = loop_target.ll_loop_code();
    if addr == 0 {
        return;
    }
    // SAFETY: see [`label_target`].
    let current = unsafe { &*(addr as *const LabelTarget) };
    if current.func_handle == func_handle {
        loop_target.set_ll_loop_code(0);
        crate::BRIDGE_DIAG[22].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Two guest words a running loop loads on its back-edge.
/// `generation` at offset 0, `slot` at offset 4.
#[repr(C)]
pub struct ResumeEntry {
    pub generation: std::sync::atomic::AtomicU32,
    pub slot: std::sync::atomic::AtomicU32,
}

impl ResumeEntry {
    pub fn new() -> Self {
        Self {
            generation: std::sync::atomic::AtomicU32::new(1),
            slot: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn addr(&self) -> u32 {
        self as *const Self as usize as u32
    }
}

/// Compiled wasm loop metadata, stored in `JitCellToken.compiled`.
pub struct CompiledWasmLoop {
    /// Owning `JitCellToken` number, used to retract this loop's
    /// CALL_ASSEMBLER target metadata on drop.
    pub token_number: u64,
    pub trace_id: u64,
    pub input_types: Vec<Type>,
    /// Shared-table slot of the materialized wasm function.  Straight-line
    /// function-entry traces may keep this at zero until their first actual
    /// execution: an invalidated trace that never reaches `execute_token`
    /// must not pay the host Wasmtime compilation cost.
    pub(crate) func_handle: Cell<u32>,
    /// `{generation, slot}` the running loop loads on each back-edge.
    /// The allocation stays put across in-place re-emission; the module
    /// bakes its address. `slot` is the table index `replace_module` keeps.
    pub(crate) resume_entry: Box<ResumeEntry>,
    /// Encoded module retained until lazy host materialization.  This is
    /// backend assembler state, not metainterpreter state: the optimized trace
    /// and all per-token descriptors have already been installed exactly as
    /// in the eager path.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) pending_wasm_bytes: RefCell<Option<Vec<u8>>>,
    /// The owning token metadata receives the guard-descriptor tracer only
    /// after the host accepts this module. Deferred root traces keep this
    /// handle until `materialize_func_handle` crosses that acceptance point.
    pub(crate) compiled_loop_token: Arc<majit_backend::CompiledLoopToken>,
    /// Serializes the eager and lazy acceptance paths. The wasm runtime is
    /// single-threaded, matching the surrounding `Cell`/`RefCell` fields.
    pub(crate) descrs_registered: Cell<bool>,
    /// This loop's own guard/finish exit descriptors (positions `[0,
    /// num_guard_cells)`, per-trace order), followed by the descr slices of
    /// every chained bridge `compile_bridge` appended (positional bookkeeping
    /// for `bridge_descr_ranges` — layouts and jitcounter hashes). An exit
    /// resolves through `jf_descr` (the cell), not by indexing this vec.
    /// `RefCell` because the
    /// append happens through the shared `&JitCellToken` the bridge attaches
    /// to; the wasm host is single-threaded so no cross-thread access occurs.
    pub fail_descrs: RefCell<Vec<Arc<WasmFailDescr>>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
    /// Number of Ref-typed values given a home slot in the frame's Ref-home
    /// region (`codegen::HOME_SLOT_BASE`). `execute_token` sizes the host
    /// frame to include this region and registers each home slot as a GC root.
    pub num_ref_homes: Cell<usize>,
    /// LABEL-capture homes this loop actually initialized. Frozen geometry
    /// may reserve more; a later bridge's published map must still cover
    /// these so a keyed tail-call cannot drop them. `Cell` so `reemit_loop`
    /// can raise it when a merge publishes a wider capture tail.
    pub used_label_homes: Cell<usize>,
    /// Geometry frozen when this token was first compiled. Every bridge
    /// chained onto it is emitted against this exact layout.
    pub frame: crate::codegen::FrameGeometry,
    /// Per-loop `jf_gcmap` for the Ref-home region.  Like RPython's assembler
    /// gcmap allocation, this remains valid after `execute_token` returns: a
    /// virtualizable token can keep that JITFRAME alive and force it later.
    /// `Cell` so `reemit_loop` can replace it when a merge widens RefHomes.
    pub home_gcmap_ptr: Cell<usize>,
    /// Base address (shared linear memory) of this loop's per-guard bridge-slot
    /// cell array — one i32 per `fail_index`, `0` = no bridge. The trace's
    /// epilogue reads `cells[fail_index]` and `compile_bridge` writes a bridge's
    /// table slot here. `0` when the trace has no in-module dispatch (native, or
    /// a guardless / straight-line trace).
    pub bridge_cells_base: Cell<u32>,
    /// Cell array of a retained pre-growth owner module. When the LABEL
    /// tail grows, the replacement is installed at a new table slot and
    /// this stays the array the old module still reads. `0` until that
    /// split. `compile_bridge` writes both aliases.
    pub retained_owner_cells_base: Cell<u32>,
    /// Byte length of the module this loop was last emitted as, own ops and
    /// every merged region together. A merge re-emits the whole owner, so this
    /// is what the next merge charges cranelift, and
    /// `inline_trip_threshold_for` turns it into the entry count that merge has
    /// to earn before it is taken.
    pub(crate) module_bytes: Cell<u32>,
    /// Number of cells in the `bridge_cells_base` array = this loop's own guard
    /// count at compile time. A bridge attaches only to one of these original
    /// guards (`source_fail_index < num_guard_cells`); descrs appended past this
    /// range belong to already-chained bridges and have no cell of their own.
    pub num_guard_cells: Cell<usize>,
    /// True when this is a peeled loop (`codegen::is_resumable_peeled`) — there
    /// is real work (a preamble = the unrolled first iteration) before the last
    /// `LABEL`, single- or multi-label. Such a loop carries the resume-at-LABEL
    /// entry `br_table` (key = label ordinal + 1) so a loop-closing bridge can
    /// re-enter at any of its labels. A loop-closing bridge re-enters through
    /// the loop's table slot (the function entry); for a peeled loop,
    /// re-running the preamble against mid-loop state would never advance the
    /// induction variable — an infinite loop. `compile_bridge` therefore
    /// declines a loop-closing bridge UNLESS its JUMP's
    /// target label resolves to a published, resumable `LabelTarget`.
    pub has_preamble: bool,
    /// Descr identity (`Arc::as_ptr`) of each `LABEL`, in ordinal order; `0`
    /// for a descr-less label. `compile_bridge` resolves a closing JUMP's
    /// target label by matching its descr identity against this list — a JUMP
    /// whose descr is not here targets ANOTHER trace's label (e.g. a sibling
    /// retrace specialization, whose start label carries the same stamped
    /// ordinal) and must not be chained into this loop.
    pub label_descrs: Vec<usize>,
    /// Per-guard (indexed by this loop's own `fail_index`), per-fail-arg:
    /// whether the value was produced by induction-advancing arithmetic after
    /// the loop-header label — fresh in the failing iteration. Consulted by
    /// `compile_bridge`'s livelock check: a loop-closing bridge that JUMPs
    /// such a fail arg verbatim still advances the chained cycle.
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// Number of fail arguments for every guard/finish exit in this trace.
    pub guard_fail_arg_counts: Vec<usize>,
    /// Whether this module transfers a compiled bridge's fail arguments as
    /// wasm call parameters instead of reloading their positional frame slots.
    pub bridge_param_dispatch: bool,
    /// `(source_trace_id, source_fail_index, start, count)` ranges into
    /// `fail_descrs` for each chained bridge `compile_bridge` appended (lib.rs
    /// extend site). Lets `compiled_bridge_fail_descr_layouts` map a source
    /// guard back to its bridge's
    /// appended descr slice — the wasm analog of dynasm's
    /// `lookup_bridge_addr` (runner.rs). Keyed by BOTH the source guard's
    /// owning trace and its per-trace fail index: with nested chaining, the
    /// loop's guard `k` and a chained bridge's guard `k` are distinct sources.
    /// Recorded in lockstep with the `extend`, inside the same `borrow_mut`
    /// critical section.
    pub bridge_descr_ranges: RefCell<Vec<(u64, u32, usize, usize)>>,
    /// Post-intern module inputs retained for a loop re-emission. Entry
    /// bridges store `None` because they tail-call another loop.
    pub reemit: RefCell<Option<crate::codegen::ModuleBuildInputs>>,
    /// Label descrs this loop published. `Drop` retracts a row that still
    /// names this loop's table slot (`TargetToken._ll_loop_code` cleared
    /// when the assembled code is freed).
    pub published_label_descrs: Vec<majit_ir::DescrRef>,
    /// `(label descr, table slot)` for every label published by a bridge
    /// chained onto this loop. The bridge module lives as long as its source
    /// loop, so `Drop` retracts the entries that still name that bridge's
    /// slot.
    pub bridge_owned_label_targets: RefCell<Vec<(majit_ir::DescrRef, u32)>>,
    /// Set when `compile_bridge` accepts a self-recursive `CallAssemblerR`
    /// bridge (`PYRE_WASM_CA`) for this loop. While set, `compile_bridge`
    /// declines chaining any FURTHER bridge into this recursion (the guard
    /// falls back to host round-trips): a chained bridge deopting inside the
    /// CA recursion trips a resume seam that reads a clobbered class — see
    /// the decline site for the failing suite shapes.
    pub ca_active: Cell<bool>,
    /// A guard reached through this loop as a wasm CALL_ASSEMBLER callee was
    /// structurally declined by `compile_bridge`.  Admission refuses this
    /// target, because entering it from compiled wasm would only blackhole.
    pub ca_terminal_declined: Cell<bool>,
    /// Compiled callers that baked this loop as their CALL_ASSEMBLER target.
    /// A terminal callee decline invalidates them for a no-CA retrace.
    /// Weak, so a retired caller artifact is not kept alive by this list.
    pub ca_callers: RefCell<Vec<std::sync::Weak<std::sync::atomic::AtomicBool>>>,
}

// Compiled loop metadata is transferred through the token's `Any + Send`
// holder, but all access to its IR snapshot and cell arrays is confined to the
// single wasm execution thread. The contained `RefCell`s enforce that runtime
// ownership model; moving the holder does not permit concurrent access.
unsafe impl Send for CompiledWasmLoop {}

impl CompiledWasmLoop {
    pub fn eager_func_handle(&self) -> u32 {
        self.func_handle.get()
    }

    /// Publish the descriptors only for a module the host can execute.
    pub(crate) fn register_descrs_once(&self) {
        if self.descrs_registered.replace(true) {
            return;
        }
        let descrs = self.fail_descrs.borrow();
        let meta: Vec<DescrRef> = descrs
            .iter()
            .filter_map(|descr| descr.meta_descr.clone())
            .collect();
        let tracer: Arc<dyn std::any::Any + Send + Sync> = Arc::new(meta);
        self.compiled_loop_token
            .asmmemmgr_gcreftracers
            .lock()
            .push(tracer);
    }

    /// Materialize a lazily-installed root trace.  The wasm host is
    /// single-threaded, matching the RefCell/Cell ownership used throughout
    /// this structure, so one trace can only cross this gate once.
    pub fn materialize_func_handle(&self) -> Result<u32, majit_backend::BackendError> {
        let current = self.func_handle.get();
        if current != 0 {
            return Ok(current);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let pending = self.pending_wasm_bytes.borrow();
            let Some(bytes) = pending.as_deref() else {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm trace has neither a function handle nor pending module bytes".into(),
                ));
            };
            let handle = crate::glue::compile_module_cached(bytes);
            if handle == 0 {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm host rejected the lazily compiled trace module".into(),
                ));
            }
            self.register_descrs_once();
            self.func_handle.set(handle);
            if self
                .resume_entry
                .slot
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
            {
                self.resume_entry
                    .slot
                    .store(handle, std::sync::atomic::Ordering::Relaxed);
            }
            let mut blocks = self.compiled_loop_token.asmmemmgr_blocks.lock();
            if let Some(resources) = blocks
                .last_mut()
                .and_then(|block| block.downcast_mut::<crate::release::LoopAsmResources>())
            {
                resources.table_slots.push(handle);
            } else {
                let mut resources = crate::release::LoopAsmResources::default();
                resources.table_slots.push(handle);
                blocks.push(Box::new(resources));
            }
            drop(pending);
            self.pending_wasm_bytes.borrow_mut().take();
            Ok(handle)
        }
    }
}

impl Drop for CompiledWasmLoop {
    fn drop(&mut self) {
        // Remove every token alias still targeting this module, including a
        // redirect source. A source redirected to a newer module survives an
        // old-loop drop because its dispatch `compiled_ptr` no longer matches.
        // Label rows and table slots are owned by `LoopAsmResources` in
        // `asmmemmgr_blocks`, dropped by `free_loop_and_bridges`.
        // Retract this loop's published label targets so a later bridge
        // cannot chain into a dropped loop's stale table slot. Guarded by
        // `func_handle`: a recompile that re-stamped the same descr onto its
        // replacement loop has already overwritten the entry, which must
        // survive the old loop's drop.
        let handle = self.func_handle.get();
        for descr in &self.published_label_descrs {
            retract_label_target_if_handle(descr, handle);
        }
        for (descr, slot) in self.bridge_owned_label_targets.get_mut().drain(..) {
            retract_label_target_if_handle(&descr, slot);
        }
    }
}
