//! Per-loop wasm resources owned by `CompiledLoopToken.asmmemmgr_blocks`.
//!
//! `llmodel.py` `AbstractLLCPU.free_loop_and_bridges` frees the loop by
//! dropping `asmmemmgr_blocks` and `asmmemmgr_gcreftracers`. A target another
//! token still jumps to stays alive through `JitCellToken.record_jump_to`
//! (`history.py`), so this token is not freed while that caller exists.

use std::any::Any;
use std::cell::{Cell, RefCell};

use majit_backend::JitCellToken;

/// One deferred inline merge and the entry counter its bridge increments.
///
/// The box lives in [`LoopAsmResources::pending_inlines`] and is freed with
/// that block (`free_loop_and_bridges`). The probe bakes this box's address
/// and the address of `counter`.
pub struct PendingInlineSlot {
    pub(crate) entry: RefCell<Option<super::PendingInline>>,
    pub(crate) counter: Cell<u64>,
}

/// Compile-time maps, table slots, label rows, fail indices, and bridge
/// cells of one emission. Pushed into `asmmemmgr_blocks`; `Drop` runs when
/// `free_loop_and_bridges` clears that list.
#[derive(Default)]
pub struct LoopAsmResources {
    pub gcmaps: Vec<Box<[usize]>>,
    /// The cpu's exit cells this emission baked as `jf_descr` immediates.
    /// Held so the cells outlive the `WasmBackend` while the code can still
    /// run (dynasm keeps the same `CpuDescrHandle` clone on its compiled
    /// loop).
    pub exit_cells: Option<std::sync::Arc<crate::failguard::CpuExitCells>>,
    /// Real `__indirect_function_table` pair bases. `0` is not a host slot.
    pub table_slots: Vec<u32>,
    /// `LabelTarget` boxes this emission published. `ll_loop_code` is the
    /// address of the box (`assembler.py` `fixup_target_tokens` writes
    /// `TargetToken._ll_loop_code`). A later compile overwrites the word
    /// with its own box; `Drop` clears the word only when it still names
    /// this box.
    pub label_targets: Vec<(majit_ir::DescrRef, Box<crate::failguard::LabelTarget>)>,
    /// One [`crate::failguard::FailDescrCell`] per guard exit. The address is
    /// what the exit stores in `jf_descr` (`get_latest_descr`). The cell
    /// outlives every module that can still leave through that exit because
    /// this block is `asmmemmgr_blocks`.
    pub fail_cells: Vec<Box<crate::failguard::FailDescrCell>>,
    /// CALL_ASSEMBLER indirect cell. Its address is `JitCellToken._ll_function_addr`.
    pub ca_entry: Option<Box<crate::failguard::WasmCaDispatchEntry>>,
    pub bridge_cells: Vec<Box<[u32]>>,
    /// `[descr_cell, gcmap]` pairs the exit loads. The address is baked
    /// into the module; the allocation does not move.
    pub exit_table: Option<Box<[usize]>>,
    /// Home-offset records the outlined guard exit reads. One box per
    /// module that admitted the helper. The address is baked into the
    /// trace; an active wasm data segment is not used, because it would
    /// be mapped into the guest heap.
    pub guard_exit_blobs: Vec<Box<[u32]>>,
    /// Deferred merges whose out-of-line bridges this emission compiled.
    /// Pushed with the bridge onto the original loop token
    /// (`push_resources`; `assembler.py` keeps bridge blocks on
    /// `original_loop_token.compiled_loop_token.asmmemmgr_blocks`).
    pub pending_inlines: Vec<Box<PendingInlineSlot>>,
}

// Loop asm resources are transferred through the token's `Any + Send`
// holder, but all access to its IR snapshot and cell arrays is confined to the
// single wasm execution thread. The contained `RefCell`s enforce that runtime
// ownership model; moving the holder does not permit concurrent access.
unsafe impl Send for LoopAsmResources {}

impl LoopAsmResources {
    pub fn park_gcmap(&mut self, map: Box<[usize]>) -> usize {
        let ptr = map.as_ptr() as usize;
        self.gcmaps.push(map);
        ptr
    }

    /// Stable address baked into `jf_descr`. The `Box` does not move for
    /// the rest of this block's life.
    pub fn alloc_fail_cell(
        &mut self,
        descr: std::sync::Arc<crate::failguard::WasmFailDescr>,
    ) -> usize {
        let cell = Box::new(crate::failguard::FailDescrCell::new(descr));
        let ptr = crate::failguard::FailDescrCell::thin_ptr(&cell);
        self.fail_cells.push(cell);
        ptr
    }

    /// `count` guard exits, two `usize` words each. Returns the guest
    /// address baked into the module, or 0 when there is nothing to name.
    pub fn alloc_exit_table(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let table = vec![0usize; count * 2].into_boxed_slice();
        let ptr = table.as_ptr() as usize;
        self.exit_table = Some(table);
        ptr
    }

    /// Stable address of a new pending-inline slot, and the address of its
    /// entry counter. Both are baked into the bridge module.
    pub(crate) fn alloc_pending_inline(&mut self, inline: super::PendingInline) -> (usize, u32) {
        let slot = Box::new(PendingInlineSlot {
            entry: RefCell::new(Some(inline)),
            counter: Cell::new(0),
        });
        let slot_addr = &*slot as *const PendingInlineSlot as usize;
        let counter_addr = &slot.counter as *const Cell<u64> as usize as u32;
        self.pending_inlines.push(slot);
        (slot_addr, counter_addr)
    }

    pub fn write_exit_slot(&mut self, index: usize, descr_cell: usize, gcmap: usize) {
        let Some(table) = self.exit_table.as_mut() else {
            return;
        };
        let base = index * 2;
        if base + 1 < table.len() {
            table[base] = descr_cell;
            table[base + 1] = gcmap;
        }
    }
}

impl Drop for LoopAsmResources {
    fn drop(&mut self) {
        for slot in self.table_slots.drain(..) {
            if slot != 0 {
                #[cfg(target_arch = "wasm32")]
                crate::glue::free(slot);
                #[cfg(not(target_arch = "wasm32"))]
                let _ = slot;
            }
        }
        for (descr, boxed) in self.label_targets.drain(..) {
            let addr = &*boxed as *const crate::failguard::LabelTarget as usize;
            if let Some(loop_target) = descr.as_loop_target_descr()
                && loop_target.ll_loop_code() == addr
            {
                loop_target.set_ll_loop_code(0);
            }
        }
    }
}

/// `codegen` publishes a home gcmap while building a module. `sink` is the
/// `LoopAsmResources` the caller will push into `asmmemmgr_blocks`. A null
/// sink is a direct `build_wasm_module` test: the map is leaked the same way
/// `allocate_gcmap` leaks via `Box::into_raw`.
pub fn park_gcmap_raw(sink: usize, map: Box<[usize]>) -> usize {
    if sink == 0 {
        return Box::into_raw(map) as *mut usize as usize;
    }
    // SAFETY: `sink` is `&mut LoopAsmResources` held by the compile that
    // called `build_wasm_module`, and that borrow lasts until the build
    // returns.
    let resources = unsafe { &mut *(sink as *mut LoopAsmResources) };
    resources.park_gcmap(map)
}

/// Park the outlined guard-exit blob and return the guest address baked
/// into the module. A null sink leaks, the same way [`park_gcmap_raw`] does
/// for a direct codegen test.
pub fn park_guard_exit_blob(sink: usize, words: Box<[u32]>) -> u32 {
    let addr = words.as_ptr() as usize as u32;
    if sink == 0 {
        Box::leak(words);
        return addr;
    }
    // SAFETY: same borrow as [`park_gcmap_raw`].
    let resources = unsafe { &mut *(sink as *mut LoopAsmResources) };
    resources.guard_exit_blobs.push(words);
    addr
}

pub fn push_resources(token: &JitCellToken, resources: LoopAsmResources) {
    let Some(clt) = token.compiled_loop_token() else {
        return;
    };
    let block: Box<dyn Any + Send> = Box::new(resources);
    clt.asmmemmgr_blocks.lock().push(block);
}
