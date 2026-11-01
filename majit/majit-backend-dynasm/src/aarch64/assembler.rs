/// aarch64/assembler.py: AssemblerARM64 — aarch64 JIT code generation backend.
///
/// Generates machine code from IR operations via dynasm-rs.
/// RPython: AssemblerARM64(ResOpAssembler) in aarch64/assembler.py.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use crate::regloc::ebp_loc_pat;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::sync::Arc;

// aarch64/assembler.py parity: aarch64-only backend.
/// `codebuf.py MachineCodeBlockWrapper`: code is assembled into a plain byte
/// vector (every relocation is PC-relative) and copied into the arena block
/// at `materialize`; no per-trace executable mapping.
pub(crate) type Assembler = dynasmrt::VecAssembler<dynasmrt::aarch64::Aarch64Relocation>;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};

use majit_backend::{
    AsmMemoryBlock, AsmMemoryManager, BackendError, JitCellToken, MachineDataBlockWrapper,
};
use majit_ir::{
    FailDescr, FailDescrStore, InputArgRc, Op, OpCode, OpRc, OpRef, OpTypeIndex, TargetArgLoc, Type,
};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::jitframe::{
    FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FRAME_OFS, JF_GCMAP_OFS,
    JF_GUARD_EXC_OFS,
};
use crate::jump::RegallocMoves;
use crate::regalloc::{RegAlloc, RegAllocAssembler, park_float_const, visit_boxes_with_frame_loc};
use crate::regloc::{Loc, RegLoc};
use crate::runner::GuardGcTypeInfo;

const AARCH64_GEN_REGS: [crate::regloc::RegLoc; 16] = crate::aarch64::registers::ALL_REGS;

const AARCH64_FLOAT_REGS: [crate::regloc::RegLoc; 8] = crate::aarch64::registers::ALL_VFP_REGS;

/// A LABEL publishes an in-buffer offset and only becomes an absolute address
/// in `fixup_target_tokens`; 0 means its target was never compiled.  The first
/// page is never mapped, so any value below it is one of those two and can only
/// be a wild branch.  Dynasm bakes the immediate at codegen, so it cannot be
/// repaired later.
const MIN_RELOCATED_JUMP_TARGET: usize = 4096;

/// Bytes the trace entry frame reserves. Layout after `_call_header`:
/// `[sp,#0]` x29/x30, `[sp,#16]` x19/x20 (`r.callee_saved_registers`),
/// `[sp,#32]` unused 16-byte hole, `[sp,#48]` thread-local base,
/// `[sp,#56]` unused 8-byte pad. The holes held x21/x22 and d8 when
/// loops pinned constants; they stay so `SAVED_THREADLOCAL_OFS` and every
/// unwind keep the same 64-byte SP delta. The prologue, the stack-overflow
/// early return and the epilogue must all agree on this size.
const CALL_FRAME_SIZE: u32 = 64;

/// Frame offset of the thread-local base a compiled entry receives in `x1`.
///
/// `aarch64/assembler.py` names this `saved_threadlocal_addr` and
/// spills the incoming `x1` there once per entry, so any point inside the
/// trace can reach `pypy_threadlocal_s` without the caller keeping a register
/// live across the body. The upstream constant is `3 * WORD` because its frame
/// reserves eight scratch words below the callee-saved area
/// (`_call_header`: `STP_rri(..., (i + 8) * WORD)`); this frame packs fp/lr
/// and `r.callee_saved_registers` from offset 0, so the slot lands above
/// them instead. The offset differs, the mechanism does not.
///
/// Reads must add any `sub sp, sp, #n` in effect at the read point, the way
/// `aarch64/callbuilder.py` `write_real_errno` adds `self.current_sp`.
const SAVED_THREADLOCAL_OFS: u32 = 48;

const _: () = assert!(
    CALL_FRAME_SIZE.is_multiple_of(16),
    "aarch64 SP stays 16-byte aligned"
);

const _: () = assert!(
    SAVED_THREADLOCAL_OFS + 8 <= CALL_FRAME_SIZE,
    "the thread-local slot lives inside the entry frame"
);

/// Resolved argument: either a frame slot (frame-pointer-relative offset) or a constant.
enum ResolvedArg {
    /// Frame-pointer-relative byte offset: [rbp + offset] on x64, [x29, #offset] on aarch64.
    Slot(i32),
    /// Immediate constant value.
    Const(i64),
}

#[derive(Clone, Copy)]
struct CallAssemblerTargetAddr {
    immediate: Option<usize>,
}

impl CallAssemblerTargetAddr {
    fn is_available(self) -> bool {
        self.immediate.is_some()
    }
}

/// Pointer-identity key for `target_tokens_currently_compiling`. PyPy
/// x86/assembler.py:93 keys it by the descr Python object itself; we use
/// the underlying allocation address of the `Arc<dyn Descr>` so two
/// distinct TargetToken descriptors are never confused.
fn loop_target_id(op: &Op) -> Option<usize> {
    op.getdescr().as_ref().map(majit_ir::descr_identity)
}

fn target_argloc_from_loc(loc: Loc) -> TargetArgLoc {
    match loc {
        Loc::Reg(r) => TargetArgLoc::Reg {
            regnum: r.value,
            is_xmm: r.is_xmm,
        },
        Loc::Ebp(e) => TargetArgLoc::Ebp {
            ebp_offset: e.value,
            is_float: e.is_float,
        },
        Loc::Esp(e) => panic!("LABEL target argloc must not be a RawEspLoc, got {e:?}"),
        Loc::Frame(f) => TargetArgLoc::Frame {
            position: f.get_position(),
            ebp_offset: f.ebp_loc.value,
            is_float: f.ebp_loc.is_float,
        },
        Loc::Immed(i) => TargetArgLoc::Immed {
            value: i.value,
            is_float: false,
        },
        Loc::ImmedFloat(i) => TargetArgLoc::Immed {
            value: i.value,
            is_float: true,
        },
        // `ConstFloatLoc` is an absolute address (`is_imm_float()`).
        Loc::ConstFloat(c) => TargetArgLoc::Immed {
            value: c.value as i64,
            is_float: true,
        },
        Loc::Addr(a) => TargetArgLoc::Addr {
            base: a.base,
            index: a.index,
            scale: a.scale,
            offset: a.offset,
        },
    }
}

fn loc_from_target_argloc(loc: &TargetArgLoc) -> Loc {
    match *loc {
        TargetArgLoc::Reg { regnum, is_xmm } => {
            Loc::Reg(crate::regloc::RegLoc::new(regnum, is_xmm))
        }
        TargetArgLoc::Ebp {
            ebp_offset,
            is_float,
        } => Loc::Ebp(crate::regloc::RawEbpLoc {
            value: ebp_offset,
            is_float,
        }),
        TargetArgLoc::Frame {
            position,
            ebp_offset,
            is_float,
        } => Loc::Frame(crate::regloc::FrameLoc::new(position, ebp_offset, is_float)),
        TargetArgLoc::Immed { value, is_float } => {
            if is_float {
                // Written by `target_argloc_from_loc` for `ConstFloatLoc`.
                Loc::ConstFloat(crate::regloc::ConstFloatLoc {
                    value: value as usize,
                })
            } else {
                Loc::immed(value)
            }
        }
        TargetArgLoc::Addr {
            base,
            index,
            scale,
            offset,
        } => Loc::Addr(crate::regloc::AddressLoc {
            base,
            index,
            scale,
            offset,
        }),
    }
}

fn all_gen_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_GEN_REGS
}

fn all_float_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_FLOAT_REGS
}

fn core_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_gen_regs()
        .iter()
        .position(|candidate| *candidate == reg)
}

fn float_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_float_regs()
        .iter()
        .position(|candidate| *candidate == reg)
        .map(|idx| all_gen_regs().len() + idx)
}

fn reg_position_in_jitframe(reg: crate::regloc::RegLoc) -> Option<usize> {
    if reg.is_xmm {
        float_reg_position(reg)
    } else {
        core_reg_position(reg)
    }
}

fn deadframe_slot_for_loc(loc: &Loc) -> Option<u16> {
    match loc {
        Loc::Reg(reg) => Some(
            reg_position_in_jitframe(*reg).expect("deadframe slot: register is not managed") as u16,
        ),
        Loc::Frame(frame) => {
            // `store_info_on_descr`: `(loc.value - base_ofs) // WORD`, with
            // the misalignment and overflow assertions; 0xFFFF is the hole.
            assert_eq!(
                frame.ebp_loc.value & (WORD as i32 - 1),
                0,
                "store_info_on_descr: misaligned"
            );
            let position = frame.get_position() + JITFRAME_FIXED_SIZE;
            assert!(
                0 < position && position < 0xFFFF,
                "store_info_on_descr: overflow!"
            );
            Some(position as u16)
        }
        Loc::Immed(_)
        | Loc::ImmedFloat(_)
        | Loc::ConstFloat(_)
        | Loc::Ebp(_)
        | Loc::Esp(_)
        | Loc::Addr(_) => None,
    }
}

/// `aarch64/assembler.py _build_propagate_exception_path` — build the
/// per-CPU exception exit once, outside every loop/bridge buffer.
pub(crate) fn build_propagate_exception_path(
    cpu_handle: &crate::guard::CpuDescrHandle,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let mut mc = Assembler::new(0);
    let propagate_descr = cpu_handle.read().descr_ptrs().propagate_exception_descr as i64;
    let exc_value_addr = crate::jit_exc_value_addr() as i64;
    let exc_type_addr = crate::jit_exc_type_addr() as i64;

    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, exc_value_addr);
    dynasm!(mc ; .arch aarch64
        ; ldr x0, [x16]
        ; str xzr, [x16]
        ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
    );
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, exc_type_addr);
    dynasm!(mc ; .arch aarch64 ; str xzr, [x16]);
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 0, propagate_descr);
    dynasm!(mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);

    // `_call_footer`: pop the shadow-stack entry, return the current
    // jitframe and unwind the compiled entry's call frame.
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, rst);
    dynasm!(mc ; .arch aarch64
        ; ldr x17, [x16]
        ; sub x17, x17, 16
        ; str x17, [x16]
        ; mov x0, x29
        ; ldp x19, x20, [sp, #16]
        ; ldp x29, x30, [sp], CALL_FRAME_SIZE as i32
        ; ret
    );

    let buffer =
        codebuf::finalize_executable(mc, arena).expect("propagate_exception_path: finalize");
    let ptr = codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `aarch64/assembler.py _build_malloc_slowpath(kind='fixed')` — the one
/// per-CPU slow path shared by `malloc_cond` and
/// `malloc_cond_varsize_frame`.
///
/// Entry follows upstream: x0 is the old nursery head, x1 the bumped head,
/// x17 the gcmap, and x29 the current jitframe.  x0/x1 are the only managed
/// registers not preserved.  The helper returns the allocated payload in x0.
pub(crate) fn build_malloc_slowpath_fixed(
    propagate_path: usize,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let mut mc = Assembler::new(0);
    let base_ofs = FIRST_ITEM_OFFSET as u32;

    // `_push_all_regs_to_jitframe(mc, [x0, x1], True)`.
    for (i, reg) in all_gen_regs().iter().enumerate() {
        if reg.value <= 1 {
            continue;
        }
        let ofs = base_ofs + i as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
    }
    let float_base = base_ofs + all_gen_regs().len() as u32 * WORD as u32;
    for reg in all_float_regs() {
        let ofs = float_base + reg.value as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
    }

    // `_build_malloc_slowpath('fixed')`: recover the byte size, publish the
    // gcmap, and call the collector helper.  Keep LR on a 16-byte-aligned C
    // stack slot; the other live state is already in the jitframe.
    dynasm!(mc ; .arch aarch64
        ; sub x0, x1, x0
        ; str x17, [x29, JF_GCMAP_OFS as u32]
        ; sub sp, sp, 16
        ; str x30, [sp]
    );
    crate::aarch64::codebuilder::emit_mov_imm64_to(
        &mut mc,
        16,
        crate::runner::dynasm_nursery_slowpath as *const () as i64,
    );
    dynasm!(mc ; .arch aarch64 ; blr x16);

    // A collection may move the jitframe.  Its shadow-stack top entry is the
    // canonical post-collection pointer (`_reload_frame_if_necessary`).
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, rst);
    dynasm!(mc ; .arch aarch64
        ; ldr x16, [x16]
        ; sub x16, x16, 8
        ; ldr x29, [x16]
    );

    let nonnull = mc.new_dynamic_label();
    dynasm!(mc ; .arch aarch64 ; cbnz x0, =>nonnull);
    // The standalone propagate path expects the compiled entry's stack, not
    // this helper's LR slot.
    dynasm!(mc ; .arch aarch64 ; add sp, sp, 16);
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, propagate_path as i64);
    dynasm!(mc ; .arch aarch64
        ; br x16
        ; =>nonnull
    );

    // `_reload_frame_if_necessary` reapplies the frame write barrier after a
    // moving collection.  No live register needs another save here: all of
    // them are already resident in the jitframe, while x0 is parked beside
    // LR across the host helper call.
    if let Some(wb) = crate::runner::dynasm_write_barrier_descr() {
        let done_wb = mc.new_dynamic_label();
        let byteofs = wb.jit_wb_if_flag_byteofs;
        if (0..4096).contains(&byteofs) {
            dynasm!(mc ; .arch aarch64 ; ldrb w16, [x29, byteofs as u32]);
        } else if (-256..0).contains(&byteofs) {
            dynasm!(mc ; .arch aarch64 ; ldurb w16, [x29, byteofs]);
        } else {
            crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 17, byteofs as i64);
            dynasm!(mc ; .arch aarch64
                ; add x17, x29, x17
                ; ldrb w16, [x17]
            );
        }
        dynasm!(mc ; .arch aarch64
            ; mov w17, wb.jit_wb_if_flag_singlebyte as u32
            ; tst w16, w17
            ; b.eq =>done_wb
            ; str x0, [sp, 8]
            ; mov x0, x29
        );
        crate::aarch64::codebuilder::emit_mov_imm64_to(
            &mut mc,
            16,
            crate::runner::dynasm_write_barrier as *const () as i64,
        );
        dynasm!(mc ; .arch aarch64
            ; blr x16
            ; ldr x0, [sp, 8]
            ; =>done_wb
        );
    }

    dynasm!(mc ; .arch aarch64 ; str xzr, [x29, JF_GCMAP_OFS as u32]);
    // `_pop_all_regs_from_jitframe(mc, [x0, x1], True)`.
    for (i, reg) in all_gen_regs().iter().enumerate().rev() {
        if reg.value <= 1 {
            continue;
        }
        let ofs = base_ofs + i as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; ldr X(reg.value), [x29, ofs]);
    }
    for reg in all_float_regs().iter().rev() {
        let ofs = float_base + reg.value as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; ldr D(reg.value), [x29, ofs]);
    }
    let (nf_addr, _) = crate::runner::dynasm_nursery_addrs();
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, nf_addr as i64);
    dynasm!(mc ; .arch aarch64
        ; ldr x1, [x16]
        ; ldr x30, [sp]
        ; add sp, sp, 16
        ; ret
    );

    let buffer = codebuf::finalize_executable(mc, arena).expect("malloc_slowpath: finalize");
    let ptr = codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `aarch64/assembler.py _push_all_regs_to_jitframe` on a bare builder, for
/// the per-CPU helpers that have no `AssemblerARM64`.
fn push_all_regs_to_jitframe_raw(
    mc: &mut Assembler,
    ignored_regs: &[RegLoc],
    withfloats: bool,
    callee_only: bool,
) {
    let base_ofs = FIRST_ITEM_OFFSET as u32;
    let regs: &[RegLoc] = if callee_only {
        &crate::aarch64::registers::CALLER_RESP
    } else {
        all_gen_regs()
    };
    for (i, reg) in regs.iter().enumerate() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let ofs = base_ofs + i as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
    }
    if withfloats {
        let float_base = base_ofs + all_gen_regs().len() as u32 * WORD as u32;
        for reg in all_float_regs() {
            let ofs = float_base + reg.value as u32 * WORD as u32;
            dynasm!(mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
        }
    }
}

/// `aarch64/assembler.py _pop_all_regs_from_jitframe` on a bare builder.
fn pop_all_regs_from_jitframe_raw(
    mc: &mut Assembler,
    ignored_regs: &[RegLoc],
    withfloats: bool,
    callee_only: bool,
) {
    let base_ofs = FIRST_ITEM_OFFSET as u32;
    let regs: &[RegLoc] = if callee_only {
        &crate::aarch64::registers::CALLER_RESP
    } else {
        all_gen_regs()
    };
    for (i, reg) in regs.iter().enumerate() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let ofs = base_ofs + i as u32 * WORD as u32;
        dynasm!(mc ; .arch aarch64 ; ldr X(reg.value), [x29, ofs]);
    }
    if withfloats {
        let float_base = base_ofs + all_gen_regs().len() as u32 * WORD as u32;
        for reg in all_float_regs() {
            let ofs = float_base + reg.value as u32 * WORD as u32;
            dynasm!(mc ; .arch aarch64 ; ldr D(reg.value), [x29, ofs]);
        }
    }
}

/// `aarch64/assembler.py _store_and_reset_exception(mc, excvalloc,
/// exctploc)`: load `pos_exc_value` into `excvalloc` and `pos_exception`
/// into `exctploc`, then clear both.
fn store_and_reset_exception_raw(mc: &mut Assembler, excvalloc: RegLoc, exctploc: RegLoc) {
    let exc_value_addr = crate::jit_exc_value_addr() as i64;
    let exc_type_addr = crate::jit_exc_type_addr() as i64;
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, exc_value_addr);
    dynasm!(mc ; .arch aarch64 ; ldr X(excvalloc.value), [x16]);
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, exc_type_addr);
    dynasm!(mc ; .arch aarch64 ; ldr X(exctploc.value), [x16]);
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, exc_value_addr);
    dynasm!(mc ; .arch aarch64 ; str xzr, [x16]);
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, exc_type_addr);
    dynasm!(mc ; .arch aarch64 ; str xzr, [x16]);
}

/// `aarch64/assembler.py _restore_exception(mc, excvalloc, exctploc)`.
fn restore_exception_raw(mc: &mut Assembler, excvalloc: RegLoc, exctploc: RegLoc) {
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, crate::jit_exc_value_addr() as i64);
    dynasm!(mc ; .arch aarch64 ; str X(excvalloc.value), [x16]);
    crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, crate::jit_exc_type_addr() as i64);
    dynasm!(mc ; .arch aarch64 ; str X(exctploc.value), [x16]);
}

/// Load the byte at `[base + byteofs]` into w16 (ip0).
fn ldrb_ip0_signed_offset(mc: &mut Assembler, base: RegLoc, byteofs: i32) {
    if (0..4096).contains(&byteofs) {
        let ofs = byteofs as u32;
        dynasm!(mc ; .arch aarch64 ; ldrb W(16), [X(base.value), ofs]);
    } else if (-256..0).contains(&byteofs) {
        dynasm!(mc ; .arch aarch64 ; ldurb W(16), [X(base.value), byteofs]);
    } else {
        // Large negative offset: compute address in x16, then load
        crate::aarch64::codebuilder::emit_mov_imm64_to(mc, 16, byteofs as i64);
        dynasm!(mc ; .arch aarch64
            ; add x16, X(base.value), x16
            ; ldrb W(16), [x16]
        );
    }
}

/// `aarch64/assembler.py _build_wb_slowpath(withcards, withfloats,
/// for_frame)`: the helper every `COND_CALL_GC_WB*` slow path `BL`s with
/// the object in x0.  It saves what the call may clobber, and the card
/// variant ends with the flag test its caller branches on.
pub(crate) fn build_wb_slowpath(
    withcards: bool,
    withfloats: bool,
    for_frame: bool,
    arena: &Arc<AsmMemoryManager>,
) -> Option<(codebuf::ArenaExecutableBuffer, usize)> {
    use crate::aarch64::registers::{CALLER_RESP, CALLER_VFP_RESP, X19, X20};
    let descr = crate::runner::dynasm_write_barrier_descr()?;
    let func = if !withcards {
        // `descr.get_write_barrier_fn(cpu)`. The frame takes the guarded
        // entry, an ordinary store the one `gc.py get_write_barrier_fn`
        // names.
        if for_frame {
            crate::runner::dynasm_write_barrier as *const () as i64
        } else {
            crate::runner::dynasm_jit_remember_young_pointer as *const () as i64
        }
    } else {
        if descr.jit_wb_cards_set == 0 {
            return None;
        }
        crate::runner::dynasm_write_barrier_from_array as *const () as i64
    };
    //
    // This builds a helper function called from the slow path of
    // write barriers.  It must save all registers, and optionally
    // all vfp registers.  It takes a single argument which is in x0.
    // It must keep stack alignment accordingly.
    let mut mc = Assembler::new(0);
    let word = WORD as u32;
    dynasm!(mc ; .arch aarch64
        ; sub sp, sp, 2 * word
        ; str x30, [sp]
    );
    let (exc0, exc1) = (X19, X20);
    let frame_save = (CALLER_RESP.len() + 2 + CALLER_VFP_RESP.len()) as u32 * word;
    if !for_frame {
        push_all_regs_to_jitframe_raw(&mut mc, &[], withfloats, true);
    } else {
        // NOTE: don't save registers on the jitframe here!  It might
        // override already-saved values that will be restored
        // later...
        //
        // we're possibly called from the slowpath of malloc
        // save the caller saved registers
        // assuming we do not collect here
        dynasm!(mc ; .arch aarch64 ; sub sp, sp, frame_save);
        for i in (0..CALLER_RESP.len()).step_by(2) {
            let ofs = i as u32 * word;
            dynasm!(mc ; .arch aarch64
                ; stp X(CALLER_RESP[i].value), X(CALLER_RESP[i + 1].value), [sp, ofs as i32]
            );
        }
        let mut cur_stack = CALLER_RESP.len() as u32;
        dynasm!(mc ; .arch aarch64
            ; stp X(exc0.value), X(exc1.value), [sp, (cur_stack * word) as i32]
        );
        cur_stack += 2;
        for reg in CALLER_VFP_RESP {
            dynasm!(mc ; .arch aarch64 ; str D(reg.value), [sp, cur_stack * word]);
            cur_stack += 1;
        }
        store_and_reset_exception_raw(&mut mc, exc0, exc1);
    }
    crate::aarch64::codebuilder::emit_mov_imm64_to(&mut mc, 16, func);
    dynasm!(mc ; .arch aarch64 ; blr x16);
    //
    if !for_frame {
        pop_all_regs_from_jitframe_raw(&mut mc, &[], withfloats, true);
    } else {
        restore_exception_raw(&mut mc, exc0, exc1);
        for i in (0..CALLER_RESP.len()).step_by(2) {
            let ofs = i as u32 * word;
            dynasm!(mc ; .arch aarch64
                ; ldp X(CALLER_RESP[i].value), X(CALLER_RESP[i + 1].value), [sp, ofs as i32]
            );
        }
        let mut cur_stack = CALLER_RESP.len() as u32;
        dynasm!(mc ; .arch aarch64
            ; ldp X(exc0.value), X(exc1.value), [sp, (cur_stack * word) as i32]
        );
        cur_stack += 2;
        for reg in CALLER_VFP_RESP {
            dynasm!(mc ; .arch aarch64 ; ldr D(reg.value), [sp, cur_stack * word]);
            cur_stack += 1;
        }
        dynasm!(mc ; .arch aarch64 ; add sp, sp, frame_save);
    }
    //
    if withcards {
        // A final TEST8 before the RET, for the caller.  Careful to
        // not follow this instruction with another one that changes
        // the status of the CPU flags!
        ldrb_ip0_signed_offset(
            &mut mc,
            crate::aarch64::registers::X0,
            descr.jit_wb_if_flag_byteofs,
        );
        dynasm!(mc ; .arch aarch64
            ; movz w17, 0x80
            ; tst w16, w17
        );
    }
    //
    dynasm!(mc ; .arch aarch64
        ; ldr x17, [sp]
        ; add sp, sp, 2 * word
        ; ret x17
    );
    let buffer = codebuf::finalize_executable(mc, arena).expect("wb_slowpath: finalize");
    let ptr = codebuf::buffer_ptr(&buffer) as usize;
    Some((buffer, ptr))
}

// ── Abstract condition codes ──
// Architecture-independent CC values used throughout the assembler.
// Converted to arch-specific encoding at emission time.
const CC_O: u8 = 0;
const CC_NO: u8 = 1;
const CC_B: u8 = 2; // unsigned <
const CC_AE: u8 = 3; // unsigned >=
const CC_E: u8 = 4; // ==
const CC_NE: u8 = 5; // !=
const CC_BE: u8 = 6; // unsigned <=
const CC_A: u8 = 7; // unsigned >
const CC_S: u8 = 8;
const CC_NS: u8 = 9;
const CC_L: u8 = 10; // signed <
const CC_GE: u8 = 11; // signed >=
const CC_LE: u8 = 12; // signed <=
const CC_G: u8 = 13; // signed >

/// Widest value accepted by `codebuilder.py`'s `CMP_ri` encoding of
/// `cmp Xn, #imm`.
const MAX_CMP_IMM12: u32 = 4095;

/// Forward reach of `b.cond`: a signed 19-bit displacement in 4-byte words.
const BCOND_FORWARD_RANGE: usize = 1 << 20;

/// Ceiling assumed for one operation's machine code when deciding whether a
/// trace can use single-instruction guard branches.  The largest emitters are
/// `Label`/`Jump` (one store per live register, ~120 bytes) and calls with a
/// full argument list; 1KB leaves roughly an 8x margin.
const MAX_BYTES_PER_OP: usize = 1024;

/// Ceiling assumed for one guard's failure-recovery stub, which
/// [`AssemblerARM64::write_pending_failure_recoveries`] emits after the body a
/// guard's `b.cond` has to reach across.  The stub is bounded by its fixed
/// part: a `bl`, the optional exception block, the descriptor `mov`+`str`,
/// the gcmap push and the call footer.
/// The exact span is still checked in [`AssemblerARM64::check_guard_reach`]
/// once every stub is written, so this only has to be right often enough to
/// keep large traces off the two-instruction form.
const MAX_BYTES_PER_GUARD_STUB: usize = 512;

fn invert_cc(cc: u8) -> u8 {
    match cc {
        CC_O => CC_NO,
        CC_NO => CC_O,
        CC_B => CC_AE,
        CC_AE => CC_B,
        CC_E => CC_NE,
        CC_NE => CC_E,
        CC_BE => CC_A,
        CC_A => CC_BE,
        CC_S => CC_NS,
        CC_NS => CC_S,
        CC_L => CC_GE,
        CC_GE => CC_L,
        CC_LE => CC_G,
        CC_G => CC_LE,
        _ => CC_E, // fallback
    }
}

/// assembler.py AssemblerARM64.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
///
/// Borrows the trace's `inputargs` and `operations` for its lifetime so
/// `OpRef → Type` resolves through `op.type_` / `inputarg.tp` directly
/// (RPython `box.type` parity); no `value_types: HashMap` side-table.
pub struct AssemblerARM64<'a> {
    /// The dynasm assembler (rx86.py + codebuf.py combined).
    pub(crate) mc: Assembler,
    /// `BaseAssembler.asmmemmgr`: destination arena for materialization.
    asm_memory_manager: Arc<AsmMemoryManager>,
    /// assembler.py:83 pending_guard_tokens — guards awaiting recovery stubs.
    pending_guard_tokens: Vec<GuardToken>,
    /// GC bitmap to push before the current collecting call (e.g.,
    /// CallMallocNursery slow path). Set by `regalloc_perform` when the
    /// regalloc captured one, cleared after.
    pending_malloc_nursery_gcmap: Option<usize>,
    /// Index of the next guard / FINISH descr in `fail_descrs`.
    fail_index: u32,
    /// Frame depth (in WORD units) for the current trace.
    frame_depth: usize,
    /// `BaseAssembler._previous_rd_locs`.
    previous_rd_locs: majit_ir::RdLocs,
    /// assembler.py:94 setup() `self.frame_depth_to_patch = []` — buffer
    /// offsets of the `gen_load_int` depth placeholders emitted by
    /// `emit_check_frame_depth`, rewritten by `patch_stack_checks` once the
    /// final `frame_depth` is known.
    frame_depth_to_patch: Vec<usize>,
    /// assembler.py:1167-1171 `_assemble`: the frame depth of a cross-loop
    /// JUMP target (its `target_frame_depth`), or 0 when the trace has no
    /// external JUMP.  The closing `br` enters the target loop's body, which
    /// may use deeper frame slots than this trace; `_assemble` grows
    /// `frame_depth` to fit so a bridge's prologue `_check_frame_depth`
    /// reallocs the in-flight JITFRAME large enough before the `br`.
    jump_target_frame_depth: usize,
    /// Fail descriptors built during assembly.
    fail_descrs: FailDescrStore,
    /// trace_id for this compilation.
    trace_id: u64,
    /// header_pc (green_key) for this compilation.
    header_pc: u64,
    /// Input argument types.
    input_types: Vec<Type>,
    /// assembler.py rebuild_faillocs_from_descr parity:
    /// bridge input locations recovered from the source guard descr.
    bridge_input_locs: Option<Vec<Loc>>,

    // ── State tracking for code generation ──
    /// Maps OpRef → jitframe slot index. `IndexMap` (O(1) get/insert), not the
    /// Vec-backed `IndexMap`: `resolve_opref` reads this per emitted op during
    /// codegen and the sync loop inserts one entry per live var, so a Vec-`get`
    /// is O(n) and makes `_assemble` O(n^2) on large traces — `get_index_of`
    /// inlines into `_assemble` and dominated compile time on a trace of tens
    /// of thousands of ops. The box→
    /// location map is a dict in the reference assembler; insertion order is
    /// preserved (no semantic change, only the lookup cost).
    opref_to_slot: indexmap::IndexMap<OpRef, usize, rustc_hash::FxBuildHasher>,
    /// Trace inputargs — borrowed for `opref_type` lookups.
    inputargs: &'a [InputArgRc],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    operations: &'a [OpRc],
    /// `arg.index` raw -> idx in inputargs, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    inputarg_pos: majit_ir::PosIndex,
    /// `op_pos[op.pos.raw()] = idx in operations`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots and Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    op_pos: majit_ir::PosIndex,
    /// Constants: OpRef index (>= 10000) → typed `Const` value. The box
    /// variant carries its own type (`Const::get_type`), so no separate
    /// constant-type map is needed.
    constants: majit_ir::ConstMap<majit_ir::Const>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// `(result location, condition code)` of the comparison emitted by the
    /// immediately preceding op, so a `GUARD_TRUE`/`GUARD_FALSE`
    /// that consumes exactly that result can branch on the live NZCV flags
    /// instead of re-testing the materialized boolean.
    /// `assembler.py:1186-1198 _walk_operations` folds the pair by handing
    /// `prevop` to `guard_operations[...]`, which reaches
    /// `regalloc.py guard_impl` -> `dispatch_comparison(prevop)` and
    /// returns the comparison's own `fcond`; the guard then emits only the
    /// conditional branch.  pyre's regalloc emits the two ops separately,
    /// so the pairing is recognised here at emit time instead.
    /// Cleared by every other op and by every `regalloc_mov`, so only an
    /// ADJACENT pair folds — the same `operations[i + 1]` window upstream uses.
    pending_cmp_cc: Option<(RegLoc, u8)>,

    /// Whether guards must use the two-instruction long-branch form.
    /// `b.cond` carries a signed 19-bit word displacement, so it reaches
    /// +1MB; the failure-recovery stubs are written straight after the body,
    /// so a guard reaches its stub whenever the trace's own code fits inside
    /// that window.  Only a trace too large for that has to pay the extra
    /// inversion and `b`.  Decided once per trace in `_assemble`.
    long_guard_branch: bool,

    /// Buffer offset of the first single-instruction `b.cond` emitted to each
    /// label, so `write_pending_failure_recoveries` can measure the real
    /// displacement to the recovery stub it binds that label to.  The earliest
    /// branch is the one that has to reach furthest, and a label with no entry
    /// was never branched to in the short form.
    short_guard_branch_offsets: IndexMap<DynamicLabel, usize, rustc_hash::FxBuildHasher>,
    /// `(branch offset, stub offset)` for every short guard branch whose stub
    /// landed outside `b.cond`'s forward reach.  Read by `check_guard_reach`.
    guard_reach_violations: Vec<(usize, usize)>,

    /// First cross-buffer JUMP whose target still points into an assembler
    /// buffer instead of executable memory.
    unrelocated_jump_target: Option<(usize, usize)>,

    /// x86/assembler.py:93 target_tokens_currently_compiling parity.
    /// Keyed by descriptor pointer identity (PyPy uses Python `is`).
    target_tokens_currently_compiling: IndexMap<usize, DynamicLabel, rustc_hash::FxBuildHasher>,
    compiled_target_tokens: Vec<majit_ir::DescrRef>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// `AbstractLLCPU.subclassrange_min_offset`. `None` keeps the
    /// TYPE_INFO arm of `emit_op_guard_subclass`.
    subclassrange_min_offset: Option<usize>,
    /// llsupport/gc.py get_typeid_from_classptr_if_gcremovetypeptr vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: IndexMap<i64, u32>,
    /// TYPE_INFO / CLASSTYPE constants for `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`, fetched by the runner from the active gc_ll_descr.
    guard_gc_type_info: Option<GuardGcTypeInfo>,
    /// Constant classptr → `(subclassrange_min, subclassrange_max)`, matching
    /// `loc_check_against_class.getint()` field reads in
    /// `aarch64/opassembler.py:695-698`.
    classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// Leaked pointer holding the resolved entry address for self-recursive
    /// CALL_ASSEMBLER via the execute trampoline. Written after finalization.
    self_entry_addr_ptr: *mut usize,
    /// opassembler.py:1177 _finish_gcmap.
    finish_gcmap: Option<*mut usize>,
    /// opassembler.py:1215 gcmap_for_finish.
    gcmap_for_finish: *mut usize,
    /// assembler.py _store_force_index parity:
    /// Pre-allocated fail descr for the next GUARD_NOT_FORCED, created
    /// at CALL_ASSEMBLER emission time so we can store its pointer to
    /// jf_force_descr before the call. Consumed by the subsequent
    /// GUARD_NOT_FORCED guard emission.
    pending_force_descr: Option<majit_ir::DescrRef>,
    /// Pre-wrapped `FailDescrCell` for the same pending guard.  Codegen
    /// bakes the cell's thin pointer into `JF_FORCE_DESCR_OFS` so that
    /// `force_token_to_dead_frame` (cranelift/compiler.rs) can
    /// recover the descr via `recover_fail_descr_cell` without the
    /// fat-pointer mismatch a bare `Arc<dyn Descr>` ptr would cause.
    /// The same cell is consumed by `append_guard_token_with_faillocs`
    /// so jf_force_descr and jf_descr resolve to the same identity.
    pending_force_cell: Option<usize>,
    /// `compile.py:665-674` + `pyjitpl.py:2283`: construction-time
    /// snapshot of the six descr pointers attached to the owning cpu
    /// instance.  Retained for constructor signature stability across
    /// runner / call_assembler callsites — emission helpers
    /// (`done_with_this_frame_descr_ptr_for_type`,
    /// `exit_frame_with_exception_descr_ref_ptr`,
    /// `propagate_exception_descr_ptr`) read live from `cpu_handle`
    /// instead so the raw ptr baked into `JF_DESCR_OFS` and the Arc
    /// stamped into `meta_descr` come from the same snapshot.
    #[allow(dead_code)]
    attached_descrs: crate::guard::AttachedDescrPtrs,
    /// `Arc` clone of the owning cpu's attachment handle.  Its heap
    /// address is baked into the CALL_ASSEMBLER helper call site as a
    /// compile-time immediate (`Arc::as_ptr`) and the `Arc` is moved
    /// into the resulting `CompiledCode` so the pointee outlives any
    /// subsequent `DynasmBackend` drop — same role as RPython's
    /// `self.cpu` attribute-access after whole-program translation,
    /// where the `cpu` object's identity is guaranteed by Python.
    cpu_handle: crate::guard::CpuDescrHandle,
    /// `assembler.py reserve_gcref_table`: one label per slot of the
    /// reference-constant table reserved at the start of this code block,
    /// which the `LoadFromGcTable` genop reads PC-relative. Empty when the
    /// trace references no reference constants.
    gcref_table: Vec<dynasmrt::DynamicLabel>,
    /// `AssemblerARM64.malloc_slowpath`: per-CPU shared slowpath built by
    /// `_build_malloc_slowpath('fixed')` and used by both fixed-size and
    /// varsize-frame nursery probes.
    malloc_slowpath_fixed: usize,
    /// `AssemblerARM64.wb_slowpath`: the per-CPU `_build_wb_slowpath`
    /// helpers, indexed `withcards + 2 * withfloats`, with `[4]` the
    /// `for_frame` one.
    wb_slowpath: [usize; 5],
    /// `assembler.py` `datablockwrapper`. Float constants live here
    /// (`VFPRegisterManager.convert_to_imm` → `ConstFloatLoc`), not in
    /// the instruction stream.
    datablockwrapper: MachineDataBlockWrapper,
}

/// `NOP` (`HINT #0`), the placeholder `_emit_guard` leaves at a
/// `GUARD_NOT_INVALIDATED` site until `invalidate_loop` writes a `B` over it.
const NOP_INSN: u32 = 0xD503_201F;

/// assembler.py GuardToken — represents a pending guard needing
/// a recovery stub to be written after the main loop body.
struct GuardToken {
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// Descr for stub bookkeeping (`set_adr_jump_offset`).
    fail_descr: majit_ir::DescrRef,
    /// [`FailDescrCell::thin_ptr`] baked into `jf_descr`. The cell lives
    /// on `Asm::fail_descrs` so the address stays valid.
    fail_cell_ptr: usize,
    /// opassembler.py:515 GuardToken.gcmap.
    gcmap: *mut usize,
    /// `opassembler.py _emit_guard`'s `token.offset = pos` — the instruction
    /// word this guard's branch is patched into.  `Some` only for
    /// `GUARD_NOT_INVALIDATED`, the one guard whose branch is written after the
    /// buffer is materialised; every other guard reaches its recovery stub
    /// through a `fail_label` dynasm resolves at finalize.
    pos_jump_offset: Option<usize>,
    /// llsupport/assembler.py must_save_exception: true for
    /// GUARD_EXCEPTION / GUARD_NO_EXCEPTION / GUARD_NOT_FORCED.  Selects the
    /// exc=True failure-recovery variant that stages pos_exc_value into
    /// jf_guard_exc (store_info_on_descr:236) so grab_exc_value can read it.
    must_save_exception: bool,
}

/// What `assembler.py process_pending_guards` reads off a guard token once the
/// buffer has been materialised.  Upstream still has the tokens there;
/// `write_pending_failure_recoveries` consumes pyre's, so the three fields the
/// walk needs survive it as this.
struct RecoveryStub {
    /// `tok.faildescr`.
    fail_descr: majit_ir::DescrRef,
    /// `tok.pos_recovery_stub`.
    pos_recovery_stub: usize,
    /// `tok.offset`, present only for `GUARD_NOT_INVALIDATED`.
    pos_jump_offset: Option<usize>,
}

fn fail_cell_capacity(ops: &[OpRc]) -> usize {
    ops.iter()
        .filter(|op| op.opcode.is_guard() || op.opcode == OpCode::Finish)
        .count()
}

/// Compiled output from assemble_loop/assemble_bridge.
pub struct CompiledCode {
    /// Executable memory buffer (keeps code alive).
    pub buffer: codebuf::ArenaExecutableBuffer,
    /// Entry point offset within the buffer.
    pub entry_offset: AssemblyOffset,
    /// Fail descriptors for guards + FINISH ops.
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at conversion
    /// from the in-progress `AssemblerARM64.fail_descrs` Vec.
    pub fail_descrs: std::sync::Arc<FailDescrStore>,
    /// Input argument types.
    pub input_types: Vec<Type>,
    /// `compile.py` parity: `Arc` clone of the owning cpu's
    /// attachment handle.  Keeps the heap pointee alive for the whole
    /// lifetime of this compiled trace so the `cpu_handle` immediate
    /// baked into the CALL_ASSEMBLER helper call site never dangles,
    /// even if the emitting `DynasmBackend` is dropped first.
    pub cpu_attachments: crate::guard::CpuDescrHandle,
    /// trace_id.
    pub trace_id: u64,
    /// header_pc (green_key).
    pub header_pc: u64,
    /// Frame depth (number of jitframe slots used).
    /// AtomicUsize for redirect_call_assembler's update_frame_info
    /// parity: may be updated through &CompiledCode (shared ref).
    pub frame_depth: std::sync::atomic::AtomicUsize,
    /// `None` for root loops; bridges set `(source_trace_id, source_fail_index_per_trace)`.
    pub source_guard: Option<(u64, u32)>,
    /// `assembler.py process_pending_guards` — the `GUARD_NOT_INVALIDATED`
    /// sites this trace left for `clt.invalidate_positions`.
    pub invalidate_positions: Vec<majit_backend::InvalidatePosition>,
    /// `MachineDataBlockWrapper.done`: the ranges that hold `ConstFloatLoc`
    /// bytes. They stay mapped for as long as this code does.
    pub data_blocks: Vec<AsmMemoryBlock>,
}

impl CompiledCode {
    /// `looptoken._ll_function_addr = rawstart + functionpos`: the first
    /// instruction, past the reserved gcref table.
    pub fn entry_ptr(&self) -> *const u8 {
        self.buffer.ptr(self.entry_offset)
    }
}

/// The `genop_*` methods here are line-by-line ports of the RPython emitters
/// (`aarch64/assembler.py` / `x86/assembler.py` `genop_*`).  Emission runs
/// through `regalloc_perform`, which works from regalloc `arglocs`, so the
/// ports whose opcode that dispatch reaches by another route carry
/// `#[allow(dead_code)]` individually.  They are annotated rather than
/// deleted so the upstream method boundary stays where a later porter looks
/// for it; the attribute is per method so the lint still reports anything
/// else in this `impl` that stops being reached.
impl<'a> AssemblerARM64<'a> {
    #[inline]
    fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        match at_op_index {
            Some(at) => type_index.opref_type_at(opref, at),
            None => type_index.opref_type(opref),
        }
    }

    /// assembler.py:54 __init__
    pub(crate) fn new(
        asm_memory_manager: Arc<AsmMemoryManager>,
        trace_id: u64,
        header_pc: u64,
        constants: majit_ir::ConstMap<majit_ir::Const>,
        vtable_offset: Option<usize>,
        subclassrange_min_offset: Option<usize>,
        classptr_to_typeid: IndexMap<i64, u32>,
        guard_gc_type_info: Option<GuardGcTypeInfo>,
        classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
        attached_descrs: crate::guard::AttachedDescrPtrs,
        cpu_handle: crate::guard::CpuDescrHandle,
        malloc_slowpath_fixed: usize,
        wb_slowpath: [usize; 5],
        inputargs: &'a [InputArgRc],
        operations: &'a [OpRc],
    ) -> Self {
        let inputarg_pos = OpTypeIndex::<OpRc, InputArgRc>::build_inputarg_pos(inputargs);
        let op_pos = OpTypeIndex::<OpRc, InputArgRc>::build_op_pos(operations);
        let datablockwrapper = MachineDataBlockWrapper::new(Arc::clone(&asm_memory_manager));
        AssemblerARM64 {
            mc: Assembler::new(0),
            asm_memory_manager,
            pending_guard_tokens: Vec::new(),
            pending_malloc_nursery_gcmap: None,
            fail_index: 0,
            frame_depth: JITFRAME_FIXED_SIZE,
            previous_rd_locs: majit_ir::RdLocs::new(),
            frame_depth_to_patch: Vec::new(),
            jump_target_frame_depth: 0,
            fail_descrs: FailDescrStore::default(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            bridge_input_locs: None,
            opref_to_slot: indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher),
            inputargs,
            operations,
            inputarg_pos,
            op_pos,
            constants,
            next_slot: 0,
            guard_success_cc: None,
            pending_cmp_cc: None,
            long_guard_branch: false,
            short_guard_branch_offsets: IndexMap::with_hasher(rustc_hash::FxBuildHasher),
            guard_reach_violations: Vec::new(),
            unrelocated_jump_target: None,
            target_tokens_currently_compiling: IndexMap::with_hasher(rustc_hash::FxBuildHasher),
            compiled_target_tokens: Vec::new(),
            vtable_offset,
            subclassrange_min_offset,
            classptr_to_typeid,
            guard_gc_type_info,
            classptr_to_subclass_range,
            self_entry_label: None,
            self_entry_addr_ptr: Box::into_raw(Box::new(0usize)),
            finish_gcmap: None,
            gcmap_for_finish: {
                let gcmap = allocate_gcmap(1, JITFRAME_FIXED_SIZE);
                gcmap_set_bit(gcmap, 0);
                gcmap
            },
            pending_force_descr: None,
            pending_force_cell: None,
            attached_descrs,
            cpu_handle,
            gcref_table: Vec::new(),
            malloc_slowpath_fixed,
            wb_slowpath,
            datablockwrapper,
        }
    }

    /// `assembler.py reserve_gcref_table`: reserve `n` zeroed words,
    /// padded to a multiple of 16 bytes, at the start of the machine code,
    /// so the `LoadFromGcTable` genop can address them PC-relative. The
    /// runner writes the gcrefs into them once the block is materialized
    /// (`patch_gcref_table`; `GcTable::in_code`). Must run before any
    /// instruction is emitted.
    pub(crate) fn reserve_gcref_table(&mut self, n: usize) {
        assert_eq!(
            self.mc.offset().0,
            0,
            "the gcref table opens the code block"
        );
        for _ in 0..n {
            let slot = self.mc.new_dynamic_label();
            dynasm!(self.mc ; =>slot ; .u64 0);
            self.gcref_table.push(slot);
        }
        if n % 2 == 1 {
            dynasm!(self.mc ; .u64 0);
        }
    }

    /// `AssemblerARM64.load` for an `is_imm_float()` value:
    /// `gen_load_int(r.ip0, adr)` then `LDR_di(loc, r.ip0, 0)`.
    /// The pool slot was allocated by `VFPRegisterManager.convert_to_imm`.
    fn load_imm_float(&mut self, loc: u8, value: crate::regloc::ConstFloatLoc) {
        self.emit_mov_imm64(16, value.value as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr D(loc), [x16]);
    }

    /// Panics on allocation failure because `load_float_arg_to_d0/d1` (`#[allow(dead_code)]` `genop_float_*`) have no error channel.
    fn const_float_from_bits(&mut self, bits: u64) -> crate::regloc::ConstFloatLoc {
        park_float_const(&mut self.datablockwrapper, bits).unwrap_or_else(|err| {
            panic!("MachineDataBlockWrapper.malloc_aligned failed parking float constant: {err}")
        })
    }

    /// `compile.py:665` parity: heap-pinned address of `self.cpu`'s
    /// attachment handle, derived from the Arc clone.  Baked into the
    /// CALL_ASSEMBLER helper call site.
    fn cpu_handle_ptr(&self) -> i64 {
        Arc::as_ptr(&self.cpu_handle) as *const () as i64
    }

    /// `compile.py:665-674` parity: attach the six metainterp descrs on
    /// the emission side.  Mirrors `self.cpu.done_with_this_frame_descr_*`
    /// reads in `rpython/jit/backend/aarch64/assembler.py`.  Reads from
    /// the live `cpu_handle` snapshot so the raw pointer baked into
    /// `JF_DESCR_OFS` and the Arc returned by
    /// `done_with_this_frame_descr_arc_for_type` resolve to the same
    /// metainterp singleton.
    fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .done_with_this_frame_descr_ptr_for_type(tp) as i64
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` Arc lookup —
    /// returns the metainterp `DoneWithThisFrameDescr*` Arc the
    /// optimizer attached for the given result type.  Used to stamp
    /// `meta_descr` on backend FINISH descrs so trait forwarding routes
    /// `is_finish` / `fail_arg_types` through the metainterp class
    /// hierarchy (`compile.py:624 final_descr=True`).
    fn done_with_this_frame_descr_arc_for_type(&self, tp: Type) -> Option<majit_ir::DescrRef> {
        let attachments = self.cpu_handle.read();
        match tp {
            Type::Void => attachments.done_with_this_frame_descr_void.clone(),
            Type::Int => attachments.done_with_this_frame_descr_int.clone(),
            Type::Ref => attachments.done_with_this_frame_descr_ref.clone(),
            Type::Float => attachments.done_with_this_frame_descr_float.clone(),
        }
    }

    /// `compile.py:658` parity: `self.cpu.exit_frame_with_exception_descr_ref`.
    fn exit_frame_with_exception_descr_ref_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .exit_frame_with_exception_descr_ref as i64
    }

    /// `pyjitpl.py` parity: `self.cpu.propagate_exception_descr`.
    /// Stamped into `jf_descr` by the inline propagate path emitted at
    /// `OpCode::CheckMemoryError` (opassembler.py:258
    /// `emit_op_check_memory_error` → `propagate_memoryerror_if_reg_is_null`).
    fn propagate_exception_descr_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .propagate_exception_descr as i64
    }

    // Helper methods

    /// Emit the inline propagate-MemoryError sequence: if `reg_x` is
    /// NULL, route through the `propagate_exception_descr` exit
    /// (`_build_propagate_exception_path`, assembler.py:559-577 —
    /// inlined per call site because pyre's dynasm backend doesn't
    /// emit a separate trampoline on aarch64).
    ///
    /// Used by:
    ///   * `OpCode::CheckMemoryError` (after the four CALL_R malloc
    ///     helpers in `gen_call_malloc_gc`).
    ///   * Each inline malloc-nursery slowpath site, where the helper
    ///     (`dynasm_nursery_slowpath` / `_varsize` / `_jitframe`)
    ///     returns x0 = 0 on real host OOM (calloc failure).  Before
    ///     this hook the OOM null was stored straight into a typed
    ///     Ref slot, corrupting subsequent generated stores.
    ///
    /// Skips emission when `propagate_exception_descr` is unattached
    /// (unit tests that bypass `MetaInterp::finish_setup`).
    fn emit_propagate_memory_error_if_null(&mut self, reg_x: u8) {
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr == 0 {
            return;
        }
        let skip = self.mc.new_dynamic_label();
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        // x0 is freely clobberable on the propagate path: _call_footer
        // (assembler.py:574) overwrites x0 with fp before returning,
        // so any live value in x0 was already dead the moment we
        // branched into the path.
        dynasm!(self.mc ; .arch aarch64
            ; cbnz X(reg_x), =>skip
        );
        // assembler.py:509-512 — load pos_exc_value into x0,
        // then clear pos_exc_value.
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x16]
            ; str xzr, [x16]
        );
        // assembler.py:535-536 — clear pos_exception.
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x16]
            // assembler.py:565 — store x0 → jf_guard_exc.
            ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
        );
        // assembler.py:572-573 — store propagate_descr → jf_descr.
        self.emit_mov_imm64(0, propagate_descr);
        dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);
        self._call_footer();
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
    }

    /// Frame-pointer-relative byte offset for a given slot index.
    /// Slots are absolute jf_frame indices, including the fixed
    /// JITFRAME-managed prefix. FIRST_ITEM_OFFSET accounts for the object
    /// header and array-length word that precede jf_frame[0].
    fn slot_offset(slot: usize) -> i32 {
        FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32
    }

    /// Resolve an OpRef to either a frame slot offset or an immediate constant.
    /// Cranelift resolve_opref parity: check constants map FIRST
    /// (regardless of CONST_BIT), then fall back to slot mapping.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        // Op results take precedence over constants (Cranelift parity).
        if let Some(&slot) = self.opref_to_slot.get(&opref) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        // history.py/268/314 — inline-Const variants carry value inline.
        if let Some(val) = opref
            .inline_const_bits()
            .or_else(|| self.constants.get(&opref.raw()).map(|c| c.as_raw_i64()))
        {
            return ResolvedArg::Const(val);
        }
        // history.py/268/314 — Const always carries a value, so a
        // constant OpRef with no resolvable value is an invariant break,
        // not a `#0`.
        if opref.is_constant() {
            panic!(
                "resolve_opref: legacy constant {opref:?} missing from constants pool — \
                 Const always carries a value (history.py:227/268/314)"
            );
        }
        // regalloc.py:102 `FrameManager.loc(must_exist=True)` raises KeyError
        // for a non-constant box that is neither register- nor frame-resident:
        // a used box is always slot-mapped or constant. Silently materializing
        // an unmapped box as `#0` would hide a wrong value — the same hazard
        // that moved genop_call_assembler onto regalloc arglocs. Fail loud.
        panic!(
            "resolve_opref: unmapped non-constant OpRef {opref:?} — every used \
             box must be slot-mapped or constant (regalloc.py:102 loc must_exist)"
        );
    }

    /// Allocate a frame slot for an OpRef and return the ABSOLUTE jitframe
    /// slot index (already offset by JITFRAME_FIXED_SIZE so slot_offset()
    /// yields the user-area byte offset directly).
    ///
    /// Reuses the existing slot if the OpRef has already been allocated.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref, slot);
        slot
    }

    /// Load a (lhs, src) pair for a 3-operand binop into registers,
    /// returning the register numbers to use. Uses x17 for lhs scratch
    /// and x16 for src scratch when the loc is Frame/Immed.
    fn load_3op_into_scratch(&mut self, lhs: &Loc, src: &Loc) -> (u8, u8) {
        let lhs_reg = self.load_loc_to_reg(lhs, 17);
        let src_reg = self.load_loc_to_reg(src, 16);
        (lhs_reg, src_reg)
    }

    /// Load a single operand `loc` into a register, returning its number.
    /// Register locs are used in place; Frame/Immed locs are materialised
    /// into the caller-supplied `scratch` register (x16/x17).
    fn load_loc_to_reg(&mut self, loc: &Loc, scratch: u8) -> u8 {
        match loc {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(scratch, f.ebp_loc.value);
                scratch
            }
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                self.emit_mov_imm64(scratch as u32, i.value);
                scratch
            }
            // The regalloc only routes Reg/Frame/Immed operands into the 3-op /
            // overflow arith paths; an Ebp/Addr loc here means a broken regalloc
            // contract. Panic instead of emitting arith on a stale scratch reg.
            other => panic!("load_loc_to_reg expected Reg/Frame/Immed, got {other:?}"),
        }
    }

    /// If `loc` is a `ConstInt` that fits the AArch64 `#imm12` add/sub
    /// immediate range, return it for the immediate instruction form.
    /// Range mirrors `check_imm_box` (`0 <= v < DEFAULT_IMM_SIZE` = 4096).
    fn addsub_imm12(loc: &Loc) -> Option<i64> {
        match loc {
            Loc::Immed(i) | Loc::ImmedFloat(i) if (0..4096).contains(&i.value) => Some(i.value),
            _ => None,
        }
    }

    /// Emit a 3-operand integer binop: `OP dst, lhs, src`.
    ///
    /// aarch64/opassembler.py parity: `ADD/SUB/AND/ORR/EOR Rd, Rn, Rm` —
    /// the hardware supports distinct Rd and Rn, unlike x86's 2-operand
    /// `SUB dst, dst, src`. Handles reg/frame/immediate forms for both
    /// operands, routing scratch via x16/x17 as needed.
    fn emit_binop_3op(&mut self, opcode: OpCode, dst_reg: u8, lhs: &Loc, src: &Loc) {
        // Fold a small non-negative ConstInt rhs into the `add/sub Rd,Rn,#imm12`
        // immediate form instead of materialising it into a scratch register.
        // The regalloc (`prepare_int_ri` / `prepare_op_int_sub`) only keeps
        // add/sub rhs as `Loc::Immed`; the range mirrors `check_imm_box`
        // (`0 <= v < DEFAULT_IMM_SIZE` = 4096, the AArch64 `#imm12` range).
        if matches!(
            opcode,
            OpCode::IntAdd | OpCode::IntSub | OpCode::NurseryPtrIncrement
        ) && let Some(im) = Self::addsub_imm12(src)
        {
            let l = self.load_loc_to_reg(lhs, 17);
            let d = dst_reg;
            // Rd/Rn take the SP-form register family for the `#imm12`
            // encoding; `XSP(r)` encodes identically to `X(r)` for every
            // non-SP register allocated to this instruction.
            match opcode {
                OpCode::IntSub => {
                    dynasm!(self.mc ; .arch aarch64 ; sub XSP(d), XSP(l), im as u32)
                }
                _ => dynasm!(self.mc ; .arch aarch64 ; add XSP(d), XSP(l), im as u32),
            }
            return;
        }
        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
        let d = dst_reg;
        let l = lhs_reg as u8;
        let s = src_reg as u8;
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                dynasm!(self.mc ; .arch aarch64 ; add X(d), X(l), X(s));
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(l), X(s));
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(l), X(s));
            }
            OpCode::IntAnd => {
                dynasm!(self.mc ; .arch aarch64 ; and X(d), X(l), X(s));
            }
            OpCode::IntOr => {
                dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(l), X(s));
            }
            OpCode::IntXor => {
                dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(l), X(s));
            }
            _ => {}
        }
    }

    /// Emit: CMP loc0, loc1
    /// `codebuilder.py ADD_ri` — `add Xd, Xn, #imm12`.  Encoded directly
    /// because dynasm's immediate form cannot take a dynamic register (it
    /// cannot tell whether the register is SP).
    pub(crate) fn emit_add_ri(&mut self, rd: u8, rn: u8, imm: u32) {
        debug_assert!(imm <= MAX_CMP_IMM12, "add immediate {imm} exceeds 12 bits");
        let word: u32 = (0b1001000100u32 << 22) | (imm << 10) | ((rn as u32) << 5) | (rd as u32);
        dynasm!(self.mc ; .arch aarch64 ; .u32 word);
    }

    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        // Load loc0 into x16 if needed, loc1 into x17 if needed
        let r0 = match loc0 {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                16
            }
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        // `opassembler.py`'s `emit_int_comp_op` takes `CMP_ri` when the
        // right-hand side is an immediate; `codebuilder.py`'s `CMP_ri` holds
        // a 12-bit unsigned field, so anything wider still needs a register.
        if let Loc::Immed(i) | Loc::ImmedFloat(i) = loc1
            && let Ok(imm) = u32::try_from(i.value)
            && imm <= MAX_CMP_IMM12
        {
            // dynasm's `cmp Xn|SP, #uimm` form cannot take a dynamic register
            // operand. `codebuilder.py`'s `CMP_ri` encodes the equivalent
            // instruction as SUBS with Rd = xzr.
            let word: u32 = (0b1111000100u32 << 22) | (imm << 10) | ((r0 as u32) << 5) | 0b11111;
            dynasm!(self.mc ; .arch aarch64 ; .u32 word);
            return;
        }
        let r1 = match loc1 {
            Loc::Reg(s) => s.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(17, f.ebp_loc.value);
                17
            }
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                self.emit_mov_imm64(17, i.value);
                17
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; cmp X(r0 as u8), X(r1 as u8));
    }

    fn guard_value_stack_offset(loc: &Loc) -> i32 {
        match loc {
            ebp_loc_pat!(e) => e.value,
            _ => panic!("emit_op_guard_value expects a stack location, got {loc:?}"),
        }
    }

    /// aarch64/opassembler.py `emit_op_guard_value`.
    /// Core: `CMP_rr`, with `ip0` holding an imm or stack arg1.
    /// Vfp: `FCMP_dd`, with arg1 a vfp register (or `vfp_ip` when stacked).
    /// The guard that follows branches on EQ.
    fn emit_op_guard_value(&mut self, arglocs: &[Loc]) {
        let v0 = arglocs[0];
        let v1 = arglocs[1];
        let ip0 = crate::aarch64::registers::IP0.value;
        let vfp_ip = crate::aarch64::registers::VFP_IP.value;
        if let Loc::Reg(r0) = v0
            && r0.is_float()
        {
            let rhs = if let Loc::Reg(r1) = v1
                && r1.is_float()
            {
                r1.value
            } else {
                assert!(
                    v1.is_stack(),
                    "emit_op_guard_value vfp arg1 must be a vfp register or stack, got {v1:?}"
                );
                self.emit_ldr_fp_d(vfp_ip, Self::guard_value_stack_offset(&v1));
                vfp_ip
            };
            dynasm!(self.mc ; .arch aarch64 ; fcmp D(r0.value), D(rhs));
            return;
        }
        // `ARMRegisterManager.return_constant` would have loaded a ConstInt
        // arg0 into a core register. An immediate or stack slot left by the
        // shared `return_constant` is compared from `ip1` so `ip0` stays
        // available for arg1.
        let lhs = if let Loc::Reg(r0) = v0
            && r0.is_core_reg()
        {
            r0.value
        } else {
            let ip1 = crate::aarch64::registers::IP1.value;
            match v0 {
                Loc::Immed(i) | Loc::ImmedFloat(i) => {
                    self.emit_mov_imm64(u32::from(ip1), i.value);
                }
                _ if v0.is_stack() => {
                    self.emit_ldr_fp(ip1, Self::guard_value_stack_offset(&v0));
                }
                _ => {
                    panic!("emit_op_guard_value arg0 must be a register, imm, or stack, got {v0:?}")
                }
            }
            ip1
        };
        let rhs = if let Loc::Reg(r1) = v1
            && r1.is_core_reg()
        {
            r1.value
        } else if let Loc::Immed(i) = v1 {
            self.emit_mov_imm64(u32::from(ip0), i.value);
            ip0
        } else {
            assert!(
                v1.is_stack(),
                "emit_op_guard_value core arg1 must be a core register, imm, or stack, got {v1:?}"
            );
            self.emit_ldr_fp(ip0, Self::guard_value_stack_offset(&v1));
            ip0
        };
        dynasm!(self.mc ; .arch aarch64 ; cmp X(lhs), X(rhs));
    }

    /// Emit: TEST loc, loc (for guard_true/guard_false)
    fn emit_test_loc(&mut self, loc: &Loc) {
        let r = match loc {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                16
            }
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; tst X(r as u8), X(r as u8));
    }

    /// Maximum unsigned immediate for `ldr/str X, [base, #imm]` (64-bit).
    /// AArch64 scaled unsigned offset: 12-bit field × 8 = 0..32760.
    const MAX_LDR_STR_UIMM: i32 = 32760;

    /// Emit `str Xsrc, [x29, #offset]`, using x16 as scratch when the
    /// offset exceeds the unsigned-immediate range of the instruction.
    fn emit_str_fp(&mut self, src: u8, offset: i32) {
        if (0..=Self::MAX_LDR_STR_UIMM).contains(&offset) {
            dynasm!(self.mc ; .arch aarch64
                ; str X(src), [x29, offset as u32]
            );
        } else {
            // Large offset: add x16, x29, #offset; str Xsrc, [x16]
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; str X(src), [x16]
            );
        }
    }

    /// Emit `str Dsrc, [x29, #offset]` (float/double), using x16 as scratch
    /// when the offset exceeds the unsigned-immediate range.
    fn emit_str_fp_d(&mut self, src: u8, offset: i32) {
        if (0..=Self::MAX_LDR_STR_UIMM).contains(&offset) {
            dynasm!(self.mc ; .arch aarch64
                ; str D(src), [x29, offset as u32]
            );
        } else {
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; str D(src), [x16]
            );
        }
    }

    /// Emit `ldr Ddst, [x29, #offset]` (float/double), using x16 as scratch
    /// when the offset exceeds the unsigned-immediate range.
    fn emit_ldr_fp_d(&mut self, dst: u8, offset: i32) {
        if (0..=Self::MAX_LDR_STR_UIMM).contains(&offset) {
            dynasm!(self.mc ; .arch aarch64
                ; ldr D(dst), [x29, offset as u32]
            );
        } else {
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; ldr D(dst), [x16]
            );
        }
    }

    /// Emit `ldr Xdst, [x29, #offset]`, using x16 as scratch when the
    /// offset exceeds the unsigned-immediate range of the instruction.
    fn emit_ldr_fp(&mut self, dst: u8, offset: i32) {
        if (0..=Self::MAX_LDR_STR_UIMM).contains(&offset) {
            dynasm!(self.mc ; .arch aarch64
                ; ldr X(dst), [x29, offset as u32]
            );
        } else {
            // Large offset: add x16, x29, #offset; ldr Xdst, [x16]
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; ldr X(dst), [x16]
            );
        }
    }

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp(0, offset);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(0, val);
            }
        }
    }

    /// Emit: load a regalloc Loc into RAX (x64) / X0 (aarch64).
    /// Unlike load_arg_to_rax, this uses the regalloc-determined location
    /// instead of resolve_opref(), so register-carried values are preserved.
    fn emit_load_to_rax(&mut self, loc: Loc) {
        let rax = Loc::Reg(crate::regloc::RegLoc {
            value: 0,
            is_xmm: false,
        });
        match loc {
            Loc::Reg(r) if r.value == 0 && !r.is_xmm => {
                // already in rax/x0
            }
            Loc::Immed(imm) | Loc::ImmedFloat(imm) => {
                self.emit_mov_imm64(0, imm.value);
            }
            _ => self.regalloc_mov(&loc, &rax),
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp(1, offset);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(1, val);
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        self.emit_str_fp(0, offset);
    }

    // assembler.py:543 _call_header — function prologue

    fn setup_input_state(&mut self, inputargs: &[InputArgRc]) {
        // opref_to_slot stores ABSOLUTE jitframe slot indices so that
        // slot_offset(slot) returns the correct byte offset directly.
        // User position `p` maps to absolute slot `p + JITFRAME_FIXED_SIZE`.
        if let Some(ref input_locs) = self.bridge_input_locs {
            let mut max_abs_slot = JITFRAME_FIXED_SIZE;
            for (ia, loc) in inputargs.iter().zip(input_locs.iter()) {
                if let Loc::Frame(floc) = loc {
                    let abs_slot = JITFRAME_FIXED_SIZE + floc.get_position();
                    self.opref_to_slot.insert(ia.opref(), abs_slot);
                    if abs_slot + 1 > max_abs_slot {
                        max_abs_slot = abs_slot + 1;
                    }
                }
            }
            self.next_slot = max_abs_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot
                    .insert(ia.opref(), JITFRAME_FIXED_SIZE + i);
            }
            self.next_slot = JITFRAME_FIXED_SIZE + inputargs.len();
        }
    }

    /// Emit the function prologue.
    /// x64: System V AMD64 ABI — first arg (jf_ptr) in RDI.
    /// aarch64: AAPCS64 — first arg (jf_ptr) in X0.
    ///
    /// Save slots: fp/lr at [sp,#0], x19/x20 at [sp,#16].  Mirrors
    /// aarch64/assembler.py:1117-1122, which iterates
    /// `r.callee_saved_registers = [x19, x20]`.
    ///
    /// aarch64/assembler.py `_call_header_with_stack_check`:
    /// inline SP probe right after the frame pointer is captured,
    /// before `gen_shadowstack_header`. On overflow, return the
    /// incoming jf_ptr in x0 without pushing to the shadow stack so
    /// the JIT glue drains the overflow flag on the way back to the
    /// interpreter.
    ///
    /// Inline probe (aarch64/assembler.py parity):
    /// ```text
    ///   gen_load_int x30, endaddr       ; load endaddr
    ///   LDR  x30, [x30]                  ; x30 = end
    ///   gen_load_int x16, lengthaddr
    ///   LDR  x16, [x16]                  ; x16 = length
    ///   MOV  x17, sp                     ; x17 = current sp (sp can't be
    ///                                    ;   the source of SUB_rr directly)
    ///   SUB  x30, x30, x17               ; x30 = ofs = end - sp
    ///   CMP  x30, x16
    ///   B.LS continue                    ; fast path: ofs <= length
    ///   MOV  x0, sp                      ; arg0 = current sp
    ///   gen_load_int x17, slowpath
    ///   BLR  x17                         ; call pyre_stack_too_big_slowpath
    ///   CBZ  w0, continue                ; slowpath: 0 = OK
    ///   ; fallthrough = real overflow → return x29 as jf_ptr
    /// ```
    fn _call_header(&mut self, inputargs: &[InputArgRc]) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, -(CALL_FRAME_SIZE as i32)]!
            ; stp x19, x20, [sp, #16]   // save callee-saved regs
            // assembler.py:1128-1129: spill the thread-local base the entry
            // received in x1, then take the jitframe out of x0.
            ; str x1, [sp, #SAVED_THREADLOCAL_OFS]
            ; mov x29, x0
        );
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr != 0
            && let Some(addrs) = crate::stack_check_addresses()
        {
            let continue_label = self.mc.new_dynamic_label();
            let exc_value_addr = crate::jit_exc_value_addr() as i64;
            let exc_type_addr = crate::jit_exc_type_addr() as i64;
            // Fast path: load end, subtract sp, compare with length.
            self.emit_mov_imm64(30, addrs.end_adr as i64); // lr holds end addr
            dynasm!(self.mc ; .arch aarch64
                ; ldr x30, [x30]                  // x30 = *end_adr
            );
            self.emit_mov_imm64(16, addrs.length_adr as i64); // ip0 holds length addr
            dynasm!(self.mc ; .arch aarch64
                ; ldr x16, [x16]                  // x16 = *length_adr
                ; mov x17, sp                     // x17 = sp (can't SUB sp directly)
                ; sub x30, x30, x17               // x30 = ofs = end - sp
                ; cmp x30, x16
                ; b.ls =>continue_label           // fast path OK: ofs <= length
                // Slow path: call pyre_stack_too_big_slowpath(sp).
                ; mov x0, sp
            );
            self.emit_mov_imm64(17, addrs.slowpath_addr as i64);
            dynasm!(self.mc ; .arch aarch64
                ; blr x17
                ; cbz w0, =>continue_label        // slowpath says OK
            );
            // aarch64/assembler.py `_build_stack_check_slowpath` jumps
            // into the same propagate-exception path as x86: move
            // pos_exc_value into jf_guard_exc, clear pos_exception, stamp
            // propagate_exception_descr, then return the incoming jf_ptr.
            self.emit_mov_imm64(16, exc_value_addr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x16]
                ; str xzr, [x16]
                ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
            );
            self.emit_mov_imm64(16, exc_type_addr);
            dynasm!(self.mc ; .arch aarch64
                ; str xzr, [x16]
            );
            self.emit_mov_imm64(0, propagate_descr);
            dynasm!(self.mc ; .arch aarch64
                ; str x0, [x29, JF_DESCR_OFS as u32]
                // Overflow fallthrough: return x29 as jf_ptr.
                // Restore the same set `_call_header` saved and
                // `_call_footer` / `gen_func_epilog` restore:
                // `r.callee_saved_registers` (x19, x20) plus fp/lr.
                ; mov x0, x29
                ; ldp x19, x20, [sp, #16]
                ; ldp x29, x30, [sp], CALL_FRAME_SIZE as i32
                ; ret
                ; =>continue_label
            );
        }
        // When addresses are not registered (tests / early startup), no
        // stack check is emitted — aarch64/assembler.py:1094-1095 parity.
        self.gen_shadowstack_header();
        self.setup_input_state(inputargs);
    }

    // assembler.py:2153 _call_footer — function epilogue

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    fn _call_footer(&mut self) {
        self.gen_footer_shadowstack();
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
            ; ldp x19, x20, [sp, #16]   // restore callee-saved regs
            ; ldp x29, x30, [sp], CALL_FRAME_SIZE as i32
            ; ret
        );
    }

    /// aarch64/assembler.py `gen_shadowstack_header` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip1.value, rst)
    ///   self.load_reg(mc, r.x8, r.ip1)   # x8 = *rst = root_stack_top
    ///   mc.gen_load_int(r.ip0.value, 1)
    ///   self.store_reg(mc, r.ip0, r.x8)  # x8[0] = 1 (is_minor marker)
    ///   self.store_reg(mc, r.fp, r.x8, WORD) # x8[WORD] = fp (jf_ptr)
    ///   mc.ADD_ri(r.x8.value, r.x8.value, 2 * WORD)
    ///   self.store_reg(mc, r.x8, r.ip1)  # *rst = x8
    /// ```
    ///
    /// Pushes two words onto the jf shadow stack: the `is_minor` marker
    /// (1) and the current jitframe pointer. Done inline on every JIT
    /// function entry so the collector can walk jf roots without ever
    /// calling into Rust.
    fn gen_shadowstack_header(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        // Register assignment mirrors assembler.py:1426-1436 literally:
        // r.ip1 = x17 holds the rst address; r.ip0 = x16 holds the
        // `1` is_minor marker; r.x8 = x8 holds the loaded top.
        self.emit_mov_imm64(17, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x8, [x17]             // x8 = *rst = root_stack_top
            ; mov x16, 1                 // is_minor marker
            ; str x16, [x8]             // [x8] = 1
            ; str x29, [x8, 8]          // [x8 + WORD] = fp (jf_ptr)
            ; add x8, x8, 16            // x8 += 2*WORD
            ; str x8, [x17]             // *rst = x8
        );
    }

    /// aarch64/assembler.py `gen_footer_shadowstack` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip1, r.ip0)  # ip1 = *rst = top
    ///   mc.SUB_ri(r.ip1.value, r.ip1.value, 2 * WORD)
    ///   self.store_reg(mc, r.ip1, r.ip0) # *rst = ip1
    /// ```
    fn gen_footer_shadowstack(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x16]            // x17 = *rst = top
            ; sub x17, x17, 16          // top -= 2*WORD
            ; str x17, [x16]            // *rst = top
        );
    }

    /// assembler.py:993 push_gcmap.
    fn push_gcmap(&mut self, gcmap: *mut usize) {
        let gcmap_ptr = gcmap as i64;
        self.emit_mov_imm64(16, gcmap_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x16, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// assembler.py:1000 pop_gcmap.
    fn pop_gcmap(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// aarch64/assembler.py `_check_frame_depth` +
    /// `build_frame_realloc_slowpath` (assembler.py:434) — emit a dynamic
    /// JITFRAME depth check at bridge entry.  If the in-flight frame is
    /// shorter than the bridge's required depth, fall into an inlined
    /// realloc slowpath that grows the frame via `dynasm_realloc_frame`.
    ///
    /// Mirrors the x86 port: the upstream backends `BL` a shared
    /// `_frame_realloc_slowpath` stub, but pyre inlines its body so the
    /// gcmap (a compile-time pointer) and the depth (a patchable
    /// immediate) need not be marshalled through the stack.
    ///
    /// The depth immediate is materialized by `emit_mov_imm64_fixed4` as a
    /// fixed 4-instruction `movz/movk` block; `frame_depth_to_patch` records
    /// each block offset (CMP operand + slowpath ARG1) and `patch_stack_checks`
    /// rewrites all four words at finalize once `frame_depth` is final.
    /// Unlike upstream's variable-length `gen_load_int` (which reserves
    /// `get_max_size_of_gen_load_int()` NOPs), this block is always four
    /// words, so no NOP padding is needed.  (The general `emit_mov_imm64`
    /// is variable length and would let the patch overrun the block.)
    fn emit_check_frame_depth(&mut self, gcmap: *mut usize) {
        let frame_len_ofs = (JF_FRAME_OFS + crate::jitframe::LENGTHOFS) as u32;
        let placeholder: i64 = 0xffffff;

        // assembler.py:935 — LDR ip0, [fp, lenofs] — current frame length.
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x29, frame_len_ofs]
        );
        // assembler.py:936-941 — gen_load_int ip1, expected_size (patched).
        let cmp_imm_ofs = self.mc.offset().0;
        self.emit_mov_imm64_fixed4(17, placeholder);
        self.frame_depth_to_patch.push(cmp_imm_ofs);

        // assembler.py:942-944 — CMP ip0, ip1; B.GE skip (fast path:
        // frame length >= required depth → no realloc).
        let continue_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64
            ; cmp x16, x17
            ; b.ge =>continue_label
        );

        // ── inlined build_frame_realloc_slowpath (assembler.py:434-493) ──
        // a) store all registers in the jitframe.
        self.push_all_regs_to_jitframe(&[], true);
        // c) store the gcmap in the jitframe (push_gcmap bakes it as imm).
        self.push_gcmap(gcmap);

        // assembler.py:461 _store_and_reset_exception(None, x19, on_frame=True):
        // jf_guard_exc ← *pos_exc_value (survives the realloc frame copy);
        // x19 ← *pos_exception (callee-saved across the C call); then clear
        // both globals so the helper sees no leftover state.  ip0/ip1 (x16/x17)
        // are scratch and x19 is saved/restored by push/pop_all_regs.
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]                            // x16 = *pos_exc_value
            ; str x16, [x29, JF_GUARD_EXC_OFS as u32]   // jf_guard_exc = excval
        );
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x19, [x16]                            // x19 = *pos_exception
            ; str xzr, [x16]                            // *pos_exception = 0
        );
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x16]                            // *pos_exc_value = 0
        );

        // assembler.py:458 MOV x0 = fp (arg0 = old frame).
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x29);
        // assembler.py:447-449 arg1 = expected_size (depth), patched imm.
        let arg1_imm_ofs = self.mc.offset().0;
        self.emit_mov_imm64_fixed4(1, placeholder);
        self.frame_depth_to_patch.push(arg1_imm_ofs);

        // assembler.py:467 BL realloc_frame.  pyre bakes the C-ABI wrapper
        // address as an immediate and calls through ip1.  `blr` clobbers lr
        // (x30), which is safe here: a bridge reuses the originating loop's
        // stack frame and reloads lr from it at `_call_footer`, treating lr
        // as call-clobbered scratch throughout the body.
        let helper_addr = crate::runner::dynasm_realloc_frame as *const () as i64;
        self.emit_mov_imm64(17, helper_addr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x17
            ; mov x29, x0                               // fp = new jitframe
        );

        // assembler.py:473 _restore_exception(None, x19):
        // *pos_exc_value ← jf_guard_exc (copied into the new frame);
        // *pos_exception ← x19 (preserved across the C call).
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x29, JF_GUARD_EXC_OFS as u32]   // x17 = jf_guard_exc
            ; str x17, [x16]                            // *pos_exc_value = x17
            ; str xzr, [x29, JF_GUARD_EXC_OFS as u32]   // reset jf_guard_exc
        );
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str x19, [x16]                            // *pos_exception = x19
        );

        // assembler.py:476-480 — update the shadow-stack top entry so the
        // GC visitor finds the post-realloc frame on the next minor
        // collection.  x0 was clobbered by _restore_exception, so write fp.
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]            // x16 = *rst = root_stack_top
            ; sub x16, x16, 8          // x16 = top - WORD
            ; str x29, [x16]           // [top - WORD] = fp (new frame)
        );

        // assembler.py:483-487 — reset jf_gcmap to 0 + restore registers.
        self.pop_gcmap();
        self.pop_all_regs_from_jitframe(&[], true);

        dynasm!(self.mc ; .arch aarch64 ; =>continue_label);
    }

    /// aarch64/assembler.py `patch_stack_checks` — rewrite each recorded
    /// `gen_load_int` depth placeholder with the final absolute frame depth
    /// (already includes `JITFRAME_FIXED_SIZE`).  No-op when no check was
    /// emitted (loops).  Takes the patch list by slice so the caller can
    /// drive it after `self.mc` has been consumed by `finalize()`.
    fn patch_stack_checks(framedepth: usize, rawstart: usize, offsets: &[usize]) {
        for &ofs in offsets {
            Self::patch_frame_depth(rawstart + ofs, framedepth);
        }
    }

    /// aarch64/assembler.py:948 `_patch_frame_depth` — overwrite the
    /// 4-instruction `movz/movk` block emitted by `emit_mov_imm64` with the
    /// finalised frame depth.  Unlike x86's flat 32-bit immediate, the
    /// aarch64 depth is materialised as four instruction words, so all four
    /// are regenerated.  The destination register is recovered from the
    /// original block (the CMP site targets x17, the ARG1 site x1) and
    /// preserved.  The patched region needs an explicit icache flush since
    /// `finalize()` made the buffer executable before this rewrite.
    fn patch_frame_depth(adr: usize, allocated_depth: usize) {
        let rd = unsafe { (adr as *const u32).read() } & 0x1F;
        let words = Self::encode_mov_imm64_words(rd, allocated_depth as i64);
        codebuf::with_writable(adr as *mut u8, 16, || {
            let p = adr as *mut u32;
            for (i, w) in words.iter().enumerate() {
                unsafe { p.add(i).write_unaligned(*w) };
            }
        });
        flush_icache(adr as *const u8, 16);
    }

    /// RPython `AbstractCallBuilder.emit`: CALL_ASSEMBLER is a collecting
    /// call, so the caller jitframe must publish the regalloc gcmap before
    /// jumping into the callee and clear it only after reloading a possibly
    /// moved frame pointer.
    fn push_pending_call_gcmap(&mut self) -> bool {
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
            true
        } else {
            false
        }
    }

    fn pop_pending_call_gcmap_after_collect(&mut self, pushed: bool) {
        self.reload_frame_if_necessary();
        if pushed {
            self.pop_gcmap();
        }
    }

    /// aarch64/assembler.py `_reload_frame_if_necessary` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip0, r.ip0)       # ip0 = *rst = root_stack_top
    ///   mc.SUB_ri(r.ip0.value, r.ip0.value, WORD)
    ///   mc.LDR_ri(r.fp.value, r.ip0.value, 0) # fp = *(top - WORD) = jf_ptr
    /// ```
    ///
    /// After a collecting helper call the GC may have moved the
    /// jitframe; the visitor rewrites the shadow-stack top entry in
    /// place during copy, so the current pointer lives at
    /// `*(root_stack_top - WORD)`. Reload fp/x29 from there.
    fn reload_frame_if_necessary(&mut self) {
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]            // ip0 = *rst_addr = root_stack_top
            ; sub x16, x16, 8           // ip0 -= WORD
            ; ldr x29, [x16]            // fp = *(top - WORD) = jf_ptr
        );
        // aarch64/assembler.py `_reload_frame_if_necessary`:
        // after a collecting helper call, re-apply the non-array write
        // barrier fast path on the current jitframe (`is_frame=True`).
        //
        //     wbdescr = self.cpu.gc_ll_descr.write_barrier_descr
        //     if gcrootmap and wbdescr:
        //
        // The `and wbdescr` is load-bearing: a collector that needs no write
        // barrier reports none (`gc.py GcLLDescr_boehm.write_barrier_descr
        // = None`), and there is no barrier to re-apply for it. Calling
        // through regardless is what `emit_write_barrier_fastpath_for_base`
        // asserts against, and it reaches here from every collecting helper —
        // `genop_call_malloc_nursery*` included — not only from a
        // `COND_CALL_GC_WB` the GC rewriter emitted. The x86 counterpart
        // already spells this gate as `if let Some(wb) = wb_descr`.
        if crate::runner::dynasm_write_barrier_descr().is_some() {
            let loc_base = crate::aarch64::registers::FP;
            self.emit_write_barrier_fastpath_for_base(loc_base, false, true, false, None);
        }
    }

    /// aarch64/assembler.py `_push_all_regs_to_jitframe` parity.
    fn push_all_regs_to_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        push_all_regs_to_jitframe_raw(&mut self.mc, ignored_regs, withfloats, false);
    }

    /// aarch64/assembler.py `_pop_all_regs_from_jitframe` parity.
    fn pop_all_regs_from_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        pop_all_regs_from_jitframe_raw(&mut self.mc, ignored_regs, withfloats, false);
    }

    /// `llsupport/assembler.py GuardToken.compute_gcmap`: skip the hole
    /// a virtual leaves (`if arg is None: continue`), mark every remaining
    /// `REF`-typed failarg, narrow nothing else.  A force token is marked like
    /// any other — `resoperation.py FORCE_TOKEN/0/r` returns the jitframe,
    /// itself a moving GC object.
    ///
    /// The skip is carried by the location rather than the type: only a
    /// `Reg` or `Frame` location names a slot, so a fail arg that has neither
    /// contributes no bit.
    fn guard_gcmap_from_faillocs(
        &self,
        fail_arg_types: &[Type],
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) -> *mut usize {
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(faillocs.iter()) {
            if *tp != Type::Ref {
                continue;
            }
            match loc {
                Some(Loc::Reg(r)) => {
                    if let Some(position) = reg_position_in_jitframe(*r) {
                        gcmap_set_bit(gcmap, position);
                    }
                }
                Some(Loc::Frame(f)) => {
                    gcmap_set_bit(gcmap, f.get_position() + JITFRAME_FIXED_SIZE);
                }
                None => {}
                Some(other) => panic!(
                    "guard_gcmap_from_faillocs: a Ref fail argument at {other:?} \
                carries no gcmap bit"
                ),
            }
        }
        gcmap
    }

    /// `aarch64/assembler.py fixup_target_tokens`.
    /// A LABEL assembled inside a bridge is a jump target for later traces exactly like a loop's.
    fn fixup_target_tokens(tokens: Vec<majit_ir::DescrRef>, frame_depth: usize, rawstart: usize) {
        for descr in tokens {
            if let Some(loop_descr) = descr.as_loop_target_descr() {
                // assembler.py:1167-1171 reads the target loop's
                // `frame_info.jfi_frame_depth` at a cross-loop JUMP; publish
                // this loop's full frame depth so a later trace's JUMP can
                // size its frame for this target.  Carries the depth grown by
                // `_assemble` for this loop's own onward JUMP.
                //
                // Store the companion `target_frame_depth` BEFORE the
                // `ll_loop_code` gate (both Release stores): a reader
                // Acquire-loads `ll_loop_code` and only reads
                // `target_frame_depth` once the gate is non-zero, so the depth
                // must become visible first — otherwise the reader pairs the
                // new code pointer with a stale 0 depth and bypasses the
                // frame-capacity check (descr.rs set_dispatch_target ordering
                // contract).  Dynasm ignores `label_block_id` (descr.rs —
                // it bakes the LABEL address straight into `ll_loop_code`), so
                // that companion is not published here.
                let old = loop_descr.ll_loop_code();
                let new = old + rawstart;
                loop_descr.set_target_frame_depth(frame_depth);
                loop_descr.set_ll_loop_code(new);
                if majit_ir::debug::have_debug_prints() {
                    majit_ir::debug::debug_print(&format!(
                        "[dynasm] fixup_target_tokens: ll_loop_code {old} -> {new:#x}"
                    ));
                }
            }
        }
    }

    fn check_unrelocated_jump_target(&self) -> Result<(), BackendError> {
        let Some((target, descr)) = self.unrelocated_jump_target else {
            return Ok(());
        };
        Err(BackendError::CompilationFailed(format!(
            "cross-buffer JUMP target {target:#x} for descr {descr:#x} is not a relocated executable address"
        )))
    }

    // assembler.py:501 assemble_loop

    /// assembler.py:501 assemble_loop: compile a loop trace.
    ///
    /// Returns compiled code with fail descriptors and entry point.
    pub fn assemble_loop(mut self) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp.get()).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        self._assemble(true)?;
        self.check_unrelocated_jump_target()?;

        // regalloc sets fail_arg_locs in append_guard_token_with_faillocs,
        // which stamps `rd_locs` from them right there
        // (`llsupport/assembler.py:279`), so no post-regalloc fixup pass runs.

        // assembler.py:553 write_pending_failure_recoveries
        let stub_offsets = self.write_pending_failure_recoveries();
        self.check_guard_reach()?;
        // `materialize_loop` calls `datablockwrapper.done()` before the
        // code block is copied out. The constants are not in that stream.
        let data_blocks = self.datablockwrapper.done();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let tokens = std::mem::take(&mut self.compiled_target_tokens);
        let frame_depth = self.frame_depth;
        let mut buffer = codebuf::finalize_writable(self.mc, &self.asm_memory_manager)?;

        // assembler.py:849 patch_pending_failure_recoveries
        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);
        let invalidate_positions = Self::collect_invalidate_positions(rawstart, &stub_offsets);

        // assembler.py:556 patch_stack_checks — rewrite any
        // `_check_frame_depth` depth placeholders with the loop's final
        // absolute frame depth.  No-op for loops (the list is empty).
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        // Write resolved entry address for self-recursive CALL_ASSEMBLER
        // trampoline. The JIT code loads from this pointer at runtime.
        unsafe { *self.self_entry_addr_ptr = rawstart + entry.0 };

        Self::fixup_target_tokens(tokens, frame_depth, rawstart);
        buffer.make_executable()?;

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: std::sync::Arc::new(self.fail_descrs),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: None,
            invalidate_positions,
            data_blocks,
        })
    }

    fn resolve_call_assembler_target_addr(
        &self,
        descr: Option<&majit_ir::DescrRef>,
    ) -> CallAssemblerTargetAddr {
        if let Some(token) = descr
            .and_then(|d| d.as_loop_token_descr())
            .and_then(|ltd| ltd.token_handle_any())
            .and_then(|any| any.downcast_ref::<Arc<JitCellToken>>())
        {
            let descr_addr = token.ll_function_addr();
            if descr_addr != 0 {
                return CallAssemblerTargetAddr {
                    immediate: Some(descr_addr),
                };
            }
            return CallAssemblerTargetAddr { immediate: None };
        }

        let _ = descr;
        CallAssemblerTargetAddr { immediate: None }
    }

    /// llsupport/assembler.py rebuild_faillocs_from_descr — reconstruct
    /// the locations of bridge inputargs from the guard's recovery layout.
    ///
    /// patch_jump_for_descr overwrites the recovery stub with a direct
    /// jump to the bridge, so the register-save subroutine never runs.
    /// The bridge sees live registers exactly as they were at guard time.
    /// Return Reg locs for register positions, matching RPython.
    pub fn rebuild_faillocs_from_descr(
        descr: &dyn majit_ir::FailDescr,
        inputargs: &[InputArgRc],
    ) -> Vec<Loc> {
        let mut locs = Vec::new();
        let gpr_regs = all_gen_regs();
        let float_regs = all_float_regs();
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        let mut input_i = 0usize;
        for &pos in descr.rd_locs() {
            if pos == 0xFFFF {
                continue;
            }
            let pos = pos as usize;
            if pos < gpr_regs.len() {
                // llsupport/assembler.py:211 — GPR: return register location
                locs.push(Loc::Reg(gpr_regs[pos]));
            } else if pos < gpr_regs.len() + float_regs.len() {
                // llsupport/assembler.py:213 — FPR: return float register
                locs.push(Loc::Reg(float_regs[pos - gpr_regs.len()]));
            } else {
                // llsupport/assembler.py:217 — frame slot
                let slot = pos - JITFRAME_FIXED_SIZE;
                let tp = inputargs
                    .get(input_i)
                    .map(|ia| ia.tp.get())
                    .unwrap_or(Type::Int);
                locs.push(Loc::Frame(crate::regloc::FrameLoc::new(
                    slot,
                    crate::regalloc::get_ebp_ofs(base_ofs, slot),
                    tp == Type::Float,
                )));
            }
            input_i += 1;
        }
        locs
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        fail_descr: &dyn FailDescr,
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] assemble_bridge: input_types={:?} arglocs={:?}",
                self.inputargs
                    .iter()
                    .map(|ia| ia.tp.get())
                    .collect::<Vec<_>>(),
                arglocs
            );
        }
        self.input_types = self.inputargs.iter().map(|ia| ia.tp.get()).collect();
        self.bridge_input_locs = if arglocs.is_empty() {
            None
        } else {
            Some(arglocs.to_vec())
        };

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(false)?;
        self.check_unrelocated_jump_target()?;
        let stub_offsets = self.write_pending_failure_recoveries();
        self.check_guard_reach()?;
        let data_blocks = self.datablockwrapper.done();

        let tokens = std::mem::take(&mut self.compiled_target_tokens);
        let frame_depth = self.frame_depth;
        let mut buffer = codebuf::finalize_writable(self.mc, &self.asm_memory_manager)?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);
        let invalidate_positions = Self::collect_invalidate_positions(rawstart, &stub_offsets);

        // assembler.py:556 patch_stack_checks — rewrite the bridge's
        // `_check_frame_depth` depth placeholder with the final absolute
        // frame depth (max of loop/bridge), now known post-finalize.
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);
        Self::fixup_target_tokens(tokens, frame_depth, rawstart);
        buffer.make_executable()?;

        if crate::majit_dump_enabled() {
            let code = unsafe { std::slice::from_raw_parts(rawstart as *const u8, buffer.len()) };
            eprintln!(
                "[dynasm] BRIDGE CODE DUMP ({} bytes at {:#x}, entry +{:?}):",
                code.len(),
                rawstart,
                entry
            );
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: std::sync::Arc::new(self.fail_descrs),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: Some((fail_descr.trace_id(), fail_descr.fail_index_per_trace())),
            invalidate_positions,
            data_blocks,
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(&mut self, emit_prologue: bool) -> Result<(), BackendError> {
        let inputargs: &'a [InputArgRc] = self.inputargs;
        let ops: &'a [OpRc] = self.operations;
        self.short_guard_branch_offsets.clear();
        self.guard_reach_violations.clear();
        self.unrelocated_jump_target = None;
        // A guard's `b.cond` has to reach a stub that is written after the
        // whole body, so the budget is the body plus every stub — not the body
        // alone.  Each stub is bounded by its fixed part, so charge
        // `MAX_BYTES_PER_GUARD_STUB` separately from `MAX_BYTES_PER_OP`.
        let guard_count = ops.iter().filter(|op| op.opcode.is_guard()).count();
        self.long_guard_branch = ops
            .len()
            .saturating_mul(MAX_BYTES_PER_OP)
            .saturating_add(guard_count.saturating_mul(MAX_BYTES_PER_GUARD_STUB))
            >= BCOND_FORWARD_RANGE;
        if emit_prologue {
            self._call_header(inputargs);
        } else {
            self.setup_input_state(inputargs);
        }
        // input_slot_depth is an ABSOLUTE jitframe slot count (already
        // offset by JITFRAME_FIXED_SIZE via setup_input_state).
        let input_slot_depth = self.next_slot;

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        // RegAlloc keeps the raw `i64` value map; project it from the
        // typed pool at this boundary (each Const carries its own type).
        let ra_constants: indexmap::IndexMap<u32, i64> = self
            .constants
            .iter()
            .map(|(&k, c)| (k, c.as_raw_i64()))
            .collect();
        let mut ra = RegAlloc::new(ra_constants, inputargs, ops);
        if let Some(ref arglocs) = self.bridge_input_locs {
            ra.prepare_bridge(arglocs);
        } else {
            ra.prepare_loop();
        }
        // assembler.py:647 — bridges emit `_check_frame_depth` between
        // `prepare_bridge` and the body so the JIT can grow the in-flight
        // JITFRAME if the bridge's frame_depth exceeds the loop's
        // allocation.  Loops skip this (assembler.py:544 uses
        // `_check_frame_depth_debug`, a no-op outside DEBUG_FRAME_DEPTH).
        if self.bridge_input_locs.is_some() {
            let gcmap = ra.get_gcmap(&[], false);
            self.emit_check_frame_depth(gcmap);
        }
        if crate::majit_ops_log_enabled() {
            eprintln!("[dynasm] _assemble: {} ops", ops.len());
        }
        // assembler.py:374 walk_operations — the regalloc drives this
        // assembler op by op (`regalloc_perform*`). The VFP manager reaches
        // `assembler.datablockwrapper` during it, so the wrapper leaves
        // `self` for the walk instead of being reachable two ways.
        self.fail_descrs = FailDescrStore::with_capacity(fail_cell_capacity(ops));
        self.fail_index = 0;
        let mut datablockwrapper = std::mem::replace(
            &mut self.datablockwrapper,
            MachineDataBlockWrapper::new(Arc::clone(&self.asm_memory_manager)),
        );
        let walked = ra.walk_operations_with_datablockwrapper(&mut datablockwrapper, self);
        self.datablockwrapper = datablockwrapper;
        walked?;
        // ra.get_final_frame_depth() returns a USER-position count; convert
        // to absolute by adding JITFRAME_FIXED_SIZE before comparing.
        let frame_slot_depth =
            input_slot_depth.max(JITFRAME_FIXED_SIZE + ra.get_final_frame_depth());
        self.frame_depth = self.frame_depth.max(frame_slot_depth);

        // Sync regalloc frame positions to opref_to_slot, the map `resolve_opref`
        // reads, once the walk is over. No live emitter reaches it any more: every remaining consumer
        // (`load_arg_to_rax` / `load_arg_to_rcx` / `resolve_const_or` and the
        // genops that call them) is `#[allow(dead_code)]`, kept for the upstream
        // method boundary. The map cannot express what the regalloc actually
        // decides — `before_call` leaves a value bound to a callee-saved member
        // of the allocation pool in its register, with no frame slot at all — so
        // an emitter that needs an operand location must take it from `arglocs`.
        // opref_to_slot stores ABSOLUTE jitframe slots (user position +
        // JITFRAME_FIXED_SIZE) so slot_offset(slot) gives the correct byte
        // offset without further adjustment.
        for (position, iarg) in inputargs.iter().enumerate() {
            self.opref_to_slot
                .insert(iarg.opref(), JITFRAME_FIXED_SIZE + position);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        visit_boxes_with_frame_loc(inputargs, ops, &ra.fm, |opref, floc| {
            self.opref_to_slot
                .insert(opref, JITFRAME_FIXED_SIZE + floc.get_position());
        });
        // frame_slot_depth is already absolute (see calculation above).
        self.next_slot = frame_slot_depth;

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                self.fail_index
            );
        }
        // Closes the last op's span, the way `--end of the loop--` closes a
        // `jit-log-opt` listing.
        if crate::majit_dump_enabled() {
            eprintln!("[dynasm] @{:#06x} end of ops", self.mc.offset().0);
        }

        // assembler.py:1167-1171 `_assemble`: grow the frame to fit a
        // cross-loop JUMP target.  The closing `br` jumps into the target
        // loop's body, which can use deeper frame slots than this trace; for
        // a bridge the prologue `_check_frame_depth` then reallocs the live
        // JITFRAME to this grown depth on entry.  `target_frame_depth` is
        // full-width (includes JITFRAME_FIXED_SIZE), matching `frame_depth`,
        // so no adjustment is needed.  Zero (no external JUMP) is a no-op.
        self.frame_depth = self.frame_depth.max(self.jump_target_frame_depth);

        Ok(())
    }

    /// assembler.py `asm_operations[opnum]` — emit code for a non-guard op
    /// from its regalloc locations (`regalloc_perform`).
    fn asm_operations(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        fail_index: u32,
        ops: &[OpRc],
    ) {
        // Only an ADJACENT comparison/guard pair may share NZCV; anything
        // emitted in between invalidates the record (the comparison arms
        // below re-arm it).
        self.pending_cmp_cc = None;
        match op.opcode {
            OpCode::IntAddOvf => {
                // RPython aarch64/opassembler.py int_add_impl parity —
                // 3-operand `ADDS Rd, Rn, Rm` (s=1 sets flags for GUARD_NO_OVERFLOW).
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    if let Some(im) = Self::addsub_imm12(src) {
                        let l = self.load_loc_to_reg(lhs, 17);
                        dynasm!(self.mc ; .arch aarch64 ; adds X(dst.value), XSP(l), im as u32);
                    } else {
                        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                        dynasm!(self.mc ; .arch aarch64
                            ; adds X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    }
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntSubOvf => {
                // aarch64/opassembler.py int_sub_impl: `SUBS Rd, Rn, Rm`.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    if let Some(im) = Self::addsub_imm12(src) {
                        let l = self.load_loc_to_reg(lhs, 17);
                        dynasm!(self.mc ; .arch aarch64 ; subs X(dst.value), XSP(l), im as u32);
                    } else {
                        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                        dynasm!(self.mc ; .arch aarch64
                            ; subs X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    }
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntMulOvf => {
                // `opassembler.py emit_comp_op_int_mul_ovf`: the product
                // overflowed iff the high half differs from the low half's
                // sign extension.  `CMP_rr_shifted(ip0, res, 63)` is
                // `cmp Xn, Xm, asr #63`, which does the shift as part of the
                // compare rather than through a scratch register.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                    dynasm!(self.mc ; .arch aarch64
                        ; smulh x15, X(lhs_reg as u8), X(src_reg as u8)
                        ; mul X(dst.value), X(lhs_reg as u8), X(src_reg as u8)
                        ; cmp x15, X(dst.value), asr #63
                    );
                    self.guard_success_cc = Some(CC_E);
                }
            }
            // ── Integer binary (aarch64 3-operand form) ──
            // RPython parity: aarch64/opassembler.py `int_sub_impl` emits
            // `SUB res, l0, l1` with three distinct locations — NOT the
            // x86 `SUB dst, dst, src` 2-operand pattern. The regalloc may
            // pick `res != arglocs[0]` (see `consider_int_sub` LEA path),
            // so we must read arglocs[0] as the true LHS.
            OpCode::IntAdd
            | OpCode::IntSub
            | OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    self.emit_binop_3op(op.opcode, dst.value, lhs, src);
                }
            }
            // ── Unary integer ──
            OpCode::IntNeg => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64 ; neg X(dst.value), X(src_reg as u8));
                }
            }
            OpCode::IntInvert => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64 ; mvn X(dst.value), X(src_reg as u8));
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(shift_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64: shifts are true 3-operand forms (`lsl/asr/lsr
                    // Rd, Rn, Rm`). The regalloc may pick `result_loc !=
                    // arglocs[0]`, so we must read the real LHS from arg0,
                    // not from the destination register.
                    let lhs_reg = self.load_loc_to_reg(lhs, 17);
                    let sr = self.load_loc_to_reg(shift_loc, 16);
                    match op.opcode {
                        OpCode::IntLshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsl X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::IntRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; asr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::UintRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        _ => {}
                    }
                }
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                // Only the right-hand operand has an immediate form, so a
                // comparison whose constant sits on the left materializes it
                // into a scratch register first. Swapping the operands folds
                // it back into the `cmp` — but a comparison is not symmetric,
                // so the swap has to move into the condition, which is what
                // `resoperation.py`'s reflex table answers. `prepare_int_ri`
                // already swaps a leading immediate for the symmetric
                // arithmetic ops, where no condition follows the operands.
                let mut opcode = op.opcode;
                if arglocs.len() >= 2 {
                    let (cmp0, cmp1) = if matches!(arglocs[0], Loc::Immed(_) | Loc::ImmedFloat(_))
                        && !matches!(arglocs[1], Loc::Immed(_) | Loc::ImmedFloat(_))
                        && let Some(reflexed) = opcode.bool_reflex()
                    {
                        opcode = reflexed;
                        (&arglocs[1], &arglocs[0])
                    } else {
                        (&arglocs[0], &arglocs[1])
                    };
                    self.emit_cmp_loc_loc(cmp0, cmp1);
                }
                let cc = Self::opcode_to_cc(opcode);
                self.flush_cc(cc, result_loc);
            }
            OpCode::IntIsTrue => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_NE, result_loc);
                }
            }
            OpCode::IntIsZero => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_E, result_loc);
                }
            }
            OpCode::UintMulHigh => {
                if let Some(Loc::Reg(dst)) = result_loc
                    && arglocs.len() >= 2
                {
                    let lhs = match arglocs[0] {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                &arglocs[0],
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    let rhs = match arglocs[1] {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                &arglocs[1],
                                &Loc::Reg(crate::regloc::RegLoc::new(17, false)),
                            );
                            17
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; umulh X(dst.value), X(lhs), X(rhs));
                }
            }
            OpCode::IntForceGeZero => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64
                        ; cmp X(src_reg as u8), xzr
                        ; csel X(dst.value), X(src_reg as u8), xzr, ge
                    );
                }
            }
            OpCode::IntSignext => {
                // aarch64/opassembler.py `emit_op_int_signext`: SXTB / SXTH / SXTW
                // for numbytes 1 / 2 / 4. The source is a register; a frame slot
                // is loaded the same way neighbouring arms read an argloc.
                let (Some(argloc), Some(numbytes_loc), Some(Loc::Reg(res))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                else {
                    panic!(
                        "int_signext: expected [arg, numbytes] and a register result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                let (Loc::Immed(numbytes) | Loc::ImmedFloat(numbytes)) = numbytes_loc else {
                    panic!("int_signext: numbytes loc must be an immediate, got {numbytes_loc:?}");
                };
                let src = self.load_loc_to_reg(argloc, 16);
                let dst = res.value;
                match numbytes.value {
                    1 => dynasm!(self.mc ; .arch aarch64 ; sxtb X(dst), W(src)),
                    2 => dynasm!(self.mc ; .arch aarch64 ; sxth X(dst), W(src)),
                    4 => dynasm!(self.mc ; .arch aarch64 ; sxtw X(dst), W(src)),
                    _ => panic!("bad number of bytes"),
                }
            }
            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                if let (Some(Loc::Reg(dst)), Some(lhs_loc), Some(rhs_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64 float ops are true 3-operand forms
                    // (`fadd/fsub/fmul/fdiv Dd, Dn, Dm`). The regalloc may
                    // choose `result_loc != arglocs[0]`, so we must read the
                    // actual lhs from arg0 rather than assuming it already
                    // lives in the destination register.
                    let lhs_reg = match lhs_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(lhs_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    let src_reg = if let Loc::Reg(s) = rhs_loc {
                        *s
                    } else {
                        // Immed or Frame — load to scratch XMM (d14/xmm14)
                        let scratch = crate::regloc::RegLoc::new(14, true);
                        self.regalloc_mov(rhs_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch aarch64 ; fadd D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch aarch64 ; fsub D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch aarch64 ; fmul D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch aarch64 ; fdiv D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg | OpCode::FloatAbs => {
                if let (Some(Loc::Reg(dst)), Some(src_loc)) = (result_loc, arglocs.first()) {
                    // aarch64 fneg/fabs are 2-operand `fneg/fabs Dd, Dn`.
                    // When regalloc chooses `dst != arg0`, reading `Dd` as
                    // the source consumes a stale destination register
                    // instead of the real input, so we must read from
                    // `arglocs[0]` like the float-binop path above.
                    let src_reg = match src_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(src_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    match op.opcode {
                        OpCode::FloatNeg => {
                            dynasm!(self.mc ; .arch aarch64 ; fneg D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatAbs => {
                            dynasm!(self.mc ; .arch aarch64 ; fabs D(dst.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                if let (Some(a_loc), Some(b_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let b_reg = match b_loc {
                        Loc::Reg(b) => Some(*b),
                        _ => None,
                    };
                    let a = if let Loc::Reg(a) = a_loc {
                        *a
                    } else {
                        let scratch = if b_reg.is_some_and(|b| b.value == 15 && b.is_xmm) {
                            crate::regloc::RegLoc::new(14, true)
                        } else {
                            crate::regloc::RegLoc::new(15, true)
                        };
                        self.regalloc_mov(a_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    let b = if let Loc::Reg(b) = b_loc {
                        *b
                    } else {
                        let scratch = if a.value == 15 && a.is_xmm {
                            crate::regloc::RegLoc::new(14, true)
                        } else {
                            crate::regloc::RegLoc::new(15, true)
                        };
                        self.regalloc_mov(b_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    dynasm!(self.mc ; .arch aarch64 ; fcmp D(a.value), D(b.value));
                    // `opassembler.py:138 emit_comp_op_float_*` returns the
                    // condition instead of materialising a boolean, so an
                    // adjacent guard branches straight off `fcmp`'s NZCV.
                    let cc = Self::float_opcode_to_cc(op.opcode);
                    self.flush_cc(cc, result_loc);
                }
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let sr = match src {
                        Loc::Reg(s) => s.value,
                        _ => {
                            self.regalloc_mov(
                                src,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; scvtf D(dst.value), X(sr));
                }
            }
            OpCode::CastFloatToInt => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    dynasm!(self.mc ; .arch aarch64 ; fcvtzs X(dst.value), D(src.value));
                }
            }
            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::CastOpaquePtr
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // `opassembler.py emit_op_load_from_gc_table`: result loc only;
            // the table index is `op.getarg(0).getint()`. Load the
            // reference constant from the slot reserved at the start of
            // this code block (`reserve_gcref_table`), PC-relative. The
            // gc_table root walker forwards the slot in place, so each load
            // observes the relocated object.
            OpCode::LoadFromGcTable => {
                let Some(Loc::Reg(dst)) = result_loc else {
                    panic!(
                        "LoadFromGcTable expects a register result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                let index =
                    op.arg(0)
                        .to_opref()
                        .const_int_value()
                        .expect("LoadFromGcTable index is ConstInt") as usize;
                let slot = *self
                    .gcref_table
                    .get(index)
                    .expect("LoadFromGcTable index inside the reserved gcref table");
                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), =>slot);
            }
            // `opassembler.py:269-270 emit_op_cast_ptr_to_int =
            // _genop_same_as` / `emit_op_cast_int_to_ptr = _genop_same_as`.
            // PyPy's aarch64 backend treats both casts as plain `mov` —
            // the AddressAsInt low-bit tag is a `blackhole.py bhimpl_cast_ptr_to_int`
            // interpreter-side invariant, not a backend codegen step.
            // See x86 sibling for the full rationale.
            OpCode::CastPtrToInt | OpCode::CastIntToPtr => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // ── Memory loads: getfield pattern ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                    let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                    // `ldar` has a base register and no offset. x16 is ip0.
                    // ArraylenGc/Strlen/Unicodelen share this emitter; their
                    // descrs never carry the Acquire bit.
                    let acquire = !dst.is_xmm
                        && op
                            .with_field_descr(|fd| fd.load_is_acquire())
                            .unwrap_or(false);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), ofs as u32]);
                    } else if acquire {
                        // `ldar` addresses a base register, not base+offset.
                        dynasm!(self.mc ; .arch aarch64 ; mov x16, X(base.value));
                        if ofs != 0 {
                            if (0..4096).contains(&ofs) {
                                dynasm!(self.mc ; .arch aarch64 ; add x16, x16, ofs as u32);
                            } else {
                                self.emit_mov_imm64(17, ofs as i64);
                                dynasm!(self.mc ; .arch aarch64 ; add x16, x16, x17);
                            }
                        }
                        // Size 4 matches the plain `ldrsw` path.
                        self.emit_ldar_from_x16(dst, field_size, field_size == 4);
                    } else {
                        match field_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), ofs as u32]);
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), ofs as u32]);
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), ofs as u32]);
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), ofs as u32]);
                            }
                        }
                    }
                }
            }
            // ── Memory loads: getarrayitem pattern ──
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                if let (Some(base_loc), Some(index_loc), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    let (base_size, item_size, signed) = op
                        .with_array_descr(|ad| {
                            (
                                ad.base_size() as i32,
                                ad.item_size() as i32,
                                op.opcode.result_type() == Type::Int && ad.is_item_signed(),
                            )
                        })
                        .unwrap_or((0, 8, false));
                    let index_reg = match index_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                index_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    if item_size != 1 {
                        self.emit_mov_imm64(17, item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul x16, X(index_reg), x17);
                    } else if index_reg != 16 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x16, X(index_reg));
                    }
                    if base_size != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; add x16, x16, base_size as u32);
                    }
                    // Resolve the base into a register. A Loc::Immed base
                    // (constant array pointer from the green-pc inline dispatch
                    // reading `program[const]`) previously failed the Loc::Reg
                    // pattern, so the address computation and load were skipped
                    // and dst kept a stale value (mirrors the GcLoad
                    // immediate-base handling below). x17 is dead here: the
                    // item_size multiply above is its only other use.
                    let base_reg = match base_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                base_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(17, false)),
                            );
                            17
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; add x16, X(base_reg), x16);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [x16]);
                    } else {
                        match item_size {
                            1 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsb X(dst.value), [x16])
                            }
                            1 => dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [x16]),
                            2 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsh X(dst.value), [x16])
                            }
                            2 => dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [x16]),
                            4 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [x16])
                            }
                            4 => dynasm!(self.mc ; .arch aarch64 ; ldr W(dst.value), [x16]),
                            _ => dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]),
                        }
                    }
                }
            }
            // ── Memory stores: setarrayitem pattern ──
            // `rewrite.py transform_to_gc_load` lowers SETARRAYITEM_GC to
            // GC_STORE_INDEXED (after `handle_write_barrier_setarrayitem`)
            // for every compiled loop, so the backend has no handler for it.
            OpCode::SetarrayitemGc => {
                panic!("dynasm: SetarrayitemGc must have been lowered by rewrite_ops_for_gc");
            }
            OpCode::SetarrayitemRaw => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let (base_size, item_size) = op
                        .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
                        .unwrap_or((0, 8));
                    let index_reg = 16u8;
                    let value_reg = 17u8;

                    self.regalloc_mov(
                        index_loc,
                        &Loc::Reg(crate::regloc::RegLoc::new(index_reg, false)),
                    );
                    if item_size != 1 {
                        self.emit_mov_imm64(value_reg.into(), item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul X(index_reg), X(index_reg), X(value_reg));
                    }
                    if base_size != 0 {
                        self.emit_mov_imm64(value_reg.into(), base_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(index_reg), X(value_reg));
                    }
                    dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(base.value), X(index_reg));

                    match value_loc {
                        Loc::Reg(val) if val.is_xmm => {
                            dynasm!(self.mc ; .arch aarch64 ; str D(val.value), [X(index_reg)]);
                        }
                        Loc::Reg(val) => match item_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; strb W(val.value), [X(index_reg)])
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; strh W(val.value), [X(index_reg)])
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; str W(val.value), [X(index_reg)])
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(index_reg)])
                            }
                        },
                        _ => {
                            self.regalloc_mov(
                                value_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(value_reg, false)),
                            );
                            match item_size {
                                1 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strb W(value_reg), [X(index_reg)])
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strh W(value_reg), [X(index_reg)])
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch aarch64 ; str W(value_reg), [X(index_reg)])
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch aarch64 ; str X(value_reg), [X(index_reg)])
                                }
                            }
                        }
                    }
                }
            }
            // ── Memory stores: opassembler.rs emit_op_setfield_regalloc ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                // `rewrite.rs transform_to_gc_load` lowers both opcodes to
                // GC_STORE / GC_STORE_INDEXED through `emit_gc_store_or_indexed`
                // before the regalloc ever sees them, and `prepare_op_setfield`
                // force-allocates a register for every non-constant base, so the
                // only shape that could reach the else arm is a constant base on
                // an op the rewriter did not consume.  Emitting it from
                // `resolve_opref` would read whichever slot the lifetime's
                // end-of-allocation `current_frame_loc` happens to name and
                // clobber x0/x1 behind the regalloc's back; decline instead.
                let [base_loc, val_loc] = arglocs else {
                    panic!(
                        "{:?} expects two regalloc locations, got {arglocs:?}",
                        op.opcode,
                    );
                };
                let Loc::Reg(base) = base_loc else {
                    panic!(
                        "{:?} reached the backend with a non-register base {base_loc:?} \
                         — `rewrite.rs transform_to_gc_load` must have consumed it",
                        op.opcode,
                    );
                };
                let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                self.emit_op_setfield_regalloc(base, val_loc, ofs, field_size);
            }
            // ── aarch64/opassembler.py _emit_op_gc_load ──
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)]
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let Some(ofs_loc) = arglocs.get(1) {
                    // aarch64/opassembler.py:371-374
                    let dst = match arglocs.get(2) {
                        Some(Loc::Reg(r)) => r,
                        _ => match result_loc {
                            Some(Loc::Reg(r)) => r,
                            _ => return,
                        },
                    };
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => op
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    let acquire = matches!(op.opcode, OpCode::RawLoadI)
                        && !dst.is_xmm
                        && op
                            .with_array_descr(|ad| ad.load_is_acquire())
                            .unwrap_or(false);
                    match arglocs.first() {
                        Some(Loc::Reg(base)) => {
                            if acquire {
                                self.emit_acquire_int_load(
                                    base,
                                    ofs_loc,
                                    dst,
                                    nsize.unsigned_abs() as usize,
                                    nsize < 0,
                                );
                            } else {
                                self.emit_op_gcload_regalloc(base, ofs_loc, dst, nsize);
                            }
                        }
                        Some(Loc::Immed(base_i) | Loc::ImmedFloat(base_i)) => {
                            // `AssemblerARM64.load`: `gen_load_int` into ip0.
                            // A PC-relative literal cannot reach a pool past
                            // the recovery stubs once the trace exceeds 1 MiB.
                            self.emit_mov_imm64(16, base_i.value);
                            let base = RegLoc {
                                value: 16,
                                is_xmm: false,
                            };
                            if acquire {
                                self.emit_acquire_int_load(
                                    &base,
                                    ofs_loc,
                                    dst,
                                    nsize.unsigned_abs() as usize,
                                    nsize < 0,
                                );
                            } else {
                                self.emit_op_gcload_regalloc(&base, ofs_loc, dst, nsize);
                            }
                        }
                        None => {}
                        Some(other) => panic!(
                            "GC_LOAD_INDEXED: unhandled base {other:?} — no load is emitted \
                        and the destination keeps its previous value"
                        ),
                    }
                }
            }
            // ── aarch64/opassembler.py emit_op_gc_store ──
            // `value_loc, base_loc, ofs_loc, size_loc = arglocs`.
            // regalloc.rs's `consider_gc_store` (mirroring RPython
            // aarch64/regalloc.py prepare_op_gc_store) always emits a
            // 4-tuple `[Reg|Immed(value), Reg(base), Reg|Immed(ofs),
            // Immed(size)]`; value_loc may be Immed when the source is a
            // Const (llsupport/regalloc.py `return_constant`), and
            // the emitter moves it to x16 before the store. Everything
            // else is tight and panics now so regalloc regressions can't
            // hide behind silent defaults.
            OpCode::GcStore | OpCode::RawStore => {
                let (value_loc, base_loc, ofs_loc, size_loc) = match arglocs {
                    [v, b, o, s] => (v, b, o, s),
                    _ => panic!(
                        "GcStore arglocs must be [value, base, ofs, size] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcStore base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                // aarch64/opassembler.py:367: scale = get_scale(size_loc.value)
                let size = match size_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value.unsigned_abs() as usize,
                    other => panic!(
                        "GcStore size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let val_reg = match value_loc {
                    Loc::Reg(r) => *r,
                    // An immediate, including a singlefloat bit pattern, is
                    // an integer store. `get_type_flag` banks SingleFloat as
                    // FLAG_UNSIGNED; only `lltype.Float` is an 8-byte xmm.
                    Loc::Immed(i) | Loc::ImmedFloat(i) => {
                        self.emit_mov_imm64(16, i.value);
                        crate::regloc::RegLoc::new(16, false)
                    }
                    Loc::ConstFloat(c) => {
                        self.load_imm_float(14, *c);
                        crate::regloc::RegLoc::new(14, true)
                    }
                    other => {
                        panic!("GcStore value_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                    }
                };
                if crate::majit_log_enabled()
                    && let Loc::Immed(i) | Loc::ImmedFloat(i) = ofs_loc
                {
                    let input0_ofs = Self::slot_offset(JITFRAME_FIXED_SIZE);
                    let input1_ofs = Self::slot_offset(JITFRAME_FIXED_SIZE + 1);
                    if i.value as i32 == input0_ofs || i.value as i32 == input1_ofs {
                        eprintln!(
                            "[dynasm][gcstore-input] ofs={} value_loc={:?} base_loc={:?} size={}",
                            i.value, value_loc, base_loc, size
                        );
                    }
                }
                // aarch64/opassembler.py: self._write_to_mem(value_loc, base_loc, ofs_loc, scale)
                self.emit_op_gcstore_regalloc(base, ofs_loc, &val_reg, size);
            }
            // ── aarch64/opassembler.py _emit_op_gc_load_indexed ──
            // arglocs = [res_loc, base_loc, index_loc, imm(nsize), imm(ofs)]
            // per `consider_gc_load_indexed` in regalloc.rs.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                if let (Some(Loc::Reg(res)), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => op
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    // aarch64/opassembler.py:402 keeps `ofs_loc.value` at
                    // the Signed word width; `_try_use_older_box` (rewrite.
                    // py:383-387) may fold large displacements into the
                    // offset immediate so the emitter handles the full
                    // range via the check_imm_arg / materialize fallback.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => 0,
                    };
                    self.emit_op_gcload_indexed_regalloc(base, index, res, ofs, nsize);
                }
            }
            // ── aarch64/opassembler.py emit_op_gc_store_indexed ──
            // arglocs = [value_loc, base_loc, index_loc, imm(size), imm(ofs)]
            OpCode::GcStoreIndexed => {
                if let (Some(value_loc), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let size = match arglocs.get(3) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value.unsigned_abs() as usize,
                        _ => 8,
                    };
                    // aarch64/opassembler.py:385-392 keeps `ofs_loc.value`
                    // at the Signed word width; large displacements folded
                    // in via `_try_use_older_box` must not be truncated.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => 0,
                    };
                    // `emit_op_gc_store_indexed` then `_write_to_mem`: upstream
                    // already has `value_loc` in a VFP / core register before it
                    // overwrites `r.ip0` with `index + ofs`. This emitter still
                    // materializes `ConstFloatLoc` here (`load_imm_float` →
                    // `AssemblerARM64.load` into `r.ip0`), so the value must
                    // land in d14 before `ADD_ri(ip0, index, ofs)`.
                    let val_reg = match value_loc {
                        Loc::Reg(r) => *r,
                        Loc::Immed(i) | Loc::ImmedFloat(i) => {
                            self.emit_mov_imm64(17, i.value);
                            crate::regloc::RegLoc::new(17, false)
                        }
                        Loc::ConstFloat(c) => {
                            self.load_imm_float(14, *c);
                            crate::regloc::RegLoc::new(14, true)
                        }
                        _ => crate::regloc::RegLoc::new(17, false),
                    };
                    // aarch64/opassembler.py:385-392: combine ofs into ip0 = index + ofs.
                    // check_imm_arg() in regalloc.py:161 allows 0..4095; for any
                    // value outside that (including negatives) RPython falls back
                    // to load(ip0, ofs_loc) + ADD_rr. ip0 = x16 (reserved scratch).
                    let combined_index = if ofs != 0 {
                        if (0..4096).contains(&ofs) {
                            // `opassembler.py:403 ADD_ri(ip0, index, ofs)` is a single
                            // instruction; dynasm's `add Xd|SP, Xn|SP, #uimm` form rejects a
                            // dynamic register operand, so encode the word directly
                            // (`codebuilder.py ADD_ri`).
                            self.emit_add_ri(16, index.value, ofs as u32);
                        } else {
                            self.emit_mov_imm64(16, ofs);
                            dynasm!(self.mc ; .arch aarch64
                                ; add x16, x16, X(index.value));
                        }
                        crate::regloc::RegLoc::new(16, false)
                    } else {
                        *index
                    };
                    // aarch64/opassembler.py:393-394:
                    //   scale = get_scale(size_loc.value)
                    //   self._write_to_mem(value_loc, base_loc, index_loc, scale)
                    self.emit_op_gcstore_regalloc(base, &Loc::Reg(combined_index), &val_reg, size);
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                let descr_arc = op.getdescr();
                let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                let target_arglocs = jump_descr
                    .map(|descr| {
                        descr
                            .target_arglocs()
                            .into_iter()
                            .map(|loc| loc_from_target_argloc(&loc))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[dynasm] JUMP remap: src={:?} dst={:?}",
                        arglocs, target_arglocs
                    );
                }
                let mut src_locations1 = Vec::new();
                let mut dst_locations1 = Vec::new();
                let mut src_locations2 = Vec::new();
                let mut dst_locations2 = Vec::new();
                // x86/regalloc.py:1287: assert len(arglocs) == jump_op.numargs()
                let remap_count = if target_arglocs.is_empty() {
                    arglocs.len()
                } else {
                    assert_eq!(
                        arglocs.len(),
                        target_arglocs.len(),
                        "JUMP args ({}) must match target LABEL args ({})",
                        arglocs.len(),
                        target_arglocs.len(),
                    );
                    target_arglocs.len()
                };
                for (i, src_loc) in arglocs[..remap_count].iter().enumerate() {
                    // One classification, two consumers. It picks the location
                    // set the pair rides — set 2 carries the floats and gets
                    // the float scratch — and, when the destination has to be
                    // synthesized below, the kind of the slot itself.
                    // Hard-coding the slot's kind instead described a slot the
                    // value never lands in: `regalloc_push` / `regalloc_pop`
                    // read exactly this field to choose their scratch
                    // register, and `loc_width` reads it for the width.
                    let arg_tp = op
                        .args_slice()
                        .get(i)
                        .and_then(|arg| self.opref_type_at(arg.to_opref(), Some(op_index)))
                        .unwrap_or(Type::Int);
                    let is_float = arg_tp == Type::Float;
                    let dst_loc = if i < target_arglocs.len() {
                        target_arglocs[i]
                    } else {
                        // The canonical base for a frame position, the
                        // one `FrameManager` was built with
                        // (`get_baseofs_of_frame_field`). Passing 0 here named
                        // a slot `FIRST_ITEM_OFFSET` bytes below the one the
                        // regalloc means by the same position, so a source and
                        // this destination could denote the same value and
                        // different storage.
                        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
                        let dst_ofs = crate::regalloc::get_ebp_ofs(base_ofs, i);
                        Loc::Frame(crate::regloc::FrameLoc::new(i, dst_ofs, is_float))
                    };
                    let (srcs, dsts) = if is_float {
                        (&mut src_locations2, &mut dst_locations2)
                    } else {
                        (&mut src_locations1, &mut dst_locations1)
                    };
                    srcs.push(*src_loc);
                    dsts.push(dst_loc);
                }
                let tmpreg1 = Loc::Reg(crate::regloc::RegLoc::new(16, false));
                let tmpreg2 = Loc::Reg(crate::regloc::RegLoc::new(15, true));
                crate::jump::remap_frame_layout_mixed(
                    self,
                    &src_locations1,
                    &dst_locations1,
                    tmpreg1,
                    &src_locations2,
                    &dst_locations2,
                    tmpreg2,
                );
                // assembler.py closing_jump
                if let Some(label) = loop_target_id(op)
                    .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
                {
                    dynasm!(self.mc ; .arch aarch64 ; b =>label);
                } else if let (Some(descr_ref), Some(descr)) = (descr_arc.as_ref(), jump_descr) {
                    let target = descr.ll_loop_code();
                    // External JUMP: direct branch to target loop code.
                    // assembler.py:2461 mc.JMP(imm(target))
                    // Use x16 (IP0 scratch) to avoid clobbering remap'd regs.
                    // assembler.py:1167-1171 `_assemble`: record the target
                    // loop's frame depth so this trace's frame grows to fit it.
                    if target < MIN_RELOCATED_JUMP_TARGET {
                        self.unrelocated_jump_target =
                            Some((target, majit_ir::descr_identity(descr_ref)));
                    } else {
                        self.jump_target_frame_depth =
                            self.jump_target_frame_depth.max(descr.target_frame_depth());
                        self.emit_mov_imm64(16, target as i64);
                        dynasm!(self.mc ; .arch aarch64 ; br x16);
                    }
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
                let result_type = if fail_arg_types.is_empty() {
                    Type::Void
                } else {
                    fail_arg_types[0]
                };
                // `compile.py` ExitFrameWithExceptionDescrRef identity:
                // route to the metainterp `exit_frame_with_exception_descr_ref`
                // when the FINISH was emitted for
                // `pyjitpl.py compile_exit_frame_with_exception`.
                let is_exit_exc = op
                    .with_fail_descr(|fd| fd.is_exit_frame_with_exception())
                    .unwrap_or(false);
                let global_descr_ptr = if is_exit_exc {
                    self.exit_frame_with_exception_descr_ref_ptr()
                } else {
                    self.done_with_this_frame_descr_ptr_for_type(result_type)
                };
                // FINISH op exit (DoneWithThisFrame* / ExitFrameWithExceptionDescr).
                // `compile.py` skips these — not a `ResumeDescr`.
                // Singleton-direct push: machine code bakes the singleton
                // ptr into jf_descr, and runner.rs::find_descr_by_ptr
                // short-circuits FINISH/Exit/Propagate to the cpu-attached
                // singleton before consulting the registry (see x86
                // counterpart for full rationale).
                let descr: majit_ir::DescrRef = if is_exit_exc {
                    self.cpu_handle
                        .read()
                        .exit_frame_with_exception_descr_ref
                        .clone()
                } else {
                    self.done_with_this_frame_descr_arc_for_type(result_type)
                }
                .expect(
                    "FINISH emission requires cpu-attached singleton — \
                     call `attach_default_test_descrs` or use `MetaInterp::new`",
                );

                // Store result to jf_frame[0]
                if let Some(result) = arglocs.first() {
                    let slot0 = Loc::Frame(crate::regloc::FrameLoc::new(
                        0,
                        crate::jitframe::FIRST_ITEM_OFFSET as i32,
                        result_type == Type::Float,
                    ));
                    self.regalloc_mov(result, &slot0);
                }

                // Store descr ptr to jf_descr.
                self.emit_mov_imm64(0, global_descr_ptr);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);

                if result_type == Type::Ref {
                    if let Some(gcmap) = self.finish_gcmap {
                        gcmap_set_bit(gcmap, 0);
                        self.push_gcmap(gcmap);
                    } else {
                        self.push_gcmap(self.gcmap_for_finish);
                    }
                } else if let Some(gcmap) = self.finish_gcmap {
                    self.push_gcmap(gcmap);
                } else {
                    self.pop_gcmap();
                }

                self._call_footer();
                // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
                // not the cell pointer.  `handle_fail_done_with_this_frame`
                // and `handle_fail_exit_frame_with_exception` match by
                // ptr-equality on the singleton, so the cell only carries
                // the keep-alive identity for `clt.asmmemmgr_gcreftracers`.
                self.fail_descrs.push(descr.clone());
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let descr_arc = op.getdescr();
                let label_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[dynasm] LABEL: new DynamicLabel({:?}) arglocs={:?}",
                        label, arglocs
                    );
                }
                dynasm!(self.mc ; =>label);
                if let Some(descr) = label_descr {
                    let stored_arglocs = arglocs
                        .iter()
                        .copied()
                        .map(target_argloc_from_loc)
                        .collect::<Vec<_>>();
                    if majit_ir::debug::have_debug_prints() {
                        majit_ir::debug::log_one(
                            "jit-backend",
                            &format!("LABEL target_arglocs={stored_arglocs:?}"),
                        );
                    }
                    descr.set_target_arglocs(stored_arglocs);
                    descr.set_ll_loop_code(self.mc.offset().0);
                    if let Some(id) = loop_target_id(op) {
                        self.target_tokens_currently_compiling.insert(id, label);
                    }
                    if let Some(descr_ref) = op.getdescr() {
                        self.compiled_target_tokens.push(descr_ref.clone());
                    }
                }
            }
            // ── Calls ──
            // `consider_call` (regalloc.rs) has already captured `arglocs =
            // [func_addr_or_descr_info..., arg_locs...]` and run `before_call`
            // for this op, so the register file is in its across-the-call shape
            // by the time the emitter sees it. Which registers that saved is the
            // `SAVE_ALL_REGS / SAVE_GCREF_REGS / SAVE_DEFAULT_REGS` choice made
            // there, not a blanket spill: `spill_or_move_registers_before_call`
            // (`regalloc.py`) drops values that die at the call, leaves
            // callee-saved ones where they are, and prefers moving the rest to a
            // free callee-saved register over spilling them.
            //
            // What is left here is the call and the result placement.
            OpCode::CallI
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                let oopspec = op.with_call_descr(|cd| cd.get_extra_info().oopspecindex);
                let is_raw_free =
                    op.opcode == OpCode::CallN && oopspec == Some(majit_ir::OopSpecIndex::RawFree);
                let is_math_sqrt = oopspec == Some(majit_ir::OopSpecIndex::MathSqrt);
                if matches!(
                    op.opcode,
                    OpCode::CallMayForceI
                        | OpCode::CallMayForceR
                        | OpCode::CallMayForceF
                        | OpCode::CallMayForceN
                        | OpCode::CallReleaseGilI
                        | OpCode::CallReleaseGilF
                        | OpCode::CallReleaseGilN
                ) {
                    self._store_force_index_if_next_guard(ops, op_index, fail_index);
                }
                if is_math_sqrt {
                    // opassembler.py `math_sqrt`: FSQRT_dd(res, arg).
                    self.genop_math_sqrt(result_loc);
                } else if is_raw_free {
                    self.genop_nursery_free_inline(op, arglocs);
                } else {
                    self.genop_call_with_arglocs(op, arglocs);
                }
            }
            OpCode::CallR => {
                let is_nursery_alloc = op.with_call_descr(|cd| cd.get_extra_info().runtime_helper)
                    == Some(majit_ir::RuntimeHelperKind::NurseryAlloc);
                if is_nursery_alloc {
                    self.genop_nursery_alloc_inline(op, arglocs);
                } else {
                    self.genop_call_with_arglocs(op, arglocs);
                }
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // assembler.py _store_force_index parity:
                // store next GUARD_NOT_FORCED's descr ptr to jf_force_descr
                // BEFORE the call, so forcing code knows which guard to resume.
                self._store_force_index_if_next_guard(ops, op_index, fail_index);
                self.genop_call_assembler(op, arglocs, result_loc);
            }
            OpCode::CondCallN => self.genop_discard_cond_call(op, arglocs),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.genop_cond_call_value(op, arglocs);
            }
            // `rewrite.py handle_new` lowers every NEW / NEW_WITH_VTABLE,
            // vtable store included (`gen_initialize_vtable`), and the
            // assembler has no genop for either.
            OpCode::New | OpCode::NewWithVtable => {
                panic!("not implemented operation with res: {:?}", op.opcode)
            }
            // ── Allocation (raw, when GC rewriter is not active) ──
            OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op, arglocs),
            OpCode::Newstr => self.genop_newstr(op, arglocs),
            OpCode::Newunicode => self.genop_newunicode(op, arglocs),
            // ── Allocation (rewritten by GC rewriter) ──
            // aarch64/regalloc.py prepare_op_call_malloc_nursery + assembler.py:682 malloc_cond parity  allow-line-citation
            OpCode::CallMallocNursery => {
                self.genop_call_malloc_nursery(op);
                if let Some(Loc::Reg(r)) = result_loc
                    && r.value != 0
                {
                    let rv = r.value;
                    dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                }
                // aarch64/regalloc.py:964 forces the result into x0.  Keep
                // it in the regalloc-assigned register; FrameManager emits a
                // spill only if a later lifetime boundary actually needs one.
            }
            OpCode::CallMallocNurseryHeaderless => {
                self.genop_call_malloc_nursery_headerless(op);
                if let Some(Loc::Reg(r)) = result_loc
                    && r.value != 0
                {
                    let rv = r.value;
                    dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                }
                // aarch64/regalloc.py:964-966 keeps malloc's result in the
                // selected result register (x0); it does not eagerly mirror
                // every result into its jitframe slot.  Regalloc emits the
                // eventual spill when a guard or register-pressure boundary
                // actually needs one.
            }
            // aarch64/assembler.py malloc_cond_varsize_frame
            OpCode::CallMallocNurseryVarsizeFrame => {
                let sizeloc = match arglocs.first() {
                    Some(Loc::Reg(sizeloc)) => *sizeloc,
                    _ => panic!("CallMallocNurseryVarsizeFrame expects register size arg"),
                };
                let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
                let slow_path = self.mc.new_dynamic_label();
                let done = self.mc.new_dynamic_label();

                if nf_addr == 0 || nt_addr == 0 {
                    if sizeloc.value != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x0, X(sizeloc.value));
                    }
                    dynasm!(self.mc ; .arch aarch64 ; b =>slow_path);
                } else {
                    let size_reg = if sizeloc == crate::aarch64::registers::X0 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x1, x0);
                        crate::aarch64::registers::X1
                    } else {
                        sizeloc
                    };
                    let gc_header_size = majit_gc::header::GcHeader::SIZE as u32;
                    self.emit_mov_imm64(0, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]);
                    dynasm!(self.mc ; .arch aarch64 ; add x1, x0, X(size_reg.value));
                    dynasm!(self.mc ; .arch aarch64 ; add x1, x1, gc_header_size);
                    self.emit_mov_imm64(16, nt_addr as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x16, [x16]
                        ; cmp x1, x16
                        ; b.hi =>slow_path
                    );
                    self.emit_mov_imm64(16, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x1, [x16]
                        ; str xzr, [x0]
                        ; add x0, x0, gc_header_size
                        ; b =>done
                    );
                }

                dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
                if nf_addr != 0 && nt_addr != 0 {
                    // `malloc_cond_varsize_frame`: each probe only loads its
                    // gcmap and calls the one `_build_malloc_slowpath('fixed')`
                    // body cached on the CPU.  x0/x1 still carry old/new
                    // nursery heads, exactly the shared stub's ABI.
                    if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                        self.emit_mov_imm64(17, gcmap as i64);
                    } else {
                        dynasm!(self.mc ; .arch aarch64 ; mov x17, 0);
                    }
                    self.emit_mov_imm64(16, self.malloc_slowpath_fixed as i64);
                    dynasm!(self.mc ; .arch aarch64 ; blr x16);
                } else {
                    // Tests without an installed nursery cannot enter the
                    // shared old/new-head ABI.  Preserve the non-GC fallback.
                    self.push_all_regs_to_jitframe(
                        &[crate::aarch64::registers::X0, crate::aarch64::registers::X1],
                        true,
                    );
                    if sizeloc.value != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x0, X(sizeloc.value));
                    }
                    if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                        self.emit_mov_imm64(16, gcmap as i64);
                        dynasm!(self.mc ; .arch aarch64
                            ; str x16, [x29, crate::jitframe::JF_GCMAP_OFS as u32]
                        );
                    } else {
                        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                    }
                    self.emit_mov_imm64(
                        16,
                        crate::runner::dynasm_nursery_slowpath_jitframe as *const () as i64,
                    );
                    dynasm!(self.mc ; .arch aarch64 ; blr x16);
                    self.reload_frame_if_necessary();
                    let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                    dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                    self.pop_all_regs_from_jitframe(
                        &[crate::aarch64::registers::X0, crate::aarch64::registers::X1],
                        true,
                    );
                    self.emit_propagate_memory_error_if_null(0);
                }
                dynasm!(self.mc ; .arch aarch64 ; =>done);
                if let Some(Loc::Reg(r)) = result_loc
                    && r.value != 0
                {
                    let rv = r.value;
                    dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                }
            }
            // aarch64/assembler.py malloc_cond_varsize
            // arglocs = [lengthloc, imm(itemsize), imm(kind)]
            OpCode::CallMallocNurseryVarsize | OpCode::CallMallocNurseryVarsizeHeaderless => {
                // Headerless allocators have no `GcHeader` and do not collect
                // on the overflow path. Cranelift and wasm do not install one.
                let headerless = op.opcode == OpCode::CallMallocNurseryVarsizeHeaderless;
                let (base_size, type_id) = op
                    .with_array_descr(|ad| (ad.base_size(), ad.type_id()))
                    .expect("CallMallocNurseryVarsize requires an ArrayDescr");
                let base_size = base_size as i64;
                let type_id = type_id as i64;
                let itemsize = match arglocs.get(1) {
                    Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                    _ => 8,
                };
                // aarch64/assembler.py `malloc_cond_varsize`: x0 is the
                // nursery/result register and x1 the size temporary reserved
                // by regalloc.  Only the two failing comparisons enter the
                // collecting helper; a short array commits the bump, stamps
                // its tid, and returns the payload directly.
                let (nf_addr, nt_addr) = if headerless {
                    crate::runner::dynasm_headerless_nursery_addrs()
                } else {
                    crate::runner::dynasm_nursery_addrs()
                };
                let max_young = crate::runner::dynasm_max_size_of_young_obj();
                let slow_path = self.mc.new_dynamic_label();
                let done = self.mc.new_dynamic_label();
                let word = std::mem::size_of::<usize>();
                // aarch64/regalloc.py `consider_call_malloc_nursery_varsize`:
                // `maxlength = (max_size_of_young_obj - WORD * 2) / itemsize`.
                // The compare below is against the item count, not the byte
                // bound x86's precheck uses.
                let header_size = if headerless {
                    0
                } else {
                    majit_gc::header::GcHeader::SIZE as i64
                };
                let max_length = if headerless {
                    max_young.saturating_sub(base_size as usize) / itemsize as usize
                } else {
                    max_young.saturating_sub(2 * word) / itemsize as usize
                };
                debug_assert!(itemsize > 0);
                debug_assert!(
                    headerless
                        || base_size as usize + majit_gc::header::GcHeader::SIZE
                            >= majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE
                );
                if nf_addr == 0 || nt_addr == 0 || max_length == 0 {
                    dynasm!(self.mc ; .arch aarch64 ; b =>slow_path);
                } else {
                    match arglocs.first() {
                        Some(Loc::Reg(len_r)) => {
                            debug_assert_ne!(len_r.value, 0);
                            debug_assert_ne!(len_r.value, 1);
                            dynasm!(self.mc ; .arch aarch64 ; mov x1, X(len_r.value));
                        }
                        Some(Loc::Immed(len_i) | Loc::ImmedFloat(len_i)) => {
                            self.emit_mov_imm64(1, len_i.value)
                        }
                        Some(Loc::Frame(len_f)) => self.emit_ldr_fp(1, len_f.ebp_loc.value),
                        Some(Loc::Ebp(len_e)) => self.emit_ldr_fp(1, len_e.value),
                        other => {
                            panic!("CallMallocNurseryVarsize length is not a value: {other:?}")
                        }
                    }
                    self.emit_mov_imm64(16, max_length as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; cmp x1, x16
                        ; b.hi =>slow_path
                    );
                    self.emit_mov_imm64(0, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]);
                    self.emit_mov_imm64(16, itemsize);
                    dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x16);
                    self.emit_mov_imm64(16, base_size + header_size + 7);
                    dynasm!(self.mc ; .arch aarch64 ; add x1, x1, x16);
                    self.emit_mov_imm64(16, -8);
                    dynasm!(self.mc ; .arch aarch64
                        ; and x1, x1, x16
                        ; add x1, x1, x0
                    );
                    self.emit_mov_imm64(16, nt_addr as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x16, [x16]
                        ; cmp x1, x16
                        ; b.hi =>slow_path
                    );
                    self.emit_mov_imm64(16, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64 ; str x1, [x16]);
                    if headerless {
                        dynasm!(self.mc ; .arch aarch64 ; b =>done);
                    } else {
                        self.emit_mov_imm64(16, type_id);
                        dynasm!(self.mc ; .arch aarch64
                            ; str x16, [x0]
                            ; add x0, x0, header_size as u32
                            ; b =>done
                        );
                    }
                }
                dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
                if headerless {
                    // Overflow opens a segment and returns the block base.
                    // No collection, so the headered varsize helper is not used.
                    // The regalloc reserves only x0/x1; the arguments below
                    // also clobber x2 and x4, so every register goes to the
                    // jitframe first (`_push_all_regs_to_jitframe`).
                    self.push_all_regs_to_jitframe(&[], true);
                    match arglocs.first() {
                        Some(Loc::Reg(len_r)) => {
                            dynasm!(self.mc ; .arch aarch64 ; mov x1, X(len_r.value));
                        }
                        Some(Loc::Immed(len_i) | Loc::ImmedFloat(len_i)) => {
                            self.emit_mov_imm64(1, len_i.value);
                        }
                        Some(Loc::Frame(len_f)) => self.emit_ldr_fp(1, len_f.ebp_loc.value),
                        Some(Loc::Ebp(len_e)) => self.emit_ldr_fp(1, len_e.value),
                        other => panic!(
                            "CallMallocNurseryVarsizeHeaderless length is not a value: {other:?}"
                        ),
                    }
                    // x1 holds the length. The helper `ovfcheck`s
                    // `length * itemsize + base_size`; the product is not
                    // computed with wrapping arithmetic here.
                    dynasm!(self.mc ; .arch aarch64 ; mov x0, x1);
                    self.emit_mov_imm64(1, itemsize);
                    self.emit_mov_imm64(2, base_size);
                    self.emit_mov_imm64(
                        4,
                        crate::runner::dynasm_nursery_slowpath_headerless_varsize as *const ()
                            as i64,
                    );
                    self.emit_malloc_slowpath_helper_call(4);
                    self.reload_frame_if_necessary();
                    // x0 carries the allocation result.
                    self.pop_all_regs_from_jitframe(&[crate::aarch64::registers::X0], true);
                    self.emit_propagate_memory_error_if_null(0);
                    if let Some(Loc::Reg(r)) = result_loc
                        && r.value != 0
                    {
                        let rv = r.value;
                        dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                    }
                    dynasm!(self.mc ; .arch aarch64 ; b =>done);
                }
                // assembler.py:254 `_push_all_regs_to_jitframe` — the helper
                // below can collect, and `emit_malloc_slowpath_helper_call`
                // only saves the volatiles to the *stack*, where the
                // collector can neither find nor rewrite them.  Spill every
                // register into the jitframe first, so the gcmap pushed below
                // describes the live references and the collector forwards
                // them.  Must precede the argument setup, which clobbers
                // x0/x1/x2.
                self.push_all_regs_to_jitframe(&[], true);
                // _build_malloc_slowpath(kind='var') parity:
                // x0 = base_size, x1 = item_size, x2 = length, x3 = type_id.
                // The tid comes from the descr: a varsize object allocated as
                // type 0 is traced with the layout of whatever registered
                // first, so its items are never walked.
                // Preserve a length held in x0/x1/x3 before those registers
                // become the helper's other arguments.
                //
                // assembler.py:745-751 `malloc_cond_varsize`:
                //
                //     if lengthloc.is_core_reg():
                //         varsizeloc = lengthloc
                //     else:
                //         assert lengthloc.is_stack()
                //         self.regalloc_mov(lengthloc, r.x1)
                //
                // The stack arm carries an assertion rather than a fallback
                // because it is the expected shape: `prepare_op_call_malloc_
                // nursery_varsize` reads `lengthloc` *after*
                // `spill_or_move_registers_before_call(SAVE_ALL_REGS)` has
                // popped every register binding, so a live length is in its
                // frame slot by the time this asks. Substituting an immediate
                // 0 sizes the block for no items at all, and the
                // `gen_initialize_len` store that follows stamps the true
                // length into it.
                match arglocs.first() {
                    Some(Loc::Reg(len_r)) => {
                        dynasm!(self.mc ; .arch aarch64 ; mov x2, X(len_r.value));
                    }
                    Some(Loc::Immed(len_i) | Loc::ImmedFloat(len_i)) => {
                        self.emit_mov_imm64(2, len_i.value);
                    }
                    Some(Loc::Frame(len_f)) => {
                        self.emit_ldr_fp(2, len_f.ebp_loc.value);
                    }
                    Some(Loc::Ebp(len_e)) => {
                        self.emit_ldr_fp(2, len_e.value);
                    }
                    other => panic!("CallMallocNurseryVarsize length is not a value: {other:?}"),
                }
                self.emit_mov_imm64(0, base_size);
                self.emit_mov_imm64(1, itemsize);
                // assembler.py:649-650 push_gcmap — a null gcmap tells the
                // collector this frame holds no references, so every pointer
                // spilled above would survive the collection unforwarded.
                // `push_gcmap` marshals through x16, which is neither an
                // argument register nor the helper register.
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                    self.push_gcmap(gcmap as *mut usize);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                }
                self.emit_mov_imm64(3, type_id);
                // x3 now carries an argument, so the helper address goes to
                // x4, which `emit_malloc_slowpath_helper_call` saves after
                // reading it.
                self.emit_mov_imm64(
                    4,
                    crate::runner::dynasm_nursery_slowpath_varsize as *const () as i64,
                );
                self.emit_malloc_slowpath_helper_call(4);
                self.reload_frame_if_necessary();
                // pop_gcmap
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                // assembler.py:283 `_pop_all_regs_from_jitframe` — the helper
                // restored the volatiles from the stack, so they still name
                // the pre-collection addresses; the jitframe slots are the
                // ones the collector rewrote.  x0 is excluded because it
                // carries the allocation result.
                self.pop_all_regs_from_jitframe(&[crate::aarch64::registers::X0], true);
                // `dynasm_nursery_slowpath_varsize` returns x0 = 0 on
                // real host OOM (calloc failure preserved as NULL per
                // runner.rs).  Route through the propagate path before
                // the result store so the null pointer never reaches
                // subsequent typed stores.
                self.emit_propagate_memory_error_if_null(0);
                if let Some(Loc::Reg(r)) = result_loc
                    && r.value != 0
                {
                    let rv = r.value;
                    dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                }
                dynasm!(self.mc ; .arch aarch64 ; =>done);
            }
            // aarch64/opassembler.py `emit_op_check_memory_error` →
            // assembler.py:342-346 `propagate_memoryerror_if_reg_is_null`
            // — emit `CBNZ reg, skip` and inline the propagate path
            // (`_build_propagate_exception_path`, assembler.py:559-577)
            // so a NULL return from a malloc helper raises a MemoryError
            // via `self.cpu.propagate_exception_descr`.
            //
            // Upstream materializes `propagate_exception_path` once per
            // backend instance and per-CHECK_MEMORY_ERROR branches to
            // it.  Pyre's dynasm doesn't have that out-of-line trampoline
            // infrastructure yet, so the path is inlined per occurrence
            // — equivalent semantics, slightly more code per site.
            // CHECK_MEMORY_ERROR is rare (only after the four CALL_R
            // malloc helpers in `gen_call_malloc_gc`), so the size
            // overhead is negligible.
            //
            // Sequence per assembler.py:559-577:
            //   1. _store_and_reset_exception(mc, r.x0)
            //      — load pos_exc_value → x0, clear pos_exc_value and
            //      pos_exception (assembler.py:501-536).
            //   2. str x0, [fp, jf_guard_exc]
            //      — transfer the saved value into the deadframe so
            //      `cpu.grab_exc_value(deadframe)` can read it back in
            //      `PropagateExceptionDescr.handle_fail` (compile.py).
            //   3. gen_load_int x0, propagate_exception_descr
            //   4. str x0, [fp, jf_descr]
            //   5. mov x0, fp + gen_func_epilog
            OpCode::CheckMemoryError => {
                let reg = match arglocs.first() {
                    Some(Loc::Reg(r)) if !r.is_xmm => r.value,
                    _ => panic!("CheckMemoryError arglocs[0] must be a non-xmm register"),
                };
                self.emit_propagate_memory_error_if_null(reg);
            }
            // aarch64/opassembler.py `emit_op_increment_debug_counter`.
            OpCode::IncrementDebugCounter => {
                let (base, value) = match arglocs {
                    [Loc::Reg(b), Loc::Reg(v)] if !b.is_xmm && !v.is_xmm => (b.value, v.value),
                    _ => panic!("IncrementDebugCounter arglocs {arglocs:?}"),
                };
                dynasm!(self.mc
                    ; .arch aarch64
                    ; ldr X(value), [X(base)]
                    ; add XSP(value), XSP(value), 1
                    ; str X(value), [X(base)]
                );
            }
            // aarch64/opassembler.py _write_barrier_fastpath parity.
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.emit_write_barrier_fastpath(regalloc, op, arglocs);
            }
            // aarch64/opassembler.py emit_op_zero_array.  This is a
            // result-less operation, but it is the initialization contract
            // for NEW_ARRAY_CLEAR when malloc_zero_filled=false.
            OpCode::ZeroArray => self.genop_discard_zero_array(op, arglocs),
            // aarch64/opassembler.py `emit_op_load_effective_address`.
            // rewrite.py turns COPYSTRCONTENT into LEA + memcpy; a silent
            // no-op here leaves the memcpy address in an unwritten register.
            OpCode::LoadEffectiveAddress => {
                self.genop_load_effective_address(arglocs, result_loc);
            }
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(r.value), x29);
                }
            }
            // assembler.py genop_save_exception IS
            // `_store_and_reset_exception(resloc)` — reuse the shared helper.
            OpCode::SaveException => self.emit_store_and_reset_exception(result_loc),
            OpCode::SaveExcClass => self.genop_save_exc_class(result_loc),
            OpCode::RestoreException => self.genop_restore_exception(arglocs),
            // Guards never reach the non-guard dispatch — the regalloc
            // hands them to `regalloc_perform_guard`.
            _ if op.opcode.is_guard() => unreachable!(
                "regalloc_perform reached with guard {:?}; guards must \
                 route through regalloc_perform_guard",
                op.opcode
            ),
            // ── No-ops ──
            _ => {}
        }
    }

    /// assembler.py `asm_guard_operations[opnum]` — emit a guard with its
    /// faillocs (`regalloc_perform_guard`).
    fn asm_guard_operations(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        faillocs: &[Option<Loc>],
        fail_index: u32,
        frame_depth: usize,
    ) {
        let guard_argloc = arglocs.first().copied();
        match op.opcode {
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                // arglocs[0] = condition location
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            // `assembler.py genop_guard_guard_false` inverts the
            // published cc, then implements: a folded IntLt that published
            // CC_L becomes a CC_L failure jump here.
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.guard_success_cc = self.guard_success_cc.map(invert_cc);
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            // aarch64/opassembler.py `emit_op_guard_value`, then EQ.
            OpCode::GuardValue => {
                if arglocs.len() >= 2 {
                    self.emit_op_guard_value(arglocs);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardClass => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardGcType => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_gc_type(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardIsObject => {
                if let Some(obj_loc) = arglocs.first() {
                    self.emit_guard_is_object(obj_loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardSubclass => {
                if arglocs.len() >= 2 {
                    self.emit_guard_subclass(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_B);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardException => {
                if let Some(expected_loc) = arglocs.first() {
                    self.emit_guard_exception(expected_loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
                self.emit_store_and_reset_exception(result_loc);
            }
            OpCode::GuardNonnullClass => {
                if arglocs.len() >= 2 {
                    self.emit_test_loc(&arglocs[0]);
                    let fail_label = self.emit_guard_jcc(CC_E);
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.emit_jcc_to_label(CC_NE, fail_label);
                    self.append_guard_token_with_faillocs(
                        op,
                        op_index,
                        fail_index,
                        fail_label,
                        guard_argloc,
                        faillocs,
                        frame_depth,
                    );
                }
            }
            OpCode::GuardNoException => {
                self.emit_guard_no_exception_check();
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNoOverflow => {
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardOverflow => {
                // aarch64/opassembler.py:547-551 aliases GUARD_NO_OVERFLOW to
                // guard_true and GUARD_OVERFLOW to guard_false. The overflow
                // arithmetic producer leaves the no-overflow success CC in
                // `guard_success_cc`, so invert it for the expected-overflow
                // arm before the common guard emitter derives its fail CC.
                let no_overflow_cc = self
                    .guard_success_cc
                    .take()
                    .expect("GuardOverflow requires a preceding overflow operation");
                self.guard_success_cc = Some(invert_cc(no_overflow_cc));
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotForced => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(16), [x29, JF_DESCR_OFS as u32] ; cmp X(16), xzr);
                self.guard_success_cc = Some(CC_E);
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotForced2 => {
                self.store_force_descr_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotInvalidated => {
                self.implement_guard_not_invalidated_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardAlwaysFails => {
                self.implement_guard_always_fails_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            _ => {
                self.implement_guard_nojump_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
        }
        // Every arm emits its own flag-setting code, so a comparison record
        // that `load_condition_into_cc` did not consume cannot describe the
        // live NZCV any more.
        self.pending_cmp_cc = None;
    }

    /// Helper: guard class comparison
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        if let Loc::Reg(obj) = obj_loc {
            if let Some(vtable_offset) = self.vtable_offset {
                let ofs = vtable_offset as u32;
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [X(obj.value), ofs]);
                self.regalloc_mov(class_loc, &Loc::Reg(crate::regloc::RegLoc::new(17, false)));
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            } else if let Loc::Immed(i) | Loc::ImmedFloat(i) = class_loc {
                let expected_typeid = self
                    .lookup_typeid_from_classptr(i.value as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "GuardClass: missing typeid for classptr {:#x}",
                            i.value as usize
                        )
                    });
                self._cmp_guard_gc_type(&Loc::Reg(*obj), &Loc::immed(expected_typeid as i64));
            }
        }
    }

    fn require_guard_gc_type_info(&self, guard_name: &'static str) -> GuardGcTypeInfo {
        self.guard_gc_type_info.unwrap_or_else(|| {
            panic!(
                "{} requires cpu.supports_guard_gc_type and a TYPE_INFO layout",
                guard_name
            )
        })
    }

    fn lookup_subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        self.classptr_to_subclass_range
            .get(&(classptr as i64))
            .copied()
    }

    fn emit_load_gc_typeid_into_x16(&mut self, obj_reg: u8) {
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i64);
        self.emit_mov_imm64(16, tid_ofs);
        dynasm!(self.mc ; .arch aarch64
            ; add x16, X(obj_reg), x16
            ; ldr w16, [x16]
        );
    }

    /// aarch64 analogue of x86/assembler.py:1893-1901
    /// `_cmp_guard_gc_type`, adjusted for majit's `obj - GcHeader::SIZE`
    /// header layout.
    fn _cmp_guard_gc_type(&mut self, obj_loc: &Loc, expected_typeid_loc: &Loc) {
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        match expected_typeid_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                self.emit_ldr_fp(17, frame.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) | Loc::ImmedFloat(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            other => panic!(
                "_cmp_guard_gc_type: unhandled expected typeid {other:?} — no cmp \
            is emitted and the guard branches on stale flags"
            ),
        }
    }

    /// aarch64/opassembler.py `emit_op_guard_is_object`.
    fn emit_guard_is_object(&mut self, obj_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_IS_OBJECT");
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        if info.shift_by > 0 {
            dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
        }
        self.emit_mov_imm64(17, (info.base_type_info + info.infobits_offset) as i64);
        let flag = info.is_object_flag as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x16, x16, x17
            ; ldrb w16, [x16]
            ; mov w17, flag
            ; tst w16, w17
        );
    }

    /// aarch64/opassembler.py `emit_op_guard_subclass`.
    fn emit_guard_subclass(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        // `cpu.vtable_offset` is set and no TYPE_INFO table is installed.
        // `offset2` is `cpu.subclassrange_min_offset`; `check_min` /
        // `check_max` are `vtable_ptr.subclassrange_min/max`.
        if self.guard_gc_type_info.is_none()
            && let (Some(vtable_offset), Some(range_off)) =
                (self.vtable_offset, self.subclassrange_min_offset)
        {
            let (Loc::Reg(obj), Loc::Immed(classptr) | Loc::ImmedFloat(classptr)) =
                (obj_loc, class_loc)
            else {
                panic!(
                    "GUARD_SUBCLASS expects [Reg object, Immed classptr] \
                     like aarch64/opassembler.py:667"
                );
            };
            let (check_min, check_max) =
                majit_backend::read_vtable_subclass_range(classptr.value, range_off);
            let offset = vtable_offset as u32;
            let offset2 = range_off as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x16, [X(obj.value), offset]
                ; ldr x16, [x16, offset2]
            );
            self.emit_mov_imm64(17, check_min);
            dynasm!(self.mc ; .arch aarch64 ; sub x16, x16, x17);
            self.emit_mov_imm64(17, check_max - check_min);
            dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            return;
        }
        let info = self.require_guard_gc_type_info("GUARD_SUBCLASS");
        let (Loc::Reg(obj), Loc::Immed(classptr) | Loc::ImmedFloat(classptr)) =
            (obj_loc, class_loc)
        else {
            panic!(
                "GUARD_SUBCLASS expects [Reg object, Immed classptr] \
                 like aarch64/opassembler.py:667"
            );
        };
        // An unbound type constant arrives as classptr 0.  Refuse to abort
        // the process: an empty range makes the unsigned compare always
        // fail, so the guard side-exits instead of panicking compile.
        let (check_min, check_max) = self
            .lookup_subclass_range(classptr.value as usize)
            .unwrap_or((0, 0));
        if let Some(vtable_offset) = self.vtable_offset {
            let offset = vtable_offset as u32;
            let offset2 = info.subclassrange_min_offset as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x16, [X(obj.value), offset]
                ; ldr x16, [x16, offset2]
            );
        } else {
            self.emit_load_gc_typeid_into_x16(obj.value);
            if info.shift_by > 0 {
                dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
            }
            self.emit_mov_imm64(
                17,
                (info.base_type_info + info.sizeof_ti + info.subclassrange_min_offset) as i64,
            );
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x16, x17
                ; ldr x16, [x16]
            );
        }
        self.emit_mov_imm64(17, check_min);
        dynasm!(self.mc ; .arch aarch64 ; sub x16, x16, x17);
        self.emit_mov_imm64(17, check_max - check_min);
        dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
    }

    /// aarch64/opassembler.py `emit_op_guard_exception`.
    fn emit_guard_exception(&mut self, expected_loc: &Loc) {
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x16]);
        match expected_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                self.emit_ldr_fp(17, frame.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) | Loc::ImmedFloat(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            other => panic!(
                "emit_guard_exception: unhandled expected location {other:?} — no \
            cmp is emitted and the guard branches on stale flags"
            ),
        }
    }

    /// `_store_and_reset_exception`: result = pos_exc_value; clear both
    /// pos_exception and pos_exc_value on the success fallthrough.
    fn emit_store_and_reset_exception(&mut self, result_loc: Option<&Loc>) {
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_value_addr);
        if let Some(loc) = result_loc {
            match loc {
                Loc::Reg(dst) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]);
                }
                Loc::Frame(frame) => {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x17, [x16]
                    );
                    self.emit_str_fp(17, frame.ebp_loc.value);
                }
                other => panic!(
                    "emit_store_and_reset_exception: unhandled result location \
                {other:?} — the exception value would be dropped"
                ),
            }
        }
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
    }

    /// Emit SETcc into a register (zero-extend to 64-bit).
    fn emit_setcc(&mut self, cc: u8, dst_reg: u8) {
        match cc {
            CC_E => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
            CC_NE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ne);
            }
            CC_L => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lt);
            }
            CC_GE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ge);
            }
            CC_LE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), le);
            }
            CC_G => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), gt);
            }
            CC_B => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lo);
            }
            CC_AE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hs);
            }
            CC_BE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ls);
            }
            CC_A => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hi);
            }
            CC_S => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), mi);
            }
            CC_NS => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), pl);
            }
            CC_O => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vs);
            }
            CC_NO => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vc);
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
        }
    }

    /// Map an integer comparison OpCode to a condition code.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => CC_E,
            OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E,
        }
    }

    /// Map a float comparison OpCode to a condition code (after `fcmp`).
    ///
    /// `fcmp` sets NZCV = 0b0011 when either operand is NaN, so C is SET and
    /// V is SET for an unordered compare.  Only conditions that are false in
    /// that state may be used for `<`, `<=`, `>`, `>=`; `==` must be false and
    /// `!=` true.  `opassembler.py` picks exactly these:
    /// `VFP_LT` = `lo` (C clear), `VFP_LE` = `ls` (C clear or Z set),
    /// `gt` (Z clear and N == V), `ge` (N == V), `eq`, `ne`.
    ///
    /// `hi`/`hs` — the x86 `seta`/`setae` spelling, correct after `ucomisd`
    /// because that sets CF on unordered — are both TRUE after an unordered
    /// `fcmp` and must not be used here.
    fn float_opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::FloatLt => CC_B,  // lo: C clear
            OpCode::FloatLe => CC_BE, // ls: C clear or Z set
            OpCode::FloatGt => CC_G,  // gt: Z clear and N == V
            OpCode::FloatGe => CC_GE, // ge: N == V
            OpCode::FloatEq => CC_E,  // eq
            OpCode::FloatNe => CC_NE, // ne
            _ => CC_E,
        }
    }

    /// `x86/regalloc.py force_allocate_reg_or_cc` hands a comparison the
    /// frame register as its result when the next op consumes the flags
    /// directly (`RegAlloc::force_allocate_reg_or_cc`).  That is a sentinel,
    /// not a real destination — `cset x29, cc` would destroy the frame
    /// pointer — so publish the condition for the following guard and emit
    /// nothing.  Otherwise materialize the boolean and remember the pair so
    /// an adjacent guard can still branch on the live flags.
    fn flush_cc(&mut self, cond: u8, result_loc: Option<&Loc>) {
        // `assembler.py:1293 flush_cc` opens with
        // `assert self.guard_success_cc == rx86.cond_none` — a condition
        // still pending here was published by an earlier op and never
        // consumed, which would make the following guard branch on it.
        debug_assert!(
            self.guard_success_cc.is_none(),
            "flush_cc: guard_success_cc already set",
        );
        let frame_reg_value = crate::aarch64::regalloc::frame_reg().value;
        if let Some(Loc::Reg(r)) = result_loc {
            if r.value == frame_reg_value {
                self.guard_success_cc = Some(cond);
                return;
            }
            self.emit_setcc(cond, r.value);
            self.pending_cmp_cc = Some((*r, cond));
        }
    }

    /// The comparison NZCV flags are still live iff the immediately
    /// preceding emission was a comparison writing exactly `loc` — the
    /// `operations[i + 1]` adjacency `assembler.py:1186 _walk_operations`
    /// requires before folding a comparison into its guard.  Every other
    /// op and every `regalloc_mov` clears the record, so nothing in between can have
    /// clobbered the flags.  Returns the condition under which the
    /// comparison was TRUE.
    fn take_pending_cmp_cc(&mut self, loc: &Loc) -> Option<u8> {
        let (cmp_reg, cc) = self.pending_cmp_cc.take()?;
        let Loc::Reg(guard_reg) = loc else {
            return None;
        };
        (cmp_reg.value == guard_reg.value && cmp_reg.is_xmm == guard_reg.is_xmm).then_some(cc)
    }

    /// `x86/regalloc.py load_condition_into_cc` on the emit side. The
    /// comparison either published its condition through the frame-register
    /// sentinel (`flush_cc`) or materialized a boolean; in the second case
    /// an ADJACENT comparison still left its flags live, so branch on those.
    /// Only when neither holds is the operand an unrelated boolean that has
    /// to be re-tested.
    fn load_condition_into_cc(&mut self, loc: &Loc) {
        if self.guard_success_cc.is_some() {
            return;
        }
        if let Some(cc) = self.take_pending_cmp_cc(loc) {
            self.guard_success_cc = Some(cc);
            return;
        }
        self.emit_test_loc(loc);
        self.guard_success_cc = Some(CC_NE);
    }

    /// Guard with faillocs — emit conditional jump and store faillocs on descr.
    fn implement_guard_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard_with_faillocs: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
    }

    /// Guard no-jump with faillocs.
    fn implement_guard_nojump_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
    }

    fn implement_guard_always_fails_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let fail_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
    }

    /// `opassembler.py emit_op_guard_not_invalidated` — the guard tests
    /// nothing.  It is a position, recorded now and turned into a branch to
    /// its recovery stub by `cpu.invalidate_loop` if the loop is ever
    /// invalidated, so a loop that keeps its quasi-immutable assumptions pays
    /// nothing per iteration for holding them.
    ///
    /// `_emit_guard` leaves a `NOP` where every other guard leaves a `BRK`,
    /// "because it is only eventually patched at a later point"; one aligned
    /// word, which is also what lets the patch land while another thread runs
    /// the loop.
    fn implement_guard_not_invalidated_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        // `_emit_guard`: `pos = self.mc.currpos()`, `token.offset = pos`.
        let pos = self.mc.offset().0;
        dynasm!(self.mc ; .arch aarch64 ; .u32 NOP_INSN);
        let fail_label = self.mc.new_dynamic_label();
        // The label is bound at the recovery stub like any other guard's;
        // under an unpatched `NOP` nothing branches to it.
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        self.pending_guard_tokens
            .last_mut()
            .expect("guard token appended")
            .pos_jump_offset = Some(pos);
    }

    /// Append guard token with regalloc faillocs instead of opref_to_slot snapshot.
    fn append_guard_token_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        fail_label: DynamicLabel,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        // assembler.py _store_force_index parity:
        // If a CALL_ASSEMBLER already pre-allocated this guard's descr
        // (stored in pending_force_descr), reuse it — same Arc, same ptr
        // that was written to jf_force_descr.
        // Stamp the per-trace fail_index and trace_id onto the metainterp
        // ResumeGuardDescr (`op.descr`).  See x86 counterpart for rationale:
        // routing the writes through the trait at codegen time lets readers
        // consume the canonical metainterp identity before
        // `stamp_guard_descr_trace_positions` (`compile.rs`) re-stamps.
        if let Some(d) = op.getdescr()
            && (d.is_resume_guard() || d.is_resume_guard_copied())
            && let Some(fd) = d.as_fail_descr()
        {
            fd.set_fail_index_per_trace(fail_index);
            fd.set_trace_id(self.trace_id);
        }
        let descr: majit_ir::DescrRef = if let Some(pre) = self.pending_force_descr.take() {
            pre
        } else if let Some(d) = op.getdescr() {
            // Guard exit — `compile.py` ResumeGuardDescr family.
            // Use the metainterp `AbstractFailDescr` Arc from `op.descr`
            // directly; per-trace fail_index / trace_id were stamped
            // above.  Refresh the descr's `fail_arg_types` slot when the
            // inferred list disagrees: `store_final_boxes_in_guard` is
            // supposed to keep it in sync, but test scaffolds and earlier
            // optimizer paths can leave it empty.  The GC map below
            // (`guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), ...)`)
            // reads back through the descr, so a stale empty list would
            // under-report `Type::Ref` slots and miss live roots.
            //
            // Do not `infer_fail_arg_types().to_vec()` just to compare:
            // the descr already holds the list on the compiled path.
            if let Some(fd) = d.as_fail_descr() {
                self.ensure_fail_arg_types(op, Some(op_index), fd);
            }
            d
        } else {
            // Test scaffold: tests synthesise guard ops without op.descr.
            // Mint a fresh metainterp ResumeGuardDescr to carry the
            // codegen-time identity (fail_index / trace_id / fail_arg_types).
            let fresh = majit_backend::make_resume_guard_descr_typed(
                self.infer_fail_arg_types(op, Some(op_index)).into_vec(),
            );
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        let descr_fd = descr.as_fail_descr().expect("guard descr is FailDescr");
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token: fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?} faillocs={:?}",
                fail_index,
                op_index,
                op.opcode,
                op.guard_fail_args(),
                descr_fd.fail_arg_types(),
                faillocs
            );
        }

        // `BaseAssembler.store_info_on_descr`: encode `None`, registers
        // and stack slots, panicking on anything else as `gen_regs.index(loc)`
        // does, and share `_previous_rd_locs` when every position matches.
        let rd_locs = crate::guard::store_info_on_descr(
            &mut self.previous_rd_locs,
            faillocs,
            deadframe_slot_for_loc,
        );
        // Stamp source_op_index directly on the meta descr (UnsafeCell slot
        // owned by ResumeGuardDescr / ResumeGuardCopiedDescr per
        // resume_guard_descr.rs); readers use `fd.source_op_index()` so
        // no side-table is needed.
        // Production deopt decodes the descr's `rd_numb`
        // (`ResumeGuardDescr.get_resumestorage`); the backend keeps no
        // recovery-layout cache.
        if descr_fd.is_resume_guard() || descr_fd.is_resume_guard_copied() {
            descr_fd.set_source_op_index(op_index);
        }
        // `llsupport/assembler.py guardtok.faildescr.rd_locs = positions`
        // — write through the trait accessor so the metainterp
        // `AbstractFailDescr` (`history.py _attrs_`) receives the
        // canonical copy when present.  Must follow the `meta_descr`
        // stamp above for the forward to reach the meta side.
        descr_fd.set_rd_locs(rd_locs);
        // `regalloc.py consider_guard_value` records
        // `all_reg_indexes[x.value]`, a deadframe slot; `llmodel.py
        // get_value_direct` reads the raw jitframe word at that slot. The
        // shared failure stub saves every managed register into the jitframe
        // before recovery runs, so the operand remains readable even when it
        // is not a fail-arg. Stamping while laying the guard out means
        // `store_hash` (`compile.py`, gated on `status == 0`) leaves it
        // alone and `must_compile` hashes the (guard, failing value) pair
        // instead of the guard alone.  Without it a guard whose failing value
        // never repeats still accumulates in one bucket and compiles another
        // bridge every `trace_eagerness` failures, without bound.
        if op.opcode == majit_ir::OpCode::GuardValue
            && let Some(slot) = guard_argloc.as_ref().and_then(deadframe_slot_for_loc)
        {
            let type_tag = match op.arg(0).to_opref().ty() {
                Some(majit_ir::Type::Ref) => majit_backend::STATUS_TY_REF,
                Some(majit_ir::Type::Float) => majit_backend::STATUS_TY_FLOAT,
                _ => majit_backend::STATUS_TY_INT,
            };
            descr_fd.make_a_counter_per_value(slot as u32, type_tag);
        }
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token-slots: fail_index={} rd_locs={:?}",
                fail_index,
                descr_fd.rd_locs()
            );
        }
        let gcmap =
            self.guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), faillocs, frame_depth);

        // Reuse the cell pre-allocated by `_store_force_index_if_next_guard`
        // when this guard is paired with a CALL_ASSEMBLER's force-store —
        // jf_force_descr and jf_descr then resolve to the same cell, and
        // `fail_descrs[fail_index]` carries exactly one entry per guard.
        let fail_cell_ptr = self
            .pending_force_cell
            .take()
            .unwrap_or_else(|| self.fail_descrs.push(descr.clone()));
        self.pending_guard_tokens.push(GuardToken {
            fail_label,
            fail_descr: descr.clone(),
            fail_cell_ptr,
            gcmap,
            pos_jump_offset: None,
            must_save_exception: matches!(
                op.opcode,
                OpCode::GuardException | OpCode::GuardNoException | OpCode::GuardNotForced
            ),
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
    }

    /// aarch64/assembler.py `store_force_descr`: GUARD_NOT_FORCED_2 arms
    /// FORCE_TOKEN; it does not branch to an ordinary guard recovery stub.
    fn store_force_descr_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let unused_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            unused_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        let token = self
            .pending_guard_tokens
            .pop()
            .expect("GUARD_NOT_FORCED_2 descriptor token");
        let descr_ptr = token.fail_cell_ptr as i64;
        self.emit_mov_imm64(16, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str X(16), [x29, JF_FORCE_DESCR_OFS as u32]
        );
        self.finish_gcmap = Some(token.gcmap);
    }

    // assembler.py:652 write_pending_failure_recoveries

    /// assembler.py:982 generate_quick_failure.
    ///
    /// RPython parity: the quick-failure stub saves managed registers into the
    /// fixed jitframe prefix before publishing jf_descr and returning.
    /// Every guard that took the single-instruction form has to reach its
    /// failure-recovery stub with a `b.cond`, whose signed 19-bit word
    /// displacement spans [`BCOND_FORWARD_RANGE`].  A stub is bound in
    /// [`Self::write_pending_failure_recoveries`], which measures each such
    /// branch against the stub it actually has to reach — by then the branches
    /// are emitted, so an overflow can no longer be answered by re-emitting
    /// them in the two-instruction form.  Refuse the trace instead: the
    /// up-front `MAX_BYTES_PER_OP` / `MAX_BYTES_PER_GUARD_STUB` estimate is a
    /// heuristic, and letting a relocation that cannot be encoded reach the
    /// assembler is not an acceptable way to find that out.  A trace that took
    /// the two-instruction form everywhere records no branches and so always
    /// passes.
    fn check_guard_reach(&self) -> Result<(), BackendError> {
        let Some(&(branch, stub)) = self.guard_reach_violations.first() else {
            return Ok(());
        };
        Err(BackendError::CompilationFailed(format!(
            "{} of {} single-instruction guard branches cannot reach their \
             recovery stub; the first branches at offset {branch} to a stub at \
             {stub}, {} bytes past b.cond's {BCOND_FORWARD_RANGE}-byte forward \
             reach, but the up-front estimate chose that form",
            self.guard_reach_violations.len(),
            self.short_guard_branch_offsets.len(),
            stub - branch - BCOND_FORWARD_RANGE,
        )))
    }

    fn generate_quick_failure(
        &mut self,
        guard_token: GuardToken,
        save_regs_label: DynamicLabel,
    ) -> RecoveryStub {
        let stub_start = self.mc.offset();

        let fail_label = guard_token.fail_label;
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("recovery stub: binding {fail_label:?}"),
            );
        }
        dynasm!(self.mc ; .arch aarch64 ; =>fail_label);

        dynasm!(self.mc ; .arch aarch64 ; bl =>save_regs_label);
        if crate::trace_call_diag_id() == Some(self.trace_id) {
            let fail_index = guard_token
                .fail_descr
                .as_fail_descr()
                .map(|fd| fd.fail_index_per_trace())
                .unwrap_or(u32::MAX);
            self.emit_push_all_volatile_regs();
            self.emit_mov_imm64(0, 1_000_000 + i64::from(fail_index) * 10);
            dynasm!(self.mc ; .arch aarch64 ; mov x1, x29);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_debug_validate_oldgen_freeblocks as *const () as usize as i64,
            );
            dynasm!(self.mc ; .arch aarch64 ; blr x2);
            self.emit_pop_all_volatile_regs();
        }

        // llsupport/assembler.py store_info_on_descr — must_save_exception
        // guards run the exc=True failure-recovery variant: stage pos_exc_value
        // into jf_guard_exc and clear both globals so grab_exc_value reads the
        // value off the deadframe (assembler.py:316-329 _build_failure_recovery).
        // x16/x17 are scratch (ip0/ip1); all managed regs were saved by the
        // bl above, so clobbering them here is safe.
        if guard_token.must_save_exception {
            let exc_value_addr = crate::jit_exc_value_addr() as i64;
            let exc_type_addr = crate::jit_exc_type_addr() as i64;
            self.emit_mov_imm64(16, exc_value_addr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x17, [x16]                            // x17 = *pos_exc_value
                ; str x17, [x29, JF_GUARD_EXC_OFS as u32]   // jf_guard_exc = excval
                ; str xzr, [x16]                            // *pos_exc_value = 0
            );
            self.emit_mov_imm64(16, exc_type_addr);
            dynasm!(self.mc ; .arch aarch64
                ; str xzr, [x16]                            // *pos_exception = 0
            );
        }

        let descr_ptr = guard_token.fail_cell_ptr as i64;
        self.emit_mov_imm64(0, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_DESCR_OFS as u32]
        );
        self.push_gcmap(guard_token.gcmap);

        self._call_footer();
        RecoveryStub {
            fail_descr: guard_token.fail_descr,
            pos_recovery_stub: stub_start.0,
            pos_jump_offset: guard_token.pos_jump_offset,
        }
    }

    /// assembler.py:1005 write_pending_failure_recoveries.
    /// Returns recovery stub offsets for post-finalize address fixup.
    fn write_pending_failure_recoveries(&mut self) -> Vec<RecoveryStub> {
        // Emit a shared _push_all_regs_to_frame routine once, then let each
        // generate_quick_failure() stub call it.
        let save_regs_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; =>save_regs_label);
        let gprs = all_gen_regs();
        for &reg in gprs.iter() {
            let save_slot = core_reg_position(reg).expect("managed aarch64 GPR");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
        }
        let fprs = all_float_regs();
        for &reg in fprs.iter() {
            let save_slot = float_reg_position(reg).expect("managed aarch64 VFP");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
        }
        dynasm!(self.mc ; .arch aarch64 ; ret);

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            let fail_label = guard_token.fail_label;
            let entry = self.generate_quick_failure(guard_token, save_regs_label);
            // The stub is bound here, so this is the first point at which the
            // displacement each `b.cond` to that label has to encode is known.
            if let Some(&branch) = self.short_guard_branch_offsets.get(&fail_label)
                && entry.pos_recovery_stub >= branch + BCOND_FORWARD_RANGE
            {
                self.guard_reach_violations
                    .push((branch, entry.pos_recovery_stub));
            }
            stub_offsets.push(entry);
        }
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("write_pending done: {} stubs", stub_offsets.len()),
            );
        }
        stub_offsets
    }

    /// assembler.py:849 patch_pending_failure_recoveries — convert
    /// buffer-relative offsets to absolute addresses after finalize.
    fn patch_pending_failure_recoveries(rawstart: usize, stubs: &[RecoveryStub]) {
        for stub in stubs {
            let abs_addr = rawstart + stub.pos_recovery_stub;
            if let Some(fd) = stub.fail_descr.as_fail_descr() {
                fd.set_adr_jump_offset(abs_addr);
            }
        }
    }

    /// `assembler.py process_pending_guards` — the `GUARD_NOT_INVALIDATED` arm
    /// of the same walk.  Rather than patching the guard now, record
    /// `(guard_pos, relative_offset)` for `clt.invalidate_positions`, already
    /// encoded as the `B imm26` the store will write.
    fn collect_invalidate_positions(
        rawstart: usize,
        stubs: &[RecoveryStub],
    ) -> Vec<majit_backend::InvalidatePosition> {
        stubs
            .iter()
            .filter_map(|stub| {
                let pos_jump_offset = stub.pos_jump_offset?;
                // `relative_offset = tok.pos_recovery_stub - tok.offset`
                let relative_offset = stub.pos_recovery_stub as i64 - pos_jump_offset as i64;
                // B imm26: signed, scaled by 4 — the same +-128 MB reach
                // `write_redirect_branch` encodes against.
                const B_REACH: i64 = 1 << 27;
                assert!(
                    (-B_REACH..B_REACH).contains(&relative_offset),
                    "guard recovery stub out of B reach: {relative_offset:#x}"
                );
                let imm26 = ((relative_offset >> 2) & 0x03FF_FFFF) as u32;
                Some(majit_backend::InvalidatePosition {
                    addr: rawstart + pos_jump_offset,
                    word: u64::from(0x1400_0000 | imm26),
                })
            })
            .collect()
    }

    // assembler.py:965-987 patch_jump_for_descr

    /// Overwrite the code at `at` with a branch to `target`.
    ///
    /// `link` selects the branch form to match the upstream method this
    /// stands in for: assembler.py patch_trace overwrites a guard stub with
    /// `BL` (branch-with-link → bridge), while redirect_call_assembler
    /// overwrites a loop entry with a plain `B`. A direct `B`/`BL imm26`
    /// reaches ±128 MB; once the code arena spans more than that, a bridge /
    /// retraced loop can land farther from the originating guard stub / loop
    /// entry, so the scaled 26-bit displacement would silently truncate and
    /// the branch would land on garbage. When the displacement does not fit,
    /// fall back to materializing the absolute target in ip0 and branching
    /// through it (`BR`/`BLR x16`) — the indirect form codebuilder.py emits
    /// for an out-of-range `B`/`BL`. ip0/x16 is call-clobbered scratch at
    /// every redirect site (the bridge prologue's `_check_frame_depth`
    /// reloads it before use).
    ///
    /// # Safety
    /// `at` must point to at least 20 writable bytes of now-dead code:
    /// both callers overwrite the head of a recovery stub / loop prologue
    /// that is longer than the emitted sequence.
    unsafe fn write_redirect_branch(at: usize, target: usize, link: bool) {
        let offset = target as isize - at as isize;
        assert!(
            at & 0b11 == 0 && target & 0b11 == 0,
            "AArch64 redirect branch endpoints must be 4-byte aligned: at={at:#x}, target={target:#x}"
        );
        // B/BL imm26: signed 26-bit, scaled by 4 → ±128 MB reach.
        const B_REACH: isize = 1 << 27;
        if (-B_REACH..B_REACH).contains(&offset) {
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            // 0x14000000 = B imm26, 0x94000000 = BL imm26.
            let opc = if link { 0x9400_0000 } else { 0x1400_0000 };
            let insn = opc | imm26;
            codebuf::with_writable(at as *mut u8, 4, || {
                unsafe { (at as *mut u32).write(insn) };
            });
            flush_icache(at as *const u8, 4);
        } else {
            // MOVZ/MOVK x16, target; BR/BLR x16 — 5 words. The four MOV words
            // come from the shared encoder so the veneer stays byte-identical
            // to emit_mov_imm64 (pinned by frame_depth_patch_words_match_emit_mov_imm64).
            let mov = Self::encode_mov_imm64_words(16, target as i64);
            // 0xD61F0000 = BR Xn, 0xD63F0000 = BLR Xn (Rn in bits 9:5).
            let br = if link { 0xD63F_0000 } else { 0xD61F_0000 };
            let words: [u32; 5] = [mov[0], mov[1], mov[2], mov[3], br | (16 << 5)];
            codebuf::with_writable(at as *mut u8, words.len() * 4, || {
                let p = at as *mut u32;
                for (i, w) in words.iter().enumerate() {
                    unsafe { p.add(i).write(*w) };
                }
            });
            flush_icache(at as *const u8, words.len() * 4);
        }
    }

    /// assembler.py:965 patch_jump_for_descr: redirect a guard to a
    /// bridge by overwriting the recovery stub with a JMP to bridge.
    ///
    /// `adr_jump_offset` is the absolute address of the recovery stub
    /// (set by patch_pending_failure_recoveries). We overwrite the
    /// stub with "MOV r11, bridge_addr; JMP r11" (x64) or "BL imm26"
    /// (aarch64), matching rpython/jit/backend/aarch64/assembler.py
    /// patch_trace().
    pub fn patch_jump_for_descr(descr: &dyn majit_ir::FailDescr, adr_new_target: usize) {
        let stub_addr = descr.adr_jump_offset();
        assert!(stub_addr != 0, "guard already patched");

        // patch_trace uses BL (branch-with-link) into the bridge.
        unsafe { Self::write_redirect_branch(stub_addr, adr_new_target, true) };

        // Verify patch was applied correctly
        if crate::majit_log_enabled() {
            let word = unsafe { (stub_addr as *const u32).read() };
            eprintln!(
                "[patch-verify] stub_addr={:#x} first_word={:#010x} target={:#x}",
                stub_addr, word, adr_new_target
            );
        }

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(
        old: &majit_backend::JitCellToken,
        new: &majit_backend::JitCellToken,
        old_addr: *const u8,
        new_addr: *const u8,
    ) {
        // redirect_call_assembler uses a plain B (no link) to the new loop.
        unsafe { Self::write_redirect_branch(old_addr as usize, new_addr as usize, false) };
        // `aarch64/assembler.py redirect_call_assembler`:
        // `jl.redirect_assembler(..., newlooptoken.number)`.
        majit_backend::redirect_assembler(old, new, new.number() as u64);
    }

    // genop_* — integer arithmetic

    /// INT_ADD: result = arg0 + arg1
    #[allow(dead_code)]
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_SUB: result = arg0 - arg1
    #[allow(dead_code)]
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; sub x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_MUL: result = arg0 * arg1
    #[allow(dead_code)]
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mul x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_AND: result = arg0 & arg1
    #[allow(dead_code)]
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; and x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_OR: result = arg0 | arg1
    #[allow(dead_code)]
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; orr x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_XOR: result = arg0 ^ arg1
    #[allow(dead_code)]
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; eor x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_NEG: result = -arg0
    #[allow(dead_code)]
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; neg x0, x0
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_INVERT: result = ~arg0
    #[allow(dead_code)]
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mvn x0, x0
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_LSHIFT: result = arg0 << arg1
    #[allow(dead_code)]
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; lsl x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    #[allow(dead_code)]
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; asr x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    #[allow(dead_code)]
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; lsr x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — overflow arithmetic (assembler.py:1413-1425)

    /// assembler.py genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    #[allow(dead_code)]
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; adds x0, x0, x1);
        self.store_rax_to_result(op.pos().get());
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py genop_int_sub_ovf.
    #[allow(dead_code)]
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; subs x0, x0, x1);
        self.store_rax_to_result(op.pos().get());
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py genop_int_mul_ovf.
    #[allow(dead_code)]
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        // aarch64/opassembler.py multiplies, computes SMULH, and then
        // compares the high word against the sign-extension of the low
        // word. regalloc.py's prepare_op_guard_no_overflow then uses
        // EQ for INT_MUL_OVF specifically.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mul x2, x0, x1
            ; smulh x3, x0, x1
            ; asr x4, x2, 63
            ; cmp x3, x4
            ; mov x0, x2
        );
        self.store_rax_to_result(op.pos().get());
        self.guard_success_cc = Some(CC_E);
    }

    // genop_* — comparisons

    /// Emit SETcc/CSET to materialize a boolean result.
    /// x64: SETcc AL; MOVZX EAX, AL
    /// aarch64: CSET X0, cc
    #[allow(dead_code)]
    fn emit_setcc_to_result(&mut self, cc: u8, result_opref: OpRef) {
        // CSET Xd, cc — sets Xd to 1 if condition is true, 0 otherwise.
        // Note: CSET Xd, cc is an alias for CSINC Xd, XZR, XZR, invert(cc).
        match cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; cset x0, lt),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; cset x0, le),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; cset x0, gt),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ge),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ne),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; cset x0, lo),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ls),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; cset x0, hi),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; cset x0, hs),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; cset x0, vs),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; cset x0, vc),
            _ => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
        }
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    #[allow(dead_code)]
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos().get().is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos().get());
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    #[allow(dead_code)]
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos().get().is_none() {
            self.emit_setcc_to_result(CC_E, op.pos().get());
        }
    }

    // genop_* — guards

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Looks up the materialized table populated by the runner from
    /// the active gc_ll_descr. RPython resolves the same value via
    /// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        self.classptr_to_typeid.get(&(classptr as i64)).copied()
    }

    fn emit_guard_jcc(&mut self, fail_cc: u8) -> DynamicLabel {
        let fail_label = self.mc.new_dynamic_label();
        self.emit_bcond_to_label(fail_cc, fail_label);
        fail_label
    }

    /// Emit a conditional branch to `label`.
    ///
    /// `opassembler.py:857 _emit_op_cond_call` and the guard emitters reserve
    /// one instruction and patch it with `B_ofs_cond`, which is what the
    /// steady-state loop should contain.  A trace whose code exceeds
    /// `b.cond`'s +1MB reach cannot do that, and falls back to
    /// `b.<inv_cc> skip; b =>label; skip:` — the unconditional `b` carries a
    /// 26-bit displacement (±128MB) at the cost of one extra instruction per
    /// guard.  logo's 70000-op trace needs it.
    fn emit_bcond_to_label(&mut self, cc: u8, label: DynamicLabel) {
        if !self.long_guard_branch {
            // Keep the earliest branch to this label: it is the one furthest
            // from the recovery stub the label is later bound to.
            let offset = self.mc.offset().0;
            self.short_guard_branch_offsets
                .entry(label)
                .or_insert(offset);
            match cc {
                CC_L => dynasm!(self.mc ; .arch aarch64 ; b.lt =>label),
                CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.le =>label),
                CC_G => dynasm!(self.mc ; .arch aarch64 ; b.gt =>label),
                CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.ge =>label),
                CC_E => dynasm!(self.mc ; .arch aarch64 ; b.eq =>label),
                CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.ne =>label),
                CC_B => dynasm!(self.mc ; .arch aarch64 ; b.lo =>label),
                CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.ls =>label),
                CC_A => dynasm!(self.mc ; .arch aarch64 ; b.hi =>label),
                CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.hs =>label),
                CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vs =>label),
                CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vc =>label),
                CC_S => dynasm!(self.mc ; .arch aarch64 ; b.mi =>label),
                CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.pl =>label),
                _ => dynasm!(self.mc ; .arch aarch64 ; b.eq =>label),
            }
            return;
        }
        let skip = self.mc.new_dynamic_label();
        // Invert: branch over the unconditional `b` when the guard succeeds
        match cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; b.ge =>skip),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.gt =>skip),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; b.le =>skip),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.lt =>skip),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; b.hs =>skip),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.hi =>skip),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; b.ls =>skip),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.lo =>skip),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vc =>skip),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vs =>skip),
            CC_S => dynasm!(self.mc ; .arch aarch64 ; b.pl =>skip),
            CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.mi =>skip),
            _ => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
        }
        // Unconditional branch to the failure target (26-bit, ±128MB)
        dynasm!(self.mc ; .arch aarch64 ; b =>label);
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
    }

    fn emit_jcc_to_label(&mut self, fail_cc: u8, fail_label: DynamicLabel) {
        self.emit_bcond_to_label(fail_cc, fail_label);
    }

    /// Infer fail_arg_types from `op.type_` (via `opref_type`) or
    /// `op.fail_arg_types`.
    /// Refresh descr types only when they disagree. Compare against a
    /// stack `SmallVec` so the compiled path (`store_final_boxes_in_guard`
    /// already wrote the list) does not `to_vec()` a 32 B copy per guard.
    fn ensure_fail_arg_types(&self, op: &Op, op_index: Option<usize>, fd: &dyn FailDescr) {
        let expected_len = op.guard_fail_args().map(|fa| fa.len()).unwrap_or(0);
        let have = fd.fail_arg_types();
        if !have.is_empty() && have.len() == expected_len {
            return;
        }
        let inferred = self.infer_fail_arg_types(op, op_index);
        if fd.fail_arg_types() != inferred.as_slice() {
            fd.set_fail_arg_types(inferred.into_vec());
        }
    }

    fn infer_fail_arg_types(&self, op: &Op, op_index: Option<usize>) -> SmallVec<[Type; 8]> {
        if (op.opcode == OpCode::Finish || op.opcode == OpCode::Jump)
            && let Some(descr_types) = op
                .with_fail_descr(|fd| {
                    let dt = fd.fail_arg_types();
                    (!dt.is_empty()).then(|| SmallVec::from_slice(dt))
                })
                .flatten()
        {
            return descr_types;
        }
        let descr_arc = op.getdescr();
        if let Some(fd) = descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            // op.descr is a ResumeGuardDescr carrying post-numbering
            // fail_arg_types installed by
            // store_final_boxes_in_guard (optimizeopt/mod.rs).
            // Prefer the descr for guards too; fall through to
            // op.fail_arg_types only for sharing-path guards
            // (optimizeopt/mod.rs) where op.descr=None.
            let dt = fd.fail_arg_types();
            let expected_len = op.guard_fail_args().map(|fa| fa.len()).unwrap_or(0);
            if dt.len() == expected_len && !dt.is_empty() {
                return SmallVec::from_slice(dt);
            }
        }
        if let Some(ts) = op.get_fail_arg_types() {
            let expected_len = if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.num_args()
            } else {
                op.guard_fail_args().map(|fa| fa.len()).unwrap_or(0)
            };
            if ts.len() == expected_len {
                SmallVec::from_slice(&ts)
            } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.args_slice()
                    .iter()
                    .map(|opref| {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    })
                    .collect()
            } else if let Some(fa) = op.guard_fail_args() {
                fa.iter()
                    .map(|opref| {
                        if opref.is_none() {
                            // resume.py parity: TAGCONST/TAGVIRTUAL
                            // slots are kept as OpRef::NONE in fail_args
                            // (PyPy filters them out; pyre keeps positional).
                            // `Type::Void` is the "hole" sentinel — value
                            // comes from the resume snapshot, not the
                            // deadframe, so downstream consumers
                            // (`guard_gcmap_from_faillocs`, `typed_outputs`
                            // reconstruction) must skip these slots. Earlier
                            // code used `Type::Ref`, which silently leaked a
                            // NULL `GcRef` into the gcmap and the shadow
                            // stack.
                            Type::Void
                        } else {
                            self.opref_type_at(opref.to_opref(), op_index).unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                     op_index={:?} (fail_arg): RPython box.type is fixed at \
                                     construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                        }
                    })
                    .collect()
            } else {
                SmallVec::new()
            }
        } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            // Finish/Jump carry no failargs; their result kind comes from
            // the argument boxes, whose types are fixed at construction
            // (resoperation.py InputArgInt/727/739).  When neither a fail descr nor
            // a preset fail_arg_types list supplies them, infer from the
            // arglist so the FINISH's done_with_this_frame_descr kind
            // matches the caller's CALL_ASSEMBLER result kind (a Void
            // mismatch routes every return through the assembler helper
            // instead of the result-loading fast path).
            op.args_slice()
                .iter()
                .map(|opref| {
                    self.opref_type_at(opref.to_opref(), op_index)
                        .unwrap_or_else(|| {
                            panic!(
                                "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                opref, op_index
                            )
                        })
                })
                .collect()
        } else if let Some(fa) = op.guard_fail_args() {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        // resume.py:411-417 parity: see comment above —
                        // Type::Void is the "hole" sentinel.
                        Type::Void
                    } else {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (fail_arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    }
                })
                .collect()
        } else {
            SmallVec::new()
        }
    }

    /// assembler.py generate_guard_no_exception:
    /// `LDR loc, [loc]; CMP loc, #0` with success on zero.
    fn emit_guard_no_exception_check(&mut self) {
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]
            ; cmp x16, xzr
        );
        self.guard_success_cc = Some(CC_E);
    }

    /// assembler.py _store_force_index: before a call that may force,
    /// store the next GUARD_NOT_FORCED's fail descr ptr to jf_force_descr,
    /// and zero jf_descr so GUARD_NOT_FORCED's CMP [jf_descr], 0 starts clean.
    fn _store_force_index_if_next_guard(&mut self, ops: &[OpRc], op_idx: usize, fail_index: u32) {
        // assembler.py _find_nearby_operation(+1)
        let next_idx = op_idx + 1;
        if next_idx >= ops.len() {
            return;
        }
        let next_op = &ops[next_idx];
        if next_op.opcode != OpCode::GuardNotForced && next_op.opcode != OpCode::GuardNotForced2 {
            return;
        }
        // Pre-allocate the fail descr for the next GUARD_NOT_FORCED.
        // The full metadata (faillocs, rd_numb, etc.) will be filled in
        // when the guard is actually emitted in append_guard_token_with_faillocs.
        // Pre-allocated GuardNotForced descr — ResumeGuardDescr family.
        // Stamp the metainterp `AbstractFailDescr` Arc from `next_op.descr`
        // here so `append_guard_token_with_faillocs` does not need a second
        // pass through `unsafe { Arc::as_ptr as *mut }`.
        if let Some(d) = next_op.getdescr()
            && (d.is_resume_guard() || d.is_resume_guard_copied())
            && let Some(fd) = d.as_fail_descr()
        {
            fd.set_fail_index_per_trace(fail_index);
            fd.set_trace_id(self.trace_id);
        }
        let descr: majit_ir::DescrRef = if let Some(d) = next_op.getdescr() {
            // Same staleness guard as the main guard-emission path: keep
            // descriptor `fail_arg_types` in sync with the inferred list
            // so downstream GC-map / rd_locs readers see the right Ref
            // slots. Do not `to_vec()` the descr's own list to compare.
            if let Some(fd) = d.as_fail_descr() {
                self.ensure_fail_arg_types(next_op, Some(next_idx), fd);
            }
            d
        } else {
            let fresh = majit_backend::make_resume_guard_descr_typed(
                self.infer_fail_arg_types(next_op, Some(next_idx))
                    .into_vec(),
            );
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        // `force_token_to_dead_frame` (cranelift/compiler.rs)
        // recovers `jf_force_descr` via `recover_fail_descr_cell`, which
        // requires a `FailDescrCell` thin pointer.  Bake the cell pointer
        // here (not the bare `Arc<dyn Descr>` fat-pointer data half) and
        // hand the cell off to `append_guard_token_with_faillocs` so the
        // inline guard-exit path bakes the same identity into jf_descr.
        let descr_ptr = self.fail_descrs.push(descr.clone()) as i64;
        self.pending_force_descr = Some(descr);
        self.pending_force_cell = Some(descr_ptr as usize);

        // x86/assembler.py:2210-2222: store descr to jf_force_descr,
        // zero jf_descr.
        self.emit_mov_imm64(16, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str X(16), [x29, JF_FORCE_DESCR_OFS as u32]
            ; str xzr, [x29, JF_DESCR_OFS as u32]
        );
    }

    // genop_* — type conversions

    // Float helpers

    /// Load a float value from `opref` into XMM0 (x64) / D0 (aarch64).
    /// Float values are stored as bit-cast i64 in frame slots.
    #[allow(dead_code)]
    fn load_float_arg_to_d0(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp_d(0, offset);
            }
            ResolvedArg::Const(val) => {
                let c = self.const_float_from_bits(val as u64);
                self.load_imm_float(0, c);
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    #[allow(dead_code)]
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp_d(1, offset);
            }
            ResolvedArg::Const(val) => {
                let c = self.const_float_from_bits(val as u64);
                self.load_imm_float(1, c);
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        self.emit_str_fp_d(0, offset);
    }

    // genop_* — float arithmetic
    // x86/assembler.py:1648 genop_float_add etc.
    // aarch64/assembler.py float equivalents

    /// FLOAT_ADD: result = arg0 + arg1
    #[allow(dead_code)]
    fn genop_float_add(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fadd d0, d0, d1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_SUB: result = arg0 - arg1
    #[allow(dead_code)]
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fsub d0, d0, d1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_MUL: result = arg0 * arg1
    #[allow(dead_code)]
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fmul d0, d0, d1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    #[allow(dead_code)]
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fdiv d0, d0, d1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    #[allow(dead_code)]
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fneg d0, d0
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    #[allow(dead_code)]
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; scvtf d0, x0
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    #[allow(dead_code)]
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fcvtzs x0, d0
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.

    // genop_* — calls
    // x86/assembler.py _genop_call

    fn argloc_imm(arglocs: &[Loc], index: usize) -> i64 {
        match arglocs.get(index) {
            Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
            _ => 0,
        }
    }

    /// aarch64/opassembler.py _emit_call + aarch64/callbuilder.py prepare_arguments
    /// prepare_arguments.
    ///
    /// Register-to-ABI-reg shuffles must go through remap_frame_layout to
    /// handle cycles: with naïve sequential `mov x0,x2; mov x1,x0` the
    /// second move reads the already-clobbered x0. RPython routes the
    /// non-float arg plan through remap_frame_layout(asm, src, dst, ip0)
    /// and preserves the fnloc across the remap via ip1 when it is a
    /// register.
    /// callbuilder.py `write_real_errno`, just before the raw call: copy the
    /// saved `errno` into the real one. `current_sp` is how far sp sits
    /// below the body sp here. x0-x7 hold the arguments and x8/x17 the
    /// callee, so this uses x11 and ip0.
    fn write_real_errno(&mut self, save_err: i64, current_sp: u32) {
        use majit_jitcode::rffi::{RFFI_ALT_ERRNO, RFFI_READSAVED_ERRNO, RFFI_ZERO_ERRNO_BEFORE};
        use majit_rlib::rthread::{
            TLFIELD_ALT_ERRNO_OFS, TLFIELD_P_ERRNO_OFS, TLFIELD_RPY_ERRNO_OFS,
        };
        let tlofs = SAVED_THREADLOCAL_OFS + current_sp;
        let p_errno = TLFIELD_P_ERRNO_OFS as u32;
        if save_err & RFFI_READSAVED_ERRNO != 0 {
            // Just before a call, read '*_errno' and write it into the
            // real 'errno'.
            let rpy_errno = if save_err & RFFI_ALT_ERRNO != 0 {
                TLFIELD_ALT_ERRNO_OFS
            } else {
                TLFIELD_RPY_ERRNO_OFS
            } as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x11, [sp, tlofs]
                ; ldr x16, [x11, p_errno]
                ; ldr w11, [x11, rpy_errno]
                ; str w11, [x16]
            );
        } else if save_err & RFFI_ZERO_ERRNO_BEFORE != 0 {
            // Same, but write zero.
            dynasm!(self.mc ; .arch aarch64
                ; ldr x11, [sp, tlofs]
                ; ldr x16, [x11, p_errno]
                ; str wzr, [x16]
            );
        }
    }

    /// callbuilder.py `read_real_errno`, just after the raw call: save the
    /// real `errno` into the thread-local copy. x0/d0 hold the result, so
    /// this uses x3 and ip0.
    fn read_real_errno(&mut self, save_err: i64, current_sp: u32) {
        use majit_jitcode::rffi::{RFFI_ALT_ERRNO, RFFI_SAVE_ERRNO};
        use majit_rlib::rthread::{
            TLFIELD_ALT_ERRNO_OFS, TLFIELD_P_ERRNO_OFS, TLFIELD_RPY_ERRNO_OFS,
        };
        if save_err & RFFI_SAVE_ERRNO != 0 {
            // Just after a call, read the real 'errno' and save a copy of
            // it inside our thread-local '*_errno'.
            let rpy_errno = if save_err & RFFI_ALT_ERRNO != 0 {
                TLFIELD_ALT_ERRNO_OFS
            } else {
                TLFIELD_RPY_ERRNO_OFS
            } as u32;
            let p_errno = TLFIELD_P_ERRNO_OFS as u32;
            let tlofs = SAVED_THREADLOCAL_OFS + current_sp;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x3, [sp, tlofs]
                ; ldr x16, [x3, p_errno]
                ; ldr w16, [x16]
                ; str w16, [x3, rpy_errno]
            );
        }
    }

    fn emit_call_from_arglocs(&mut self, arglocs: &[Loc], func_index: usize, save_err: i64) {
        let arg_count = arglocs.len();

        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        // aarch64/callbuilder.py:54-60 — if fnloc is a core reg or stack,
        // save it to ip1 (x17) so later remap can clobber its source reg.
        let fnloc = arglocs.get(func_index).copied();
        let fnloc_in_ip1 = match fnloc {
            Some(Loc::Reg(r)) if !r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; mov x17, X(r.value));
                true
            }
            Some(Loc::Frame(f)) if !f.ebp_loc.is_float => {
                self.emit_ldr_fp(17, f.ebp_loc.value);
                true
            }
            _ => false,
        };

        // aarch64/callbuilder.py:29-41 — build arg plan.  Core and float
        // arguments have independent x0..x7 / d0..d7 banks; overflow from
        // either bank is written to the outgoing stack area in source order.
        let mut non_float_src: Vec<Loc> = Vec::new();
        let mut non_float_dst: Vec<Loc> = Vec::new();
        let mut float_src: Vec<Loc> = Vec::new();
        let mut float_dst: Vec<Loc> = Vec::new();
        let mut immed_args: Vec<(u8, i64)> = Vec::new(); // (abi_idx, value)
        let mut stack_args: Vec<Loc> = Vec::new();
        let mut next_core = 0u8;
        let mut next_float = 0u8;

        for &arg in &arglocs[(func_index + 1)..arg_count] {
            let is_float = match arg {
                Loc::Frame(f) => f.ebp_loc.is_float,
                Loc::Reg(r) => r.is_xmm,
                Loc::Immed(_) => false,
                Loc::ImmedFloat(_) | Loc::ConstFloat(_) => true,
                _ => false,
            };
            let abi_idx = if is_float {
                if next_float == 8 {
                    stack_args.push(arg);
                    continue;
                }
                let idx = next_float;
                next_float += 1;
                idx
            } else {
                if next_core == 8 {
                    stack_args.push(arg);
                    continue;
                }
                let idx = next_core;
                next_core += 1;
                idx
            };
            match arg {
                Loc::Frame(f) if f.ebp_loc.is_float => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Frame(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Reg(r) if r.is_xmm => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Reg(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Immed(im) => {
                    immed_args.push((abi_idx, im.value));
                }
                Loc::ImmedFloat(_) => {
                    // An aarch64 walk installs the data block on `xrm`, so
                    // `VFPRegisterManager.convert_to_imm` yields `ConstFloatLoc`
                    // and remaps above. `ImmedFloat` is the no-wrapper arm;
                    // no aarch64 walk produces it as a call argument.
                    panic!(
                        "call argument {abi_idx} is ImmedFloat; aarch64 walk \
                         float constants are ConstFloatLoc"
                    );
                }
                Loc::ConstFloat(_) => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                // See the x86 twin: Reg/Frame/Immed is the whole range the
                // regalloc produces, and silently skipping anything else
                // passes a stale register as that argument.
                other => panic!("call argument {abi_idx} has unsupported location {other:?}"),
            }
        }

        // aarch64/callbuilder.py:42-50: reserve a 16-byte-aligned outgoing
        // stack area and copy every register-bank overflow argument before
        // remapping the register arguments (whose moves may clobber sources).
        let stack_bytes = (stack_args.len() * WORD + 15) & !15;
        if stack_bytes != 0 {
            dynasm!(self.mc ; .arch aarch64 ; sub sp, sp, stack_bytes as u32);
            for (slot, arg) in stack_args.into_iter().enumerate() {
                let offset = (slot * WORD) as u32;
                match arg {
                    Loc::Reg(r) if r.is_xmm => {
                        dynasm!(self.mc ; .arch aarch64 ; str D(r.value), [sp, offset]);
                    }
                    Loc::Reg(r) => {
                        dynasm!(self.mc ; .arch aarch64 ; str X(r.value), [sp, offset]);
                    }
                    Loc::Frame(f) if f.ebp_loc.is_float => {
                        self.emit_ldr_fp_d(15, f.ebp_loc.value);
                        dynasm!(self.mc ; .arch aarch64 ; str d15, [sp, offset]);
                    }
                    Loc::Frame(f) => {
                        self.emit_ldr_fp(16, f.ebp_loc.value);
                        dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, offset]);
                    }
                    Loc::Immed(im) | Loc::ImmedFloat(im) => {
                        self.emit_mov_imm64(16, im.value);
                        dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, offset]);
                    }
                    Loc::ConstFloat(c) => {
                        self.load_imm_float(15, c);
                        dynasm!(self.mc ; .arch aarch64 ; str d15, [sp, offset]);
                    }
                    other => panic!("unsupported AArch64 stack call argument {other:?}"),
                }
            }
        }

        // aarch64/callbuilder.py:62-65 — remap non-float then float args.
        let tmp_nf = Loc::Reg(crate::regloc::RegLoc::new(16, false)); // x16 (ip0)
        let tmp_fp = Loc::Reg(crate::regloc::RegLoc::new(15, true)); // d15
        crate::jump::remap_frame_layout(self, &non_float_src, &non_float_dst, tmp_nf);
        crate::jump::remap_frame_layout(self, &float_src, &float_dst, tmp_fp);

        // Immediate args after remap (each targets a distinct ABI GPR).
        // A float constant from `VFPRegisterManager.convert_to_imm` is
        // `Loc::ConstFloat` and remaps above (`AssemblerARM64.load`).
        for (abi_idx, val) in immed_args {
            self.emit_mov_imm64(abi_idx as u32, val);
        }

        // llsupport/callbuilder.py `emit_call_release_gil`:
        // write_real_errno(); emit_raw_call(); read_real_errno().
        let current_sp = 16 + stack_bytes as u32;
        // aarch64/callbuilder.py:67 — restore fnloc from ip1 / emit call.
        if fnloc_in_ip1 {
            self.write_real_errno(save_err, current_sp);
            dynasm!(self.mc ; .arch aarch64 ; blr x17);
        } else {
            // Unlike x86, which always ends in `call rax`, this arm is the
            // only place a `blr` is emitted — an unhandled fnloc spelling
            // would emit no call at all and fall through with a stale x0.
            let Some(Loc::Immed(i) | Loc::ImmedFloat(i)) = fnloc else {
                panic!("unsupported AArch64 call target {fnloc:?}");
            };
            let val = i.value;
            self.emit_mov_imm64(8, val);
            self.write_real_errno(save_err, current_sp);
            dynasm!(self.mc ; .arch aarch64 ; blr x8);
        }
        // `CallBuilderARM64.load_result` reads the one return word in x0.
        self.read_real_errno(save_err, current_sp);

        if stack_bytes != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add sp, sp, stack_bytes as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    fn ensure_call_result_bit_extension(&mut self, arglocs: &[Loc]) {
        let size = Self::argloc_imm(arglocs, 1) as usize;
        let signed = Self::argloc_imm(arglocs, 2) != 0;
        if size >= WORD {
            return;
        }

        match size {
            4 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; asr x0, x0, 32);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; lsr x0, x0, 32);
                }
            }
            2 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 48 ; asr x0, x0, 48);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFFFF);
                }
            }
            1 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 56 ; asr x0, x0, 56);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFF);
                }
            }
            _ => {}
        }
    }

    fn _genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        // [resloc, size, sign, saveerr, func, args...] for CALL_RELEASE_GIL.
        let is_call_release_gil = op.opcode.is_call_release_gil();
        let save_err = if is_call_release_gil {
            Self::argloc_imm(arglocs, 3)
        } else {
            0
        };
        let func_index = 3 + usize::from(is_call_release_gil);
        // aarch64/callbuilder.py `prepare_arguments` sorts arguments by
        // location type alone, and `load_result` asserts on a `'S'` result
        // ("not supported yet"): the AArch64 CPU keeps model.py's
        // `supports_singlefloats = False`, so no `'S'` call reaches here.
        assert!(
            !op.getdescr().is_some_and(|descr| {
                descr
                    .as_call_descr()
                    .is_some_and(|cd| cd.result_class() == 'S' || cd.arg_classes().contains('S'))
            }),
            "aarch64 does not support singlefloat calls"
        );
        self.emit_call_from_arglocs(arglocs, func_index, save_err);
        if op.opcode.result_type() == Type::Int {
            self.ensure_call_result_bit_extension(arglocs);
        }
    }

    /// opassembler.py `math_sqrt`: `FSQRT_dd(res, arg)`.
    /// The shared regalloc force-results into the source register.
    fn genop_math_sqrt(&mut self, result_loc: Option<&Loc>) {
        if let Some(Loc::Reg(r)) = result_loc {
            dynasm!(self.mc ; .arch aarch64 ; fsqrt D(r.value), D(r.value));
        }
    }

    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        let can_collect = op
            .with_call_descr(|descr| descr.get_extra_info().check_can_collect())
            .unwrap_or(false);
        if can_collect && let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        }
        self._genop_call_with_arglocs(op, arglocs);
        if can_collect {
            self.reload_frame_if_necessary();
            self.pop_gcmap();
        }
        if crate::trace_call_diag_id() == Some(self.trace_id) {
            self.emit_push_all_volatile_regs();
            self.emit_mov_imm64(0, op.pos().get().raw() as i64);
            dynasm!(self.mc ; .arch aarch64 ; mov x1, x29);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_debug_validate_oldgen_freeblocks as *const () as usize as i64,
            );
            dynasm!(self.mc ; .arch aarch64 ; blr x2);
            self.emit_pop_all_volatile_regs();
        }
    }

    /// Inline nursery bump for a call tagged
    /// [`majit_ir::RuntimeHelperKind::NurseryAlloc`].
    ///
    /// The tag declares a two-word headerless node taken from the
    /// interpreter's own nursery, whose payload is the call's two arguments:
    /// the value at offset 0 and the successor link at offset 8.
    ///
    /// The fast path takes the head of the allocator's recycle list when it
    /// reports one and has one, otherwise advances `nursery_free`, and stores
    /// those two words. Neither can collect, so it emits no gcmap. The slow
    /// path is the ordinary residual call wrapper, which may collect inside
    /// the callee; the callee is free to treat the second argument as a
    /// keep-root, since that is the object the new node links to. This is
    /// sound because the op is still a call: the optimizer's residual-call
    /// emission fences pending setfields before the allocation, so a collector
    /// that walks the interpreter's own structures finds roots that are
    /// current. A collector that is not headerless-aware
    /// (`dynasm_headerless_nursery_addrs`) gets the residual call alone.
    fn genop_nursery_alloc_inline(&mut self, op: &Op, arglocs: &[Loc]) {
        // Two-word headerless node: value@0, link@8.
        const NURSERY_ALLOC_NODE_SIZE: u32 = 16;

        let (nf_addr, nt_addr) = crate::runner::dynasm_headerless_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }

        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        let (Some(&value_loc), Some(&next_loc)) =
            (arglocs.get(func_index + 1), arglocs.get(func_index + 2))
        else {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        };

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        let bump = self.mc.new_dynamic_label();
        let init = self.mc.new_dynamic_label();

        // Fast path uses ONLY reserved IP regs (x14/x15/x16/x17); never touches a
        // regalloc-managed register, so the slow path's original arglocs stay
        // intact. x16=slot address, x17=cell, x14=newf/link(temp), x15=top(temp).
        //
        // Take from the recycle list first, exactly as the callee's own
        // allocation order does. A bump-only fast path would agree with the
        // callee only while the current chunk has room: once it fills, an
        // allocator that recycles serves every request from the list, the bump
        // pointer stays at the limit, and the fast path never fires again.
        let recycle_addr = crate::runner::dynasm_nursery_recycle_list_addr();
        if recycle_addr != 0 {
            self.emit_mov_imm64(16, recycle_addr as i64); // x16 = &recycle_head
            dynasm!(self.mc ; .arch aarch64
                ; ldr x17, [x16]                    // x17 = cell = *recycle_head
                ; cbz x17, =>bump                   // empty -> bump instead
                ; ldr x14, [x17, 8]                 // x14 = cell's link word
                ; str x14, [x16]                    // *recycle_head = link
                ; b =>init
            );
        }
        dynasm!(self.mc ; .arch aarch64 ; =>bump);
        self.emit_mov_imm64(16, nf_addr as i64); // x16 = &nursery_free
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x16]                        // x17 = base = *nursery_free
            ; add x14, x17, NURSERY_ALLOC_NODE_SIZE // x14 = newf = base + 16
        );
        self.emit_mov_imm64(15, nt_addr as i64); // x15 = &nursery_top
        dynasm!(self.mc ; .arch aarch64
            ; ldr x15, [x15]                        // x15 = top = *nursery_top
            ; cmp x14, x15                          // newf vs top
            ; b.hi =>slow_path                      // newf > top -> exhausted
            ; str x14, [x16]                        // *nursery_free = newf
        );
        dynasm!(self.mc ; .arch aarch64 ; =>init);
        // Init node fields; value/next loaded from ORIGINAL arglocs (no pre-clobber).
        // load_loc_to_reg returns a managed reg untouched, or loads Frame/Immed into
        // the IP scratch we pass (x15 and x14 are free here: top consumed, newf stored).
        let vreg = self.load_loc_to_reg(&value_loc, 15);
        dynasm!(self.mc ; .arch aarch64 ; str X(vreg), [x17]); // value @ base+0
        let nreg = self.load_loc_to_reg(&next_loc, 14);
        dynasm!(self.mc ; .arch aarch64 ; str X(nreg), [x17, 8]); // next @ base+8
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x17); // result = base
        dynasm!(self.mc ; .arch aarch64 ; b =>done);

        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        self.genop_call_with_arglocs(op, arglocs);
        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// Inline recycle push for a `raw_free` whose allocator publishes a
    /// recycle window.
    ///
    /// Publishing the window and the list head declares that releasing a cell
    /// inside the window is exactly `cell.link = *head; *head = cell`, with
    /// the link word at offset 8 — the same two-word headerless cell
    /// [`Self::genop_nursery_alloc_inline`] hands out. An allocator that
    /// publishes neither keeps the whole release on the residual call, and so
    /// does a cell the window rejects, which is how a null pointer and a cell
    /// the allocator no longer owns reach the callee's own tests.
    ///
    /// The op stays a call, so `before_call` has already put the register file
    /// in its across-the-call shape; what this replaces is the branch to the
    /// callee and the callee's own body. It cannot collect and emits no
    /// gcmap, matching the descr the release carries.
    fn genop_nursery_free_inline(&mut self, op: &Op, arglocs: &[Loc]) {
        let window_addr = crate::runner::dynasm_nursery_recycle_window_addr();
        let head_addr = crate::runner::dynasm_nursery_recycle_list_addr();
        if window_addr == 0 || head_addr == 0 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }
        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        let Some(&cell_loc) = arglocs.get(func_index + 1) else {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        };

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();

        // Fast path uses ONLY reserved IP regs (x14/x15/x16/x17), so the slow
        // path's original arglocs stay intact. `load_loc_to_reg` returns a
        // managed register untouched, or loads a Frame/Immed operand into the
        // x17 scratch; neither can be x14/x15/x16.
        let cell = self.load_loc_to_reg(&cell_loc, 17);
        self.emit_mov_imm64(16, window_addr as i64); // x16 = &[base, width]
        dynasm!(self.mc ; .arch aarch64
            ; ldr x14, [x16]                    // x14 = base
            ; ldr x15, [x16, 8]                 // x15 = width
            ; sub x14, X(cell), x14             // x14 = cell - base
            ; cmp x14, x15
            ; b.hs =>slow_path                  // outside the window
        );
        self.emit_mov_imm64(16, head_addr as i64); // x16 = &recycle_head
        dynasm!(self.mc ; .arch aarch64
            ; ldr x14, [x16]                    // x14 = *recycle_head
            ; str x14, [X(cell), 8]             // cell's link word = old head
            ; str X(cell), [x16]                // *recycle_head = cell
            ; b =>done
        );

        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        self.genop_call_with_arglocs(op, arglocs);
        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// assembler.py call_assembler: invoke a compiled JIT loop.
    ///
    /// RPython fast path (assembler.py:295-360):
    ///   1. _call_assembler_emit_call — call the target trace
    ///   2. _call_assembler_check_descr — CMP jf_descr == done_with_this_frame_descr
    ///   3. Path A (slow): call assembler_helper
    ///   4. Path B (fast): MOV result, [frame + ofs]
    ///   5. join paths
    ///
    /// RPython allocates the callee jitframe via malloc_jitframe (heap).
    /// We use malloc for parity: the callee's frame-slot model needs
    /// frame_depth slots (not just num_args), and heap allocation avoids
    /// stack overflow on deep recursion.
    /// assembler.py call_assembler parity.
    /// Uses regalloc-provided arglocs to load callee arguments instead of
    /// resolve_opref(), which drops register-carried values to Const(0).
    fn genop_call_assembler(&mut self, op: &Op, arglocs: &[Loc], result_loc: Option<&Loc>) {
        // handle_call_assembler (rewrite.py) always pre-builds the
        // callee jitframe — storing every inputarg, and for a virtualizable
        // passing the forced vable object as the second arg — so the backend
        // only loads arglocs[0] (the rewritten frame) and invokes the target.
        // There is no genop-side vable "expansion": materialising vable fields
        // into callee frame slots here clobbered the inputargs the rewrite had
        // already stored (double-materialisation).
        {
            let frame_loc = arglocs
                .first()
                .copied()
                .expect("call_assembler missing rewritten jitframe arg");
            let vable_loc = arglocs.get(1).copied();
            self.emit_load_to_rax(frame_loc);

            let descr_arc = op.getdescr();
            let target_addr = self.resolve_call_assembler_target_addr(descr_arc.as_ref());
            let is_resolved = target_addr.is_available() || self.self_entry_label.is_some();
            let result_type = op.opcode.result_type();
            let done_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
            let helper_addr = crate::call_assembler_helper_addr() as i64;
            let green_key = self.header_pc as i64;

            if !is_resolved {
                let force_addr = crate::call_assembler_force_fn_addr() as i64;
                if force_addr != 0 {
                    if let Some(vloc) = vable_loc {
                        self.emit_load_to_rax(vloc);
                    } else {
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr x0, [x0, FIRST_ITEM_OFFSET as u32]
                        );
                    }
                    self.emit_mov_imm64(2, force_addr);
                    let pushed_gcmap = self.push_pending_call_gcmap();
                    dynasm!(self.mc ; .arch aarch64
                        ; blr x2
                    );
                    self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                } else {
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, xzr
                    );
                }
                self.move_call_assembler_result(op.opcode.result_type(), result_loc);
                return;
            }

            // llsupport/assembler.py:320 +
            // aarch64/opassembler.py `_call_assembler_emit_call`:
            // call the target assembler entry directly with the callee
            // jitframe in x0.  The extra Rust execute trampoline was a
            // pre-existing adaptation; it is not part of the RPython fast
            // path and is too expensive for recursive CALL_ASSEMBLER.
            let pushed_gcmap = self.push_pending_call_gcmap();
            // `_call_assembler_emit_call` opens with `LDR x1, [sp, ofs]` — the
            // callee's prologue re-spills x1 into its own frame, so the base
            // has to be reloaded here rather than assumed live. The target
            // address goes through ip0 because x1 is now an argument.
            dynasm!(self.mc ; .arch aarch64
                ; ldr x1, [sp, #SAVED_THREADLOCAL_OFS]
            );
            if let Some(addr) = target_addr.immediate {
                let addr = addr as i64;
                self.emit_mov_imm64(16, addr);
                dynasm!(self.mc ; .arch aarch64 ; blr x16);
            } else if let Some(entry_label) = self.self_entry_label {
                dynasm!(self.mc ; .arch aarch64 ; bl =>entry_label);
            }
            // aarch64/callbuilder.py `CallBuilderARM64.pop_gcmap` runs
            // `_reload_frame_if_necessary` first: a collection during the
            // call can move the caller jitframe that the callee footer
            // restored into x29, so x29 is re-read from the shadow stack.
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);

            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            self.emit_mov_imm64(2, done_descr_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x1, [x0, JF_DESCR_OFS as u32]
                ; cmp x1, x2
                ; b.eq =>fast_path
            );
            {
                // `compile.py:665` parity: helper signature is
                // `(cpu_handle, callee_jf_ptr, green_key)`. Pass cpu_ptr as
                // arg0 so the trampoline can resolve the attached
                // `done_with_this_frame_descr_*` /
                // `exit_frame_with_exception_descr_ref` identities. blr
                // through x3 so x0/x1/x2 stay live as the helper args.
                let cpu_ptr = self.cpu_handle_ptr();
                self.emit_mov_imm64(3, helper_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x1, x0                    // arg1 = callee_jf_ptr
                );
                self.emit_mov_imm64(2, green_key); // arg2 = green_key
                self.emit_mov_imm64(0, cpu_ptr); // arg0 = cpu_handle
                let pushed_gcmap = self.push_pending_call_gcmap();
                dynasm!(self.mc ; .arch aarch64
                    ; blr x3                        // x0 = helper result
                );
                self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                dynasm!(self.mc ; .arch aarch64 ; b =>merge);
            }
            {
                dynasm!(self.mc ; .arch aarch64
                    ; =>fast_path
                );
                if result_type == Type::Float {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr d0, [x0, FIRST_ITEM_OFFSET as u32]
                        ; fmov x0, d0
                    );
                } else {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x0, [x0, FIRST_ITEM_OFFSET as u32]
                    );
                }
                dynasm!(self.mc ; .arch aarch64
                    ; =>merge
                );
            }
            // RPython `BaseAssembler.call_assembler` joins helper/fast paths
            // here without another frame reload.  The collecting target/helper
            // calls above already reload in `Aarch64CallBuilder.pop_gcmap`.
            self.move_call_assembler_result(result_type, result_loc);
        }
    }

    /// `llsupport/assembler.py call_assembler` keeps the result in
    /// `result_loc`; it does not mirror every call result into a fresh frame
    /// slot.  The local helper ABI joins both paths with the raw result bits in
    /// x0, so materialize those bits in the location chosen by regalloc.
    fn move_call_assembler_result(&mut self, result_type: Type, result_loc: Option<&Loc>) {
        match (result_type, result_loc) {
            (Type::Void, None) => {}
            (Type::Float, Some(Loc::Reg(r))) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; fmov D(r.value), x0);
            }
            (_, Some(Loc::Reg(r))) if !r.is_xmm => {
                if r.value != 0 {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(r.value), x0);
                }
            }
            _ => panic!(
                "CALL_ASSEMBLER result must use its regalloc result register: type={result_type:?} loc={result_loc:?}"
            ),
        }
    }

    /// aarch64/opassembler.py _write_barrier_fastpath parity.
    fn emit_write_barrier_fastpath(&mut self, regalloc: &RegAlloc<'_>, op: &Op, arglocs: &[Loc]) {
        // `mc.LDRB_ri(r.ip0.value, loc_base.value, ...)` indexes the base as
        // a core register; upstream guarantees that in
        // `ARMRegisterManager.return_constant` (aarch64/regalloc.py), which
        // materializes every Const into a scratch register. The shared
        // `RegisterManager::return_constant` follows the llsupport spelling
        // (llsupport/regalloc.py) and can hand back a bare `Loc::Immed`,
        // so state the contract instead of emitting nothing — a barrier that
        // assembles to zero bytes stays invisible until it corrupts memory.
        let loc_base = match arglocs.first() {
            Some(Loc::Reg(r)) => *r,
            other => {
                panic!("write barrier base loc must be Loc::Reg (regalloc contract), got {other:?}")
            }
        };
        let is_array = op.opcode == majit_ir::OpCode::CondCallGcWbArray;
        // `loc_index = arglocs[1]` — the location kind is discriminated at
        // the card-marking block, not here.
        let loc_index = if is_array {
            arglocs.get(1).copied()
        } else {
            None
        };
        // `len(self._regalloc.vfprm.reg_bindings)`.
        let withfloats = regalloc.xrm.reg_bindings_len() > 0;
        self.emit_write_barrier_fastpath_for_base(loc_base, is_array, false, withfloats, loc_index);
    }

    /// aarch64/opassembler.py `_write_barrier_fastpath(mc, descr, arglocs,
    /// array, is_frame)`: test the flag byte and `BL` the
    /// `wb_slowpath[helper_num]` helper built by `_build_wb_slowpath`.
    fn emit_write_barrier_fastpath_for_base(
        &mut self,
        loc_base: crate::regloc::RegLoc,
        is_array: bool,
        is_frame: bool,
        withfloats: bool,
        loc_index: Option<Loc>,
    ) {
        // `assert cls is not None and isinstance(descr, cls)`.
        // `COND_CALL_GC_WB` only exists because the GC rewriter emitted it,
        // so a missing descriptor here means the two disagree; returning
        // would drop the barrier without a trace.
        let wb = crate::runner::dynasm_write_barrier_descr()
            .expect("COND_CALL_GC_WB emitted without a write barrier descriptor");
        //
        let mut card_marking = false;
        let mut mask = wb.jit_wb_if_flag_singlebyte as i64;
        if is_array && wb.jit_wb_cards_set != 0 {
            // assumptions the rest of the function depends on:
            assert_eq!(wb.jit_wb_cards_set_byteofs, wb.jit_wb_if_flag_byteofs);
            assert_eq!(wb.jit_wb_cards_set_singlebyte, -0x80);
            card_marking = true;
            mask = wb.jit_wb_if_flag_singlebyte as i64 | -0x80;
        }
        //
        if is_frame {
            assert_eq!(loc_base, crate::aarch64::registers::FP);
        }
        ldrb_ip0_signed_offset(&mut self.mc, loc_base, wb.jit_wb_if_flag_byteofs);
        mask &= 0xFF;
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64
            ; movz w17, mask as u32
            ; tst w16, w17
            ; b.eq =>done
        );

        // for cond_call_gc_wb_array, also add another fast path:
        // if GCFLAG_CARDS_SET, then we can just set one bit and be done
        let js_location = self.mc.new_dynamic_label();
        if card_marking {
            // GCFLAG_CARDS_SET is in this byte at 0x80
            dynasm!(self.mc ; .arch aarch64
                ; movz w17, 0x80
                ; tst w16, w17
                ; b.ne =>js_location
            );
        }

        // Write only a CALL to the helper prepared in advance, passing it as
        // argument the address of the structure we are writing into
        // (the first argument to COND_CALL_GC_WB).
        let helper_num = if is_frame {
            4
        } else {
            usize::from(card_marking) + 2 * usize::from(withfloats)
        };
        let helper = self.wb_slowpath[helper_num];
        assert_ne!(helper, 0, "wb_slowpath[{helper_num}] was not built");
        //
        let x0 = crate::aarch64::registers::X0;
        if loc_base != x0 {
            // push two registers to keep stack aligned
            dynasm!(self.mc ; .arch aarch64
                ; sub sp, sp, 2 * WORD as u32
                ; str x0, [sp, WORD as u32]
                ; str X(loc_base.value), [sp]
                ; mov x0, X(loc_base.value)
            );
        }
        self.emit_mov_imm64(16, helper as i64);
        dynasm!(self.mc ; .arch aarch64 ; blr x16);
        if loc_base != x0 {
            dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [sp, WORD as u32]
                ; ldr X(loc_base.value), [sp]
                ; add sp, sp, 2 * WORD as u32
            );
        }

        if card_marking {
            // The helper ends again with a check of the flag in the object.  So
            // here, we can simply write again a conditional jump, which will be
            // taken if GCFLAG_CARDS_SET is still not set.
            dynasm!(self.mc ; .arch aarch64
                ; b.eq =>done
                ; =>js_location
            );
            //
            // case GCFLAG_CARDS_SET: emit a few instructions to do
            // directly the card flag setting
            match loc_index {
                // `assert loc_index.is_core_reg()`
                Some(Loc::Reg(loc_index)) => {
                    let shift = 3 + wb.jit_wb_card_page_shift;
                    dynasm!(self.mc ; .arch aarch64
                        ; lsr x16, X(loc_index.value), shift
                        ; mvn x30, x16
                        ; lsr x16, X(loc_index.value), wb.jit_wb_card_page_shift
                        ; and x17, x16, 7
                        ; mov x16, 1
                        ; lsl x17, x16, x17
                        ; sub x30, x30, majit_gc::header::GcHeader::SIZE as u32
                        ; ldrb w16, [X(loc_base.value), x30]
                        ; orr w16, w16, w17
                        ; strb w16, [X(loc_base.value), x30]
                    );
                }
                _ => panic!("card index {loc_index:?} is not a core register"),
            }
        }

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// _push_all_regs_to_jitframe(also_push_vfp=True) parity: save x0-x15
    /// (GPR) + d0-d15 (VFP) on the C stack. Total 256 bytes (16-aligned).
    /// The pop counterpart is [`emit_pop_all_volatile_regs`].
    fn emit_push_all_volatile_regs(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x0, x1, [sp, -256]!
            ; stp x2, x3, [sp, 16]
            ; stp x4, x5, [sp, 32]
            ; stp x6, x7, [sp, 48]
            ; stp x8, x9, [sp, 64]
            ; stp x10, x11, [sp, 80]
            ; stp x12, x13, [sp, 96]
            ; stp x14, x15, [sp, 112]
            ; stp d0, d1, [sp, 128]
            ; stp d2, d3, [sp, 144]
            ; stp d4, d5, [sp, 160]
            ; stp d6, d7, [sp, 176]
            ; stp d8, d9, [sp, 192]
            ; stp d10, d11, [sp, 208]
            ; stp d12, d13, [sp, 224]
            ; stp d14, d15, [sp, 240]
        );
    }

    /// _pop_all_regs_from_jitframe parity: restore VFP then GPR saved by
    /// [`emit_push_all_volatile_regs`].
    fn emit_pop_all_volatile_regs(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; ldp d14, d15, [sp, 240]
            ; ldp d12, d13, [sp, 224]
            ; ldp d10, d11, [sp, 208]
            ; ldp d8, d9, [sp, 192]
            ; ldp d6, d7, [sp, 176]
            ; ldp d4, d5, [sp, 160]
            ; ldp d2, d3, [sp, 144]
            ; ldp d0, d1, [sp, 128]
            ; ldp x14, x15, [sp, 112]
            ; ldp x12, x13, [sp, 96]
            ; ldp x10, x11, [sp, 80]
            ; ldp x8, x9, [sp, 64]
            ; ldp x6, x7, [sp, 48]
            ; ldp x4, x5, [sp, 32]
            ; ldp x2, x3, [sp, 16]
            ; ldp x0, x1, [sp], 256
        );
    }

    /// `_build_malloc_slowpath()` parity: preserve live state across the
    /// helper call. RPython's JIT-generated slowpath explicitly preserves all
    /// registers apart from the result convention registers; our Rust helper
    /// call must do the same at the call site.
    ///
    /// The nursery slowpath helpers communicate through `x0` (result) and may
    /// clobber `x1`, so we preserve `x2..x15` and `d0..d15` here.
    fn emit_malloc_slowpath_helper_call(&mut self, helper_reg: u8) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [sp, -240]!
            ; stp x4, x5, [sp, 16]
            ; stp x6, x7, [sp, 32]
            ; stp x8, x9, [sp, 48]
            ; stp x10, x11, [sp, 64]
            ; stp x12, x13, [sp, 80]
            ; stp x14, x15, [sp, 96]
            ; stp d0, d1, [sp, 112]
            ; stp d2, d3, [sp, 128]
            ; stp d4, d5, [sp, 144]
            ; stp d6, d7, [sp, 160]
            ; stp d8, d9, [sp, 176]
            ; stp d10, d11, [sp, 192]
            ; stp d12, d13, [sp, 208]
            ; stp d14, d15, [sp, 224]
            ; blr X(helper_reg)
            ; ldp d14, d15, [sp, 224]
            ; ldp d12, d13, [sp, 208]
            ; ldp d10, d11, [sp, 192]
            ; ldp d8, d9, [sp, 176]
            ; ldp d6, d7, [sp, 160]
            ; ldp d4, d5, [sp, 144]
            ; ldp d2, d3, [sp, 128]
            ; ldp d0, d1, [sp, 112]
            ; ldp x14, x15, [sp, 96]
            ; ldp x12, x13, [sp, 80]
            ; ldp x10, x11, [sp, 64]
            ; ldp x8, x9, [sp, 48]
            ; ldp x6, x7, [sp, 32]
            ; ldp x4, x5, [sp, 16]
            ; ldp x2, x3, [sp], 240
        );
    }

    /// aarch64/assembler.py malloc_cond parity.
    ///
    /// Inline nursery bump allocation. total_size (from op.arg(0))
    /// includes GcHeader. Result in x0 = object pointer (after header).
    fn genop_call_malloc_nursery(&mut self, op: &Op) {
        let size_ref = op.arg(0).to_opref();
        // history.py ConstInt.value carried inline — prefer the inline
        // payload before falling through to the legacy pool / raw u32.
        let total_size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .map(|c| c.as_raw_i64())
                .unwrap_or(size_ref.raw() as i64)
        });
        let gc_hdr = majit_gc::header::GcHeader::SIZE as i64;
        // gc.py:525-531 — read nursery slot addresses from the active GC
        // descriptor (cpu.gc_ll_descr.get_nursery_free_addr() parity), not
        // from a process-global singleton.
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.emit_mov_imm64(0, total_size);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_nursery_slowpath as *const () as i64,
            );
            self.emit_malloc_slowpath_helper_call(2);
            self.reload_frame_if_necessary();
            return;
        }

        // aarch64/assembler.py malloc_cond uses only x0/x1 plus the reserved IP
        // scratch registers.  Keep x2..x13 available to regalloc on the fast
        // path; the collecting slow path spills/restores them below.
        self.emit_mov_imm64(16, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x16]);

        if total_size < 4096 {
            let ts = total_size as u32;
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, ts);
        } else {
            self.emit_mov_imm64(1, total_size);
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, x1);
        }

        self.emit_mov_imm64(17, nt_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x17]);
        dynasm!(self.mc ; .arch aarch64 ; cmp x1, x17);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.hi =>slow_path);

        // Fast path: x16 still holds &nursery_free; bump it and return the
        // payload pointer. `gen_initialize_tid` writes the whole header word.
        dynasm!(self.mc ; .arch aarch64
            ; str x1, [x16]      // *nursery_free = new_free
        );
        let hs = gc_hdr as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, hs     // x0 = payload pointer (after header)
            ; b =>done
        );

        // `malloc_cond`: publish the site-specific gcmap in ip1 and call the
        // one per-CPU `_build_malloc_slowpath('fixed')` body.  x0/x1 still
        // hold old/new nursery heads; the shared body owns all register
        // preservation, collection, frame reload, write barrier and OOM exit.
        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.emit_mov_imm64(17, gcmap as i64);
        } else {
            dynasm!(self.mc ; .arch aarch64 ; mov x17, 0);
        }
        self.emit_mov_imm64(16, self.malloc_slowpath_fixed as i64);
        dynasm!(self.mc ; .arch aarch64 ; blr x16);

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// Headerless fixed-size nursery allocation.  `size` excludes any GC
    /// header; result is the old nursery base.
    ///
    /// The fast path raw-bumps the nursery, so it is correct only when the
    /// active dynasm GC is headerless-aware — its nursery must yield a raw base
    /// carrying no `GcHeader` — what an interpreter that runs its own collector
    /// over its own object arena supplies.  A headered collector such as
    /// MiniMarkGC must never back this op: its nursery walk reads a `GcHeader`
    /// at `base - GcHeader::SIZE`, which a raw base lacks.
    /// `dynasm_headerless_nursery_addrs` answers no nursery for such a
    /// collector, so the op goes straight to the slowpath, where
    /// `alloc_nursery_headerless`'s panicking default fails loudly.
    fn genop_call_malloc_nursery_headerless(&mut self, op: &Op) {
        let size_ref = op.arg(0).to_opref();
        let size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .map(|c| c.as_raw_i64())
                .unwrap_or(size_ref.raw() as i64)
        });
        let (nf_addr, nt_addr) = crate::runner::dynasm_headerless_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.emit_mov_imm64(0, size);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_nursery_slowpath_headerless as *const () as i64,
            );
            self.emit_malloc_slowpath_helper_call(2);
            self.reload_frame_if_necessary();
            return;
        }

        // As in upstream malloc_cond, x0/x1 are the only managed registers
        // clobbered by the fast path; nursery addresses live in reserved IP
        // scratch registers.
        self.emit_mov_imm64(16, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x16]);

        if size < 4096 {
            let sz = size as u32;
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, sz);
        } else {
            self.emit_mov_imm64(1, size);
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, x1);
        }

        // When the allocator reports the top slot immediately after the free
        // slot, reuse x16 as the address base; allocators that keep the two
        // slots independent retain the general path.
        if nt_addr == nf_addr.wrapping_add(std::mem::size_of::<usize>()) {
            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x16, 8]);
        } else {
            self.emit_mov_imm64(17, nt_addr as i64);
            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x17]);
        }
        dynasm!(self.mc ; .arch aarch64 ; cmp x1, x17);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.hi =>slow_path);

        // x16 still holds nf_addr from the load above: the intervening add,
        // top load, compare, and branch only write x0, x1, x17, and flags.
        // Keep that address live on the fast arm instead of materializing the
        // same 64-bit constant a second time.  The slow arm has its own call
        // sequence and does not join at this store.
        dynasm!(self.mc ; .arch aarch64
            ; str x1, [x16]      // *nursery_free = new_free
            ; b =>done
        );

        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [x29, base_ofs + 2 * 8]
            ; stp x4, x5, [x29, base_ofs + 4 * 8]
            ; stp x6, x7, [x29, base_ofs + 6 * 8]
            ; stp x8, x9, [x29, base_ofs + 8 * 8]
            ; stp x10, x11, [x29, base_ofs + 10 * 8]
            ; stp x12, x13, [x29, base_ofs + 12 * 8]
            ; stp x19, x20, [x29, base_ofs + 14 * 8]
        );
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
            dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        }
        self.emit_mov_imm64(0, size);
        self.emit_mov_imm64(
            2,
            crate::runner::dynasm_nursery_slowpath_headerless as *const () as i64,
        );
        self.emit_malloc_slowpath_helper_call(2);
        self.reload_frame_if_necessary();
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        let base_ofs_r = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        dynasm!(self.mc ; .arch aarch64
            ; ldp x2, x3, [x29, base_ofs_r + 2 * 8]
            ; ldp x4, x5, [x29, base_ofs_r + 4 * 8]
            ; ldp x6, x7, [x29, base_ofs_r + 6 * 8]
            ; ldp x8, x9, [x29, base_ofs_r + 8 * 8]
            ; ldp x10, x11, [x29, base_ofs_r + 10 * 8]
            ; ldp x12, x13, [x29, base_ofs_r + 12 * 8]
            ; ldp x19, x20, [x29, base_ofs_r + 14 * 8]
        );
        self.emit_propagate_memory_error_if_null(0);

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate a typed GC array.
    ///
    /// Rewrite normally replaces these with `CALL_MALLOC_NURSERY_VARSIZE` /
    /// `malloc_array`. This leftover path must still stamp the collector
    /// type id and write length at the descr's `lendescr` offset — not
    /// libc malloc with a string-header store at +8.
    fn genop_new_array(&mut self, op: &Op, arglocs: &[Loc]) {
        let [len_loc, ..] = arglocs else {
            panic!("varsize allocation expects a length location, got {arglocs:?}");
        };
        let (base_size, item_size, type_id, len_ofs) = op
            .with_array_descr(|ad| {
                (
                    ad.base_size() as i64,
                    ad.item_size() as i64,
                    ad.type_id() as i64,
                    ad.len_descr().map_or(0, |fd| fd.offset() as i64),
                )
            })
            .unwrap_or((8, 8, 0, 0));
        let clear = matches!(op.opcode, OpCode::NewArrayClear) as i64;
        self.emit_load_to_rax(*len_loc);
        dynasm!(self.mc ; .arch aarch64 ; mov x4, x0);
        self.emit_mov_imm64(0, base_size);
        self.emit_mov_imm64(1, item_size);
        self.emit_mov_imm64(2, len_ofs);
        self.emit_mov_imm64(3, type_id);
        self.emit_mov_imm64(5, clear);
        self.emit_mov_imm64(
            8,
            crate::runner::dynasm_malloc_new_array as *const () as i64,
        );
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-16]!
            ; blr x8
            ; ldp x29, x30, [sp], #16
        );
    }

    // genop_* — misc

    // assembler.py genop_save_exc_class / genop_save_exception

    /// assembler.py genop_save_exc_class:
    /// `MOV resloc, [pos_exception]`.  The regalloc always assigns the
    /// result a register (`consider_no_arg_result`).
    fn genop_save_exc_class(&mut self, result_loc: Option<&Loc>) {
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        match result_loc {
            Some(Loc::Reg(dst)) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]);
            }
            Some(Loc::Frame(frame)) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x16]);
                self.emit_str_fp(17, frame.ebp_loc.value);
            }
            None => {}
            Some(other) => {
                panic!("genop_save_exc_class: unhandled result location {other:?}")
            }
        }
    }

    /// assembler.py:1845-1850 `_restore_exception`:
    /// `MOV [pos_exc_value], excvalloc; MOV [pos_exception], exctploc`.
    /// arglocs = [class, value]; the regalloc brings both into registers
    /// (`consider_restore_exception`).  x16/x17 are the scratch pair.
    fn genop_restore_exception(&mut self, arglocs: &[Loc]) {
        if arglocs.len() < 2 {
            return;
        }
        let load_to_x17 = |this: &mut Self, loc: &Loc| match loc {
            Loc::Reg(src) => {
                let src = src.value;
                dynasm!(this.mc ; .arch aarch64 ; mov x17, X(src));
            }
            Loc::Frame(frame) => this.emit_ldr_fp(17, frame.ebp_loc.value),
            Loc::Immed(imm) | Loc::ImmedFloat(imm) => this.emit_mov_imm64(17, imm.value),
            other => panic!(
                "genop_restore_exception: unhandled operand {other:?} — x17 still \
            holds the previously loaded operand, which the store below writes"
            ),
        };
        load_to_x17(self, &arglocs[1]); // value
        self.emit_mov_imm64(16, crate::jit_exc_value_addr() as i64);
        dynasm!(self.mc ; .arch aarch64 ; str x17, [x16]);
        load_to_x17(self, &arglocs[0]); // class
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        dynasm!(self.mc ; .arch aarch64 ; str x17, [x16]);
    }

    // genop_* — extended integer arithmetic

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    #[allow(dead_code)]
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; umulh x0, x0, x1
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — extended float operations

    /// FLOAT_ABS: result = |arg0|
    #[allow(dead_code)]
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fabs d0, d0
        );
        self.store_d0_to_result(op.pos().get());
    }

    // genop_* — GC memory operations

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    #[allow(dead_code)]
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        match size {
            1 => dynasm!(self.mc ; .arch aarch64 ; strb w1, [x0]),
            2 => dynasm!(self.mc ; .arch aarch64 ; strh w1, [x0]),
            4 => dynasm!(self.mc ; .arch aarch64 ; str w1, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]),
        }
    }

    /// Resolve an OpRef that is expected to be a compile-time constant.
    fn resolve_const_or(&self, opref: OpRef, default: i64) -> i64 {
        match self.resolve_opref(opref) {
            ResolvedArg::Const(v) => v,
            _ => default,
        }
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    #[allow(dead_code)]
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self
            .resolve_const_or(op.arg(3).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    #[allow(dead_code)]
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4).to_opref(), 0) as i32;
        let itemsize = self
            .resolve_const_or(op.arg(5).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        if scale != 1 {
            self.emit_mov_imm64(2, scale as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    // genop_* — interior field operations

    // genop_* — call variants

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    ///
    /// arglocs come from the regalloc plan (`prepare_op_copystrcontent`):
    /// `[condition, fn, call_args...]`. aarch64 has no cc-fusion (the compare
    /// result is always materialised into a register), so the condition can
    /// be register-resident — `resolve_opref` only handles slots/constants and
    /// would fail on a register. Use the regalloc locations directly:
    /// `emit_load_to_rax` carries register/slot/immediate, and
    /// `emit_call_from_arglocs` (func_index 1) loads the call args from their
    /// regalloc locations rather than re-resolving the op operands.
    /// COND_CALL: if arg(0) != 0, call arg(1)(arg(2..)); discard result.
    ///
    /// `prepare_op_copystrcontent` emits no `before_call`, so the regalloc
    /// does NOT spill caller-saved registers across this op. On the taken
    /// path the callee clobbers x0..x13 + d0..d7, which would destroy any
    /// value live across the cond_call (e.g. a residual-call result stored
    /// into a frame after the cond_call). Spill every register into the
    /// jitframe and publish `jf_gcmap` — `_build_cond_call_slowpath` plus
    /// `_emit_op_cond_call`'s `push_gcmap` / `pop_gcmap`.
    fn genop_discard_cond_call(&mut self, op: &Op, arglocs: &[Loc]) {
        let _ = op;
        let skip_label = self.mc.new_dynamic_label();
        // `opassembler.py:864 _emit_op_cond_call` skips the CMP when
        // `arglocs[0] is None` — the condition is already in the flags.
        // Here that case is signalled by `guard_success_cc`, published by
        // `flush_cc` when the regalloc handed the comparison the
        // frame-register sentinel.
        if let Some(cc) = self.guard_success_cc.take() {
            self.emit_jcc_to_label(invert_cc(cc), skip_label);
        } else {
            // Test the condition in a scratch register (ip0/x16), not an
            // allocatable register, so the test never clobbers a live value
            // before it is saved.
            self.emit_load_loc_to_ip0(arglocs[0]);
            dynasm!(self.mc ; .arch aarch64 ; cbz x16, =>skip_label);
        }

        // aarch64/opassembler.py `_emit_op_cond_call`:
        //   gcmap = self._regalloc.get_gcmap([res_loc])
        //   self.push_gcmap(self.mc, gcmap)
        //   BL cond_call_slowpath  # `_push_all_regs_to_jitframe` then BLR
        //   self.pop_gcmap(self.mc)
        // `clear_vable_token` → `force_now` allocates; a CPU-stack save of
        // volatiles is not a GC root, and a null `jf_gcmap` leaves every
        // spilled Ref slot unforwarded.
        self.push_all_regs_to_jitframe(&[], true);
        let pushed_gcmap = self.push_pending_call_gcmap();
        self.emit_call_from_arglocs(arglocs, 1, 0);
        self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
        self.pop_all_regs_from_jitframe(&[], true);

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);
    }

    /// Load `loc` into the ip0 scratch register (x16) without touching any
    /// allocatable register.
    fn emit_load_loc_to_ip0(&mut self, loc: Loc) {
        match loc {
            Loc::Reg(r) if !r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; mov x16, X(r.value));
            }
            Loc::Immed(imm) | Loc::ImmedFloat(imm) => {
                self.emit_mov_imm64(16, imm.value);
            }
            Loc::Frame(f) if !f.ebp_loc.is_float => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
            }
            _ => {
                self.emit_load_to_rax(loc);
                dynasm!(self.mc ; .arch aarch64 ; mov x16, x0);
            }
        }
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    ///
    /// aarch64/opassembler.py `_emit_op_cond_call` / `emit_op_cond_call_value_i`
    /// (`_r` is the same). Arglocs are `[argloc, resloc]` from
    /// `_prepare_op_cond_call`; extra args already sit in `argument_regs`.
    /// `CMP argloc, 0` then `B.NE` over the slow path; on miss the helper
    /// returns a plain word that is moved into `res_loc`. No
    /// `store_rax_to_result`.
    fn genop_cond_call_value(&mut self, op: &Op, arglocs: &[Loc]) {
        let (argloc, resloc) = match arglocs {
            [argloc, resloc, ..] => (*argloc, *resloc),
            other => panic!(
                "COND_CALL_VALUE arglocs are [argloc, resloc] (aarch64/regalloc.py _prepare_op_cond_call), got {other:?}"
            ),
        };
        let skip_label = self.mc.new_dynamic_label();
        // Test in ip0 so a miss-path extra arg already in x0 is not clobbered.
        self.emit_load_loc_to_ip0(argloc);
        dynasm!(self.mc ; .arch aarch64 ; cbnz x16, =>skip_label);

        self.push_all_regs_to_jitframe(&[], true);
        let pushed_gcmap = self.push_pending_call_gcmap();
        self.emit_cond_call_value_helper(op);
        // `_build_cond_call_slowpath`: `_reload_frame_if_necessary` does not
        // touch x0; stash the helper word to ip1 (reload uses ip0) so the
        // restore can put every managed register back, then `MOV res_loc, ip1`.
        self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
        dynasm!(self.mc ; .arch aarch64 ; mov x17, x0);
        self.pop_all_regs_from_jitframe(&[], true);
        self.regalloc_mov(&Loc::Reg(crate::aarch64::registers::IP1), &resloc);

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);
    }

    /// Inline `cond_call_slowpath` body: extra args already in
    /// `argument_regs`, func is `op.getarg(1)` Const, `BLR ip1`.
    fn emit_cond_call_value_helper(&mut self, op: &Op) {
        let func = match self.resolve_opref(op.arg(1).to_opref()) {
            ResolvedArg::Const(val) => val,
            ResolvedArg::Slot(_) => {
                panic!("COND_CALL_VALUE func is Const (aarch64/regalloc.py _prepare_op_cond_call)")
            }
        };
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
        self.emit_mov_imm64(17, func);
        dynasm!(self.mc ; .arch aarch64
            ; blr x17
            ; ldp x29, x30, [sp], #16
        );
    }

    // genop_* — string/array operations

    /// aarch64/opassembler.py `emit_op_load_effective_address` /
    /// `_gen_address`: `result = base + (index << scale) + static_offset`.
    /// `resoperation.py` args `[v_gcptr, v_index, c_baseofs, c_shift]`.
    fn genop_load_effective_address(&mut self, arglocs: &[Loc], result_loc: Option<&Loc>) {
        let Some(Loc::Reg(dst)) = result_loc else {
            panic!("LoadEffectiveAddress result_loc must be Loc::Reg, got {result_loc:?}");
        };
        let [base, index, baseofs, shift] = match arglocs {
            [a, b, c, d, ..] => [a, b, c, d],
            other => panic!("LoadEffectiveAddress expects 4 arglocs, got {other:?}"),
        };
        let shift_amt = match shift {
            Loc::Immed(i) | Loc::ImmedFloat(i) => i.value,
            other => panic!(
                "LoadEffectiveAddress shift must be Immed (rewrite.py ConstInt), got {other:?}"
            ),
        };
        let ofs = match baseofs {
            Loc::Immed(i) | Loc::ImmedFloat(i) => i.value,
            other => panic!(
                "LoadEffectiveAddress baseofs must be Immed (rewrite.py ConstInt), got {other:?}"
            ),
        };
        let base_reg = self.load_loc_to_reg(base, 16);
        let index_reg = self.load_loc_to_reg(index, 17);
        if shift_amt > 0 {
            if !(0..=3).contains(&shift_amt) {
                panic!(
                    "LoadEffectiveAddress shift must be 0..=3 (rewrite.py itemscale), got {shift_amt}"
                );
            }
            // `_gen_address`: LSL into ip0 (x16) when scale > 0.  Reload
            // the base first if it was staged in that same scratch.
            let scaled = 16u8;
            if base_reg == scaled {
                dynasm!(self.mc ; .arch aarch64 ; mov X(dst.value), X(base_reg));
                dynasm!(self.mc ; .arch aarch64
                    ; lsl X(scaled), X(index_reg), shift_amt as u32
                    ; add X(dst.value), X(dst.value), X(scaled)
                );
            } else {
                dynasm!(self.mc ; .arch aarch64
                    ; lsl X(scaled), X(index_reg), shift_amt as u32
                    ; add X(dst.value), X(base_reg), X(scaled)
                );
            }
        } else {
            dynasm!(self.mc ; .arch aarch64 ; add X(dst.value), X(base_reg), X(index_reg));
        }
        if ofs == 0 {
            return;
        }
        self.emit_mov_imm64(16, ofs);
        dynasm!(self.mc ; .arch aarch64 ; add X(dst.value), X(dst.value), x16);
    }

    /// NEWSTR: allocate a byte string of given length.
    /// `base_size` / `item_size` come from the injected ArrayDescr
    /// (`builtin_string_array_descr` in `runner.rs`), which encodes
    /// `get_array_token(rstr.STR, ...)` — basesize includes the +1
    /// extra_item_after_alloc null terminator.
    fn genop_newstr(&mut self, op: &Op, arglocs: &[Loc]) {
        let (base_size, item_size, type_id) = Self::array_token_from_descr(op, 16, 1);
        self.genop_alloc_lowlevel_string(arglocs, type_id, base_size, item_size);
    }

    /// NEWUNICODE: allocate a unicode string (4-byte chars).
    /// Basesize = 16 (no extra_item_after_alloc), itemsize = 4.
    fn genop_newunicode(&mut self, op: &Op, arglocs: &[Loc]) {
        let (base_size, item_size, type_id) = Self::array_token_from_descr(op, 16, 4);
        self.genop_alloc_lowlevel_string(arglocs, type_id, base_size, item_size);
    }

    /// Read `(base_size, item_size)` from the injected ArrayDescr.
    /// Fallback used only when the descr is missing (should never happen
    /// for NEWSTR/NEWUNICODE after `inject_builtin_string_descrs`).
    fn array_token_from_descr(op: &Op, fallback_base: i64, fallback_item: i64) -> (i64, i64, i64) {
        op.with_array_descr(|ad| {
            (
                ad.base_size() as i64,
                ad.item_size() as i64,
                ad.type_id() as i64,
            )
        })
        .unwrap_or((fallback_base, fallback_item, 0))
    }

    fn genop_alloc_lowlevel_string(
        &mut self,
        arglocs: &[Loc],
        type_id: i64,
        base_size: i64,
        item_size: i64,
    ) {
        // The length comes from its regalloc location: `consider_raw_call_like` plans
        // these opcodes, and `before_call` leaves a value bound to a
        // callee-saved member of the allocation pool in its register, where
        // `resolve_opref` cannot see it.
        let [len_loc, ..] = arglocs else {
            panic!("lowlevel string allocation expects a length location, got {arglocs:?}");
        };
        self.emit_load_to_rax(*len_loc);
        dynasm!(self.mc ; .arch aarch64 ; mov x3, x0);
        self.emit_mov_imm64(0, type_id);
        self.emit_mov_imm64(1, base_size);
        self.emit_mov_imm64(2, item_size);
        self.emit_mov_imm64(
            8,
            crate::runner::dynasm_malloc_lowlevel_string as *const () as i64,
        );
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-16]!
            ; blr x8
            ; ldp x29, x30, [sp], #16
        );
    }

    /// ZERO_ARRAY: zero a range in an array.
    /// arg(0)=base, arg(1)=start, arg(2)=size, arg(3)=scale_start, arg(4)=scale_size.
    fn genop_discard_zero_array(&mut self, op: &Op, arglocs: &[Loc]) {
        let [
            base_loc,
            start_loc,
            size_loc,
            scale_start_loc,
            scale_size_loc,
        ] = arglocs
        else {
            panic!("ZERO_ARRAY expects five regalloc locations, got {arglocs:?}");
        };
        if matches!(size_loc, Loc::Immed(i) | Loc::ImmedFloat(i) if i.value == 0) {
            return;
        }
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));

        // The scale operands are the `st.const_int(scale)` pair that
        // `rewrite.rs` emits for every ZERO_ARRAY, so `make_sure_var_in_reg`
        // hands them back as bare immediates (`return_constant` with no
        // selected_reg).  Read them from there rather than re-resolving the
        // op: a non-constant scale is an invariant break the emitter cannot
        // encode, and failing loud declines the trace instead of silently
        // scaling by one.
        let (Loc::Immed(scale_start) | Loc::ImmedFloat(scale_start)) = scale_start_loc else {
            panic!("ZERO_ARRAY scale_start must be an immediate, got {scale_start_loc:?}");
        };
        let (Loc::Immed(scale_size) | Loc::ImmedFloat(scale_size)) = scale_size_loc else {
            panic!("ZERO_ARRAY scale_size must be an immediate, got {scale_size_loc:?}");
        };
        let (scale_start, scale_size) = (scale_start.value, scale_size.value);

        // aarch64/opassembler.py emit_op_zero_array: first compute the byte destination
        // in ip0/x16.  ip0/ip1 are never managed by regalloc.
        self.regalloc_mov(base_loc, &Loc::Reg(crate::aarch64::registers::X16));
        if let Loc::Immed(start) | Loc::ImmedFloat(start) = start_loc {
            let byte_offset = base_size + start.value * scale_start;
            if byte_offset != 0 {
                if (0..4096).contains(&byte_offset) {
                    dynasm!(self.mc ; .arch aarch64 ; add x16, x16, byte_offset as u32);
                } else {
                    self.emit_mov_imm64(17, byte_offset);
                    dynasm!(self.mc ; .arch aarch64 ; add x16, x16, x17);
                }
            }
        } else {
            self.regalloc_mov(start_loc, &Loc::Reg(crate::aarch64::registers::X17));
            if scale_start != 1 {
                self.emit_mov_imm64(2, scale_start);
                dynasm!(self.mc ; .arch aarch64 ; mul x17, x17, x2);
            }
            if (0..4096).contains(&base_size) {
                dynasm!(self.mc ; .arch aarch64
                    ; add x16, x16, x17
                    ; add x16, x16, base_size as u32
                );
            } else {
                self.emit_mov_imm64(2, base_size);
                dynasm!(self.mc ; .arch aarch64
                    ; add x16, x16, x17
                    ; add x16, x16, x2
                );
            }
        }

        let natural_size = if item_size & 1 != 0 {
            1
        } else if item_size & 2 != 0 {
            2
        } else if item_size & 4 != 0 {
            4
        } else {
            8
        };
        let constant_start = matches!(start_loc, Loc::Immed(_) | Loc::ImmedFloat(_));
        let limit = if natural_size < 8 && constant_start {
            8
        } else {
            natural_size
        };
        let constant_bytes = match size_loc {
            Loc::Immed(size) | Loc::ImmedFloat(size) => Some(size.value * scale_size),
            _ => None,
        };

        if let Some(total_size) = constant_bytes.filter(|size| *size >= 0 && *size <= 14 * limit) {
            // aarch64/opassembler.py:796-835 — inline STRB/STRH/STRW/STR.
            // For a constant byte start, group naturally aligned runs into
            // eight-byte stores exactly as upstream does.
            let start_byte = match start_loc {
                Loc::Immed(start) | Loc::ImmedFloat(start) => start.value * scale_start,
                _ => -1,
            };
            let mut next_group = -1;
            if natural_size < 8 && start_byte >= 0 {
                next_group = (-start_byte) & 7;
            }
            let mut i = 0;
            let mut dst_i = 0;
            while i < total_size {
                let mut size = natural_size;
                if i == next_group {
                    next_group += 8;
                    if next_group <= total_size {
                        size = 8;
                        if dst_i % 8 != 0 {
                            dynasm!(self.mc ; .arch aarch64 ; add x16, x16, dst_i as u32);
                            dst_i = 0;
                        }
                    }
                }
                let offset = dst_i as u32;
                match size {
                    8 => dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16, offset]),
                    4 => dynasm!(self.mc ; .arch aarch64 ; str wzr, [x16, offset]),
                    2 => dynasm!(self.mc ; .arch aarch64 ; strh wzr, [x16, offset]),
                    _ => dynasm!(self.mc ; .arch aarch64 ; strb wzr, [x16, offset]),
                }
                i += size;
                dst_i += size;
            }
            return;
        }

        // Residual non-collecting call.  Regalloc has already applied its
        // normal `before_call(SAVE_DEFAULT_REGS)` policy, so no blanket
        // jitframe save/restore is needed here.
        let memset_ptr = libc::memset as *const () as i64;
        self.regalloc_mov(size_loc, &Loc::Reg(crate::aarch64::registers::X2));
        if scale_size != 1 {
            self.emit_mov_imm64(1, scale_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x2, x2, x1);
        }
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x16                        // dest
            ; mov x1, 0
            ; stp x29, x30, [sp, #-16]!
        );
        self.emit_mov_imm64(8, memset_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
        );
    }

    // genop_* — address computation
}

/// Flush icache — aarch64 only.
fn flush_icache(addr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        unsafe { sys_icache_invalidate(addr as *mut u8, len) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(addr as *mut u8, (addr as *mut u8).add(len)) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use majit_backend::{Backend, JitCellToken};
    use majit_ir::forwarding::bound_operand_from_opref;
    use majit_ir::operand::Operand;
    use majit_ir::{
        GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value, make_array_descr_signed,
        make_loop_target_descr,
    };

    use crate::regalloc::rebind_trace_arg_boxes;
    use crate::runner::DynasmBackend;
    use dynasmrt::{DynasmApi, dynasm};

    /// Encode a 32-bit aarch64 instruction with the same dynasm path the
    /// prologue uses, so a search through the compiled buffer matches the
    /// word `_call_header` would have emitted.
    fn aarch64_insn_bytes(emit: impl FnOnce(&mut super::Assembler)) -> [u8; 4] {
        let mut mc = super::Assembler::new(0);
        emit(&mut mc);
        let buf = mc.finalize().unwrap();
        assert_eq!(buf.len(), 4, "expected a single 32-bit instruction");
        [buf[0], buf[1], buf[2], buf[3]]
    }

    /// `_call_header` saves fp/lr and `r.callee_saved_registers` (x19, x20).
    /// The overflow fallthrough and `_call_footer` must restore that same
    /// set (`gen_func_epilog`). Reloading x21/x22 or d8 from slots the
    /// prologue no longer writes would return to native code with those
    /// callee-saved registers overwritten.
    #[test]
    fn call_header_overflow_return_restores_only_saved_callee_regs() {
        static STACK_END: AtomicUsize = AtomicUsize::new(usize::MAX);
        static STACK_LEN: AtomicUsize = AtomicUsize::new(usize::MAX);
        extern "C" fn dummy_slowpath(_sp: usize) -> u8 {
            0
        }
        crate::register_stack_check_addresses(
            &STACK_END as *const AtomicUsize as usize,
            &STACK_LEN as *const AtomicUsize as usize,
            dummy_slowpath as *const () as usize,
        );

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(0));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());
        let token = JitCellToken::new(530);
        let rebind_ops_1 = [OpRc::new(finish)];
        rebind_trace_arg_boxes(&[], &rebind_ops_1);
        backend
            .compile_loop(&[], &rebind_ops_1, &token)
            .expect("compile a loop whose prologue includes the stack check");

        let compiled = token
            .compiled
            .get()
            .expect("compiled code")
            .downcast_ref::<super::CompiledCode>()
            .expect("dynasm compiled code");
        let code = unsafe {
            std::slice::from_raw_parts(
                compiled.buffer.ptr(dynasmrt::AssemblyOffset(0)),
                compiled.buffer.len(),
            )
        };

        let restore_x19_x20 = aarch64_insn_bytes(|mc| {
            dynasm!(mc ; .arch aarch64 ; ldp x19, x20, [sp, #16]);
        });
        let restore_fp_lr = aarch64_insn_bytes(|mc| {
            dynasm!(mc ; .arch aarch64 ; ldp x29, x30, [sp], super::CALL_FRAME_SIZE as i32);
        });
        let forbidden_x21_x22 = aarch64_insn_bytes(|mc| {
            dynasm!(mc ; .arch aarch64 ; ldp x21, x22, [sp, #32]);
        });
        let forbidden_d8 = aarch64_insn_bytes(|mc| {
            dynasm!(mc ; .arch aarch64 ; ldr d8, [sp, #56]);
        });

        assert!(
            code.windows(4).any(|w| w == restore_x19_x20),
            "overflow return / epilogue must restore x19/x20"
        );
        assert!(
            code.windows(4).any(|w| w == restore_fp_lr),
            "overflow return / epilogue must restore fp/lr with CALL_FRAME_SIZE"
        );
        assert!(
            !code.windows(4).any(|w| w == forbidden_x21_x22),
            "must not LDP x21, x22 from a slot _call_header no longer writes"
        );
        assert!(
            !code.windows(4).any(|w| w == forbidden_d8),
            "must not LDR d8 from a slot _call_header no longer writes"
        );

        let mut restore_pair = [0u8; 8];
        restore_pair[..4].copy_from_slice(&restore_x19_x20);
        restore_pair[4..].copy_from_slice(&restore_fp_lr);
        assert!(
            code.windows(8).any(|w| w == restore_pair),
            "each x19/x20 restore must be followed by the fp/lr pop, matching _call_footer"
        );
    }

    /// aarch64/regalloc.py prepare_op_call_malloc_nursery keeps CALL_MALLOC_NURSERY's result in x0;
    /// only FrameManager may decide to spill it later.  The old hybrid emitter
    /// also called `allocate_slot` unconditionally, growing every recursive
    /// loop's JitFrame for an otherwise register-resident allocation result.
    #[test]
    fn malloc_nursery_result_does_not_grow_frame_depth() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        let malloc = OpRc::new(Op::new(
            OpCode::CallMallocNursery,
            &[Operand::from_opref(OpRef::const_int(32))],
        ));
        malloc.pos().set(OpRef::ref_op(0));

        let finish = Op::new(OpCode::Finish, &[Operand::from_bound_op(&malloc)]);
        finish.pos().set(OpRef::void_op(1));
        finish.set_fail_arg_types(vec![Type::Ref]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(517);
        let rebind_ops_2 = [malloc, OpRc::new(finish)];
        rebind_trace_arg_boxes(&[], &rebind_ops_2);
        backend
            .compile_loop(&[], &rebind_ops_2, &token)
            .expect("compile register-resident nursery result trace");

        let compiled = token
            .compiled
            .get()
            .expect("compiled code")
            .downcast_ref::<super::CompiledCode>()
            .expect("dynasm compiled code");
        assert_eq!(
            compiled.frame_depth.load(Ordering::Acquire),
            super::JITFRAME_FIXED_SIZE,
            "a register-resident nursery result must not allocate a shadow frame slot",
        );
    }

    /// `VFPRegisterManager.convert_to_imm` parks the bits in the machine
    /// data block (`ConstFloatLoc`) and `AssemblerARM64.load` does
    /// `gen_load_int(ip0, adr)` then `LDR_di`. The eight bytes of the
    /// constant are not in the instruction stream.
    #[test]
    fn float_constant_loads_from_literal_pool() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let inputargs = vec![InputArg::new_float_rc(0)];
        let add = OpRc::new(Op::new(
            OpCode::FloatAdd,
            &[
                bound_operand_from_opref(OpRef::input_arg_float(0)),
                Operand::from_opref(OpRef::const_float(1.5)),
            ],
        ));
        add.pos().set(OpRef::float_op(1));
        let finish = Op::new(OpCode::Finish, &[Operand::from_bound_op(&add)]);
        finish.pos().set(OpRef::void_op(2));
        finish.set_fail_arg_types(vec![Type::Float]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(522);
        let rebind_ops_3 = [add, OpRc::new(finish)];
        rebind_trace_arg_boxes(&inputargs, &rebind_ops_3);
        backend
            .compile_loop(&inputargs, &rebind_ops_3, &token)
            .expect("compile float-add constant");

        let frame = backend.execute_token(&token, &[Value::Float(2.0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_float_value(&frame, 0).to_bits(),
            3.5f64.to_bits()
        );

        let compiled = token
            .compiled
            .get()
            .expect("compiled code")
            .downcast_ref::<super::CompiledCode>()
            .expect("dynasm compiled code");
        let code = unsafe {
            std::slice::from_raw_parts(
                compiled.buffer.ptr(dynasmrt::AssemblyOffset(0)),
                compiled.buffer.len(),
            )
        };
        let bits = 1.5f64.to_le_bytes();
        assert!(
            !code.windows(8).any(|window| window == bits),
            "float bits must not sit in the code stream"
        );
        let parked = compiled.data_blocks.iter().any(|block| {
            let bytes = unsafe { std::slice::from_raw_parts(block.ptr(), block.len()) };
            bytes.windows(8).any(|window| window == bits)
        });
        assert!(parked, "float bits must live in the machine data block");
    }

    fn eval_breaker_poll_ops(word_addr: usize, first_pos: u32) -> (OpRc, OpRc, OpRc) {
        let word = OpRc::new(Op::with_descr(
            OpCode::RawLoadI,
            &[
                Operand::from_opref(OpRef::const_int(word_addr as i64)),
                Operand::from_opref(OpRef::const_int(0)),
            ],
            make_array_descr_signed(0, 8, Type::Int, true),
        ));
        word.pos().set(OpRef::int_op(first_pos));
        let armed = OpRc::new(Op::new(OpCode::IntIsTrue, &[Operand::from_bound_op(&word)]));
        armed.pos().set(OpRef::int_op(first_pos + 1));
        let guard = OpRc::new(Op::new(
            OpCode::GuardFalse,
            &[Operand::from_bound_op(&armed)],
        ));
        guard.pos().set(OpRef::void_op(first_pos + 2));
        guard.set_fail_arg_types(vec![]);
        guard.setfailargs(vec![].into());
        (word, armed, guard)
    }

    fn compile_eval_breaker_poll_trace(
        trace_id: u64,
        word_addr: usize,
    ) -> (DynasmBackend, JitCellToken) {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let token = JitCellToken::new(trace_id as i64);

        let (word, armed, guard) = eval_breaker_poll_ops(word_addr, 0);

        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let ops = vec![word, armed, guard, OpRc::new(finish)];
        rebind_trace_arg_boxes(&[], &ops);
        backend
            .compile_loop(&[], &ops, &token)
            .expect("compile eval-breaker poll IR");
        (backend, token)
    }

    #[test]
    fn eval_breaker_poll_deopts_when_bitmask_set() {
        let test_word = AtomicUsize::new(0);
        let word_addr = &test_word as *const AtomicUsize as usize;

        let (backend, token) = compile_eval_breaker_poll_trace(518, word_addr);
        let frame = backend.execute_token(&token, &[]);
        assert!(
            backend.get_latest_descr(&frame).is_finish(),
            "a zero eval-breaker word must stay on the guard fast path"
        );

        test_word.fetch_or(majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);
        let frame = backend.execute_token(&token, &[]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "the STW bit must use the existing guard-failure exit"
        );
        test_word.fetch_and(!majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);

        let (backend_after_clear, token_after_clear) =
            compile_eval_breaker_poll_trace(519, word_addr);
        let frame = backend_after_clear.execute_token(&token_after_clear, &[]);
        assert!(
            backend_after_clear.get_latest_descr(&frame).is_finish(),
            "a freshly compiled trace must pass after the STW bit clears"
        );

        test_word.fetch_or(majit_ir::eval_breaker_word::EB_ASYNC, Ordering::Relaxed);
        let frame = backend_after_clear.execute_token(&token_after_clear, &[]);
        assert!(
            !backend_after_clear.get_latest_descr(&frame).is_finish(),
            "the async bit must use the same guard-failure exit"
        );
    }

    #[test]
    fn eval_breaker_poll_cross_trace_uses_same_word() {
        let test_word = AtomicUsize::new(0);
        let word_addr = &test_word as *const AtomicUsize as usize;

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        // Compile the destination first so its LABEL descriptor carries the
        // external entry address used by the source trace's closing JUMP.
        let target_descr = make_loop_target_descr(520, false);
        let label = Op::new(OpCode::Label, &[]);
        label.pos().set(OpRef::void_op(0));
        label.setdescr(target_descr.clone());

        let (word, armed, guard) = eval_breaker_poll_ops(word_addr, 1);

        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(4));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let target_token = JitCellToken::new(520);
        let rebind_ops_4 = [OpRc::new(label), word, armed, guard, OpRc::new(finish)];
        rebind_trace_arg_boxes(&[], &rebind_ops_4);
        backend
            .compile_loop(&[], &rebind_ops_4, &target_token)
            .expect("compile cross-trace bitmask target");

        test_word.fetch_or(majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);

        // Enter the target loop body through an external branch. The ordinary
        // IR poll must still load the same constant-address word.
        let jump = Op::new(OpCode::Jump, &[]);
        jump.pos().set(OpRef::void_op(0));
        jump.setdescr(target_descr);
        let source_token = JitCellToken::new(521);
        let rebind_ops_5 = [OpRc::new(jump)];
        rebind_trace_arg_boxes(&[], &rebind_ops_5);
        backend
            .compile_loop(&[], &rebind_ops_5, &source_token)
            .expect("compile external JUMP source");

        let frame = backend.execute_token(&target_token, &[]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "the first trace must deopt through its baked bitmask address"
        );

        let frame = backend.execute_token(&source_token, &[]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "external entry must deopt on the same bitmask word"
        );
    }

    // Drives the process-global STW machinery (`quiesce_mutators`) to prove a
    // running compiled loop parks and resumes through its baked back-edge poll.
    // Held-open global STW makes every sibling collection test in this binary
    // take the active-STW path and race the shared STW_DEPTH, so it cannot run
    // under default parallel `cargo test`. Not a production concern: real
    // mutators park at their safepoint before they could collect concurrently.
    // Run on demand: `cargo test -- --ignored --test-threads=1`.
    #[test]
    #[ignore = "drives process-global STW; run serially via --ignored --test-threads=1"]
    fn compiled_eval_breaker_parks_and_resumes_mutator() {
        let test_word = Arc::new(AtomicUsize::new(0));
        let deopted = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel();
        let (resumed_tx, resumed_rx) = mpsc::channel();
        let worker_deopted = deopted.clone();
        let worker_test_word = test_word.clone();
        let worker = std::thread::spawn(move || {
            majit_gc::gc_sync::register_thread();
            let mut backend = DynasmBackend::new();
            backend.attach_default_test_descrs();
            let target_descr = make_loop_target_descr(522, false);
            let label = Op::new(OpCode::Label, &[]);
            label.pos().set(OpRef::void_op(0));
            label.setdescr(target_descr.clone());
            let (word, armed, guard) =
                eval_breaker_poll_ops(Arc::as_ptr(&worker_test_word) as usize, 1);
            let jump = Op::new(OpCode::Jump, &[]);
            jump.pos().set(OpRef::void_op(4));
            jump.setdescr(target_descr);
            let token = JitCellToken::new(522);
            let rebind_ops_6 = [OpRc::new(label), word, armed, guard, OpRc::new(jump)];
            rebind_trace_arg_boxes(&[], &rebind_ops_6);
            backend
                .compile_loop(&[], &rebind_ops_6, &token)
                .expect("compile polling loop on worker");

            ready_tx.send(()).unwrap();
            let frame = backend.execute_token(&token, &[]);
            assert!(
                !backend.get_latest_descr(&frame).is_finish(),
                "the running compiled loop must deopt on the STW bit"
            );
            worker_deopted.store(true, Ordering::Release);
            majit_gc::gc_sync::safepoint_poll();
            assert_eq!(
                worker_test_word.load(Ordering::Relaxed) & majit_ir::eval_breaker_word::EB_STW,
                0,
                "the resumed mutator must observe a balanced STW bit"
            );
            resumed_tx.send(()).unwrap();
            majit_gc::gc_sync::unregister_thread();
        });

        ready_rx.recv().unwrap();
        let armer_word = test_word.clone();
        let armer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            armer_word.fetch_or(majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);
        });
        let stw = majit_gc::gc_sync::quiesce_mutators();
        assert!(
            deopted.load(Ordering::Acquire),
            "quiescence must be reached through the compiled guard exit"
        );
        assert_ne!(
            test_word.load(Ordering::Relaxed) & majit_ir::eval_breaker_word::EB_STW,
            0,
            "the STW bit must remain armed while the mutator is parked"
        );
        test_word.fetch_and(!majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);
        assert_eq!(
            test_word.load(Ordering::Relaxed) & majit_ir::eval_breaker_word::EB_STW,
            0,
            "the test-local STW mirror must clear before mutator resume"
        );
        drop(stw);
        armer.join().unwrap();
        resumed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the compiled-loop worker must resume cleanly");
        worker.join().unwrap();
    }

    // ── COND_CALL_GC_WB_ARRAY inline card marking ──────────────────────

    /// Array length that satisfies both clauses of the card question that a
    /// fixture controls (incminimark.py:1017-1019).
    ///
    /// Indices must land in more than one card byte at the default
    /// `card_page_indices = 128` (incminimark.py:275): eight cards per byte
    /// means index 1024 is the first index in card byte 1.
    ///
    /// And the array must be a *large* object — the third clause denies cards
    /// to anything the arena arm could have taken. A whole card page above
    /// `(16384 + 512) * WORD` clears it with the base and length words
    /// included; the indices below still land in bytes 0 and 1, and the card
    /// bytes past them stay clean.
    const CARD_ARRAY_LENGTH: usize = 17024;

    /// incminimark.py `external_malloc` with card bits: an
    /// old-gen varsize array whose items are GC pointers gets GCFLAG_HAS_CARDS
    /// and a run of zeroed card bytes in front of the header.
    fn alloc_old_card_array(gc: &mut majit_gc::collector::MiniMarkGC, type_id: u32) -> GcRef {
        let item_size = std::mem::size_of::<GcRef>();
        let total_size = majit_gc::header::GcHeader::SIZE + 8 + item_size * CARD_ARRAY_LENGTH;
        // Said out loud rather than left to the card assertions: a length that
        // stopped clearing the threshold would disarm this test into asserting
        // that two clean arrays match.
        assert!(
            total_size >= majit_gc::GcAllocator::max_nursery_object_size(gc),
            "CARD_ARRAY_LENGTH no longer describes a large object, so it gets no cards"
        );
        let obj = gc.alloc_in_oldgen_with_cards(type_id, total_size, CARD_ARRAY_LENGTH, true);
        assert!(
            unsafe {
                (*majit_gc::header::header_of(obj.0)).has_flag(majit_gc::GcFlags::GCFLAG_HAS_CARDS)
            },
            "the fixture array must carry cards"
        );
        // `dirty_cards` reads the length out of the array's own length field.
        unsafe { *(obj.0 as *mut usize) = CARD_ARRAY_LENGTH };
        obj
    }

    /// Compile and run a one-operation trace holding a single
    /// `COND_CALL_GC_WB_ARRAY` against `obj`.
    ///
    /// `index_in_register` selects which argloc kind the emitter sees:
    /// a non-constant `InputArg` is forced into a core register, while a
    /// `ConstInt` reaches `RegisterManager::return_constant`
    /// (llsupport/regalloc.py) with no selected register and comes back
    /// as a bare `Loc::Immed`.
    fn run_cond_call_gc_wb_array(trace_id: u64, obj: GcRef, index: i64, index_in_register: bool) {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        let mut inputargs = vec![InputArg::new_ref_rc(0)];
        let mut values = vec![Value::Ref(obj)];
        let index_operand = if index_in_register {
            inputargs.push(InputArg::new_int_rc(1));
            values.push(Value::Int(index));
            bound_operand_from_opref(OpRef::input_arg_int(1))
        } else {
            bound_operand_from_opref(OpRef::const_int(index))
        };

        let barrier = Op::new(
            OpCode::CondCallGcWbArray,
            &[
                bound_operand_from_opref(OpRef::input_arg_ref(0)),
                index_operand,
            ],
        );
        barrier.pos().set(OpRef::void_op(2));

        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(trace_id as i64);
        let rebind_ops_7 = [OpRc::new(barrier), OpRc::new(finish)];
        rebind_trace_arg_boxes(&inputargs, &rebind_ops_7);
        backend
            .compile_loop(&inputargs, &rebind_ops_7, &token)
            .expect("compile COND_CALL_GC_WB_ARRAY trace");
        let frame = backend.execute_token(&token, &values);
        assert!(
            backend.get_latest_descr(&frame).is_finish(),
            "the barrier trace must run to its FINISH"
        );
    }

    /// opassembler.py:996-1015 inline card marking, immediate-index arm.
    ///
    /// The register arm shifts the index at runtime; the immediate arm folds
    /// the same two quantities — card byte displacement and card bit — at
    /// assembly time (x86/assembler.py:2382-2386). Both must dirty exactly
    /// the card `mark_card` (incminimark.py:1574-1598) would dirty.
    ///
    /// Regression cover: while the whole card sequence sat under a match that
    /// only admitted `Loc::Reg`, an immediate index assembled to zero bytes
    /// and the array kept a clean card across a barrier that was supposed to
    /// dirty one.
    #[test]
    fn cond_call_gc_wb_array_immed_index_marks_same_card_as_reg_index() {
        // gc.py:273 JIT_WB_CARDS_SET — zero means the backend emits no card
        // sequence at all, which would leave this test asserting nothing.
        let wb = crate::runner::dynasm_write_barrier_descr()
            .expect("a write barrier descriptor must be resolvable");
        assert_ne!(
            wb.jit_wb_cards_set, 0,
            "card marking must be enabled for this test to exercise the card arms"
        );
        let card_page_shift = wb.jit_wb_card_page_shift;

        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let item_size = std::mem::size_of::<GcRef>();
        let type_id = gc.register_type(majit_gc::TypeInfo::varsize(
            8,
            item_size,
            0,
            true,
            Vec::new(),
        ));
        let obj_immed = alloc_old_card_array(&mut gc, type_id);
        let obj_reg = alloc_old_card_array(&mut gc, type_id);
        let obj_interp = alloc_old_card_array(&mut gc, type_id);

        // opassembler.py:943-949 branches straight to the inline card block
        // when GCFLAG_CARDS_SET is already set, so the compiled arms never
        // reach the `jit_remember_young_pointer_from_array` helper.
        // `mark_card` sets the same flag on the interpreter's object itself.
        for obj in [obj_immed, obj_reg] {
            unsafe {
                (*majit_gc::header::header_of(obj.0)).set_flag(majit_gc::GcFlags::GCFLAG_CARDS_SET);
            }
        }
        for obj in [obj_immed, obj_reg, obj_interp] {
            assert!(
                gc.dirty_cards(obj).is_empty(),
                "a freshly allocated card array starts with every card clean"
            );
        }

        // Indices chosen to span two card bytes and several bits within them.
        const INDICES: [i64; 5] = [0, 5, 200, 1152, 2047];
        for (n, &index) in INDICES.iter().enumerate() {
            let trace_id = 9100 + 2 * n as u64;
            run_cond_call_gc_wb_array(trace_id, obj_immed, index, false);
            run_cond_call_gc_wb_array(trace_id + 1, obj_reg, index, true);
            gc.do_write_barrier_card(obj_interp, index as usize, card_page_shift);
            // Compare after every index, not only at the end: an aggregate
            // comparison would accept two arms that dirty the same set of
            // cards while pairing them with different indices.
            assert_eq!(
                gc.dirty_cards(obj_immed),
                gc.dirty_cards(obj_reg),
                "index {index} must dirty the same cards through both arms"
            );
        }

        let mut expected: Vec<usize> = INDICES
            .iter()
            .map(|&index| (index as usize) >> card_page_shift)
            .collect();
        expected.sort_unstable();
        expected.dedup();

        let immed_cards = gc.dirty_cards(obj_immed);
        let reg_cards = gc.dirty_cards(obj_reg);
        let interp_cards = gc.dirty_cards(obj_interp);

        assert_eq!(
            immed_cards, reg_cards,
            "an immediate index must dirty the same cards as the register arm"
        );
        assert_eq!(
            immed_cards, interp_cards,
            "the compiled card bits must match remember_young_pointer_from_array2"
        );
        assert_eq!(
            immed_cards, expected,
            "each index must dirty exactly its own card, and nothing else"
        );
    }
}

/// `jump.py`'s three primitives. The parallel-move algorithm that drives
/// them is one shared implementation in `crate::jump`; only these differ per
/// backend, which is what makes them the trait and it the free function.
impl<'a> crate::jump::RegallocMoves for AssemblerARM64<'a> {
    /// aarch64/jump.py `_move`.
    fn _move(&mut self, src: &Loc, dst: &Loc, tmpreg: &Loc) {
        let mut src = *src;
        if dst.is_stack() && src.is_stack() {
            self.regalloc_mov(&src, tmpreg);
            src = *tmpreg;
        }
        self.regalloc_mov(&src, dst);
    }

    fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one("jit-backend", &format!("remap-mov: {src:?} -> {dst:?}"));
        }
        // A reload/spill/register move between a comparison and its guard
        // leaves them no longer the adjacent pair `_walk_operations` folds.
        self.pending_cmp_cc = None;
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            (Loc::Reg(s), Loc::Reg(d)) => {
                if s.is_xmm && d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), D(s.value));
                } else if !s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(d.value), X(s.value));
                } else if s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov X(d.value), D(s.value));
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(s.value));
                }
            }
            (Loc::Reg(s), ebp_loc_pat!(e)) => {
                let ofs = e.value;
                if s.is_xmm {
                    self.emit_str_fp_d(s.value, ofs);
                } else {
                    self.emit_str_fp(s.value, ofs);
                }
            }
            (ebp_loc_pat!(e), Loc::Reg(d)) => {
                let ofs = e.value;
                if d.is_xmm {
                    self.emit_ldr_fp_d(d.value, ofs);
                } else {
                    self.emit_ldr_fp(d.value, ofs);
                }
            }
            (Loc::ConstFloat(c), Loc::Reg(d)) if d.is_xmm => {
                self.load_imm_float(d.value, *c);
            }
            (Loc::ConstFloat(c), ebp_loc_pat!(e)) => {
                // `_mov_imm_float_to_loc`: load into `vfp_ip` (d15) then store.
                self.load_imm_float(15, *c);
                self.emit_str_fp_d(15, e.value);
            }
            (Loc::Immed(i) | Loc::ImmedFloat(i), Loc::Reg(d)) => {
                if d.is_xmm {
                    self.emit_mov_imm64(16, i.value); // x16 = scratch
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(16));
                } else {
                    self.emit_mov_imm64(d.value as u32, i.value);
                }
            }
            (Loc::Immed(i) | Loc::ImmedFloat(i), ebp_loc_pat!(e)) => {
                let ofs = e.value;
                self.emit_mov_imm64(16, i.value);
                self.emit_str_fp(16, ofs);
            }
            (ebp_loc_pat!(e1), ebp_loc_pat!(e2)) if e1.value == e2.value => {}
            (ebp_loc_pat!(e1), ebp_loc_pat!(e2)) => {
                let o1 = e1.value;
                let o2 = e2.value;
                self.emit_ldr_fp(16, o1);
                self.emit_str_fp(16, o2);
            }
            _ => panic!(
                "parallel move {src:?} -> {dst:?} is outside the RegallocMoves \
                 operand contract; emitting nothing here would leave the \
                 destination stale",
            ),
        }
    }

    fn regalloc_push(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; str D(r.value), [sp, #-16]!);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; str X(r.value), [sp, #-16]!);
            }
            ebp_loc_pat!(e) if e.is_float => {
                self.emit_ldr_fp_d(15, e.value);
                dynasm!(self.mc ; .arch aarch64 ; str D(15), [sp, #-16]!);
            }
            ebp_loc_pat!(e) => {
                self.emit_ldr_fp(16, e.value);
                dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, #-16]!);
            }
            _ => panic!(
                "parallel move cannot park {loc:?} on the stack; emitting nothing \
                 here would leave the matching pop unbalanced",
            ),
        }
    }

    fn regalloc_pop(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(r.value), [sp], #16);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(r.value), [sp], #16);
            }
            ebp_loc_pat!(e) if e.is_float => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(15), [sp], #16);
                self.emit_str_fp_d(15, e.value);
            }
            ebp_loc_pat!(e) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [sp], #16);
                self.emit_str_fp(16, e.value);
            }
            _ => panic!(
                "parallel move cannot restore {loc:?} from the stack; emitting \
                 nothing here would leave the stack pointer shifted",
            ),
        }
    }
}

/// aarch64/assembler.py `_walk_operations`: the regalloc hands each op's
/// locations to its emitter while the register managers still hold the
/// state of that op.
impl RegAllocAssembler for AssemblerARM64<'_> {
    fn regalloc_perform(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
            eprintln!(
                "[dynasm] emit[{}]: {:?} args=[{}] result={:?}",
                op_index,
                op.opcode,
                al.join(", "),
                result_loc
            );
        }
        // Byte offset where this op's code begins: the span from
        // the last LABEL to the JUMP is one loop iteration, the
        // window `jit-log-opt` brackets with its `+508:` prefixes.
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        self.pending_malloc_nursery_gcmap = gcmap;
        let fail_index = self.fail_index;
        self.asm_operations(
            regalloc,
            op,
            op_index,
            arglocs,
            result_loc.as_ref(),
            fail_index,
            ops,
        );
        self.pending_malloc_nursery_gcmap = None;
    }

    fn regalloc_perform_discard(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            let al: Vec<String> = arglocs.iter().map(|l| format!("{l:?}")).collect();
            eprintln!(
                "[dynasm] discard[{}]: {:?} args=[{}]",
                op_index,
                op.opcode,
                al.join(", ")
            );
        }
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        let fail_index = self.fail_index;
        self.asm_operations(regalloc, op, op_index, arglocs, None, fail_index, ops);
        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            self.fail_index += 1;
        }
    }

    fn regalloc_perform_guard(
        &mut self,
        _regalloc: &RegAlloc<'_>,
        op_index: usize,
        faillocs: &[Option<Loc>],
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        frame_depth: usize,
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            eprintln!(
                "[dynasm] guard[{}]: {:?} args=[{}] faillocs={}",
                op_index,
                op.opcode,
                arglocs
                    .iter()
                    .map(|l| format!("{:?}", l))
                    .collect::<Vec<_>>()
                    .join(", "),
                faillocs.len()
            );
        }
        // A guard owns bytes like any other op, so it opens its
        // own span. Without this line the guard's code is read as
        // part of the preceding op's span and every span after the
        // trace's first guard names the wrong operation.
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        let fail_index = self.fail_index;
        self.asm_guard_operations(
            op,
            op_index,
            arglocs,
            result_loc.as_ref(),
            faillocs,
            fail_index,
            frame_depth,
        );
        self.fail_index += 1;
    }
}
