//! Rust split of `pyjitpl.py` frame types.
//!
//! RPython `pyjitpl.py` keeps `MIFrame` and `MIFrameStack` in the same
//! file as `MetaInterp`. Rust splits the frame data into this submodule
//! while preserving the upstream object shape: a jitcode reference, PC,
//! and three register arrays (int/ref/float), plus the stack of nested
//! inline-call frames.

use std::sync::Arc;

use majit_ir::{OpRef, Type};

use crate::jitcode::{JitArgKind, JitCode, read_u8, read_u16};
use crate::opencoder::{Box as OpBox, TraceRecordBuffer};
use crate::recorder::SnapshotTagged;

/// Encode one register box for `_encode`. Constants carry their value
/// inline (`ConstInt.getint` / `ConstPtr.getref_base` /
/// `ConstFloat.getfloat_storage`); a value op / inputarg is TAGBOX
/// (`FrontendOp.get_position()`).
///
/// `ConstPtr.getref_base` reads the intern table. Call this after any
/// earlier `_encode` / `new_array` that may have minor-collected.
#[inline]
fn register_to_box(opref: OpRef) -> OpBox {
    match opref.inline_const_to_value() {
        Some(majit_ir::Value::Int(v)) => OpBox::ConstInt(v),
        Some(majit_ir::Value::Float(f)) => OpBox::ConstFloat(f.to_bits()),
        Some(majit_ir::Value::Ref(g)) => OpBox::ConstPtr(g.as_usize() as u64),
        Some(majit_ir::Value::Void) => {
            panic!("register_to_box: constant {opref:?} has Void type")
        }
        None => OpBox::ResOp(opref.raw()),
    }
}

/// pyjitpl.py `copy_constants`: grow `registers` only when too small.
///
/// A too-small bank is replaced (`registers = [missing] * n`), not
/// extended in place. Extending would keep the previous jitcode's
/// constant slots as the new jitcode's register prefix.
fn grow_reg_file<T>(regs: &mut Vec<Option<T>>, n: usize) {
    if regs.len() < n {
        regs.clear();
        regs.resize_with(n, || None);
    }
}

/// A single execution frame for jitcode bytecode.
///
/// RPython pyjitpl.py: class MIFrame
///
/// TODO: RPython holds `self.metainterp` as a
/// back-pointer to the owning MetaInterp; pyre takes `&mut MetaInterp`
/// as a parameter on call sites instead so that the borrow checker
/// allows `MetaInterp::framestack` to own a `Vec<MIFrame>` without
/// self-referential aliasing.  `jitcode` is held as `Arc<JitCode>` so
/// the frame can outlive any single function-scope borrow — this
/// matches the upstream model where frames live as long as
/// `MetaInterp.framestack` keeps them alive.
pub struct MIFrame {
    pub jitcode: Arc<JitCode>,
    /// pyjitpl.py `MIFrame.setup`: `self.bytecode = jitcode.code`, the
    /// code the opcode and operand readers index. It points into the body
    /// of `jitcode`, which this frame's `Arc` keeps alive and which
    /// `JitCode::set_body` writes only once; `new` and `setup_reused` set
    /// it together with `jitcode`.
    bytecode: *const [u8],
    pub pc: usize,
    pub code_cursor: usize,
    /// Bytecode position of the operation currently being executed.
    /// Exception unwinding uses this instead of the post-decode cursor when
    /// the host records the Python traceback for this frame.
    pub last_opcode_position: usize,
    pub int_regs: Vec<Option<OpRef>>,
    pub ref_regs: Vec<Option<OpRef>>,
    pub float_regs: Vec<Option<OpRef>>,
    pub inline_frame: bool,
    /// \[FR\] Saved caller sym scalar/fixed-array state for a recursive-portal
    /// INLINE frame; restored into the shared sym when the frame returns so the
    /// callee's in-place mutation of the single sym doesn't corrupt the caller.
    pub portal_scalar_state: Option<Vec<(OpRef, i64)>>,
    /// \[FR\] True only for a frame whose push recorded `ENTER_PORTAL_FRAME`
    /// (an inline-pushed portal, dispatch.rs). Its normal-return and
    /// exception-return pops record the matching `LEAVE_PORTAL_FRAME`, so every
    /// enter_portal_frame pairs with a leave_portal_frame (pyjitpl.py:
    /// newframe→enter_portal_frame, popframe→leave_portal_frame default True on
    /// both the finishframe and finishframe_exception paths). Distinct from
    /// `inline_frame`, which is also true for ordinary `BC_INLINE_CALL` frames
    /// that record no ENTER and must record no LEAVE. The merge-point cut is the
    /// sole `leave_portal_frame=False` site and re-emits LEAVE itself.
    pub portal_entered: bool,
    /// True when the matching `portal_trace_positions` open entry was
    /// recorded. Leave must push a close only then, or find_biggest_function
    /// sees an unmatched close.
    pub portal_trace_logged: bool,
    /// \[FR\] The jd_index carried in this frame's `LEAVE_PORTAL_FRAME` op, set at
    /// the same push that set `portal_entered`. Unused when `portal_entered` is
    /// false.
    pub portal_jd: usize,
    pub return_i: Option<usize>,
    pub return_r: Option<usize>,
    pub return_f: Option<usize>,
    /// pyjitpl.py `MIFrame.greenkey` — set when this frame is a
    /// recursive portal call (pyjitpl.py).
    pub greenkey: Option<super::PortalGreenKey>,
    /// pyjitpl.py:91 `self._result_argcode = 'v'`.
    ///
    /// Single-byte argcode of the *previous* opimpl's result type
    /// (`b'i'` / `b'r'` / `b'f'` / `b'v'`).  Updated by call-recording
    /// opimpls before they advance the pc; consulted by `_try_tco`
    /// (pyjitpl.py) to decide whether the call's result type
    /// matches a following `*_return` opcode.  Initialized to `b'v'`
    /// because a fresh frame's "previous opimpl" is the implicit
    /// frame setup (returns void).
    pub _result_argcode: u8,
    /// Rust bytecode adaptation for `pyjitpl.py ord(self.bytecode[self.pc - 1])`.
    ///
    /// RPython encodes the result register as the last byte before the
    /// post-call LIVE marker, so `get_list_of_active_boxes(in_a_call=True)`
    /// can recover it from `self.pc - 1`.  pyre's grouped inline-call
    /// encoding stores one result register as the last operand,
    /// so the exact result index is carried here while preserving
    /// `_result_argcode` for the line-by-line clear logic.
    pub result_arg_index: Option<usize>,
    /// pyjitpl.py `MIFrame.pushed_box` — the box that the previous
    /// `_opimpl_any_push` instruction parked on the frame.  Reset to
    /// `None` by `cleanup_registers` so a recycled frame does not keep
    /// the parked box alive.
    pub pushed_box: Option<OpRef>,
    /// pyjitpl.py `self.parent_snapshot = -1`.
    ///
    /// Set by `TraceRecordBuffer::_ensure_parent_resumedata` to the
    /// snapshot_index returned by the paired `create_snapshot(back,
    /// is_last)` call — so that a later `capture_resumedata` whose
    /// framestack shares this frame as an ancestor can short-circuit
    /// the parent chain walk by patching with `snapshot_add_prev(
    /// target.parent_snapshot)` instead of re-emitting the snapshot.
    pub parent_snapshot: i64,
    /// pyjitpl.py `self.unroll_iterations = 1`.
    pub unroll_iterations: usize,
}

impl MIFrame {
    pub fn new(jitcode: Arc<JitCode>, pc: usize) -> Self {
        // RPython `pyjitpl.py` `MIFrame.setup` creates the typed
        // register arrays sized to `num_regs_and_consts_*()`.  pyre
        // keeps the raw allocation step separate so low-level tests and
        // resume paths can construct a frame without immediately
        // interning constants into a `TraceCtx`.
        //
        // `MetaInterp::newframe`, `interpret`,
        // `trace_jitcode`, and inline-call dispatch should call
        // `MIFrame::setup(...)` below, which mirrors the rest of
        // `pyjitpl.py`.
        //
        // RPython `pyjitpl.py` `MIFrame.setup`:
        //   self.registers_i = [None] * jitcode.num_regs_and_consts_i()
        //   self.registers_r = [None] * jitcode.num_regs_and_consts_r()
        //   self.registers_f = [None] * jitcode.num_regs_and_consts_f()
        // The register file spans working registers AND the constants area
        // (`[num_regs_X .. num_regs_and_consts_X)`).
        let regs_and_consts_i = jitcode.num_regs_and_consts_i();
        let regs_and_consts_r = jitcode.num_regs_and_consts_r();
        let regs_and_consts_f = jitcode.num_regs_and_consts_f();
        let bytecode: *const [u8] = jitcode.code.as_slice();
        Self {
            jitcode,
            bytecode,
            pc,
            code_cursor: 0,
            last_opcode_position: pc,
            int_regs: vec![None; regs_and_consts_i],
            ref_regs: vec![None; regs_and_consts_r],
            float_regs: vec![None; regs_and_consts_f],
            inline_frame: false,
            portal_scalar_state: None,
            portal_entered: false,
            portal_trace_logged: false,
            portal_jd: 0,
            return_i: None,
            return_r: None,
            return_f: None,
            greenkey: None,
            _result_argcode: b'v',
            result_arg_index: None,
            pushed_box: None,
            parent_snapshot: -1,
            unroll_iterations: 1,
        }
    }

    /// `history.py AbstractValue.getint()` / `blackhole.py`
    /// `_copy_data_from_miframe` `box.getint()`. A `ConstInt` is inline.
    /// `freeze_values_into_const_boxes` copies a stamped `_resint` onto
    /// that Const; an unfrozen live box is `IntOp.getint` of unstamped
    /// `_resint` (0).
    pub fn int_value_for_blackhole(&self, index: usize) -> Option<i64> {
        self.int_regs.get(index).copied().flatten().map(|op| {
            op.inline_const_to_value()
                .map(|v| v.as_raw_i64())
                .unwrap_or(0)
        })
    }

    /// `history.py AbstractValue.getref_base()` / `_copy_data_from_miframe`
    /// `box.getref_base()`. Same box-owner rule as
    /// [`Self::int_value_for_blackhole`]: a ConstPtr is inline; an unfrozen
    /// live box is `RefOp.getref_base` of unstamped `_resref` (nullptr).
    pub fn ref_value_for_blackhole(&self, index: usize) -> Option<i64> {
        self.ref_regs.get(index).copied().flatten().map(|op| {
            op.inline_const_to_value()
                .map(|v| v.as_raw_i64())
                .unwrap_or(0)
        })
    }

    /// `history.py AbstractValue.getfloatstorage()` /
    /// `_copy_data_from_miframe` `box.getfloatstorage()`. Same box-owner
    /// rule as [`Self::int_value_for_blackhole`].
    pub fn float_value_for_blackhole(&self, index: usize) -> Option<i64> {
        self.float_regs.get(index).copied().flatten().map(|op| {
            op.inline_const_to_value()
                .map(|v| v.as_raw_i64())
                .unwrap_or(0)
        })
    }

    /// Publish a forwarding update into a Ref register's ConstPtr box.
    ///
    /// The packed blackhole-entry root array can itself be forwarded before
    /// `_copy_data_from_miframe` runs.  `ConstPtr.value` lives in
    /// `const_ptr_table`; this writes that slot (`history.py` `ConstPtr`).
    pub fn set_forwarded_ref_value(&mut self, index: usize, value: i64) {
        if let Some(OpRef::ConstPtr(slot)) = self.ref_regs[index] {
            majit_ir::const_ptr_table::set_slot(slot, majit_ir::GcRef(value as usize));
        }
    }

    /// Rewrite non-constant register boxes to inline Consts so
    /// `getint()` / `getref_base()` survive `abort_trace` dropping the
    /// recorder. `IntOp.getint` lives on the FrontendOp; blackhole
    /// copies after abort need that value on the box that remains in
    /// the register (`pyjitpl.py` `registers_i[i]` is that object).
    pub fn freeze_values_into_const_boxes(&mut self, ctx: &crate::trace_ctx::TraceCtx) {
        for slot in &mut self.int_regs {
            if let Some(op) = *slot
                && !op.is_constant()
                && let Some(v) = ctx.box_bits(op)
            {
                *slot = Some(OpRef::const_int(v));
            }
        }
        for slot in &mut self.ref_regs {
            if let Some(op) = *slot
                && !op.is_constant()
                && let Some(v) = ctx.box_bits(op)
            {
                *slot = Some(OpRef::const_ptr(majit_ir::GcRef(v as usize)));
            }
        }
        for slot in &mut self.float_regs {
            if let Some(op) = *slot
                && !op.is_constant()
                && let Some(v) = ctx.box_bits(op)
            {
                *slot = Some(OpRef::const_float(f64::from_bits(v as u64)));
            }
        }
    }

    /// `IntOp.getint` / `ConstInt.getint` / `InputArgInt.getint` for one
    /// int register.
    pub fn getint(&self, ctx: &crate::trace_ctx::TraceCtx, index: usize) -> Option<i64> {
        self.int_regs
            .get(index)
            .copied()
            .flatten()
            .and_then(|op| ctx.box_bits(op))
    }

    /// `RefOp.getref_base` / `ConstPtr.getref_base` / `InputArgRef.getref_base`.
    pub fn getref_base(&self, ctx: &crate::trace_ctx::TraceCtx, index: usize) -> Option<i64> {
        self.ref_regs
            .get(index)
            .copied()
            .flatten()
            .and_then(|op| ctx.box_bits(op))
    }

    /// `FloatOp.getfloat_storage` / `ConstFloat.getfloat_storage`.
    pub fn getfloat_storage(&self, ctx: &crate::trace_ctx::TraceCtx, index: usize) -> Option<i64> {
        self.float_regs
            .get(index)
            .copied()
            .flatten()
            .and_then(|op| ctx.box_bits(op))
    }

    /// RPython `pyjitpl.py` `MIFrame.setup(jitcode, greenkey=None)`.
    ///
    /// This is the entry-point that should back normal frame creation:
    /// allocate the typed register files, remember the recursive-portal
    /// greenkey, copy constants into the register window, and reset the
    /// per-frame bookkeeping fields that upstream initializes in
    /// `setup()`.
    pub fn setup(
        jitcode: Arc<JitCode>,
        pc: usize,
        greenkey: Option<super::PortalGreenKey>,
        ctx: Option<&mut crate::trace_ctx::TraceCtx>,
    ) -> Self {
        let mut frame = Self::new(jitcode, pc);
        frame.greenkey = greenkey;
        if let Some(ctx) = ctx {
            frame.copy_constants(ctx);
        }
        frame._result_argcode = b'v';
        frame.result_arg_index = None;
        frame.parent_snapshot = -1;
        frame.unroll_iterations = 1;
        frame
    }

    /// `pyjitpl.py MIFrame.setup` on an object taken from
    /// `MetaInterp.free_frames_list`.
    ///
    /// `copy_constants` grows a bank only when it is too small and, when
    /// translated, leaves live slots alone. Tests still fill, matching
    /// `elif not we_are_translated()`. Constants are recopied; per-call
    /// bookkeeping returns to the values `setup` writes.
    pub fn setup_reused(
        &mut self,
        jitcode: Arc<JitCode>,
        pc: usize,
        greenkey: Option<super::PortalGreenKey>,
        ctx: Option<&mut crate::trace_ctx::TraceCtx>,
    ) {
        let regs_and_consts_i = jitcode.num_regs_and_consts_i();
        let regs_and_consts_r = jitcode.num_regs_and_consts_r();
        let regs_and_consts_f = jitcode.num_regs_and_consts_f();

        self.bytecode = jitcode.code.as_slice();
        self.jitcode = jitcode;
        self.pc = pc;
        self.code_cursor = 0;
        self.last_opcode_position = pc;
        // pyjitpl.py `copy_constants`: grow if too small; the translated
        // path does not zero live slots (`elif not we_are_translated()`).
        grow_reg_file(&mut self.int_regs, regs_and_consts_i);
        grow_reg_file(&mut self.ref_regs, regs_and_consts_r);
        grow_reg_file(&mut self.float_regs, regs_and_consts_f);
        #[cfg(debug_assertions)]
        {
            // pyjitpl.py `copy_constants`: `elif not we_are_translated()`.
            self.int_regs.fill(None);
            self.ref_regs.fill(None);
            self.float_regs.fill(None);
        }
        self.inline_frame = false;
        self.portal_scalar_state = None;
        self.portal_entered = false;
        self.portal_trace_logged = false;
        self.portal_jd = 0;
        self.return_i = None;
        self.return_r = None;
        self.return_f = None;
        self.greenkey = greenkey;
        self._result_argcode = b'v';
        self.result_arg_index = None;
        self.pushed_box = None;
        self.parent_snapshot = -1;
        self.unroll_iterations = 1;
        if let Some(ctx) = ctx {
            self.copy_constants(ctx);
        }
    }

    /// pyjitpl.py `MIFrame.bytecode`.
    #[inline]
    pub fn bytecode(&self) -> &[u8] {
        debug_assert!(std::ptr::eq(self.bytecode, self.jitcode.code.as_slice()));
        // SAFETY: `bytecode` points into `self.jitcode`'s body, which the
        // `Arc` held in `self.jitcode` keeps alive and unchanged.
        unsafe { &*self.bytecode }
    }

    pub fn next_u8(&mut self) -> u8 {
        let code: *const [u8] = self.bytecode;
        // SAFETY: as in `bytecode`.
        read_u8(unsafe { &*code }, &mut self.code_cursor)
    }

    /// Read one register operand ([`crate::jitcode::JitcodeReg`]); register
    /// operands are always one byte. Use this at register reads instead of
    /// `next_u8` so the 1-byte register width stays enforced in one place.
    pub fn next_reg(&mut self) -> crate::jitcode::JitcodeReg {
        self.next_u8()
    }

    pub fn next_u16(&mut self) -> u16 {
        let code: *const [u8] = self.bytecode;
        // SAFETY: as in `bytecode`.
        read_u16(unsafe { &*code }, &mut self.code_cursor)
    }

    /// Peek a u16 at absolute position `pos` without advancing the
    /// cursor.  Mirrors the canonical decode helper on
    /// `BlackholeInterpreter::peek_u16_at` so trace dispatch can do the
    /// same dual-encoding auto-detect for vable opcodes.
    pub fn peek_u16_at(&self, pos: usize) -> Option<u16> {
        let code = self.bytecode();
        if pos + 1 >= code.len() {
            return None;
        }
        Some((code[pos] as u16) | ((code[pos + 1] as u16) << 8))
    }

    /// Resolve a `d`-argcode descr index against the per-jitcode runtime
    /// descr pool (`exec.descrs`) — falling through to the shared global
    /// build-time pool for LLBC-extracted jitcodes (via [`JitCode::descr_at`])
    /// — and return the canonical `BhDescr` if present.  Trace dispatch uses
    /// this to detect when a vable opcode was emitted in canonical form (descr
    /// lives in the pool) versus pyre's pre-orthodox legacy form (descr
    /// operand is the field index inline).
    pub fn runtime_bh_descr(&self, descr_idx: usize) -> Option<&crate::blackhole::BhDescr> {
        self.jitcode
            .descr_at(descr_idx)
            .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
    }

    /// Return the optimizer descriptor stored beside the canonical `d`
    /// operand.  `Assembler.descrs` upstream stores the one descriptor object
    /// both blackhole and tracing dispatch consume; this is the tracing view
    /// of pyre's split serialized/runtime representation.
    pub fn runtime_optimizer_descr(&self, descr_idx: usize) -> Option<majit_ir::DescrRef> {
        self.jitcode
            .descr_at(descr_idx)
            .and_then(crate::jitcode::RuntimeBhDescr::as_optimizer_descr)
            .cloned()
    }

    /// Read the static-field index from a canonical `VableField` descr
    /// pool entry at `pos`. Stage 3c-3 dropped the dual-mode
    /// auto-detect, so the bytes must already be canonical
    /// (`assembler.py:165-167` + `:197-207`).
    pub fn vable_field_index_at(&self, pos: usize) -> usize {
        let idx = self
            .peek_u16_at(pos)
            .expect("vable_field_index_at: descr operand out of bounds");
        match self.runtime_bh_descr(idx as usize) {
            Some(crate::blackhole::BhDescr::VableField { index }) => *index,
            other => {
                panic!("vable_field_index_at: expected VableField at descr {idx}, got {other:?}")
            }
        }
    }

    /// Read the array-field index from a canonical
    /// (`VableArray`, `Array`) descr pool pair.
    pub fn vable_array_index_pair_at(&self, field_pos: usize, array_pos: usize) -> usize {
        let field_idx = self
            .peek_u16_at(field_pos)
            .expect("vable_array_index_pair_at: field descr out of bounds");
        let array_idx = self
            .peek_u16_at(array_pos)
            .expect("vable_array_index_pair_at: array descr out of bounds");
        let field_descr = self.runtime_bh_descr(field_idx as usize);
        let array_descr = self.runtime_bh_descr(array_idx as usize);
        match (field_descr, array_descr) {
            (
                Some(crate::blackhole::BhDescr::VableArray { index }),
                Some(crate::blackhole::BhDescr::Array { .. }),
            ) => *index,
            other => panic!(
                "vable_array_index_pair_at: expected (VableArray, Array) at descrs ({field_idx}, {array_idx}), got {other:?}"
            ),
        }
    }

    /// Decode a `getfield_vable_<kind>/rd>X` operand triple, returning
    /// `(vable_reg, field_idx, dest_reg)` per
    /// `assembler.py::Assembler.write_insn` and `Assembler.write_op_live`.
    /// Canonical layout: 1B vable_reg + 2B descr_pool_idx
    /// + 1B dest_reg. The leading `r` operand carries the live struct register
    /// consumed as the `struct` argument by the `_opimpl_setfield_vable_*`
    /// family in `pyjitpl.py`.
    pub fn read_vable_getfield(&mut self) -> (usize, usize, usize) {
        let field_idx = self.vable_field_index_at(self.code_cursor + 1);
        let base = self.next_u8() as usize;
        self.code_cursor += 2;
        let dst = self.next_u8() as usize;
        (base, field_idx, dst)
    }

    /// Decode a `new/d>r` (or `new_with_vtable/d>r`) operand pair,
    /// returning `(descr_pool_idx, dest_reg)`. Canonical layout: 2B
    /// descr_pool_idx + 1B dest_reg (`blackhole.py:1301-1310`).
    pub fn read_new(&mut self) -> (usize, usize) {
        let descr_idx = self.next_u16() as usize;
        let dst = self.next_u8() as usize;
        (descr_idx, dst)
    }

    /// Decode `new_array/id>r` and `new_array_clear/id>r`:
    /// `(length_reg, descr_idx, dest)`. Same operand order as
    /// `bhimpl_new_array` (`blackhole.py`).
    pub fn read_new_array(&mut self) -> (usize, usize, usize) {
        let length_reg = self.next_u8() as usize;
        let descr_idx = self.next_u16() as usize;
        let dst = self.next_u8() as usize;
        (length_reg, descr_idx, dst)
    }

    /// Decode a `setfield_gc_<kind>/rXd` operand triple, returning
    /// `(struct_reg, value_reg, descr_pool_idx)`. Canonical layout: 1B
    /// struct_reg + 1B value_reg + 2B descr_pool_idx
    /// (`blackhole.py bhimpl_setfield_gc_i`).
    pub fn read_setfield_gc(&mut self) -> (usize, usize, usize) {
        let struct_reg = self.next_u8() as usize;
        let value_reg = self.next_u8() as usize;
        let descr_idx = self.next_u16() as usize;
        (struct_reg, value_reg, descr_idx)
    }

    /// Decode a `newlist_clear/idddd>r` operand septuple, returning
    /// `(length_reg, struct_descr_idx, length_descr_idx, items_descr_idx,
    /// array_descr_idx, dest)`. Canonical layout: 1B length_reg + 2B
    /// structdescr_idx + 2B lengthdescr_idx + 2B itemsdescr_idx + 2B
    /// arraydescr_idx + 1B dest_reg — the exact read order
    /// `handler_newlist_clear` (`blackhole.py:1173-1180`) and the
    /// codewriter emit (`assembler.rs` NewListClear arm) agree on.
    pub fn read_newlist_clear(&mut self) -> (usize, usize, usize, usize, usize, usize) {
        let length_reg = self.next_u8() as usize;
        let struct_descr = self.next_u16() as usize;
        let length_descr = self.next_u16() as usize;
        let items_descr = self.next_u16() as usize;
        let array_descr = self.next_u16() as usize;
        let dest = self.next_u8() as usize;
        (
            length_reg,
            struct_descr,
            length_descr,
            items_descr,
            array_descr,
            dest,
        )
    }

    /// Decode a `getfield_gc_<kind>/rd>X` operand triple, returning
    /// `(struct_reg, descr_pool_idx, dest_reg)`. Canonical layout: 1B
    /// struct_reg + 2B descr_pool_idx + 1B dest_reg
    /// (`blackhole.py bhimpl_getfield_gc_i`).
    pub fn read_getfield_gc(&mut self) -> (usize, usize, usize) {
        let struct_reg = self.next_u8() as usize;
        let descr_idx = self.next_u16() as usize;
        let dst = self.next_u8() as usize;
        (struct_reg, descr_idx, dst)
    }

    /// Decode a `setfield_vable_<kind>/rXd` operand triple, returning
    /// `(vable_reg, field_idx, value_reg)`. Canonical layout: 1B
    /// vable_reg + 1B value_reg + 2B descr_pool_idx.
    pub fn read_vable_setfield(&mut self) -> (usize, usize, usize) {
        let field_idx = self.vable_field_index_at(self.code_cursor + 2);
        let base = self.next_u8() as usize;
        let src = self.next_u8() as usize;
        self.code_cursor += 2;
        (base, field_idx, src)
    }

    /// `setfield_vable_i_imm/rddd`: `(vable_reg, field_idx, immediate)`.
    /// Layout: 1B vable_reg + u32 value + 2B descr.
    pub fn read_vable_setfield_imm(&mut self) -> (usize, usize, i64) {
        let field_idx = self.vable_field_index_at(self.code_cursor + 5);
        let base = self.next_u8() as usize;
        let lo = self.next_u16() as u32;
        let hi = self.next_u16() as u32;
        self.code_cursor += 2;
        (base, field_idx, (lo | (hi << 16)) as i64)
    }

    /// Decode a `getarrayitem_vable_<kind>/ridd>X` operand quintuple,
    /// returning `(vable_reg, array_idx, index_reg, dest_reg)`.
    /// Canonical layout: 1B vable_reg + 1B index_reg + 2B fdescr + 2B
    /// adescr + 1B dest.
    pub fn read_vable_getarrayitem(&mut self) -> (usize, usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 2, self.code_cursor + 4);
        let base = self.next_u8() as usize;
        let index_reg = self.next_u8() as usize;
        self.code_cursor += 4;
        let dst = self.next_u8() as usize;
        (base, array_idx, index_reg, dst)
    }

    /// Decode a `setarrayitem_vable_<kind>/riXdd` operand quintuple,
    /// returning `(vable_reg, array_idx, index_reg, value_reg)`.
    /// Canonical layout: 1B vable_reg + 1B index_reg + 1B value_reg +
    /// 2B fdescr + 2B adescr.
    pub fn read_vable_setarrayitem(&mut self) -> (usize, usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 3, self.code_cursor + 5);
        let base = self.next_u8() as usize;
        let index_reg = self.next_u8() as usize;
        let src = self.next_u8() as usize;
        self.code_cursor += 4;
        (base, array_idx, index_reg, src)
    }

    /// Decode an `arraylen_vable/rdd>i` operand triple, returning
    /// `(vable_reg, array_idx, dest_reg)`. Canonical layout: 1B
    /// vable_reg + 2B fdescr + 2B adescr + 1B dest.
    pub fn read_vable_arraylen(&mut self) -> (usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 1, self.code_cursor + 3);
        let base = self.next_u8() as usize;
        self.code_cursor += 4;
        let dst = self.next_u8() as usize;
        (base, array_idx, dst)
    }

    /// pyjitpl.py `MIFrame.verify_green_args(jitdriver_sd, varargs)`.
    ///
    /// ```python
    /// def verify_green_args(self, jitdriver_sd, varargs):
    ///     num_green_args = jitdriver_sd.num_green_args
    ///     assert len(varargs) == num_green_args
    ///     for i in range(num_green_args):
    ///         assert isinstance(varargs[i], Const)
    /// ```
    ///
    /// Called from `opimpl_jit_merge_point` (pyjitpl.py) so the
    /// metainterp aborts immediately if a green arg arrives as a Box
    /// (non-constant) — that would mean the codewriter / annotator
    /// failed to mark it as compile-time-known. The pyre port lifts
    /// `varargs` from the upstream `boxes3` typed-list to a flat
    /// `&[OpRef]` (the trace-side representation of green args after
    /// `make_three_lists` flattens by kind in jtransform). Each element
    /// must satisfy [`OpRef::is_constant`] — the constant-namespace
    /// flag set by `OpRef::const_int` / `const_ptr` / `const_float`.
    ///
    /// Production callers land with a follow-up (the metainterp-side
    /// `opimpl_jit_merge_point` port). For now the helper is dead code
    /// outside tests; it locks the contract upstream relies on so a
    /// future caller cannot silently accept a non-Const green.
    pub fn verify_green_args(
        jitdriver_sd: &crate::jitdriver::JitDriverStaticData,
        varargs: &[majit_ir::OpRef],
    ) {
        let num_green_args = jitdriver_sd.num_green_args();
        assert_eq!(
            varargs.len(),
            num_green_args,
            "verify_green_args: expected {} greens, got {}",
            num_green_args,
            varargs.len()
        );
        for (i, opref) in varargs.iter().enumerate() {
            assert!(
                opref.is_constant(),
                "verify_green_args: greens[{i}] = {opref:?} is not a Const \
                 (upstream pyjitpl.py:1534 asserts isinstance(varargs[i], Const))",
            );
        }
    }

    /// pyjitpl.py `MIFrame.copy_constants`.
    ///
    /// RPython copies each entry of `jitcode.constants_{i,r,f}` into the
    /// register file at `registers_X[num_regs_X + i]`, wrapping it in
    /// `Const{Int,Ptr,Float}`. `MIFrame.setup` (pyjitpl.py) invokes
    /// this inline right after sizing the register arrays to
    /// `num_regs_and_consts_X`.
    ///
    /// pyre stores the Const box in the register; `ctx` interns it into
    /// the trace context's constant pool. The value lives on the box
    /// (`ConstInt.value`).
    pub fn copy_constants(&mut self, ctx: &mut crate::trace_ctx::TraceCtx) {
        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        for (i, &value) in self.jitcode.constants_i.iter().enumerate() {
            let slot = num_regs_i + i;
            #[cfg(feature = "jit-audits")]
            majit_ir::reg_write_audit::note_int_write(self.int_regs.as_ptr() as usize, slot, None);
            self.int_regs[slot] = Some(ctx.const_int(value));
        }
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        for i in 0..self.jitcode.constants_r.len() {
            let value = self.jitcode.constants_r[i].get();
            let slot = num_regs_r + i;
            self.ref_regs[slot] = Some(ctx.const_ref(value));
        }
        let num_regs_f = self.jitcode.c_num_regs_f as usize;
        for (i, &value) in self.jitcode.constants_f.iter().enumerate() {
            let slot = num_regs_f + i;
            self.float_regs[slot] = Some(ctx.const_float(value));
        }
    }

    /// `pyjitpl.py` `add_box_to_storage(self.registers_r[index])`.
    ///
    /// `copy_constants` interned the constants-area slots as `ConstPtr`
    /// (`history.py` `ConstPtr.value`). Re-resolve that intern after
    /// `new_array` / an earlier `_encode`. A raw `constants_r` word is
    /// only the fallback when `copy_constants` did not fill the slot.
    fn ref_liveness_to_box(&self, idx: usize, num_regs_r: usize) -> OpBox {
        if idx < num_regs_r {
            return match self.ref_regs[idx] {
                Some(opref) => register_to_box(opref),
                None => OpBox::ConstPtr(0),
            };
        }
        if let Some(opref) = self.ref_regs.get(idx).copied().flatten() {
            return register_to_box(opref);
        }
        OpBox::ConstPtr(self.jitcode.constants_r[idx - num_regs_r].get() as u64)
    }

    /// `pyjitpl.py` `self.registers_i[index]` after `copy_constants`.
    fn int_liveness_to_box(&self, idx: usize, num_regs_i: usize) -> OpBox {
        if idx < num_regs_i {
            let opref =
                self.int_regs[idx].expect("get_list_of_active_boxes: int register uninitialized");
            return register_to_box(opref);
        }
        if let Some(opref) = self.int_regs.get(idx).copied().flatten() {
            return register_to_box(opref);
        }
        OpBox::ConstInt(self.jitcode.constants_i[idx - num_regs_i])
    }

    /// `pyjitpl.py` `self.registers_f[index]` after `copy_constants`.
    fn float_liveness_to_box(&self, idx: usize, num_regs_f: usize) -> OpBox {
        if idx < num_regs_f {
            let opref = self.float_regs[idx]
                .expect("get_list_of_active_boxes: float register uninitialized");
            return register_to_box(opref);
        }
        if let Some(opref) = self.float_regs.get(idx).copied().flatten() {
            return register_to_box(opref);
        }
        OpBox::ConstFloat(self.jitcode.constants_f[idx - num_regs_f] as u64)
    }

    /// Side-table counterpart of [`Self::ref_liveness_to_box`].
    ///
    /// `resume.py` `ResumeDataLoopMemo._number_boxes` keys on box identity:
    /// a live InputArg / ResOp stays `SnapshotTagged::Box`. `isinstance(box,
    /// Const)` goes through `getconst` — constants-area slots
    /// (`copy_constants`) and a Const already in the register stay Const.
    /// Do not reverse-map a ConstPtr's address onto a live InputArg.
    fn ref_liveness_to_tagged(&self, idx: usize, num_regs_r: usize) -> SnapshotTagged {
        let opref = if idx < num_regs_r {
            match self.ref_regs[idx] {
                Some(op) => op,
                None => return SnapshotTagged::Const(0, Type::Ref),
            }
        } else if let Some(op) = self.ref_regs.get(idx).copied().flatten() {
            op
        } else {
            return SnapshotTagged::const_ref(
                self.jitcode.constants_r[idx - num_regs_r].get() as usize
            );
        };
        if opref.is_input_arg() || !opref.is_constant() {
            return SnapshotTagged::Box(opref, Type::Ref);
        }
        if let Some(index) = opref.const_ptr_index() {
            SnapshotTagged::Const(i64::from(index), Type::Ref)
        } else if let Some(v) = opref.inline_const_to_value() {
            SnapshotTagged::const_ref(v.as_raw_i64() as usize)
        } else {
            SnapshotTagged::Box(opref, Type::Ref)
        }
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.bytecode().len()
    }

    /// pyjitpl.py `MIFrame.cleanup_registers()`.
    ///
    /// ```python
    /// def cleanup_registers(self):
    ///     for i in range(self.jitcode.num_regs_r()):
    ///         self.registers_r[i] = None
    ///     self.pushed_box = None
    /// ```
    ///
    /// Iterates `0..num_regs_r()` (RPython skips the constants area
    /// that lives past `num_regs_r`); pyre's `ref_regs` is sized
    /// exactly to `num_regs_r` so the loop scans the same slots.
    pub fn cleanup_registers(&mut self) {
        let num_regs_r = self.jitcode.num_regs_r();
        for i in 0..num_regs_r {
            self.ref_regs[i] = None;
        }
        self.pushed_box = None;
    }

    /// pyjitpl.py `MIFrame.setup_resume_at_op(pc)`.
    ///
    /// `last_opcode_position` follows `pc`, the way `setup` seats both and
    /// `BlackholeInterpreter::reset_position_state` does for a resumed
    /// blackhole frame. `prepare_resume_from_failure` runs
    /// `finishframe_exception` on the rebuilt frame before any opcode has
    /// dispatched, and the traceback node it attaches reads this cursor.
    pub fn setup_resume_at_op(&mut self, pc: usize) {
        self.pc = pc;
        self.last_opcode_position = pc;
    }

    /// pyjitpl.py `MIFrame.handle_rvmprof_enter_on_resume`.
    ///
    /// Skip a leading `-live-`. When the resumed opcode is `rvmprof_code/ii`
    /// and arg1 == 1, execution will call `jit_rvmprof_code(1, unique_id)`,
    /// so emit the matching enter `jit_rvmprof_code(0, arg2)` first.
    /// `op_live` / `op_rvmprof_code` are `staticdata` lookups
    /// (`insns.get(..., -1)`). `-1` matches no opcode byte.
    pub fn handle_rvmprof_enter_on_resume(&mut self, op_live: i32, op_rvmprof_code: i32) {
        let code = &self.jitcode.code;
        let mut position = self.pc;
        let mut opcode = code[position];
        if op_live >= 0 && opcode == op_live as u8 {
            position += majit_jitcode::liveness::OFFSET_SIZE + 1;
            opcode = code[position];
        }
        if op_rvmprof_code >= 0 && opcode == op_rvmprof_code as u8 {
            let arg1 = self.int_regs[code[position + 1] as usize]
                .and_then(|op| op.inline_const_to_value())
                .map(|v| v.as_raw_i64())
                .unwrap_or_else(|| {
                    panic!(
                        "handle_rvmprof_enter_on_resume: registers_i[{}] has no int",
                        code[position + 1]
                    )
                });
            let arg2 = self.int_regs[code[position + 2] as usize]
                .and_then(|op| op.inline_const_to_value())
                .map(|v| v.as_raw_i64())
                .unwrap_or_else(|| {
                    panic!(
                        "handle_rvmprof_enter_on_resume: registers_i[{}] has no int",
                        code[position + 2]
                    )
                });
            if arg1 == 1 {
                // pyjitpl.py `cintf.jit_rvmprof_code(0, arg2)`.
                majit_rlib::rvmprof::cintf::jit_rvmprof_code(0, arg2);
            }
        }
    }

    /// pyjitpl.py `MIFrame.make_result_of_lastop(resultbox)`.
    ///
    /// Stores the result of the last opimpl into the typed register at
    /// `target_index`. RPython reads `target_index = ord(self.bytecode[self.pc-1])`
    /// from the bytecode; pyre's call BC encodes `dst` explicitly so
    /// callers pass it directly.
    pub fn make_result_of_lastop(
        &mut self,
        kind: JitArgKind,
        target_index: usize,
        opref: OpRef,
        _concrete: i64,
    ) {
        #[cfg(debug_assertions)]
        {
            // pyjitpl.py:260-264 non-translated check:
            //
            //     if not we_are_translated():
            //         typeof = {'i': history.INT, 'r': history.REF,
            //                   'f': history.FLOAT}
            //         assert typeof[self.jitcode._resulttypes[self.pc]] == got_type
            //
            // Rust `debug_assertions` is the closest equivalent of
            // `not we_are_translated()`: fail loudly in non-translated
            // developer builds, but elide the whole check in release.
            let got = match kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            };
            let resulttypes = self
                .jitcode
                .core()
                .body()
                .resulttypes
                .as_ref()
                .expect("make_result_of_lastop: _resulttypes is None");
            let recorded = *resulttypes
                .get(&self.pc)
                .expect("make_result_of_lastop: missing _resulttypes[pc]");
            assert_eq!(
                recorded, got,
                "make_result_of_lastop: jitcode._resulttypes[{}] = {recorded:?} but runtime kind = {got:?}",
                self.pc
            );
        }
        match kind {
            JitArgKind::Int => {
                #[cfg(feature = "jit-audits")]
                majit_ir::reg_write_audit::note_int_write(
                    self.int_regs.as_ptr() as usize,
                    target_index,
                    Some(opref),
                );
                self.int_regs[target_index] = Some(opref);
            }
            JitArgKind::Ref => {
                self.ref_regs[target_index] = Some(opref);
            }
            JitArgKind::Float => {
                self.float_regs[target_index] = Some(opref);
            }
        }
    }

    /// pyjitpl.py `MIFrame.setup_call(argboxes)`.
    ///
    /// Resets `pc` to 0 and copies each argbox into the first slot of
    /// its typed register bank in declaration order. RPython's
    /// `setup_call` consults `box.type`; pyre's `OpRef` does not carry
    /// type info, so the caller passes a typed `(kind, value, concrete)`
    /// tuple per arg.
    ///
    /// Also resets `parent_snapshot = -1`. In RPython this lives in
    /// `MIFrame.setup()` (pyjitpl.py) which always precedes
    /// `setup_call`; pyre's `MIFrame::new` already sets it, but any
    /// future frame-recycling path (upstream `free_frames_list`) would
    /// reuse an MIFrame and skip `new()`, so we reset here to match the
    /// RPython "every call-entry clears parent_snapshot" invariant.
    /// pyjitpl.py `MIFrame.get_list_of_active_boxes`.
    ///
    /// Reads the LIVE-op liveness header preceding the current pc and
    /// pushes each live register onto the trace's snapshot-array data
    /// in int → ref → float order.  Returns the `_snapshot_array_data`
    /// offset produced by `new_array` (RPython `storage`).
    ///
    /// `op_live` and `all_liveness` are threaded through from
    /// `MetaInterpStaticData` (RPython
    /// `self.metainterp.staticdata.op_live` /
    /// `.liveness_info`) — pyre passes them explicitly so this method
    /// does not depend on `MetaInterpStaticData` structurally.
    ///
    /// `clear_result_register` selects the in_a_call clear strategy.
    /// When `true` the branch mints `history.CONST_FALSE` / `CONST_NULL`
    /// / `history.CONST_FZERO` inline-Const OpRefs and writes them into
    /// the cleared register slot exactly as pyjitpl.py does.
    /// Tests that only care about the snapshot output may pass `false` —
    /// the fallback path then substitutes `Box::Const*(0)` directly into
    /// the snapshot array (byte-identical to the structural path) but
    /// leaves the register slot untouched.
    pub fn get_list_of_active_boxes(
        &mut self,
        in_a_call: bool,
        trace: &mut TraceRecordBuffer,
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> i64 {
        const SIZE_LIVE_OP: usize = majit_jitcode::liveness::OFFSET_SIZE + 1;
        use majit_jitcode::liveness::{LivenessIterator, decode_offset};

        // pyjitpl.py:180-193 — in_a_call branch.  The frame that holds
        // the in-flight CALL instruction has a "result" register slot
        // that is not yet defined (the call has not returned).  RPython
        // clears that slot to a zero constant (history.CONST_FALSE /
        // CONST_NULL / history.CONST_FZERO) so the snapshot captures a
        // well-defined placeholder instead of pre-call stale data.
        //
        // When `clear_result_register` is set we mirror RPython exactly:
        // mint the zero constant, write it into the register slot, and
        // let the bank loop below emit it through the normal register →
        // OpBox path.  This makes
        // a subsequent `get_list_of_active_boxes` (e.g. a re-snapshot
        // of the same in-flight call before `make_result_of_lastop`)
        // see the cleared slot rather than the pre-call stale contents.
        //
        // When it is clear (test fixtures that only check snapshot
        // bytes) we fall back to substituting the cleared box directly
        // into the snapshot array via `clear_*_idx`.  The snapshot bytes
        // are identical, but the register slot retains its pre-call
        // contents.
        let (clear_int_idx, clear_ref_idx, clear_float_idx) = if in_a_call {
            let argcode = self._result_argcode;
            let index = self
                .result_arg_index
                .take()
                .unwrap_or_else(|| self.jitcode.code[self.pc - 1] as usize);
            // pyjitpl.py `self._result_argcode = '?'` — mark cleared.
            self._result_argcode = b'?';
            if clear_result_register {
                // pyjitpl.py:184-192 register clearing via inline-Const.
                match argcode {
                    b'i' => {
                        let opref = OpRef::const_int(0);
                        #[cfg(feature = "jit-audits")]
                        majit_ir::reg_write_audit::note_int_write(
                            self.int_regs.as_ptr() as usize,
                            index,
                            Some(opref),
                        );
                        self.int_regs[index] = Some(opref);
                        (None, None, None)
                    }
                    b'r' => {
                        let opref = OpRef::const_ptr(majit_ir::GcRef::NULL);
                        self.ref_regs[index] = Some(opref);
                        (None, None, None)
                    }
                    b'f' => {
                        let opref = OpRef::const_float(0.0);
                        self.float_regs[index] = Some(opref);
                        (None, None, None)
                    }
                    _ => (None, None, None),
                }
            } else {
                // Test-fallback: substitute zero in the snapshot output
                // without mutating the register slot.
                match argcode {
                    b'i' => (Some(index), None, None),
                    b'r' => (None, Some(index), None),
                    b'f' => (None, None, Some(index)),
                    _ => (None, None, None),
                }
            }
        } else {
            (None, None, None)
        };

        // pyjitpl.py:194-198 — pick the pc of the preceding LIVE op.
        let pc = if in_a_call || after_residual_call {
            self.pc
        } else {
            self.pc - SIZE_LIVE_OP
        };

        // pyjitpl.py:199 `assert ord(self.jitcode.code[pc]) == op_live`.
        // RPython's `assert` survives translation; a debug-only check
        // lets release decode operand bytes as an `all_liveness` offset.
        assert_eq!(self.jitcode.code[pc], op_live);

        // pyjitpl.py:202-207 — decode offset + per-type lengths.
        let mut offset = decode_offset(&self.jitcode.code, pc + 1);
        let length_i = all_liveness[offset] as u32;
        let length_r = all_liveness[offset + 1] as u32;
        let length_f = all_liveness[offset + 2] as u32;
        offset += 3;

        // pyjitpl.py:209-214 — pre-allocate the storage array.
        let total = (length_i + length_r + length_f) as usize;
        if majit_gc::diag_p92_enabled() && total != 4 && (self.pc == 836 || pc == 836) {
            eprintln!(
                "P92_ACTIVE_BOXES jc={:?} frame_pc={} live_pc={pc} in_a_call={in_a_call} after_residual={after_residual_call} total={total} li={length_i} lr={length_r} lf={length_f}",
                self.jitcode.try_index(),
                self.pc
            );
            eprintln!("{}", std::backtrace::Backtrace::force_capture());
        }
        let storage = trace.new_array(total);

        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        let num_regs_f = self.jitcode.c_num_regs_f as usize;

        // pyjitpl.py:216-221 — push live int registers.  Liveness
        // indices are interpreted against the `num_regs_and_consts_i`
        // layout (RPython pyjitpl.py:82-83): the first `num_regs_i`
        // slots are runtime registers, the remaining are jitcode
        // constants copied in by `setup()`.  pyre keeps registers
        // separate from the jitcode's `constants_i` Vec, so a liveness
        // index past `num_regs_i` reads directly from
        // `self.jitcode.constants_i[idx - num_regs_i]` and emits the
        // corresponding `ConstInt` box — matching what RPython's
        // `copy_constants` + `registers_i[index]` lookup would return.
        if length_i > 0 {
            // SAFETY: `all_liveness` bank at `offset` is a complete
            // `encode_liveness` record (`liveness.py` `LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_i, all_liveness) };
            for index in it.by_ref() {
                let idx = index as usize;
                let b = if Some(idx) == clear_int_idx {
                    // pyjitpl.py:184-185 history.CONST_FALSE clearing.
                    OpBox::ConstInt(0)
                } else {
                    self.int_liveness_to_box(idx, num_regs_i)
                };
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset();
        }

        // pyjitpl.py `get_list_of_active_boxes` — push live ref registers.
        // `add_box_to_storage(self.registers_r[index])`: constants live
        // in the same file after `copy_constants`.
        if length_r > 0 {
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_r, all_liveness) };
            for index in it.by_ref() {
                let idx = index as usize;
                let b = if Some(idx) == clear_ref_idx {
                    // pyjitpl.py:186-187 CONST_NULL clearing.
                    OpBox::ConstPtr(0)
                } else {
                    self.ref_liveness_to_box(idx, num_regs_r)
                };
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset();
        }

        // pyjitpl.py:228-233 — push live float registers.
        if length_f > 0 {
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_f, all_liveness) };
            for index in it.by_ref() {
                let idx = index as usize;
                let b = if Some(idx) == clear_float_idx {
                    // pyjitpl.py:188-189 history.CONST_FZERO clearing.
                    OpBox::ConstFloat(0)
                } else {
                    self.float_liveness_to_box(idx, num_regs_f)
                };
                trace._add_box_to_storage_box(b);
            }
            let _ = it.offset(); // offset no longer read after the last bank
        }

        storage
    }

    /// Side-table snapshot counterpart of
    /// [`Self::get_list_of_active_boxes`].
    ///
    /// This follows the same pyjitpl.py control flow but returns
    /// `recorder::SnapshotTagged` entries for the legacy `TraceCtx`
    /// snapshot side table instead of writing opencoder byte arrays.
    pub fn get_list_of_active_snapshot_boxes(
        &mut self,
        in_a_call: bool,
        clear_result_register: bool,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> Vec<SnapshotTagged> {
        const SIZE_LIVE_OP: usize = majit_jitcode::liveness::OFFSET_SIZE + 1;
        use majit_jitcode::liveness::{LivenessIterator, decode_offset};

        let (clear_int_idx, clear_ref_idx, clear_float_idx) = if in_a_call {
            let argcode = self._result_argcode;
            let index = self
                .result_arg_index
                .take()
                .unwrap_or_else(|| self.jitcode.code[self.pc - 1] as usize);
            self._result_argcode = b'?';
            if clear_result_register {
                match argcode {
                    b'i' => {
                        let opref = OpRef::const_int(0);
                        #[cfg(feature = "jit-audits")]
                        majit_ir::reg_write_audit::note_int_write(
                            self.int_regs.as_ptr() as usize,
                            index,
                            Some(opref),
                        );
                        self.int_regs[index] = Some(opref);
                    }
                    b'r' => {
                        let opref = OpRef::const_ptr(majit_ir::GcRef::NULL);
                        self.ref_regs[index] = Some(opref);
                    }
                    b'f' => {
                        let opref = OpRef::const_float(0.0);
                        self.float_regs[index] = Some(opref);
                    }
                    _ => {}
                }
                (None, None, None)
            } else {
                match argcode {
                    b'i' => (Some(index), None, None),
                    b'r' => (None, Some(index), None),
                    b'f' => (None, None, Some(index)),
                    _ => (None, None, None),
                }
            }
        } else {
            (None, None, None)
        };

        let pc = if in_a_call || after_residual_call {
            self.pc
        } else {
            self.pc - SIZE_LIVE_OP
        };
        // `get_list_of_active_boxes` asserts `code[pc] == op_live`.
        // pyjitpl.py get_list_of_active_boxes: the assert survives
        // translation; a debug-only check lets release decode operand
        // bytes as an `all_liveness` offset.
        assert_eq!(self.jitcode.code[pc], op_live);

        let mut offset = decode_offset(&self.jitcode.code, pc + 1);
        let length_i = all_liveness[offset] as u32;
        let length_r = all_liveness[offset + 1] as u32;
        let length_f = all_liveness[offset + 2] as u32;
        offset += 3;

        let total = (length_i + length_r + length_f) as usize;
        if majit_gc::diag_p92_enabled() && total != 4 && (self.pc == 836 || pc == 836) {
            eprintln!(
                "P92_ACTIVE_SNAP_BOXES jc={:?} frame_pc={} live_pc={pc} in_a_call={in_a_call} after_residual={after_residual_call} total={total} li={length_i} lr={length_r} lf={length_f}",
                self.jitcode.try_index(),
                self.pc
            );
            eprintln!("{}", std::backtrace::Backtrace::force_capture());
        }
        let mut boxes = Vec::with_capacity(total);

        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        let num_regs_f = self.jitcode.c_num_regs_f as usize;

        if length_i > 0 {
            // SAFETY: `all_liveness` bank at `offset` is a complete
            // `encode_liveness` record (`liveness.py` `LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_i, all_liveness) };
            for index in it.by_ref() {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_int_idx {
                    #[cfg(feature = "jit-audits")]
                    majit_ir::reg_write_audit::note_int_read_diverted(idx, "clear_int_idx");
                    SnapshotTagged::Const(0, Type::Int)
                } else if idx < num_regs_i {
                    let opref = self.int_regs[idx]
                        .expect("get_list_of_active_snapshot_boxes: int register uninitialized");
                    #[cfg(feature = "jit-audits")]
                    majit_ir::reg_write_audit::note_int_read(
                        self.int_regs.as_ptr() as usize,
                        idx,
                        opref,
                    );
                    if let Some(v) = opref.inline_const_to_value() {
                        SnapshotTagged::Const(v.as_raw_i64(), Type::Int)
                    } else {
                        SnapshotTagged::Box(opref, Type::Int)
                    }
                } else {
                    #[cfg(feature = "jit-audits")]
                    majit_ir::reg_write_audit::note_int_read_diverted(idx, "constants-pool");
                    SnapshotTagged::Const(self.jitcode.constants_i[idx - num_regs_i], Type::Int)
                };
                boxes.push(tagged);
            }
            offset = it.offset();
        }

        if length_r > 0 {
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_r, all_liveness) };
            for index in it.by_ref() {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_ref_idx {
                    SnapshotTagged::Const(0, Type::Ref)
                } else {
                    self.ref_liveness_to_tagged(idx, num_regs_r)
                };
                boxes.push(tagged);
            }
            offset = it.offset();
        }

        if length_f > 0 {
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(offset, length_f, all_liveness) };
            for index in it {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_float_idx {
                    SnapshotTagged::Const(0, Type::Float)
                } else if idx < num_regs_f {
                    let opref = self.float_regs[idx]
                        .expect("get_list_of_active_snapshot_boxes: float register uninitialized");
                    if let Some(v) = opref.inline_const_to_value() {
                        SnapshotTagged::Const(v.as_raw_i64(), Type::Float)
                    } else {
                        SnapshotTagged::Box(opref, Type::Float)
                    }
                } else {
                    SnapshotTagged::Const(self.jitcode.constants_f[idx - num_regs_f], Type::Float)
                };
                boxes.push(tagged);
            }
        }

        boxes
    }

    /// pyjitpl.py `MIFrame.replace_active_box_in_frame(oldbox, newbox)`.
    ///
    /// ```python
    /// def replace_active_box_in_frame(self, oldbox, newbox):
    ///     if oldbox.type == 'i':
    ///         count = self.jitcode.num_regs_i()
    ///         registers = self.registers_i
    ///     elif oldbox.type == 'r':
    ///         count = self.jitcode.num_regs_r()
    ///         registers = self.registers_r
    ///     elif oldbox.type == 'f':
    ///         count = self.jitcode.num_regs_f()
    ///         registers = self.registers_f
    ///     else:
    ///         assert 0, oldbox
    ///     if not count:
    ///         return
    ///     for i in range(count):
    ///         if registers[i] is oldbox:
    ///             registers[i] = newbox
    /// ```
    ///
    /// pyjitpl.py dispatches on `oldbox.type` — `'i'` / `'r'` / `'f'`
    /// pick the matching register bank, the `else` arm is `assert 0,
    /// oldbox`. Pyre's OpRef does not carry a type tag at the call
    /// boundary, so the bank is selected by an explicit `oldbox_type`
    /// parameter that the caller (`MetaInterp::replace_box`) resolves
    /// once via `TraceCtx::get_opref_type`. `Type::Void` panics with the
    /// upstream assertion message.
    pub fn replace_active_box_in_frame(&mut self, oldbox: OpRef, newbox: OpRef, oldbox_type: Type) {
        // All three banks are flat `Vec<Option<OpRef>>`; the shared replace
        // logic compares by OpRef value (pyre's flat-OpRef adaptation of
        // pyjitpl.py `registers[i] is oldbox`). Only the working registers
        // `[0, num_regs_X)` are scanned; the constants area above them
        // holds Const boxes that are never `oldbox`.
        let (count, registers) = match oldbox_type {
            Type::Int => (self.jitcode.num_regs_i(), &mut self.int_regs),
            Type::Float => (self.jitcode.num_regs_f(), &mut self.float_regs),
            Type::Ref => (self.jitcode.num_regs_r(), &mut self.ref_regs),
            // pyjitpl.py:236-244 `else: assert 0, oldbox` — RPython rejects
            // any box whose `type` attribute is not 'i' / 'r' / 'f'.
            // Mirroring that assertion strength keeps the contract: the
            // caller must resolve a typed Box; passing a Void-typed
            // OpRef indicates the caller's type oracle returned a
            // semantically impossible answer.
            Type::Void => panic!(
                "replace_active_box_in_frame: oldbox {oldbox:?} resolved to Type::Void; \
                 RPython parity rejects unknown/void box types (pyjitpl.py:236)"
            ),
        };
        if count == 0 {
            return;
        }
        for slot in &mut registers[..count] {
            if *slot == Some(oldbox) {
                *slot = Some(newbox);
            }
        }
    }

    /// pyjitpl.py `MIFrame.fill_registers`.
    ///
    /// The `newframe` / `newframe2` / `newframe3` decoder writes each
    /// callee slot as it reads a register index from the caller bytecode
    /// ("instead of allocating the list of boxes"). `length` source
    /// indices start at this frame's `code_cursor`.
    pub fn fill_registers(&mut self, callee: &mut MIFrame, length: usize, argcode: u8) {
        for i in 0..length {
            let index = self.next_reg() as usize;
            match argcode {
                b'I' => {
                    #[cfg(feature = "jit-audits")]
                    majit_ir::reg_write_audit::note_int_write(
                        callee.int_regs.as_ptr() as usize,
                        i,
                        self.int_regs[index],
                    );
                    callee.int_regs[i] = self.int_regs[index];
                }
                b'R' => {
                    callee.ref_regs[i] = self.ref_regs[index];
                }
                b'F' => {
                    if i >= callee.float_regs.len() || index >= self.float_regs.len() {
                        panic!(
                            "fill_registers: float arg {i} of {} into {} (callee float regs {}, caller float regs {}, caller index {index})",
                            self.jitcode.name(),
                            callee.jitcode.name(),
                            callee.float_regs.len(),
                            self.float_regs.len(),
                        );
                    }
                    callee.float_regs[i] = self.float_regs[index];
                }
                other => panic!("fill_registers: argcode {other} is not I, R, or F"),
            }
        }
    }

    pub fn setup_call(&mut self, argboxes: &[(JitArgKind, OpRef, i64)]) {
        self.pc = 0;
        self.parent_snapshot = -1;
        let mut need_i = 0usize;
        let mut need_r = 0usize;
        let mut need_f = 0usize;
        for (kind, _, _) in argboxes {
            match kind {
                JitArgKind::Int => need_i += 1,
                JitArgKind::Ref => need_r += 1,
                JitArgKind::Float => need_f += 1,
            }
        }
        if self.int_regs.len() < need_i {
            self.int_regs.resize(need_i, None);
        }
        if self.ref_regs.len() < need_r {
            self.ref_regs.resize(need_r, None);
        }
        if self.float_regs.len() < need_f {
            self.float_regs.resize(need_f, None);
        }
        let mut count_i = 0;
        let mut count_r = 0;
        let mut count_f = 0;
        for (kind, value, _concrete) in argboxes {
            match kind {
                JitArgKind::Int => {
                    #[cfg(feature = "jit-audits")]
                    majit_ir::reg_write_audit::note_int_write(
                        self.int_regs.as_ptr() as usize,
                        count_i,
                        Some(*value),
                    );
                    self.int_regs[count_i] = Some(*value);
                    count_i += 1;
                }
                JitArgKind::Ref => {
                    self.ref_regs[count_r] = Some(*value);
                    count_r += 1;
                }
                JitArgKind::Float => {
                    self.float_regs[count_f] = Some(*value);
                    count_f += 1;
                }
            }
        }
    }
}

/// RPython pyjitpl.py: MetaInterp.framestack
#[derive(Default)]
pub struct MIFrameStack {
    pub frames: Vec<MIFrame>,
    /// `pyjitpl.py MetaInterp.free_frames_list` for the generated
    /// `JitCodeMachine` adapter.  The machine borrows this wrapper instead of
    /// the whole `MetaInterp`, so the wrapper must carry the paired free list
    /// alongside the live `framestack`; they still have one owner and one
    /// lifetime, just as the two upstream `MetaInterp` attributes do.
    free_frames: Vec<MIFrame>,
}

impl MIFrameStack {
    /// Empty framestack.  Mirrors `self.framestack = []` in
    /// `MetaInterp.initialize_state_from_start` (pyjitpl.py) and
    /// `rebuild_state_after_failure` (pyjitpl.py).
    pub fn empty() -> Self {
        Self {
            frames: Vec::new(),
            free_frames: Vec::new(),
        }
    }

    /// Build a stack pre-seeded with one root frame.  Pyre's
    /// `JitCodeMachine` always opens with a single root frame, so this
    /// constructor mirrors `MIFrameStack::new(root)` from before the
    /// `Arc<JitCode>` migration.
    pub fn new(root: MIFrame) -> Self {
        Self {
            frames: vec![root],
            free_frames: Vec::new(),
        }
    }

    pub fn current_mut(&mut self) -> &mut MIFrame {
        self.frames.last_mut().expect("empty JitCode frame stack")
    }

    pub fn push(&mut self, frame: MIFrame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<MIFrame> {
        self.frames.pop()
    }

    /// `pyjitpl.py MetaInterp.newframe`: take a cleaned MIFrame from
    /// `free_frames_list` when possible, then run `MIFrame.setup` on it.
    pub fn take_frame(
        &mut self,
        jitcode: Arc<JitCode>,
        pc: usize,
        greenkey: Option<super::PortalGreenKey>,
        ctx: Option<&mut crate::trace_ctx::TraceCtx>,
    ) -> MIFrame {
        if let Some(mut frame) = self.free_frames.pop() {
            frame.setup_reused(jitcode, pc, greenkey, ctx);
            frame
        } else {
            MIFrame::setup(jitcode, pc, greenkey, ctx)
        }
    }

    /// `pyjitpl.py MetaInterp.popframe`: clear live references before
    /// returning the frame object to `free_frames_list`.
    pub fn recycle_frame(&mut self, mut frame: MIFrame) {
        frame.cleanup_registers();
        self.free_frames.push(frame);
    }

    /// Park the free list so a standalone walk can hand it to the next
    /// `StandaloneFrameStack` (`MetaInterp.free_frames_list` has no owner
    /// on that path).
    pub fn take_free_frames(&mut self) -> Vec<MIFrame> {
        std::mem::take(&mut self.free_frames)
    }

    pub fn restore_free_frames(&mut self, frames: Vec<MIFrame>) {
        self.free_frames = frames;
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode::JitCodeBuilder;
    use majit_ir::OpRef;

    fn make_jitcode_with_regs(num_i: u16, num_r: u16, num_f: u16) -> Arc<JitCode> {
        let mut builder = JitCodeBuilder::new();
        for i in 0..num_i {
            builder.load_const_i_value(i, 0);
        }
        for i in 0..num_r {
            builder.load_const_r_value(i, 0);
        }
        for i in 0..num_f {
            builder.load_const_f_value(i, 0);
        }
        let mut jitcode = builder.finish();
        jitcode.body_mut().resulttypes = Some([(0, 'i'), (1, 'r'), (2, 'f')].into_iter().collect());
        Arc::new(jitcode)
    }

    #[test]
    fn int_value_for_blackhole_reads_constint() {
        let jitcode = make_jitcode_with_regs(2, 0, 0);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef::const_int(42));
        frame.int_regs[1] = Some(OpRef::int_op(7));

        assert_eq!(frame.int_value_for_blackhole(0), Some(42));
        assert_eq!(frame.int_value_for_blackhole(1), Some(0));
        assert_eq!(frame.int_value_for_blackhole(2), None);
    }

    #[test]
    fn float_value_for_blackhole_reads_constfloat() {
        let jitcode = make_jitcode_with_regs(0, 0, 2);
        let mut frame = MIFrame::new(jitcode, 0);
        let bits = 1.5f64.to_bits() as i64;
        frame.float_regs[0] = Some(OpRef::const_float(1.5));
        frame.float_regs[1] = Some(OpRef::float_op(7));

        assert_eq!(frame.float_value_for_blackhole(0), Some(bits));
        assert_eq!(frame.float_value_for_blackhole(1), Some(0));
        assert_eq!(frame.float_value_for_blackhole(2), None);
    }

    #[test]
    fn ref_value_for_blackhole_reads_constptr_owner() {
        let jitcode = make_jitcode_with_regs(0, 2, 0);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.ref_regs[0] = Some(OpRef::const_ptr(majit_ir::GcRef(0xBEEF)));
        frame.ref_regs[1] = Some(OpRef::ref_op(7));

        assert_eq!(frame.ref_value_for_blackhole(0), Some(0xBEEF));
        // `RefOp.getref_base` of unstamped `_resref` is nullptr.
        assert_eq!(frame.ref_value_for_blackhole(1), Some(0));

        let kept = frame.ref_regs[0];
        frame.set_forwarded_ref_value(0, 0xF00D);
        // The register keeps its table index. `set_slot` writes the address.
        assert_eq!(frame.ref_regs[0], kept);
        assert_eq!(
            frame.ref_regs[0].and_then(|r| r.as_const_ptr()),
            Some(majit_ir::GcRef(0xF00D))
        );
        assert_eq!(frame.ref_value_for_blackhole(0), Some(0xF00D));
    }

    #[test]
    fn setup_call_distributes_argboxes_by_kind_in_declaration_order() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 99;
        frame.setup_call(&[
            (JitArgKind::Int, OpRef::int_op(10), 100),
            (JitArgKind::Ref, OpRef::ref_op(20), 200),
            (JitArgKind::Int, OpRef::int_op(11), 101),
            (JitArgKind::Float, OpRef::float_op(30), 300),
            (JitArgKind::Ref, OpRef::ref_op(21), 201),
        ]);

        assert_eq!(frame.pc, 0);
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(10)));
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(11)));
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(20)));
        assert_eq!(frame.ref_regs[1], Some(OpRef::ref_op(21)));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(30)));
    }

    #[test]
    fn frame_stack_reuses_cleaned_register_storage() {
        // pyjitpl.py `MetaInterp.newframe` / `popframe`: a popped MIFrame is
        // cleaned, appended to `free_frames_list`, and selected for the next
        // call.  The generated-machine adapter must preserve that lifecycle
        // without leaking any per-call state into the reused frame.
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut stack = MIFrameStack::empty();
        let mut frame = stack.take_frame(jitcode.clone(), 7, None, None);
        let int_storage = frame.int_regs.as_ptr();
        let ref_storage = frame.ref_regs.as_ptr();
        let float_storage = frame.float_regs.as_ptr();
        frame.int_regs[0] = Some(OpRef::int_op(11));
        frame.ref_regs[0] = Some(OpRef::ref_op(12));
        frame.float_regs[0] = Some(OpRef::float_op(13));
        frame.inline_frame = true;
        frame.return_i = Some(1);
        frame.pushed_box = Some(OpRef::ref_op(14));

        stack.recycle_frame(frame);
        assert_eq!(stack.free_frames.len(), 1);

        let frame = stack.take_frame(jitcode, 3, None, None);
        assert_eq!(stack.free_frames.len(), 0);
        assert_eq!(frame.int_regs.as_ptr(), int_storage);
        assert_eq!(frame.ref_regs.as_ptr(), ref_storage);
        assert_eq!(frame.float_regs.as_ptr(), float_storage);
        assert!(frame.int_regs.iter().all(Option::is_none));
        assert!(frame.ref_regs.iter().all(Option::is_none));
        assert!(frame.float_regs.iter().all(Option::is_none));
        assert!(!frame.inline_frame);
        assert_eq!(frame.return_i, None);
        assert_eq!(frame.pushed_box, None);
        assert_eq!(frame.pc, 3);
    }

    /// `MIFrame::get_list_of_active_boxes` (non-in_a_call
    /// path) pushes live int / ref / float registers onto the trace's
    /// `_snapshot_array_data` in declaration order, returning the
    /// `new_array` offset. Tests with `after_residual_call=true` so the
    /// LIVE op sits exactly at `self.pc` (pyjitpl.py).
    #[test]
    fn get_list_of_active_boxes_populates_trace_snapshot_array() {
        use crate::opencoder::{Box as OpBox, TraceRecordBuffer};
        use majit_ir::OpRef;
        use std::sync::Arc;

        // 1 int reg + 1 ref reg + 0 float regs live at pc=0.
        let mut builder = JitCodeBuilder::new();
        let mut jitcode = builder.finish();
        jitcode.body_mut().c_num_regs_i = 1;
        jitcode.body_mut().c_num_regs_r = 1;
        jitcode.body_mut().c_num_regs_f = 0;
        const LIVE_OP: u8 = 0x42;
        // bytecode: [LIVE_OP, offset_lo, offset_hi] at pc 0..3.
        // decode_offset reads 2 bytes → liveness info at offset 0.
        jitcode.body_mut().code = vec![LIVE_OP, 0x00, 0x00];
        let jitcode = Arc::new(jitcode);

        // all_liveness at offset 0:
        //   len_i=1 len_r=1 len_f=0  →  3 header bytes
        //   int bitmask byte: bit 0 = register 0 live
        //   ref bitmask byte: bit 0 = register 0 live
        let all_liveness: Vec<u8> = vec![1, 1, 0, 0b0000_0001, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode, 0);
        // int_regs[0] non-constant OpRef::int_op(5) → Box::ResOp(5).
        frame.int_regs[0] = Some(OpRef::int_op(5));
        // ref_regs[0] constant pointer addr=0xdead_beef → Box::ConstPtr.
        frame.ref_regs[0] = Some(OpRef::const_ptr(majit_ir::GcRef(0xdead_beef)));

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(16, sd);
        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* clear_result_register */ false,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        // Two boxes pushed → array length prefix + 2 varints.
        // new_array(2) returns the offset before the length prefix.
        assert!(
            storage > 0,
            "storage offset must be non-zero for non-empty array"
        );

        // Verify the encoded tagged values match what we expect:
        // int: ResOp(5) encodes via _encode_box_position(5).
        // ref: ConstPtr(0xdead_beef) encodes via _encode_ptr.
        // Decode directly from _snapshot_array_data at `storage`.
        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 2, "array length prefix");
        let p0 = storage as usize + consumed;
        let (tag0, c0) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p0..]);
        assert_eq!(
            tag0,
            TraceRecordBuffer::_encode_box_position(5),
            "int register → TAGBOX(5)"
        );
        let p1 = p0 + c0;
        let (tag1, _) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p1..]);
        assert!(tag1 != 0, "ref register should encode to non-zero tag");
    }

    /// pyjitpl.py `add_box_to_storage(self.registers_r[index])` snapshots
    /// the box in `ref_regs`. `ref_values` is an optional pyre mirror.
    #[test]
    fn get_list_of_active_boxes_reads_ref_regs_when_values_are_none() {
        use crate::opencoder::{TAGCONSTPTR, TraceRecordBuffer, tag};
        use majit_ir::OpRef;
        use std::sync::Arc;

        const LIVE_OP: u8 = 0x42;
        let mut builder = JitCodeBuilder::new();
        let mut jitcode = builder.finish();
        jitcode.body_mut().c_num_regs_i = 0;
        jitcode.body_mut().c_num_regs_r = 2;
        jitcode.body_mut().c_num_regs_f = 0;
        jitcode.body_mut().code = vec![LIVE_OP, 0x00, 0x00];
        let jitcode = Arc::new(jitcode);

        // length_r=2; bitset lights registers 0 and 1.
        let all_liveness: Vec<u8> = vec![0, 2, 0, 0b0000_0011];

        let mut frame = MIFrame::new(jitcode, 0);
        frame.ref_regs[0] = Some(OpRef::const_ptr(majit_ir::GcRef(0xdead_beef)));
        frame.ref_regs[1] = Some(OpRef::ref_op(7));

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(16, sd);
        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* clear_result_register */ false,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 2, "array length prefix");
        let p0 = storage as usize + consumed;
        let (tag0, c0) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p0..]);
        let ptr_idx = *trace
            ._refs_dict
            .get(&0xdead_beef)
            .expect("ConstPtr bits come from ref_regs, not the missing mirror");
        assert_eq!(tag0, tag(TAGCONSTPTR, ptr_idx) as i64);
        let p1 = p0 + c0;
        let (tag1, _) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p1..]);
        assert_eq!(
            tag1,
            TraceRecordBuffer::_encode_box_position(7),
            "ResOp ref register → TAGBOX(7) with no ref_values"
        );
    }

    /// pyjitpl.py:180-193 in_a_call branch — the parent frame in a
    /// nested `capture_resumedata` walk clears the result register of
    /// the in-flight CALL to a zero constant.  When the liveness list
    /// INCLUDES the cleared slot, the snapshot array must record
    /// `Box::ConstInt(0)` (RPython's `history.CONST_FALSE`) instead
    /// of the pre-call register contents.
    ///
    /// Bytecode layout around `self.pc`:
    /// `[... call_op][dst_idx]_at_pc-1 [LIVE_byte]_at_pc
    ///  [off_lo][off_hi] ...` — the LIVE op carries the 2-byte offset
    /// to the bank-size + index bytes inside `all_liveness`.
    #[test]
    fn get_list_of_active_boxes_in_a_call_emits_const_for_cleared_slot() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        // `[0x88_call, 0x00_dst_idx, OP_LIVE, 0x00, 0x00]`
        //            ^^ pc - 1          ^^ pc
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2; // index of OP_LIVE byte

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        // `decode_offset(code, live_pc + 1)` reads offset bytes at
        // indices 3 and 4 — both 0, so `offset` into `all_liveness`
        // is 0.  Layout: `[length_i, length_r, length_f,
        //                 int_bitset_byte]`.  `length_i=1` means "1
        // live int bit set"; bitset byte `0b0000_0001` lights up
        // register index 0 (the cleared slot).
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        // Pre-call stale contents — must NOT leak into the snapshot.
        frame.int_regs[0] = Some(OpRef::int_op(777));

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* clear_result_register */ false,
            OP_LIVE,
            &all_liveness,
            /* after_residual_call */ false,
        );

        // pyjitpl.py:193 — `_result_argcode` flips to `b'?'` after
        // the clear.
        assert_eq!(frame._result_argcode, b'?');

        // Snapshot array content: `[length=1][ConstInt(0) tag]`.
        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // `ConstInt(0)` encodes as `tag(TAGINT, 0) = 0`.
        assert_eq!(tag0, 0, "cleared slot must encode as ConstInt(0)");
    }

    /// pyjitpl.py:184-192 — when `clear_result_register` is set, the
    /// in_a_call branch mutates the register slot itself to the
    /// zero inline-Const.  A second snapshot of the same in-flight
    /// call (after `_result_argcode` flips to `b'?'`) therefore also
    /// sees the cleared register, instead of the pre-call stale box.
    #[test]
    fn get_list_of_active_boxes_in_a_call_clear_flag_mutates_register() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        frame.int_regs[0] = Some(OpRef::int_op(777));

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);

        let _ = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* clear_result_register */ true,
            OP_LIVE,
            &all_liveness,
            /* after_residual_call */ false,
        );

        // Register slot now holds the zero ConstInt inline-Const OpRef,
        // and the parallel value mirror is 0.
        let cleared = frame.int_regs[0].expect("register cleared, not unset");
        assert!(
            cleared.is_constant(),
            "cleared register must be a constant OpRef"
        );
        assert_eq!(frame.int_value_for_blackhole(0), Some(0));
        assert_eq!(frame._result_argcode, b'?');
    }

    /// Complement: when the liveness walk lists a register OTHER than
    /// the cleared slot, the snapshot records the register's pre-
    /// existing box (not ConstInt(0)), while `_result_argcode` still
    /// flips to `b'?'`.
    #[test]
    fn get_list_of_active_boxes_in_a_call_preserves_other_registers() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(2, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        // Liveness: live int idx 1 (NOT the cleared slot 0).
        // bitset byte 0b0000_0010 lights up bit 1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0010];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        frame.int_regs[0] = Some(OpRef::int_op(10)); // cleared — not listed.
        frame.int_regs[1] = Some(OpRef::int_op(11)); // live — recorded as ResOp(11).

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(2, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* clear_result_register */ false,
            OP_LIVE,
            &all_liveness,
            false,
        );
        assert_eq!(frame._result_argcode, b'?');

        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // OpRef::int_op(11) encodes as `tag(TAGBOX, 11) = 11 << 2 | 3 = 47`.
        assert_eq!(tag0, TraceRecordBuffer::_encode_box_position(11));
    }

    /// pyjitpl.py `copy_constants` — liveness indices past
    /// `num_regs_i` read from `jitcode.constants_i`.  pyre does not
    /// copy the constants into the `int_regs` Vec; instead the
    /// liveness walk reads `jitcode.constants_i[idx - num_regs_i]`
    /// directly and emits `Box::ConstInt(v)`.  This test confirms
    /// parity-equivalent snapshot content for the constants area.
    #[test]
    fn get_list_of_active_boxes_reads_constants_area() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![OP_LIVE, 0x00, 0x00];
        let live_pc = 0;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            // `num_regs_i == 1`; constants_i has a single entry at
            // "register" index 1 (= num_regs_i).
            jc_mut.body_mut().constants_i = vec![1234];
            jc
        };

        // `self.pc = live_pc + SIZE_LIVE_OP` (in_a_call=false path:
        // `pc = self.pc - SIZE_LIVE_OP`).
        let current_pc = live_pc + majit_jitcode::codewriter::liveness::OFFSET_SIZE + 1;

        // Liveness lists register index `1` — falls into the
        // constants area.  bitset byte 0b0000_0010 lights up bit 1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0010];

        let mut frame = MIFrame::new(jitcode_arc, current_pc);
        frame.int_regs[0] = Some(OpRef::int_op(5)); // real register — not referenced.

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* clear_result_register */ false,
            OP_LIVE,
            &all_liveness,
            false,
        );

        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // `ConstInt(1234)` — fits in SMALL_INT range so TAGINT with
        // shifted value.  `tag(TAGINT, 1234) = 1234 << 2 = 4936`.
        assert_eq!(tag0, 1234 << 2);
    }

    /// `copy_constants` interned the constants-area slot as `ConstPtr`.
    /// `get_list_of_active_boxes` must encode that intern, not the raw
    /// `constants_r` word, after a later encode-window forward.
    #[test]
    fn get_list_of_active_boxes_encodes_interned_constants_r_after_forward() {
        use crate::opencoder::TraceRecordBuffer;
        use majit_ir::GcRef;

        struct Restore {
            index: u32,
            addr: GcRef,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                majit_ir::const_ptr_table::set_slot(self.index, self.addr);
            }
        }

        const OP_LIVE: u8 = 0x42;
        let original = GcRef(0x96E2_A000);
        let forwarded = GcRef(0x96E2_B000);
        let interned = OpRef::const_ptr(original);
        let index = interned.const_ptr_index().expect("non-null const ptr");
        let _restore = Restore {
            index,
            addr: original,
        };

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(0, 1, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = vec![OP_LIVE, 0x00, 0x00];
            jc_mut.body_mut().constants_r = vec![original.0 as i64]
                .into_iter()
                .map(Into::into)
                .collect();
            jc
        };
        let current_pc = majit_jitcode::codewriter::liveness::OFFSET_SIZE + 1;
        // Liveness lists register index `1` — constants area after
        // `num_regs_r == 1`. bitset byte 0b0000_0010 lights up bit 1.
        let all_liveness: Vec<u8> = vec![0, 1, 0, 0b0000_0010];

        let mut frame = MIFrame::new(jitcode_arc, current_pc);
        assert_eq!(frame.ref_regs.len(), 2);
        frame.ref_regs[1] = Some(interned);
        majit_ir::const_ptr_table::set_slot(index, forwarded);
        assert_eq!(
            frame.jitcode.constants_r[0].get() as usize,
            original.0,
            "constants_r still holds the from-space word"
        );

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);
        let _storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* clear_result_register */ false,
            OP_LIVE,
            &all_liveness,
            false,
        );
        assert_eq!(trace.current_ref(1), forwarded.0 as u64);
    }

    #[test]
    fn cleanup_registers_clears_ref_slots_and_pushed_box() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.int_regs[0] = Some(OpRef::int_op(1));
        frame.ref_regs[0] = Some(OpRef::ref_op(2));
        frame.float_regs[0] = Some(OpRef::float_op(3));
        frame.pushed_box = Some(OpRef::int_op(99));

        frame.cleanup_registers();

        // pyjitpl.py cleanup_registers: int and float slots are untouched.
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(1)));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(3)));
        // pyjitpl.py:124-126: ref slots [0, num_regs_r()) are cleared.
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        // pyjitpl.py:127: pushed_box is reset to None.
        assert_eq!(frame.pushed_box, None);
    }

    #[test]
    fn fill_registers_copies_nine_int_slots_from_caller_bytecode() {
        // pyjitpl.py fill_registers: nine source indices (the 72 B
        // Vec<usize> the typed inline-call path used to allocate) are
        // read from the caller bytecode and written to callee slots
        // 0..9 with no intermediate list.
        let mut caller_jc = make_jitcode_with_regs(9, 0, 0);
        {
            let jc = Arc::get_mut(&mut caller_jc).expect("fresh Arc");
            jc.body_mut().code = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
            jc.body_mut().c_num_regs_i = 9;
        }
        let callee_jc = make_jitcode_with_regs(9, 0, 0);
        let mut caller = MIFrame::new(caller_jc, 0);
        let mut callee = MIFrame::new(callee_jc, 0);
        for i in 0..9 {
            caller.int_regs[i] = Some(OpRef::int_op(10 + i as u32));
        }
        caller.fill_registers(&mut callee, 9, b'I');
        for i in 0..9 {
            assert_eq!(callee.int_regs[i], Some(OpRef::int_op(10 + i as u32)));
        }
        assert_eq!(caller.code_cursor, 9);
    }

    /// pyjitpl.py `MIFrame.replace_active_box_in_frame`.
    ///
    /// Bank dispatch: `oldbox.type` selects which register array to scan;
    /// the other banks must NOT be touched.  Walk replaces every slot
    /// whose `Some(opref)` matches `oldbox` with `Some(newbox)`; non-
    /// matching slots stay untouched.
    #[test]
    fn replace_active_box_in_frame_replaces_only_matching_oprefs() {
        let jitcode = make_jitcode_with_regs(3, 2, 1);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef::int_op(7));
        frame.int_regs[1] = Some(OpRef::int_op(8));
        frame.int_regs[2] = Some(OpRef::int_op(7)); // duplicate — must also flip.
        frame.ref_regs[0] = Some(OpRef::ref_op(7)); // same raw u32 but different bank/variant.
        frame.float_regs[0] = Some(OpRef::float_op(7));

        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Int);

        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(42)));
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(8)));
        assert_eq!(frame.int_regs[2], Some(OpRef::int_op(42)));
        // Ref / float banks untouched — bank dispatch is by oldbox.type.
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(7)));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(7)));
    }

    /// Empty bank short-circuit: pyjitpl.py:248 `if not count: return`.
    #[test]
    fn replace_active_box_in_frame_returns_early_when_bank_empty() {
        let jitcode = make_jitcode_with_regs(0, 1, 0);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.ref_regs[0] = Some(OpRef::ref_op(7));
        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Int);
        // Int bank empty — ref bank stays untouched.
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(7)));
    }

    /// Type::Void is not a valid box type (pyjitpl.py:246 `assert 0,
    /// oldbox`).  The Rust port mirrors RPython's strength — a Void-typed
    /// oldbox indicates the caller's type oracle returned a semantically
    /// impossible answer, so panic rather than silently swallow.
    #[test]
    #[should_panic(expected = "Type::Void")]
    fn replace_active_box_in_frame_void_type_panics() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef::int_op(7));
        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Void);
    }

    #[test]
    fn make_result_of_lastop_stores_into_typed_slot() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);

        frame.pc = 0;
        frame.make_result_of_lastop(JitArgKind::Int, 1, OpRef::int_op(7), 77);
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(7)));

        frame.pc = 1;
        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef::ref_op(8), 88);
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(8)));

        frame.pc = 2;
        frame.make_result_of_lastop(JitArgKind::Float, 0, OpRef::float_op(9), 99);
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(9)));
    }

    #[test]
    fn make_result_of_lastop_accepts_matching_recorded_resulttype() {
        // RPython `pyjitpl.py:260-265`: in non-translated builds the
        // assertion compares the resultbox kind against
        // `jitcode._resulttypes[frame.pc]`.  pyre's writer-side
        // `JitCodeBuilder::record_resulttype` populates the map at
        // end-of-instruction position; this test asserts the reader
        // accepts a matching kind without panicking.
        use crate::JitCallArg;
        let mut builder = JitCodeBuilder::new();
        // Emit a canonical Pure residual_call that records 'i' at
        // end-of-instr per `assembler.py`.
        let fn_idx = builder.add_fn_ptr(std::ptr::dangling::<()>());
        builder.call_pure_int_canonical_via_target(fn_idx, &[JitCallArg::int(0)], 0);
        let jitcode = Arc::new(builder.finish());
        let post_call_pc = jitcode.body().code.len();
        let recorded = jitcode
            .body()
            .resulttypes
            .as_ref()
            .and_then(|resulttypes| resulttypes.get(&post_call_pc).copied());
        assert_eq!(
            recorded,
            Some('i'),
            "writer must record 'i' at end-of-instruction PC for residual_call_int"
        );

        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.pc = post_call_pc;
        // Matching kind — assertion passes silently.
        frame.make_result_of_lastop(JitArgKind::Int, 0, OpRef::int_op(7), 77);
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(7)));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "_resulttypes")]
    fn make_result_of_lastop_panics_on_recorded_resulttype_mismatch() {
        // Same writer setup as the matching variant — but the reader
        // passes `Ref` against a recorded `'i'`, which fires the
        // RPython `pyjitpl.py` assertion port.
        use crate::JitCallArg;
        let mut builder = JitCodeBuilder::new();
        let fn_idx = builder.add_fn_ptr(std::ptr::dangling::<()>());
        builder.call_pure_int_canonical_via_target(fn_idx, &[JitCallArg::int(0)], 0);
        let jitcode = Arc::new(builder.finish());
        let post_call_pc = jitcode.body().code.len();

        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.pc = post_call_pc;
        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef::ref_op(7), 77);
    }

    #[test]
    fn setup_resume_at_op_assigns_pc() {
        let jitcode = make_jitcode_with_regs(0, 0, 0);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.setup_resume_at_op(123);
        assert_eq!(frame.pc, 123);
    }

    #[test]
    fn setup_call_with_empty_argboxes_only_resets_pc() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 42;
        frame.setup_call(&[]);
        assert_eq!(frame.pc, 0);
        assert!(frame.int_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.float_regs.iter().all(|r| r.is_none()));
    }

    #[test]
    fn setup_populates_constants_and_greenkey_inline() {
        let mut jitcode = make_jitcode_with_regs(1, 1, 1);
        {
            let jc = Arc::get_mut(&mut jitcode).expect("fresh Arc");
            jc.body_mut().constants_i = vec![123];
            jc.body_mut().constants_r = vec![0x1234.into()];
            jc.body_mut().constants_f = vec![1.5f64.to_bits() as i64];
        }
        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut recorder = crate::recorder::Trace::new();
        let _ = recorder.record_input_arg(Type::Int);
        let mut ctx = crate::trace_ctx::TraceCtx::new(recorder, 0, sd);

        let frame = MIFrame::setup(jitcode, 7, Some((0xfeed, None)), Some(&mut ctx));

        assert_eq!(frame.pc, 7);
        assert_eq!(frame.greenkey, Some((0xfeed, None)));
        assert_eq!(frame._result_argcode, b'v');
        assert_eq!(frame.parent_snapshot, -1);
        assert_eq!(frame.unroll_iterations, 1);
        assert_eq!(frame.int_value_for_blackhole(1), Some(123));
        assert_eq!(frame.ref_value_for_blackhole(1), Some(0x1234));
        assert_eq!(
            frame.float_value_for_blackhole(1),
            Some(1.5f64.to_bits() as i64)
        );
        assert!(frame.int_regs[1].is_some());
        assert!(frame.ref_regs[1].is_some());
        assert!(frame.float_regs[1].is_some());
    }

    /// pyjitpl.py — `verify_green_args` accepts only Const
    /// OpRefs whose count matches `jitdriver_sd.num_green_args`.
    #[test]
    fn verify_green_args_accepts_constants_matching_num_greens() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Int)],
            vec![("r0", Type::Int)],
        );
        let greens = vec![OpRef::const_int(0), OpRef::const_int(1)];
        // Must not panic — both opref are Const-tagged and length matches.
        MIFrame::verify_green_args(&jd, &greens);
    }

    #[test]
    #[should_panic(expected = "expected 2 greens")]
    fn verify_green_args_rejects_wrong_count() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Int)],
            vec![],
        );
        // Only one green provided — must fail count check.
        MIFrame::verify_green_args(&jd, &[OpRef::const_int(0)]);
    }

    #[test]
    #[should_panic(expected = "is not a Const")]
    fn verify_green_args_rejects_non_const_opref() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(vec![("g0", Type::Int)], vec![]);
        // Plain OpRef::int_op(0) is in the operation namespace (no CONST_BIT) —
        // upstream pyjitpl.py:1534 asserts isinstance(varargs[i], Const).
        MIFrame::verify_green_args(&jd, &[OpRef::int_op(0)]);
    }
}
