//! SSAReprEmitter — setup-side carrier for the walker.
//!
//! The walker (`codewriter.rs::transform_graph_to_jitcode`) accumulates
//! per-op work directly into a walker-local `SSARepr`; this emitter
//! survives only to host the `JitCodeBuilder` setup state
//! (register-file sizing, constant pools, fn-pointer table,
//! virtualizable setup, jitcode name, abort flag) that
//! `Assembler::assemble` needs as its starting builder. The per-op
//! methods that used to mirror `JitCodeBuilder` and push `Insn::Op`
//! values are gone — the dual emitter was collapsed into the
//! single walker-local `SSARepr`.
//!
//! Reference: `rpython/jit/codewriter/codewriter.py`.
//!
//! Portal `jit_merge_point` emission now runs through the graph layer:
//! `codewriter.rs::portal_jit_merge_point_graph_args` builds the
//! upstream-orthodox 7-arg shape; `record_graph_op` stores it; and
//! `GraphFlattener::serialize_op` lowers it into the SSARepr
//! with a custom `lower_constant` callback that routes pycode through
//! the runtime constant pool.  The SSARepr-direct helper that used to
//! live here is gone.

use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};

use super::assembler::{Assembler, NumRegs};
use super::flatten::SSARepr;

/// Setup-side carrier. Every method is either a `JitCodeBuilder`
/// passthrough or `finish_with_positions_from` — the finalization hook
/// that hands the builder to `Assembler::assemble` alongside the
/// walker-local `SSARepr`.
pub(super) struct SSAReprEmitter {
    /// Setup-state builder — carries `fn_ptrs`, constant pools,
    /// register-file sizing, and the jitcode name. `Assembler::assemble`
    /// consumes this builder after the walker is done; the finished
    /// `JitCode` keeps these tables intact.
    builder: JitCodeBuilder,
    /// Counter for walker-side label-id allocation. `new_label()`
    /// returns the next id; walker macros format it into the TLabel
    /// name (`catch_landing_{id}`) that `Assembler::assemble` resolves
    /// against `state.label_positions`.
    next_label_id: u16,
}

impl SSAReprEmitter {
    pub fn new() -> Self {
        Self {
            builder: JitCodeBuilder::default(),
            next_label_id: 0,
        }
    }

    // setup passthrough (mirrors JitCodeBuilder setup API)

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.builder.set_name(name);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.builder.ensure_i_regs(count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.builder.ensure_r_regs(count);
    }

    /// Stage `(fnaddr, calldescr)` from `CallControl::get_jitcode_calldescr`
    /// (`call.py`) so the values land on the constructed `JitCode`
    /// **before** `Assembler::assemble` runs `set_body`.  RPython
    /// `call.py` builds `JitCode(name, fnaddr, calldescr)` ahead of
    /// `assembler.assemble(...)`; pyre stages on the builder so
    /// `JitCodeBuilder::finish()` can stamp the body atomically.
    pub fn set_fnaddr_and_calldescr(
        &mut self,
        fnaddr: i64,
        calldescr: majit_jitcode::jitcode::BhCallDescr,
    ) {
        self.builder.set_fnaddr(fnaddr);
        self.builder.set_calldescr(calldescr);
    }

    /// Register a helper fn pointer without a slot classification —
    /// used by `register_helper_fn_pointers::bind` for `MayForce` /
    /// `ReleaseGil` flavors whose runtime-resolved EI cannot be
    /// represented as an [`majit_metainterp::EffectInfoSlot`]
    /// (`pyjitpl.py do_conditional_call` asserts forces
    /// never reach the slot-reading dispatcher).  The descriptor
    /// stays at the analyzer-absent default
    /// [`majit_metainterp::EffectInfoSlot::CanRaise`].
    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.builder.add_fn_ptr(ptr)
    }

    /// Register a helper fn pointer with its per-callee
    /// [`majit_metainterp::EffectInfoSlot`] classification — used by
    /// `register_helper_fn_pointers` to thread each helper's
    /// `CallFlavor` into the [`majit_metainterp::JitCallTarget`]
    /// runtime descriptor so the trace dispatcher can pick the
    /// matching `EffectInfo` (`call.py getcalldescr` parity).
    pub fn add_fn_ptr_with_slot(
        &mut self,
        ptr: *const (),
        slot: majit_metainterp::EffectInfoSlot,
    ) -> u16 {
        self.builder.add_fn_ptr_with_slot(ptr, slot)
    }

    pub fn has_abort_flag(&self) -> bool {
        self.builder.has_abort_flag()
    }

    // label id allocation

    /// Allocate the next u16 label id. The walker formats the id into
    /// a TLabel name (`catch_landing_{id}`) that `Assembler::assemble`
    /// resolves against the matching `Insn::Label` pushed into the
    /// walker-local `SSARepr`.
    pub fn new_label(&mut self) -> u16 {
        let id = self.next_label_id;
        self.next_label_id = self
            .next_label_id
            .checked_add(1)
            .expect("label id overflow");
        id
    }

    // portal jit_merge_point

    // finalization

    /// Translate an insn-index position into the corresponding JitCode
    /// byte offset using the `ssarepr.insns_pos` table that
    /// `Assembler::assemble` populates (`assembler.py`).
    pub fn insn_pos_to_byte_offset(
        ssarepr: &SSARepr,
        positions: impl IntoIterator<Item = usize>,
    ) -> Vec<usize> {
        let pos_table = ssarepr
            .insns_pos
            .as_ref()
            .expect("ssarepr.insns_pos not populated — call after assemble()");
        positions
            .into_iter()
            .map(|i| {
                *pos_table.get(i).unwrap_or_else(|| {
                    panic!(
                        "insn_pos_to_byte_offset: insn index {} out of range (len {})",
                        i,
                        pos_table.len()
                    )
                })
            })
            .collect()
    }

    /// Feed the walker-local `SSARepr` into `Assembler::assemble`
    /// against the pre-populated builder, translate the walker's
    /// per-PC insn-index map into byte offsets, and return the
    /// finished `JitCode` alongside the translated positions. Returns `None`
    /// when the single-byte register/constant namespace cannot encode it.
    ///
    /// `num_regs` is the post-regalloc per-kind ceiling computed by
    /// `RegAllocator::find_num_colors` from `max(color)+1`
    /// (`codewriter.py:62-67`). Passing pre-regalloc builder values
    /// would over-allocate the `JitCode.num_regs_*` slots that
    /// `Assembler::emit_reg`'s 256-bound assertion (`assembler.py`)
    /// checks against.
    pub fn finish_with_positions_from(
        self,
        assembler: &mut Assembler,
        mut ssarepr: SSARepr,
        insn_positions: &[usize],
        num_regs: NumRegs,
    ) -> Option<(JitCode, Vec<usize>)> {
        let jitcode = assembler.try_assemble(&mut ssarepr, self.builder, Some(num_regs))?;
        let byte_positions =
            Self::insn_pos_to_byte_offset(&ssarepr, insn_positions.iter().copied());
        Some((jitcode, byte_positions))
    }
}

impl Default for SSAReprEmitter {
    fn default() -> Self {
        Self::new()
    }
}
