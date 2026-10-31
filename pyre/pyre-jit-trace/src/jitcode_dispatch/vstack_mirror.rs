//! Value-stack mirror: classify / reconcile / reseed / seed / step /
//! enter-handler.
//!
//! **Parity:** pyre-specific — mirrors the interpreter value-stack shape
//! for the FBW walker; PyPy's `MIFrame` uses register banks with no stack
//! mirror.
//!
//! Tracks the interpreter value-stack shape alongside the walker so a
//! guard resume coordinate and the catch-landing depth can be
//! reconstructed. Includes the pc-map block-head helpers the mirror
//! steps through.

use super::*;

/// #73: classify `instr` for the [`VstackOpClass`] taxonomy.  Mirrors the
/// stack-effect grouping in [`crate::liveness`]'s `stack_effects`, but
/// collapsed to the three categories the operand-stack box maintenance
/// cares about.  `op_arg` is read only where the net effect depends on it
/// (LOAD_GLOBAL's NULL-sentinel low bit, LOAD_SUPER_ATTR's method bit).
pub(crate) fn classify_vstack_opcode(
    instr: &pyre_interpreter::bytecode::Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> VstackOpClass {
    use pyre_interpreter::bytecode::Instruction;
    match instr {
        // Trivia / no stack effect — neither produces a TOS box nor pops.
        // Treat as pop-only-or-side-store: truncate to the (unchanged)
        // depth, leave the surviving slots intact.
        Instruction::Nop
        | Instruction::Resume { .. }
        // COPY_FREE_VARS copies closure cells into the frame's locals/cells
        // region and leaves the operand stack untouched.  It is the first
        // opcode of a closure body, so classifying it as Unmodeled disabled
        // the mirror for the whole function.  A later conditional expression
        // could then abort after an earlier side effect and replay from entry,
        // skipping the remaining work when that side effect was a seen-set
        // insertion.  PyPy's MIFrame changes no value-stack register here;
        // preserve the same no-op stack shape.
        | Instruction::CopyFreeVars { .. }
        | Instruction::Cache
        | Instruction::NotTaken
        // Pyre's END_FOR is a no-op; the following POP_ITER removes the
        // iterator (codewriter.rs / pyopcode.rs).
        | Instruction::EndFor
        | Instruction::ExtendedArg => VstackOpClass::PopOnlyOrSideStore,

        // Single value lands on the new TOS = the last Ref written.
        Instruction::LoadConst { .. }
        | Instruction::LoadSmallInt { .. }
        // `return_generator` pushes the `None` the following `POP_TOP`
        // discards (`liveness.rs` net +1). Leaving it unmodeled dropped the
        // mirror, so the later `YIELD_VALUE` finished with that `None`
        // instead of the value `LOAD_SMALL_INT` pushed.
        | Instruction::ReturnGenerator
        | Instruction::LoadFast { .. }
        | Instruction::LoadFastBorrow { .. }
        | Instruction::LoadFastCheck { .. }
        | Instruction::LoadFastAndClear { .. }
        | Instruction::LoadName { .. }
        | Instruction::LoadDeref { .. }
        | Instruction::LoadLocals
        // LOAD_BUILD_CLASS shares one codewriter arm and one liveness arm with
        // LOAD_LOCALS (`(d + 1, d + 1)`), and the `abort_permanent` arm there is
        // gated on `is_locals && !is_true_portal`, so it always lowers to the
        // `load_build_class` residual plus `emit_pushvalue_ref!`.  Left
        // unmodeled it killed the mirror at the first `class` statement in a
        // loop-owning frame.
        | Instruction::LoadBuildClass
        | Instruction::UnaryNegative
        | Instruction::UnaryNot
        | Instruction::UnaryInvert
        // TO_BOOL is net 0 (liveness `(d, d)`) and emits no JitCode
        // (codewriter `Instruction::ToBool => {}`), so it stamps no
        // `write_ref_reg` box of its own.  It still belongs here: the
        // unchanged value is exactly `vstack_last_ref`, and where that is
        // NONE — the tested value came from an unboxed bank —
        // `loadconst_operand_ref` fail-closes to NONE, leaving the
        // intentional hole the capture overlay omits and resume
        // rematerializes.  What it must NOT be is `Unmodeled`: that latched
        // `vstack_valid = false` whenever the after-residual guard-capture
        // reconcile re-entered at the TO_BOOL pc (`prev_pypc == new_pypc`),
        // dropping the kept accumulator below a residual-call condition
        // (`s = s + (i if f(...) else 0)`).
        | Instruction::ToBool
        | Instruction::GetLen
        // MATCH_SEQUENCE / MATCH_MAPPING / MATCH_KEYS peek their operands and
        // push one result (liveness `(d + 1, d + 1)`), the same shape as GET_LEN
        // and IMPORT_FROM in this group.  `match_keys` is bound MayForce — it
        // runs the subject's `get` — so a replay of a region containing one can
        // re-enter a Python frame.
        | Instruction::MatchSequence
        | Instruction::MatchMapping
        | Instruction::MatchKeys
        | Instruction::LoadAttr { .. }
        | Instruction::ImportFrom { .. }
        | Instruction::BinaryOp { .. }
        | Instruction::CompareOp { .. }
        | Instruction::IsOp { .. }
        | Instruction::ContainsOp { .. }
        | Instruction::Call { .. }
        | Instruction::BuildTuple { .. }
        | Instruction::BuildList { .. }
        | Instruction::BuildSet { .. }
        | Instruction::BuildMap { .. }
        | Instruction::BuildString { .. }
        // Compute opcodes that pop their operands and leave a single result on
        // the new TOS (= the last Ref written, captured via `write_ref_reg` or
        // the operand-stack push chokepoint) — same shape as the arithmetic /
        // build group above.  PUSH_NULL's sole new TOS is the pushed NULL
        // marker; FORMAT_SIMPLE/FORMAT_WITH_SPEC/CONVERT_VALUE/BINARY_SLICE/
        // IMPORT_NAME each push exactly one result.
        | Instruction::PushNull
        | Instruction::FormatSimple
        | Instruction::FormatWithSpec
        | Instruction::ConvertValue { .. }
        | Instruction::BinarySlice
        | Instruction::ImportName { .. }
        // MATCH_CLASS pops the keyword-names tuple, the class and the subject
        // and pushes one result (liveness `(d - 2, d - 2)`), so the new TOS sits
        // where the popped `subject` box was and `ResultToTos` overwrites it.
        // `MultiResultFromShadow` would be wrong here: its pop point is
        // `vstack_depth - 1`, which for a net-negative opcode is an EMPTY clear
        // range, and both hole-fill helpers skip non-NONE slots — the stale
        // `subject` box would survive as the result.
        | Instruction::MatchClass { .. }
        // MAKE_FUNCTION pops the code object and pushes the built function
        // (net 0, `stack_effects` `(d, d)`).  The `make_function_value`
        // residual's Ref result reaches the new TOS through the operand-stack
        // push chokepoint (`emit_pushvalue_ref!`, codewriter.rs), so it is
        // the same `ResultToTos` shape as the value producers above.  Left
        // unmodeled it killed the mirror for the rest of the walk at the first
        // nested `def`, declining any later depth > 1 kept-stack branch guard.
        | Instruction::MakeFunction
        // SET_FUNCTION_ATTRIBUTE follows MAKE_FUNCTION for defaults,
        // annotations, and closures, and pushes the updated function back to
        // TOS.  The remaining ops here also leave one result on the new TOS;
        // their arg-dependent depths are already baked into
        // `pyre/pyre-jit-trace/src/liveness.rs`'s depth table.
        | Instruction::SetFunctionAttribute { .. }
        | Instruction::CallKw { .. }
        | Instruction::BuildSlice { .. }
        | Instruction::CallFunctionEx
        | Instruction::CallIntrinsic1 { .. }
        | Instruction::LoadCommonConstant { .. }
        | Instruction::LoadFromDictOrGlobals { .. }
        // #73: LOAD_FAST/STORE_FAST super-instructions.  Their net
        // result still lands on the new TOS as the LAST Ref written (the
        // second load, resp. the load following the store), so `ResultToTos`
        // models the top slot correctly.  A two-push pair
        // (`LoadFast(Borrow)LoadFast(Borrow)`, net +2) additionally leaves the
        // slot BELOW the new TOS a NONE hole; the general hole-fill in
        // `reconcile_vstack_at_boundary` recovers it from the virtualizable
        // shadow (or leaves it NONE when unsourceable — the overlay then
        // omits the slot, which resume re-materializes) WITHOUT invalidating
        // the mirror.  Net-0 `StoreFastLoadFast` overwrites the
        // consumed TOS with the loaded value (no hole).  Before this slice
        // these fell through to `Unmodeled`, killing the mirror at the first
        // super-instruction in a short-circuit / condexpr loop body.
        | Instruction::LoadFastLoadFast { .. }
        | Instruction::LoadFastBorrowLoadFastBorrow { .. }
        | Instruction::StoreFastLoadFast { .. } => VstackOpClass::ResultToTos,

        // Pop-only / side-store / control transfer: the surviving TOS box
        // is already in `vstack_boxes`, do NOT overwrite it from the last
        // ref (which targets a local/global/attr, not the new stack TOS).
        Instruction::PopTop
        | Instruction::PopIter
        | Instruction::PopExcept
        | Instruction::StoreFast { .. }
        // STORE_FAST__STORE_FAST: two consecutive local stores, pops 2 with no
        // stack result — a pure side-store, the surviving slots just truncate.
        | Instruction::StoreFastStoreFast { .. }
        | Instruction::StoreName { .. }
        | Instruction::StoreGlobal { .. }
        | Instruction::StoreDeref { .. }
        | Instruction::StoreAttr { .. }
        | Instruction::DeleteAttr { .. }
        | Instruction::StoreSubscr
        | Instruction::DeleteSubscr
        | Instruction::StoreSlice
        // LIST_APPEND / SET_ADD / MAP_ADD / LIST_EXTEND and the dict/set
        // update opcodes pop their value operand(s) and mutate the collection
        // PEEK'd in place below them — a side-store, same shape as
        // STORE_SUBSCR: the surviving TOS box stays put. MAKE_CELL stores its
        // result into the frame-local virtualizable slot, not operand TOS.
        | Instruction::ListAppend { .. }
        | Instruction::SetAdd { .. }
        | Instruction::MapAdd { .. }
        | Instruction::ListExtend { .. }
        | Instruction::DictUpdate { .. }
        | Instruction::DictMerge { .. }
        | Instruction::SetUpdate { .. }
        | Instruction::MakeCell { .. }
        | Instruction::DeleteFast { .. }
        | Instruction::DeleteName { .. }
        | Instruction::DeleteGlobal { .. }
        | Instruction::DeleteDeref { .. }
        | Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. }
        | Instruction::JumpForward { .. }
        | Instruction::JumpBackward { .. }
        | Instruction::JumpBackwardNoInterrupt { .. }
        | Instruction::ReturnValue => VstackOpClass::PopOnlyOrSideStore,

        // LOAD_GLOBAL: pyre pushes the global and then, for `namei & 1`, the
        // NULL call sentinel. A guard emitted while lowering the following
        // CALL can observe both slots, so method form keeps a distinct shape
        // instead of treating NULL as an unsourceable result hole.
        Instruction::LoadGlobal { namei } if namei.get(op_arg) & 1 != 0 => {
            VstackOpClass::LoadGlobalMethod
        }
        Instruction::LoadGlobal { .. } => VstackOpClass::ResultToTos,

        // LOAD_SUPER_ATTR pops `global_super`, `__class__`, and `self`
        // (`liveness.rs` `Instruction::LoadSuperAttr`). Non-method form pushes
        // one result (net -2); `ResultToTos` truncates to that depth and writes
        // `vstack_last_ref` on the new TOS, leaving `global_super` in the
        // surviving lower slot. Method form (`op_arg & 1`) pushes `func` then
        // `self` (net -1). Under `ResultToTos` that truncate-and-stamp keeps
        // `global_super` under TOS (`self`) instead of installing `func`;
        // `LoadSuperAttrMethod` clears both pushed slots so the hole-fill
        // sources them from the virtualizable shadow (`setarrayitem_vable_r`
        // of `super_attr_unwrap` 0 and 1).
        Instruction::LoadSuperAttr { namei } if u32::from(namei.get(op_arg)) & 1 != 0 => {
            VstackOpClass::LoadSuperAttrMethod
        }
        Instruction::LoadSuperAttr { .. } => VstackOpClass::ResultToTos,

        // LOAD_SPECIAL pops the context-manager object at `prev_depth - 1`
        // and pushes the special method followed by the call self/NULL slot
        // upward from that position.  Clear that popped slot and the pushed
        // range so each pushed slot is sourced from the virtualizable shadow,
        // except the trailing `self_or_null` the reconcile stamps as an
        // explicit NULL constant.
        Instruction::LoadSpecial { method }
            if matches!(
                method.get(op_arg),
                pyre_interpreter::bytecode::SpecialMethod::Enter
                    | pyre_interpreter::bytecode::SpecialMethod::Exit
            ) =>
        {
            VstackOpClass::LoadSpecialMethod
        }

        // GET_ITER pops the iterable and pushes iterator then index-or-NULL,
        // the same two-push shape as LOAD_SPECIAL's method form.
        Instruction::GetIter { .. } => VstackOpClass::LoadSpecialMethod,

        // SWAP(i): exchange TOS with the box `i` positions below.  A pure
        // permutation (net depth 0); the decoded `i` drives the
        // `vstack_boxes` exchange in `reconcile_vstack_at_boundary`.
        Instruction::Swap { i } => VstackOpClass::Swap(i.get(op_arg) as usize),

        // COPY(i): duplicate the box `i` positions from the top onto the new
        // TOS (net +1).  The decoded `i` drives the duplicate-from-slot copy
        // in `reconcile_vstack_at_boundary` (sources `vstack_boxes[depth-1-i]`,
        // not `vstack_last_ref`, so `COPY i>1` is faithful).
        Instruction::Copy { i } => VstackOpClass::Copy(i.get(op_arg) as usize),

        // Exception machinery inside a handler body.  The unwinder + exc-info
        // operations rewrite the operand stack in ways a producer/pop model
        // cannot express, but every resulting slot is written through
        // `setarrayitem_vable_r`, so the virtualizable shadow is authoritative
        // — `ShadowReseed` reconciles by reseeding from it.  Inert on the
        // non-exception path: these are reached only via the unwind/catch
        // edge, where the mirror is already invalid unless
        // `vstack_enter_exception_handler` re-seeded it at handler entry.
        Instruction::PushExcInfo
        | Instruction::CheckExcMatch
        | Instruction::Reraise { .. }
        | Instruction::RaiseVarargs { .. }
        | Instruction::WithExceptStart => VstackOpClass::ShadowReseed,

        // UNPACK_SEQUENCE / UNPACK_EX: pop one sequence, push its elements
        // (net push > 1).  Every pushed element is in the virtualizable
        // shadow, so reconcile reseeds the pushed range from it.
        Instruction::UnpackSequence { .. } | Instruction::UnpackEx { .. } => {
            VstackOpClass::MultiResultFromShadow
        }

        // BUILD_LIST_FROM_ARG pops the iterator and pushes list then iterator.
        // TOS is the restored iterator, not the list, so ResultToTos (the
        // BuildList class) would overwrite the wrong slot.
        Instruction::BuildListFromArg => VstackOpClass::MultiResultFromShadow,

        // FOR_ITER (continue arm): peeks the iterator (kept on the stack) and
        // pushes the yielded item on the new TOS (net +1) — the same shape as
        // the value producers above, so the item is `vstack_last_ref`.  The
        // item never reaches either mirror chokepoint on its own (the
        // `for_iter_next` residual result is stamped via `set_opref_concrete`,
        // not `write_ref_reg`, and the item lands on TOS through the
        // codewriter's `pin!` slot binding, not a `setarrayitem_vable_r`
        // push), so the residual-execution path seeds `vstack_last_ref` with
        // the item OpRef explicitly (the `ForIterNext` capture site).  The
        // exhaustion arm pushes no item, but it is a non-fallthrough guard
        // exit, so the boundary's `sequential` gate suppresses this per-op
        // effect there.
        Instruction::ForIter { .. } => VstackOpClass::ResultToTos,

        // Everything else is not modeled — decline; the overlay then omits
        // the affected slots, which resume re-materializes.  The forward walk
        // maps JitCode pc -> py_pc and so never lands ON a no-JitCode opcode
        // (TO_BOOL), but the after-residual guard-capture reconcile passes an
        // explicit resume py_pc that CAN coincide with the TO_BOOL pc, so
        // TO_BOOL is modeled explicitly above rather than left to this arm.
        _ => VstackOpClass::Unmodeled,
    }
}

/// The boxed constant a value `LOAD_CONST` / `LOAD_SMALL_INT` /
/// prebuilt `LOAD_COMMON_CONSTANT` pushes, as a trace-constant OpRef, or
/// `OpRef::NONE` when `instr` is none of those (or its constant is
/// unresolvable). Realizes the constant the same way `bh_load_const_fn` /
/// `w_small_int_const` / `load_common_constant_value` do, so the mirror
/// carries the identical box the resume path would.
fn loadconst_operand_ref<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &pyre_interpreter::CodeObject,
    instr: &pyre_interpreter::bytecode::Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> OpRef {
    if let pyre_interpreter::bytecode::Instruction::LoadSmallInt { i } = instr {
        let val = i.get(op_arg) as u32 as i64;
        return ctx
            .trace_ctx
            .const_ref(pyre_object::w_small_int_const(val) as i64);
    }
    if let pyre_interpreter::bytecode::Instruction::LoadCommonConstant { idx } = instr {
        let cc = pyre_interpreter::pyopcode::common_constant_arg(*idx, op_arg);
        if !pyre_interpreter::opcode_ops::common_constant_is_prebuilt(cc) {
            return OpRef::NONE;
        }
        let w_const = pyre_interpreter::opcode_ops::load_common_constant_value(cc);
        return ctx.trace_ctx.const_ref(w_const as i64);
    }
    let pyre_interpreter::bytecode::Instruction::LoadConst { consti } = instr else {
        return OpRef::NONE;
    };
    let idx = usize::from(consti.get(op_arg));
    let w_code = if ctx.is_top_level {
        let session = ctx.session.borrow();
        let frame = session.recording_frame_ptr as *const pyre_interpreter::PyFrame;
        if frame.is_null() {
            0
        } else {
            unsafe { (*frame).pycode as usize }
        }
    } else {
        ctx.inline_w_code()
    };
    let w_const = if w_code == 0 || w_code == usize::MAX {
        // Malformed/test-only walks have no enclosing PyCode. Keep the
        // historical materializer fallback; production LOAD_CONST always has
        // the per-frame red `pycode` owner.
        pyre_interpreter::pyframe::load_const_from_code(code, idx)
    } else {
        unsafe { pyre_interpreter::pycode::w_code_const(w_code as pyre_object::PyObjectRef, idx) }
    };
    if w_const.is_null() {
        return OpRef::NONE;
    }
    ctx.trace_ctx.const_ref(w_const as i64)
}

/// Apply pyre's two-push method-form `LOAD_GLOBAL` stack shape.  The callable
/// slot remains a hole until the ordinary shadow fill below supplies its box;
/// the following NULL is already a complete Ref constant and must not be
/// confused with an unresolved `OpRef::NONE` slot.
pub(crate) fn reconcile_load_global_method_shape(
    boxes: &mut Vec<OpRef>,
    old_depth: usize,
    new_depth: usize,
    null: OpRef,
) {
    let old_depth = old_depth.min(new_depth);
    boxes.truncate(new_depth);
    if boxes.len() < new_depth {
        boxes.resize(new_depth, OpRef::NONE);
    }
    for slot in &mut boxes[old_depth..new_depth] {
        *slot = OpRef::NONE;
    }
    if new_depth > old_depth {
        boxes[new_depth - 1] = null;
    }
}

/// Box currently in the register the codewriter assigned to operand-stack
/// slot `stack_slot` at `jit_pc`.
///
/// `pyopcode.py` `pushvalue` writes `locals_cells_stack_w`;
/// `_opimpl_setarrayitem_vable` then names that box.  Pyre's walk-level
/// mirror follows the same slot through the pcdep color map
/// (`put_back_virtualizable_mapped_colors`).  A coalesced `LOAD_FAST` is a
/// register move that executes no jitcode op, so `write_ref_reg` never
/// stamps `vstack_last_ref`; this read is the box the pushed slot holds
/// now.
fn vstack_slot_box_from_pcdep_registers<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jit_pc: usize,
    stack_slot: usize,
) -> OpRef {
    if ctx.fbw_mode.inline_subwalk {
        return OpRef::NONE;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return OpRef::NONE;
    }
    // SAFETY: pointer live for the full-body walk; read-only nlocals / jitcode.
    let sym = unsafe { &*sym_ptr };
    if !sym.owns_virtualizable_shadow() || sym.jitcode().is_null() {
        return OpRef::NONE;
    }
    let jc = unsafe { &*sym.jitcode() };
    let pcdep = jc.payload.pcdep_for_jitcode_pc(jit_pc).unwrap_or_default();
    if pcdep.is_empty() {
        return OpRef::NONE;
    }
    let slot = sym.nlocals() + stack_slot;
    let Some(color) = crate::state::semantic_slot_color_for_ref_slot(&pcdep, slot) else {
        return OpRef::NONE;
    };
    ctx.registers_r.get(color).unwrap_or(OpRef::NONE)
}

/// #73: reconcile the PREVIOUS Python opcode's stack effect into
/// [`WalkFrameStateData::vstack_boxes`] at an opcode boundary, BEFORE the new
/// opcode (`new_pypc`) is walked.  Running this before the new op means
/// that when the new op is a branch guard, `vstack_boxes` already holds
/// the correct boxes for the guard's resume depth.
///
/// `code` is the Python `CodeObject` of the outer (full-body) jitcode;
/// `new_pypc` is the Python pc the walk is about to enter; `new_depth` is
/// `depth_at_py_pc[new_pypc]` (stack-only).  The previous opcode is
/// decoded from `code` at `ctx.vstack_cur_pypc`.  `jit_pc` is the JitCode
/// offset of that new opcode, used to read the pcdep color of the pushed
/// stack slot.
///
/// On any unmodeled effect (or a structurally impossible depth) the
/// function latches `ctx.vstack_valid = false` so the `stack_sync`
/// overlay omits every operand slot, which resume re-materializes (zero
/// regression).
pub(crate) fn reconcile_vstack_at_boundary<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &pyre_interpreter::CodeObject,
    new_pypc: u32,
    new_depth: usize,
    after_residual_call: bool,
    jit_pc: usize,
) {
    if !ctx.vstack_valid {
        return;
    }
    let prev_pypc = ctx.vstack_cur_pypc as usize;
    let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, prev_pypc) else {
        ctx.vstack_valid = false;
        return;
    };
    let class = classify_vstack_opcode(&instr, op_arg);
    // RPython's MIFrame follows one flow-graph link at a time; the target
    // block receives its register state from that link's inputargs
    // (pyjitpl.py `interpret` / `run_one_step`).  The full-body
    // walk instead traverses flattened JitCode whose layout can switch to a
    // different source block without the Python PCs forming a CFG edge.  Do
    // not replay the previous Python opcode across such a layout switch: its
    // `vstack_last_ref` belongs to the other block.  Doing so at the two arms
    // of a conditional expression overwrote the loop-carried FOR_ITER slot
    // with the selected local, and a later branch guard serialized that
    // foreign box into the virtualizable resume image.
    //
    // Enter a shadow-reseed region until the walk passes both endpoints of
    // the out-of-order transition.  This subsumes the former SWAP/COPY-only
    // detection for non-monotonic comprehension lowering and applies the
    // same block-input rule to ordinary conditional-expression arms.
    let fallthrough = crate::pyjitpl::semantic_fallthrough_pc(code, prev_pypc);
    use pyre_interpreter::Instruction;
    let has_fallthrough = !matches!(
        instr,
        Instruction::JumpForward { .. }
            | Instruction::JumpBackward { .. }
            | Instruction::JumpBackwardNoInterrupt { .. }
            | Instruction::ReturnValue
            | Instruction::Reraise { .. }
            | Instruction::RaiseVarargs { .. }
    );
    // A boundary that reports the SAME py_pc is not a block transition at all:
    // the walk has not left this Python opcode, so no source block was visited
    // out of order.  One opcode's jitcode expansion can carry more than one
    // boundary marker, and treating the repeat as a backed-off transition armed
    // the permutation region on the spot, which forces `ShadowReseed` below and
    // clears every mirror slot — including the ones this opcode did not touch.
    // The `layout_only_boundary` arm already models a repeat correctly (the
    // observed depth matches neither successor of the previous opcode, so it
    // preserves the surviving slots), and it is the arm the reorder region
    // excludes.  A genuine self-branch is unaffected: `target_pc` reports it,
    // so the clause below already accepts it.
    // The handler an exception raised at `prev_pypc` transfers to is a CFG
    // successor as much as a branch delta is; it just lives in
    // `co_exceptiontable` instead of the opcode's operand, so `target_pc`
    // cannot report it.  Left out, an unwind to the handler read as a block
    // permutation and armed the reorder region over the whole handler body,
    // where `ShadowReseed` drops every operand box the shadow does not carry.
    // One Python opcode's jitcode expansion can carry more than one boundary
    // marker, so a boundary can report the py_pc it came from.  The walk has
    // not left that opcode there, so the opcode has not retired.
    let repeat_boundary = new_pypc as usize == prev_pypc;
    let cfg_successor = repeat_boundary
        || (has_fallthrough && new_pypc as usize == fallthrough)
        || crate::liveness::target_pc(code, &instr, prev_pypc, op_arg) == Some(new_pypc as usize)
        || crate::liveness::exception_target_pc(code, prev_pypc) == Some(new_pypc as usize);
    // #389(b): leave the out-of-order permutation region once the walk has
    // advanced PAST the py_pc it backed off from — py order is monotonic again
    // and the per-op reconcile is valid from here, INCLUDING at this boundary.
    // Tested after the class had already been applied, the region covered one
    // boundary too many: the exiting step is an ordinary sequential one whose
    // previous opcode really did produce `vstack_last_ref`, and `ShadowReseed`
    // dropped that box for the reseed, which cannot recover a slot the shadow
    // does not carry.  Ordered before the arming below so a boundary that both
    // passes the old ceiling and is itself out of order opens a new region
    // instead of running unprotected.
    if ctx.vstack_reorder_ceiling != u32::MAX && new_pypc > ctx.vstack_reorder_ceiling {
        ctx.vstack_reorder_ceiling = u32::MAX;
        ctx.frame_state.borrow_mut().vstack_reorder_saved = None;
    }
    if !cfg_successor && ctx.vstack_reorder_ceiling == u32::MAX {
        ctx.vstack_reorder_ceiling = (new_pypc as usize).max(prev_pypc) as u32;
        let mut state = ctx.frame_state.borrow_mut();
        state.vstack_reorder_saved = Some((
            prev_pypc as u32,
            ctx.vstack_depth,
            state.vstack_boxes.clone(),
            Vec::new(),
        ));
    }
    let in_reorder_region = ctx.vstack_reorder_ceiling != u32::MAX;
    let (fallthrough_depth, branch_depth) =
        crate::liveness::stack_effects(&instr, op_arg, ctx.vstack_depth);
    if vstack_diag_enabled() {
        eprintln!(
            "[vstack-reconcile] code={} sub={} prev_pypc={prev_pypc} new_pypc={new_pypc} \
             new_depth={new_depth} prev_depth={} class={class:?} reorder={in_reorder_region} \
             succ=(ft={},br={:?},exc={:?}) last_ref={:?} instr={instr:?}",
            code.obj_name,
            ctx.fbw_mode.inline_subwalk,
            ctx.vstack_depth,
            if has_fallthrough {
                fallthrough as isize
            } else {
                -1
            },
            crate::liveness::target_pc(code, &instr, prev_pypc, op_arg),
            crate::liveness::exception_target_pc(code, prev_pypc),
            ctx.frame_state.borrow().vstack_last_ref
        );
    }
    // A JitCode's block layout can visit source-PC floor segments out of
    // Python bytecode order.  In that case `vstack_cur_pypc` is only the
    // preceding layout segment, not the opcode whose stack effect produced
    // this boundary.  Applying its effect is unsound: a LOAD_FAST segment
    // whose expected depth is `d + 1` can otherwise overwrite a surviving
    // FOR_ITER iterator when the observed depth stayed at `d`.
    //
    // RPython reads the live MIFrame registers and never reconstructs this
    // transition from source PCs.  Preserve the slots that demonstrably
    // survive at the observed depth whenever neither real successor depth
    // matches; holes remain conservative and are handled by the shadow fill
    // below.
    let layout_only_boundary = new_depth != fallthrough_depth && new_depth != branch_depth;

    // The excursion came back to the coordinate it left.  A `LOAD_ATTR` +
    // `CALL` pair puts the walk here: the CALL's lowering re-enters the
    // LOAD_ATTR source segment, so the floor lookup reports the earlier py_pc,
    // and the return boundary reports the CALL's again at the same depth.  No
    // Python opcode retired in between — the stack the walk left is the stack
    // it resumes with — so the saved boxes, not a reseed, are this boundary's
    // reconcile.  The reseed cannot serve this shape: the operands the
    // in-flight opcode already popped read NULL in the virtualizable shadow,
    // so `reseed_vstack_from_shadow` rejects them and the callable slides down
    // into the `self_or_null` slot with the TOS left a hole.  The restore
    // replaces only the reconcile; the shadow-backed hole-fill below still
    // runs over the result.
    //
    // The excursion itself is an artifact of walking a flattened jitcode:
    // `pyopcode.py` `LOAD_ATTR` pops and pushes the live value stack in
    // place, so the interpreter has no walk position to leave and return to,
    // and nothing here to mirror.
    let returned_to_arm_point = matches!(
        &ctx.frame_state.borrow().vstack_reorder_saved,
        Some((pc, depth, _, _)) if *pc == new_pypc && *depth == new_depth
    );
    let restored = in_reorder_region && returned_to_arm_point;
    if restored {
        let (_, _, saved, mask) = ctx
            .frame_state
            .borrow_mut()
            .vstack_reorder_saved
            .take()
            .expect("checked by returned_to_arm_point");
        // `vstack_boxes` has two producers: the executed
        // `setarrayitem_vable_r` store, and this pc-derived reconstruction.
        // `_opimpl_setarrayitem_vable` (`pyjitpl.py`) is the only writer
        // of `virtualizable_boxes` upstream and takes its index from the
        // descr, never from a pc, so no precedence rule was ever needed there.
        // Here one is: a slot the walk actually executed a store into is
        // authoritative, and the snapshot — taken at a boundary that can be
        // reported mid-opcode, so it can hold a state no opcode ever left
        // behind — must not overwrite it.  Same-pc-and-same-depth cannot tell
        // the two apart: `obj.m()` with the method-load fold suppressed saves
        // `[NULL, method]` mid-`LOAD_ATTR` and returns to the same coordinate
        // once the stores have produced `[method, NULL]`, so restoring the
        // snapshot wholesale hands `CALL` a null callable.
        //
        // With no executed store in the window the mask is empty and the
        // restore is unchanged, which keeps the slots it legitimately
        // recovers: `ShadowReseed` clears the mirror at the arming boundary
        // and the shadow-backed hole-fill declines any slot the virtualizable
        // does not carry, and those holes are exactly what the snapshot
        // restores.
        //
        // Such a restore can replace a live-looking Ref with a NULL const-ptr,
        // and that is the intended outcome rather than a lost box.  A `with`
        // block keeps the resolved `__exit__` and its `self_or_null` NULL side
        // by side at the base of the block's stack for the whole body, and the
        // NULL's slot is the one the callable's own push wrote last, so a
        // reseed inside the region refills it from the shadow with the
        // callable.  The snapshot is what puts the NULL back.  Extending the
        // mask over these slots would keep the callable instead — the shape a
        // blackhole resuming into WITH_EXCEPT_START calls with the receiver
        // twice.
        if mask.iter().all(|&w| !w) {
            ctx.frame_state.borrow_mut().vstack_boxes = saved;
        } else {
            // Merge per slot rather than keeping either side whole: one window
            // can hold both an executed store and a slot the reseed dropped.
            let cur = std::mem::replace(&mut ctx.frame_state.borrow_mut().vstack_boxes, saved);
            if ctx.frame_state.borrow().vstack_boxes.len() < new_depth {
                ctx.frame_state
                    .borrow_mut()
                    .vstack_boxes
                    .resize(new_depth, OpRef::NONE);
            }
            for s in 0..new_depth {
                if mask.get(s).copied().unwrap_or(false)
                    && let Some(&v) = cur.get(s)
                {
                    ctx.frame_state.borrow_mut().vstack_boxes[s] = v;
                }
            }
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .truncate(new_depth);
        }
        ctx.vstack_reorder_ceiling = u32::MAX;
    }

    // PER-OP RECONCILE.  In the SEQUENTIAL case the previous opcode's stack
    // effect explains the depth change: a producer (`ResultToTos`) lands its
    // result box (`vstack_last_ref`) on the new TOS; a pop / side-store just
    // truncates.  This captures the kept boxes from the walk register file
    // (LOAD_FAST / LOAD_NAME / COPY results) — values that may NOT be present
    // in the virtualizable shadow (function-local LOAD_FAST temps live only
    // in the walk register bank, never written through to the portal array).
    // Inside the out-of-order permutation region the per-op replay is invalid;
    // reseed from the shadow (same shape as `ShadowReseed`).
    let effective_class = if in_reorder_region && !restored {
        VstackOpClass::ShadowReseed
    } else {
        class
    };
    // GET_ITER / LOAD_SPECIAL after-residual: the residual's result is the
    // first pushed value (`make_result_of_lastop`), and neither push has
    // run yet. Fill from `vstack_last_ref` after the hole-fill below, and
    // only while the slot is still NONE, so a looked-inside GET_ITER that
    // already wrote the iterator through `setarrayitem_vable_r` is not
    // overwritten by a previous opcode's residual.
    let mut fill_first_push_from_residual = false;
    match effective_class {
        // The saved boxes ARE the reconcile for this boundary; the previous
        // opcode's effect belongs to the excursion, not to the arm point.  The
        // shadow-backed hole-fill below still runs: a slot the arm point
        // already carried as a hole is no more recoverable from the saved
        // boxes than from a reseed, and skipping the fill left the operand
        // `LOAD_GLOBAL` pushed unsourced in the resume image.
        _ if restored => {}
        // A layout-only boundary (the observed depth matches neither real
        // successor of the previous opcode) means the per-op effect cannot
        // explain this transition; preserve the surviving slots instead of
        // replaying a stale effect.  The reorder region is already served by
        // the `ShadowReseed` arm, so exclude it here.
        _ if layout_only_boundary && !in_reorder_region => {
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .resize(new_depth, OpRef::NONE);
        }
        VstackOpClass::ResultToTos => {
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .resize(new_depth, OpRef::NONE);
            if new_depth > 0 {
                // `NONE` means the result was produced in an unboxed bank;
                // leave an intentional hole so the capture overlay around
                // stack_sync omits the slot and resume rematerializes it.
                let mut top = ctx.frame_state.borrow().vstack_last_ref;
                // `pyopcode.py` `pushvalue` / `_opimpl_setarrayitem_vable`:
                // the snapshot names the box the pushed slot holds now.
                // Prefer the pcdep color at this py_pc over `vstack_last_ref`
                // — a coalesced `LOAD_FAST` executes no op, so
                // `write_ref_reg` never stamps the last-ref.
                let from_reg = vstack_slot_box_from_pcdep_registers(ctx, jit_pc, new_depth - 1);
                if from_reg != OpRef::NONE {
                    top = from_reg;
                }
                // `LOAD_CONST` / `LOAD_SMALL_INT` / a prebuilt
                // `LOAD_COMMON_CONSTANT` push that constant. A `ref_copy`
                // renaming does not stamp `vstack_last_ref`, so the previous
                // opcode's ref (the `None` a generator send pushed) would
                // otherwise become this TOS and the yield would finish with
                // it.  Skip this mint when the pcdep register already holds
                // the pushed box, so the snapshot names that identity.
                let const_result = matches!(
                    instr,
                    pyre_interpreter::bytecode::Instruction::LoadSmallInt { .. }
                        | pyre_interpreter::bytecode::Instruction::LoadConst { .. }
                        | pyre_interpreter::bytecode::Instruction::LoadCommonConstant { .. }
                );
                if from_reg == OpRef::NONE && (top == OpRef::NONE || const_result) {
                    let from_const = loadconst_operand_ref(ctx, code, &instr, op_arg);
                    if !from_const.is_none() {
                        top = from_const;
                    }
                }
                ctx.frame_state.borrow_mut().vstack_boxes[new_depth - 1] = top;
            }
        }
        VstackOpClass::LoadGlobalMethod => {
            // `opcode_load_global`: push the callable, then NULL.  The
            // callable's vable store fills the preceding hole below; NULL is
            // semantically live even though it carries no GC pointer.
            let null = ctx.trace_ctx.const_null();
            reconcile_load_global_method_shape(
                &mut ctx.frame_state.borrow_mut().vstack_boxes,
                ctx.vstack_depth,
                new_depth,
                null,
            );
        }
        VstackOpClass::PopOnlyOrSideStore => {
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .truncate(new_depth);
        }
        VstackOpClass::Swap(i) => {
            // SWAP is net-depth-0 (prev_depth == new_depth).  Exchange the
            // TOS box with the box `i` positions below it, matching
            // `swap_values` (`localsplus[depth-1] <-> localsplus[depth-i]`).
            // A NONE in either slot is just permuted (the later hole-fill /
            // legacy-defer handles it); a malformed / out-of-range arg
            // declines (latch invalid).
            //
            // The exchange belongs to the boundary that RETIRES the opcode.  A
            // repeat boundary has not left it, and SWAP is the one class whose
            // replay is not idempotent: exchanging the same pair twice restores
            // the pre-SWAP order, so the mirror ends up disagreeing with the
            // `setarrayitem_vable_r` stores the walk executed for that SWAP,
            // and the capture overlay then publishes the stale order into a
            // branch guard's virtualizable resume section.  A repeat is invisible
            // to `layout_only_boundary` here precisely because the net-zero
            // depth matches the fallthrough successor.  The depth normalization
            // still runs: it is what keeps mirror slot `s` naming operand-stack
            // slot `s` at the observed depth.
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .resize(new_depth, OpRef::NONE);
            if repeat_boundary {
                // The retiring boundary performs the exchange.
            } else if new_depth >= 1 && i >= 1 && i <= new_depth {
                let top = new_depth - 1;
                let other = new_depth - i;
                ctx.frame_state.borrow_mut().vstack_boxes.swap(top, other);
            } else {
                ctx.vstack_valid = false;
            }
        }
        VstackOpClass::Copy(i) => {
            // COPY(i): duplicate the box `i` positions from the top onto the
            // new TOS (net +1).  The duplicated box is the COPIED slot
            // `vstack_boxes[new_depth-1-i]` (`opcode_copy_value` =
            // `push(peek_at(i-1))`), sourced directly rather than from
            // `vstack_last_ref` so `COPY i>1` (duplicating a deeper operand)
            // is faithful; `COPY 1` reduces to dup-of-TOS.  A missing source
            // slot or out-of-range arg declines (latch invalid).
            match new_depth.checked_sub(1 + i) {
                Some(src_idx)
                    if i >= 1 && src_idx < ctx.frame_state.borrow().vstack_boxes.len() =>
                {
                    let src = ctx.frame_state.borrow().vstack_boxes[src_idx];
                    ctx.frame_state
                        .borrow_mut()
                        .vstack_boxes
                        .resize(new_depth, OpRef::NONE);
                    ctx.frame_state.borrow_mut().vstack_boxes[new_depth - 1] = src;
                }
                _ => ctx.vstack_valid = false,
            }
        }
        VstackOpClass::ShadowReseed => {
            // Resize to the post-opcode depth, leaving every slot a NONE
            // hole; the shadow-backed hole-fill below sources each slot from
            // the virtualizable shadow the exception lowering just wrote.  An
            // unsourceable slot (genuine NULL exc-info / Int temp) stays NONE
            // and `mirror_covers_kept` declines for it — the conservative
            // fallback, never a corrupt box.
            ctx.frame_state.borrow_mut().vstack_boxes.clear();
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .resize(new_depth, OpRef::NONE);
        }
        VstackOpClass::MultiResultFromShadow
        | VstackOpClass::LoadSpecialMethod
        | VstackOpClass::LoadSuperAttrMethod => {
            // UNPACK_* pops ONE sequence (at `prev_depth - 1`) and pushes its
            // elements upward.  Clear only the affected range
            // `[pop_point .. new_depth)` to NONE so the hole-fill below
            // sources each pushed element from the shadow (all were written
            // through `setarrayitem_vable_r`); slots BELOW the popped sequence
            // keep their mirror-tracked boxes.
            //
            // LOAD_SPECIAL has that same shape but its second push is the
            // call's `self_or_null` slot, a live NULL for the bound method it
            // just resolved.  The shadow cannot give that slot back:
            // `reseed_vstack_from_shadow` reads a dense array in which an
            // absent slot and a written NULL are the same word, so it rejects
            // NULL and the slot stays a NONE hole `stack_sync` omits.  A
            // blackhole resuming into the `WITH_EXCEPT_START` that reads it
            // then sees whatever the slot held before — the bound `__exit__`
            // this same opcode pushed one slot below — and calls it with the
            // receiver twice.  Stamp the constant, as the method-form
            // LOAD_GLOBAL arm above does for the other callable/NULL pair.
            //
            // Method-form LOAD_SUPER_ATTR pops three and pushes two (`func`,
            // `self`); both pushed slots are already in the shadow, so clear
            // `[new_depth - 2 .. new_depth)` and leave `trailing_null` unset.
            let trailing_null = match effective_class {
                VstackOpClass::LoadSpecialMethod => Some(ctx.trace_ctx.const_null()),
                _ => None,
            };
            let pop_point = if matches!(effective_class, VstackOpClass::LoadSuperAttrMethod) {
                new_depth.saturating_sub(2)
            } else {
                ctx.vstack_depth.saturating_sub(1)
            };
            ctx.frame_state
                .borrow_mut()
                .vstack_boxes
                .resize(new_depth, OpRef::NONE);
            for s in pop_point..new_depth {
                ctx.frame_state.borrow_mut().vstack_boxes[s] = OpRef::NONE;
            }
            if new_depth > 0 {
                if let Some(null) = trailing_null {
                    ctx.frame_state.borrow_mut().vstack_boxes[new_depth - 1] = null;
                }
            }
            fill_first_push_from_residual =
                after_residual_call && trailing_null.is_some() && new_depth >= 2;
        }
        VstackOpClass::Unmodeled => {
            ctx.vstack_valid = false;
        }
    }
    // The FBW walk follows jitcode control flow, not just sequential opcodes:
    // an `and`/`or` chain's short-circuit continuation jumps BACKWARD to a
    // deeper merge point, so the previous opcode did NOT produce the slots
    // below the new TOS — the per-op reconcile leaves a NONE hole there.
    // Recover those slots from the virtualizable shadow (kept current by the
    // portal `setarrayitem_vable_r` pushes for values that ARE written
    // through).  `reseed_vstack_from_shadow` rejects a NULL-const shadow slot
    // (a function-local temp the portal never wrote), so a genuinely
    // unrecoverable kept slot fails the re-seed.
    //
    // A non-reseedable hole does NOT latch `vstack_valid = false`: an
    // Int/Float-bank operand-stack temp (e.g. the `while i < N` loop
    // condition's `LoadConst N`, a transient `BINARY_OP` int result) is not
    // a Ref the Ref-only mirror can ever hold, but it is CONSUMED before the
    // all-Ref short-circuit guard region — invalidating the whole mirror
    // there made it die at the loop condition, never reaching the kept-stack
    // guard.  Instead keep the mirror TRACKING (advance position / depth)
    // with the NONE slot left in place; `stack_sync` (USE) omits any NONE
    // mirror slot, which resume re-materializes.
    if ctx.vstack_valid {
        // Admission decides the shadow owner: filling a callee hole from the
        // outer portal virtualizable would record a caller operand as the
        // callee's kept-stack value. A sub-walk mirror only
        // exists once `seed_callee_vstack_mirror` / `step_vstack_mirror` have
        // admitted it, so `inline_subwalk` alone is that predicate here.
        let callee_local_shadow = ctx.fbw_mode.inline_subwalk;
        let hole = ctx
            .frame_state
            .borrow()
            .vstack_boxes
            .get(..new_depth)
            .map(|s| s.iter().any(|&b| b == OpRef::NONE))
            .unwrap_or(true);
        if hole {
            // Best-effort fill from this frame's own storage; leave
            // un-fillable slots NONE.  An inline callee's locals/stack array
            // belongs to its CalleeLocalsShadow, never to the outer portal
            // virtualizable.
            if callee_local_shadow {
                let _ = reseed_vstack_from_callee_shadow(ctx, code, new_depth);
            } else {
                let _ = reseed_vstack_from_shadow(ctx, new_depth);
            }
        }
    }
    if fill_first_push_from_residual && ctx.vstack_valid && new_depth >= 2 {
        let result = ctx.frame_state.borrow().vstack_last_ref;
        let hole = ctx
            .frame_state
            .borrow()
            .vstack_boxes
            .get(new_depth - 2)
            .copied()
            == Some(OpRef::NONE);
        // LoadSpecialMethod's last store is the trailing NULL sentinel, not
        // the first push; a null-constant result would fill the hole wrongly.
        if hole && result != OpRef::NONE && !opref_is_null_const_ptr(result) {
            ctx.frame_state.borrow_mut().vstack_boxes[new_depth - 2] = result;
        }
    }
    if ctx.vstack_valid {
        ctx.vstack_cur_pypc = new_pypc;
        ctx.vstack_depth = new_depth;
        ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
    }
}

/// Fill missing operand-stack boxes from the active inline frame's own
/// localsplus shadow.  `CalleeLocalsShadow` is the Rust owner corresponding
/// to that MIFrame's register/frame state; indexing begins after this code
/// object's locals and cells, so no portal-frame slot can leak across the
/// inline boundary.
fn reseed_vstack_from_callee_shadow<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &pyre_interpreter::CodeObject,
    new_depth: usize,
) -> bool {
    if ctx.frame_state.borrow().vstack_boxes.len() < new_depth {
        ctx.frame_state
            .borrow_mut()
            .vstack_boxes
            .resize(new_depth, OpRef::NONE);
    }
    let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let mut state = ctx.frame_state.borrow_mut();
    let WalkFrameStateData {
        callee_shadow,
        vstack_boxes,
        ..
    } = &mut *state;
    let Some(shadow) = callee_shadow.as_ref() else {
        return false;
    };
    let mut all_present = true;
    for (s, slot) in vstack_boxes[..new_depth].iter_mut().enumerate() {
        if *slot != OpRef::NONE {
            continue;
        }
        // The NULL const-ptr rejection here is stricter than the shadow can
        // justify: `opref` is a sparse map, so a PRESENT key is already the
        // proof that this walk wrote the slot, and a present key holding
        // CONST_NULL is a deliberately written NULL — PUSH_NULL's `self_or_null`
        // ahead of a call inside the inlined callee.  `reseed_vstack_from_shadow`
        // needed a per-slot live-NULL side table to draw that same distinction
        // only because its source is a dense array, where absent and NULL are
        // the same word.  Kept as-is regardless: dropping the clause leaves the
        // whole dynasm corpus at 386/386 with NO jit-stats movement, so nothing
        // measures the difference, and an unwitnessed widening of what counts as
        // a resolved mirror slot is the direction that turns a decline into a
        // wrong answer.
        match shadow.opref.get(&((stack_base + s) as i64)).copied() {
            Some(value) if value != OpRef::NONE && !opref_is_null_const_ptr(value) => {
                *slot = value;
            }
            _ => all_present = false,
        }
    }
    all_present
}

/// #73: re-seed `ctx.vstack_boxes[0..new_depth]` from the virtualizable
/// shadow's operand-stack slots (`virtualizable_box_at(nvs + nlocals + s)`).
/// Used when a control-flow edge makes the per-opcode reconcile model
/// inapplicable (a backward/forward jump landing at a different stack
/// level).  The portal `pyframe.pushvalue` lowers every Ref push to
/// `setarrayitem_vable_r(locals_cells_stack_w, depth, w_obj)`, so the
/// shadow holds the live operand stack at a merge point.
///
/// Returns `true` on success (every slot `0..new_depth` sourced as a
/// non-NONE box), `false` if any slot is unsourceable (caller then
/// latches `vstack_valid = false`).
pub(crate) fn reseed_vstack_from_shadow<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    new_depth: usize,
) -> bool {
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return false;
    }
    // SAFETY: pointer live for the full-body walk; read-only nlocals.
    let nlocals = unsafe { (*full_body_sym).nlocals() };
    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    // Only FILL the NONE holes the per-op reconcile could not source from
    // the walk register file; keep the boxes the reconcile DID capture (the
    // shadow may carry a stale value in a non-hole slot).  A hole the shadow
    // also cannot source (NONE / NULL const-ptr — a function-local temp the
    // portal never wrote through) fails the whole re-seed so the caller
    // leaves the slot NONE; `stack_sync` then omits it (resume
    // re-materializes).
    //
    // A NULL const-ptr is rejected because it cannot be told apart from a slot
    // the portal never wrote — EXCEPT where the slot carries the live-NULL
    // marker, which says the last executed store into it wrote a NULL on
    // purpose.  That is PUSH_NULL's `self_or_null` sentinel: it stays live
    // across the whole callable/args/kwargs build ahead of a CALL, and the
    // reorder region reseeds the mirror from the shadow in the middle of that
    // build.  Rejecting it left slot NONE, `capture_vstack_mirror_image`
    // refuses an image with any unresolved slot, and an escape inside the call
    // then had no blackhole image at all and fell back to the legacy entry
    // replay — which re-runs the residuals the walk already executed.
    if ctx.frame_state.borrow().vstack_boxes.len() < new_depth {
        ctx.frame_state
            .borrow_mut()
            .vstack_boxes
            .resize(new_depth, OpRef::NONE);
    }
    let mut all_present = true;
    for s in 0..new_depth {
        if ctx.frame_state.borrow().vstack_boxes[s] != OpRef::NONE {
            continue;
        }
        let flat = nvs + nlocals + s;
        match ctx.trace_ctx.virtualizable_box_at(flat) {
            Some(b)
                if b != OpRef::NONE
                    && (!opref_is_null_const_ptr(b)
                        || ctx.trace_ctx.virtualizable_slot_stored_live_null(flat)) =>
            {
                ctx.frame_state.borrow_mut().vstack_boxes[s] = b;
            }
            // Fill what we can; an unsourceable hole (NONE / NULL const-ptr —
            // an Int/Float-bank temp or a function-local the portal never
            // wrote) stays NONE.  `stack_sync` omits a NONE slot (resume
            // re-materializes), so an unfilled slot is never a corrupt box.
            _ => all_present = false,
        }
    }
    all_present
}

/// #73: map a jitcode pc to the Python opcode whose lowering region
/// CONTAINS it, WITHOUT the `containing_py_pc_for_jitcode_pc` block-head marker
/// special-case.  For the operand-stack mirror we want the containing
/// opcode (where the walk physically is), not the resume block-head a
/// `-live-` marker names — the marker case returns an EARLIER py_pc and
/// makes the mirror's boundary detection oscillate.  Uses only the
/// JitCode-PC floor pivot (largest floor boundary at-or-before `jit_pc`).
pub(crate) fn vstack_containing_py_pc(metadata: &crate::PyJitCodeMetadata, jit_pc: usize) -> u32 {
    if !metadata.py_floor_by_jit_pc.is_empty() {
        return crate::pyjitcode::floor_segment_for_jitcode_pc(
            &metadata.py_floor_by_jit_pc,
            jit_pc,
        )
        .expect("drained JitCode PC floor pivot must begin at byte offset zero")
        .1;
    }
    0
}

pub(crate) fn vstack_initial_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
    permuted_for_iter_entry: bool,
) -> u32 {
    if !permuted_for_iter_entry {
        return vstack_containing_py_pc(metadata, jit_pc);
    }
    metadata_block_head_py_pc(metadata, jit_pc)
        .unwrap_or_else(|| vstack_containing_py_pc(metadata, jit_pc))
}

pub(crate) fn metadata_block_head_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> Option<u32> {
    metadata
        .block_head_py_by_jit_pc
        .binary_search_by_key(&jit_pc, |&(off, _)| off)
        .ok()
        .map(|i| metadata.block_head_py_by_jit_pc[i].1)
}

/// Read depth for the same Python opcode selected by vstack_step_py_pc.
/// The floor twin is valid only while its owner equals that opcode; disjoint
/// emission regions can select a different exact owner. Both portal and
/// inline frames use their own code's liveness for that owner.
pub(crate) fn vstack_step_depth(pjc: &crate::PyJitCode, jit_pc: usize, py_pc: u32) -> usize {
    let raw_depth = || {
        crate::liveness::liveness_for(pjc.code_ptr)
            .depth_at_py_pc()
            .get(py_pc as usize)
            .copied()
            .unwrap_or(0) as usize
    };
    if pjc.depth_containing_populated() && py_pc == vstack_containing_py_pc(&pjc.metadata, jit_pc) {
        let depth = pjc.depth_containing_for_jitcode_pc(jit_pc).unwrap_or(0) as usize;
        if pcmap_containing_audit_enabled() {
            assert_eq!(
                depth,
                raw_depth(),
                "vstack step depth at jit_pc {jit_pc}, py {py_pc}"
            );
        }
        depth
    } else {
        raw_depth()
    }
}

pub(crate) fn vstack_step_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
    current_py_pc: u32,
) -> u32 {
    // A block-head entry is a control-flow marker, not the lowering of a
    // Python opcode.  Keep the current mirror coordinate until the first
    // actual operation in the destination block.  In particular, the floor
    // segment at a marker can still name the terminal opcode of the preceding
    // block (for example its RETURN_VALUE); applying that opcode here corrupts
    // the live caller stack before the destination block is entered.
    if metadata_block_head_py_pc(metadata, jit_pc).is_some() {
        return current_py_pc;
    }
    // Prefer the per-emission segmentation over the floor tier.  The floor
    // table keys each Python PC to its FIRST jitcode offset, so a PC that
    // emits in two disjoint regions is collapsed onto the earlier one and the
    // later region reads as belonging to whichever PC last opened a segment.
    // The mirror then sees a boundary the walk never crossed and replays that
    // opcode's stack effect against a stack it does not describe — the same
    // shape as applying a SWAP at a boundary that did not retire it.
    //
    // `py_exact_by_jit_pc` records the owning PC at every point the emitted
    // stream changes owner, so it answers directly.  Empty for skeleton /
    // fixture metadata, where the floor tier remains the only source.
    if let Some(exact) =
        crate::pyjitcode::exact_py_pc_for_jitcode_pc(&metadata.py_exact_by_jit_pc, jit_pc)
        && !exact_segmentation_disabled()
    {
        return exact;
    }
    vstack_containing_py_pc(metadata, jit_pc)
}

/// `PYRE_VSTACK_DIAG`, read once: the mirror reconciles at every boundary,
/// on every walk, and an environment lookup per boundary is not free.
fn vstack_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_VSTACK_DIAG").is_some())
}

/// `PYRE_VSTACK_NO_EXACT`: fall back to the floor tier for the mirror's
/// coordinate. The escape hatch for A/B-ing the segmentation switchover
/// against the tier it replaces.
fn exact_segmentation_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("PYRE_VSTACK_NO_EXACT").is_some())
}

/// GET_ITER / method-form LOAD_SPECIAL: pop one, push two (iterator or
/// special method, then index-or-NULL / self_or_null). The backward
/// `-live-` skip below only applies to these opcodes; CHECK_EXC_MATCH and
/// other after-residual captures still take the reorder reseed.
fn two_push_get_iter_or_load_special(code: &pyre_interpreter::CodeObject, pc: usize) -> bool {
    let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc) else {
        return false;
    };
    classify_vstack_opcode(&instr, op_arg) == VstackOpClass::LoadSpecialMethod
}

/// True when the operand-stack shadow already holds a non-null box in
/// `first_slot..depth`. After-residual GET_ITER / LOAD_SPECIAL pop before
/// their two pushes land; an empty pair is the `jit_next(NULL)` reseed,
/// while a populated pair is the later marker that should ShadowReseed a
/// stale capture image. Outer operand slots are ignored so a nested
/// GET_ITER is not treated as complete just because an enclosing iterator
/// already occupies a lower slot.
fn vstack_operand_shadow_occupied<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    code: &pyre_interpreter::CodeObject,
    first_slot: usize,
    depth: usize,
) -> bool {
    if first_slot >= depth {
        return false;
    }
    if ctx.fbw_mode.inline_subwalk {
        let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
        let state = ctx.frame_state.borrow();
        let Some(shadow) = state.callee_shadow.as_ref() else {
            return false;
        };
        (first_slot..depth).any(
            |s| match shadow.opref.get(&((stack_base + s) as i64)).copied() {
                Some(value) if value != OpRef::NONE && !opref_is_null_const_ptr(value) => true,
                _ => false,
            },
        )
    } else {
        if ctx.fbw_mode.snapshot_sym.is_null() {
            return false;
        }
        let nlocals = unsafe { (*ctx.fbw_mode.snapshot_sym).nlocals() };
        let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        (first_slot..depth).any(
            |s| match ctx.trace_ctx.virtualizable_box_at(nvs + nlocals + s) {
                Some(b)
                    if b != OpRef::NONE
                        && (!opref_is_null_const_ptr(b)
                            || ctx
                                .trace_ctx
                                .virtualizable_slot_stored_live_null(nvs + nlocals + s)) =>
                {
                    true
                }
                _ => false,
            },
        )
    }
}

/// #73: step the walk-level operand-stack box mirror at
/// the top of every jitcode `step`.  Detects a Python-opcode boundary by
/// mapping the current `jit_pc` back to its containing Python opcode; when
/// that differs from `ctx.vstack_cur_pypc`, reconciles the previous
/// opcode's stack effect into `vstack_boxes` (see
/// [`reconcile_vstack_at_boundary`]).
///
/// No-op unless the outer full-body sym owns the virtualizable shadow and
/// `vstack_valid` is still set.  Reached only when the outer full-body sym
/// owns the shadow (`fbw_mode.snapshot_sym` non-null).  Writes
/// only the `vstack_*` side-fields; never the registers / snapshot.
pub(crate) fn step_vstack_mirror<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, jit_pc: usize) {
    if !ctx.vstack_valid {
        return;
    }
    // On genuine callee sub-walk paths, `jit_pc` is a callee coordinate with
    // no meaning in the outer (`fbw_mode.snapshot_sym`) jitcode's py_pc→jitcode
    // tables.  `inline_subwalk` is also set for the carrier walk of root code,
    // where that premise does not hold and the mirror is simply never seeded.
    // The explicit diagnostic gate and the recursive-closure admission are
    // shared with `seed_callee_vstack_mirror`.
    let (new_pypc, code_ptr, new_depth) = if ctx.fbw_mode.inline_subwalk {
        let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) else {
            ctx.vstack_valid = false;
            return;
        };
        // A carrier-resume walk is the rebuilt MIFrame of
        // `resume.py rebuild_from_resumedata` being driven forward after guard
        // failure. Its per-frame operand stack must read the exact boxes the
        // reconstructed frame owns, just like an ordinary inline sub-walk.
        let Some(coord) = frame.vstack_step_coordinate_for_jitcode_pc(jit_pc, ctx.vstack_cur_pypc)
        else {
            ctx.vstack_valid = false;
            return;
        };
        coord
    } else {
        let full_body_sym = ctx.fbw_mode.snapshot_sym;
        if full_body_sym.is_null() {
            return;
        }
        // SAFETY: the pointer is live for the lifetime of the full-body walk
        // (set in `dispatch_via_miframe`); read-only access to immutable
        // layout fields (jitcode / code_ptr / metadata).
        let sym = unsafe { &*full_body_sym };
        if sym.jitcode().is_null() {
            return;
        }
        unsafe {
            let jc = &*sym.jitcode();
            if jc.payload.code_ptr.is_null() {
                return;
            }
            let py_pc = vstack_step_py_pc(&jc.payload.metadata, jit_pc, ctx.vstack_cur_pypc);
            let depth = vstack_step_depth(&jc.payload, jit_pc, py_pc);
            // Stage gate for the per-emission exact segmentation.  The
            // boundary the mirror acts on is currently INFERRED — `py_pc` is
            // the floor segment's owner, and whether an opcode retired is then
            // guessed from (prev, new, depth).  `py_exact_by_jit_pc` answers it
            // directly.  Compare the two at every step before any consumer
            // trusts the new table: a disagreement is a boundary the floor tier
            // reports differently from the emission record, and each one is a
            // case the inference has to special-case.
            if vstack_exact_audit_enabled() {
                if let Some(exact) = crate::pyjitcode::exact_py_pc_for_jitcode_pc(
                    &jc.payload.metadata.py_exact_by_jit_pc,
                    jit_pc,
                ) && exact != py_pc
                {
                    let code = &*jc.payload.code_ptr;
                    eprintln!(
                        "[vstack-exact] code={} jit_pc={jit_pc} floor_py={py_pc} \
                         exact_py={exact} cur_py={} boundary={}",
                        code.obj_name,
                        ctx.vstack_cur_pypc,
                        py_pc != ctx.vstack_cur_pypc,
                    );
                }
            }
            (py_pc, jc.payload.code_ptr, depth)
        }
    };
    if new_pypc == ctx.vstack_cur_pypc {
        // Reporting the mirror's own coordinate: whatever landing block the
        // walk was crossing is behind it, so stop holding that py_pc.
        ctx.vstack_handler_landing_py = None;
        return;
    }
    // After-residual capture already applied this opcode's stack effect and
    // advanced `vstack_cur_pypc` to the resume py_pc. A later `-live-` still
    // owned by the retired opcode reports that earlier py_pc, which looks
    // like a backward layout switch. GET_ITER / LOAD_SPECIAL have popped
    // but not yet pushed: reseeding from that empty pair leaves the
    // iterator NULL (`jit_next(NULL)`). Keep the capture boxes until the
    // two pushed slots appear in the shadow; then fall through so the
    // reorder reseed can replace a stale image. Other opcodes, including
    // CHECK_EXC_MATCH, take the reorder reseed on this marker so a stale
    // capture box is not published as live.
    if new_pypc < ctx.vstack_cur_pypc {
        let code = unsafe { &*code_ptr };
        let depth = ctx.vstack_depth;
        if crate::pyjitpl::semantic_fallthrough_pc(code, new_pypc as usize)
            == ctx.vstack_cur_pypc as usize
            && two_push_get_iter_or_load_special(code, new_pypc as usize)
            && !vstack_operand_shadow_occupied(ctx, code, depth.saturating_sub(2), depth)
        {
            return;
        }
    }
    // Still inside the landing block a catch target opened: these bytes belong
    // to no Python opcode, so hold the handler-entry coordinate rather than
    // treating the segment they were laid out in as a boundary.  Any other
    // py_pc means the walk has left them; the mirror resumes stepping from the
    // handler entry, which is where the reconstruction placed it.  Together
    // with the clear above the hold lasts exactly as far as the landing block:
    // a later boundary that happens to name the same py_pc — the opcode those
    // bytes were laid out inside really is walked eventually — reconciles
    // normally.
    if ctx.vstack_handler_landing_py == Some(new_pypc) {
        return;
    }
    ctx.vstack_handler_landing_py = None;
    // The Python-opcode boundary: sample the executed-effect odometer for any
    // abort leg that resumes the interpreter AT this opcode, which re-executes
    // it whole (`FBW_OPCODE_ENTRY_EFFECTS`).  Top-level only — a sub-walk's
    // `py_pc` indexes the callee's code object.
    if !ctx.fbw_mode.inline_subwalk {
        fbw_note_opcode_entry_effects(new_pypc as usize);
    }
    let code = unsafe { &*code_ptr };
    reconcile_vstack_at_boundary(ctx, code, new_pypc, new_depth, false, jit_pc);
}

/// Advance the operand-stack mirror to the post-step resume coordinate.
///
/// [`step_vstack_mirror`] runs at the top of the next `step`.  Aborting
/// first leaves the mirror on the completed opcode's on-entry stack.
/// RPython's MIFrame after `run_one_step` already holds the post-step
/// registers (`_copy_data_from_miframe`).
///
/// A block-head merge point is the measured hole: `vstack_step_py_pc`
/// keeps the previous opcode so the walk does not apply that opcode's
/// effect before the destination block runs.  `blackhole_if_trace_too_long`
/// aborts AT that marker (`pyjitpl.py`), and the CRN green is the
/// destination `next_instr`.  FOR_ITER continue then publishes
/// `[total, iter]` while STORE_NAME needs `[total, iter, n]`.
pub(crate) fn reconcile_vstack_to_resume_pc<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    step_vstack_mirror(ctx, jit_pc);
    if !ctx.vstack_valid || ctx.fbw_mode.inline_subwalk {
        return;
    }
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return;
    }
    let (dest_py, code_ptr, depth) = unsafe {
        let sym = &*full_body_sym;
        if sym.jitcode().is_null() {
            return;
        }
        let jc = &*sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            return;
        }
        let Some(dest_py) = metadata_block_head_py_pc(&jc.payload.metadata, jit_pc) else {
            return;
        };
        if dest_py == ctx.vstack_cur_pypc {
            return;
        }
        let depth = vstack_step_depth(&jc.payload, jit_pc, dest_py);
        (dest_py, jc.payload.code_ptr, depth)
    };
    if !ctx.fbw_mode.inline_subwalk {
        fbw_note_opcode_entry_effects(dest_py as usize);
    }
    reconcile_vstack_at_boundary(ctx, unsafe { &*code_ptr }, dest_py, depth, false, jit_pc);
}

pub(crate) fn seed_callee_vstack_mirror<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame: &ActiveResumeFrame,
) {
    let Some((first_pypc, _code_ptr, _depth)) = frame.vstack_coordinate_for_jitcode_pc(0) else {
        return;
    };
    ctx.frame_state.borrow_mut().vstack_boxes.clear();
    ctx.vstack_depth = 0;
    ctx.vstack_cur_pypc = first_pypc;
    ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
    ctx.vstack_handler_landing_py = None;
    ctx.vstack_valid = true;
}

/// #73: seed the walk-level operand-stack box mirror
/// ([`WalkFrameStateData::vstack_boxes`]) at full-body-walk entry.  Enables the
/// mirror (`vstack_valid = true`) only when the outer `sym` owns the
/// virtualizable shadow AND the entry operand stack can be fully sourced
/// from that shadow.  Sets `vstack_cur_pypc = entry_py_pc` and
/// `vstack_depth = depth_at_py_pc[entry_py_pc]`, filling
/// `vstack_boxes[0..depth]` from the virtualizable shadow's operand-stack
/// slots (`virtualizable_box_at(nvs + nlocals + s)`) — the SAME source
/// `collect_outer_active_boxes` / `stack_sync` read.  Any unsourceable
/// slot leaves `vstack_valid = false`; the overlay then omits operand
/// slots, which resume re-materializes (zero regression).
pub(crate) fn seed_vstack_mirror<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    sym: &Sym,
    start_pc: usize,
) {
    if sym.jitcode().is_null() || !sym.owns_virtualizable_shadow() {
        return;
    }
    // Ordinarily seed at the opcode containing the first walked jitcode op,
    // so the first step is a no-op and reconciliation starts only after an
    // opcode actually runs. A SWAP/COPY immediately preceding a FOR_ITER
    // block-head marker is different: the shadow already contains the
    // post-permutation stack at trace entry. Seed that marker at FOR_ITER and
    // let `vstack_step_py_pc` ignore the marker itself, preventing the
    // predecessor permutation from being applied to the mirror a second time.
    let (first_pypc, depth, nlocals) = unsafe {
        let jc = &*sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            return;
        }
        let containing_pypc = vstack_containing_py_pc(&jc.payload.metadata, start_pc);
        let predecessor_permuted_stack = pyre_interpreter::decode_instruction_at(
            &*jc.payload.code_ptr,
            containing_pypc as usize,
        )
        .is_some_and(|(instr, op_arg)| {
            matches!(
                classify_vstack_opcode(&instr, op_arg),
                VstackOpClass::Swap(_) | VstackOpClass::Copy(_)
            )
        });
        let target_is_for_iter = metadata_block_head_py_pc(&jc.payload.metadata, start_pc)
            .and_then(|target| {
                let code = &*jc.payload.code_ptr;
                let target = target as usize;
                if pyre_interpreter::instruction_logical_start(code, target) != target {
                    return None;
                }
                pyre_interpreter::decode_instruction_at(
                    code,
                    pyre_interpreter::instruction_opcode_pc(code, target),
                )
            })
            .is_some_and(|(instr, _)| {
                matches!(
                    instr,
                    pyre_interpreter::bytecode::Instruction::ForIter { .. }
                )
            });
        let permuted = predecessor_permuted_stack && target_is_for_iter;
        let first_pypc = vstack_initial_py_pc(&jc.payload.metadata, start_pc, permuted);
        let raw_depth = || {
            crate::liveness::liveness_for(jc.payload.code_ptr)
                .depth_at_py_pc()
                .get(first_pypc as usize)
                .copied()
                .unwrap_or(0) as usize
        };
        let d = if jc.payload.depth_containing_populated() {
            // Mirror `vstack_initial_py_pc`'s branch: the exact block-head
            // marker depth for a permuted FOR_ITER entry (falling back to the
            // floor when `start_pc` is not a marker), else the floor segment
            // depth for a normal entry.
            let twin = if permuted {
                jc.payload
                    .depth_block_head_for_jitcode_pc(start_pc)
                    .unwrap_or_else(|| {
                        jc.payload
                            .depth_containing_for_jitcode_pc(start_pc)
                            .unwrap_or(0)
                    })
            } else {
                jc.payload
                    .depth_containing_for_jitcode_pc(start_pc)
                    .unwrap_or(0)
            } as usize;
            if pcmap_containing_audit_enabled() {
                assert_eq!(
                    twin,
                    raw_depth(),
                    "PYRE_PCMAP_CONTAINING_AUDIT: vstack-initial depth twin diverged (start_pc {start_pc}, py {first_pypc})"
                );
            }
            twin
        } else {
            raw_depth()
        };
        (first_pypc, d, sym.nlocals())
    };
    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    let mut boxes = Vec::with_capacity(depth);
    for s in 0..depth {
        // Read the operand-stack slot `s` from the virtualizable shadow.
        // A missing / NONE slot means the entry stack box is not
        // reconstructible here — decline the whole mirror for this walk.
        match ctx.trace_ctx.virtualizable_box_at(nvs + nlocals + s) {
            Some(b) if b != OpRef::NONE => boxes.push(b),
            _ => return,
        }
    }
    ctx.frame_state.borrow_mut().vstack_boxes = boxes;
    ctx.vstack_depth = depth;
    ctx.vstack_cur_pypc = first_pypc;
    ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
    ctx.vstack_handler_landing_py = None;
    ctx.vstack_valid = true;
}

/// Python-opcode coordinate of the handler a catch target enters.
///
/// The obvious source — the floor segment of the catch target's JitCode offset
/// — is not reliable here.  A catch target can be an out-of-line block that
/// only performs the unwind bookkeeping and then jumps to the handler proper,
/// and such a block carries no py pivot of its own, so the floor answers with
/// whatever segment it happened to be laid out inside.  Observed: a catch
/// target near the end of the JitCode floored onto the `RERAISE` that ENDS the
/// handler (py 53) while the walk went on to report the handler's
/// `PUSH_EXC_INFO` entry (py 35), which armed the reorder region across the
/// whole handler body.
///
/// `co_exceptiontable` states the same edge exactly, at the Python level the
/// mirror models: the unwind target of the entry covering the raising opcode.
/// Fall back to the floor when no entry covers it — the JitCode-level catch
/// then belongs to a construct the table does not describe.
fn handler_entry_py_pc(
    code_ptr: *const pyre_interpreter::CodeObject,
    raising_py_pc: u32,
    floor_py: u32,
) -> u32 {
    // SAFETY: the caller resolved `code_ptr` from a live jitcode payload.
    let code = unsafe { &*code_ptr };
    crate::liveness::exception_target_pc(code, raising_py_pc as usize)
        .and_then(|py| u32::try_from(py).ok())
        .unwrap_or(floor_py)
}

/// Static-liveness operand-stack depth on entry to the Python opcode at `py`.
fn py_stack_depth(code_ptr: *const pyre_interpreter::CodeObject, py: u32) -> usize {
    crate::liveness::liveness_for(code_ptr)
        .depth_at_py_pc()
        .get(py as usize)
        .copied()
        .unwrap_or(0) as usize
}

/// Drop any armed out-of-order region.
///
/// The region's snapshot is keyed on the `(py_pc, depth)` coordinate the walk
/// left, and it is restored when the walk returns to that same coordinate on
/// the premise that no Python opcode retired in between.  A caller that
/// rewrites `vstack_boxes` wholesale and moves the coordinate — an exception
/// handler entry, a call-assembler return — breaks that premise: the stack the
/// snapshot describes no longer exists, so a later boundary that happens to
/// land on the same `(py_pc, depth)` would restore it over the live mirror.
pub(crate) fn disarm_vstack_reorder_region<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) {
    if keep_reorder_region_across_reseed() {
        return;
    }
    ctx.vstack_reorder_ceiling = u32::MAX;
    ctx.frame_state.borrow_mut().vstack_reorder_saved = None;
}

/// `PYRE_VSTACK_KEEP_REORDER`: leave an armed region in place across a mirror
/// re-seed. The escape hatch for A/B-ing the disarm against the behaviour it
/// replaces.
fn keep_reorder_region_across_reseed() -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var_os("PYRE_VSTACK_KEEP_REORDER").is_some())
}

/// The handler-entry coordinate the mirror adopted, under `PYRE_VSTACK_DIAG`.
/// `floor_py` is reported next to it: the two disagreeing is what an
/// out-of-line catch target looks like from the walk's side, and without both
/// numbers a spurious reorder region downstream cannot be attributed.
fn vstack_handler_diag(
    arm: &str,
    handler_jit_pc: usize,
    from_pypc: u32,
    handler_py: u32,
    floor_py: u32,
    handler_depth: usize,
) {
    if vstack_diag_enabled() {
        eprintln!(
            "[vstack-handler] {arm} handler_jit_pc={handler_jit_pc} from_pypc={from_pypc} \
             handler_py={handler_py} floor_py={floor_py} handler_depth={handler_depth}"
        );
    }
}

/// #370: model the exception-unwind boundary on the operand-stack mirror.
/// When a raised exception is caught by THIS frame's handler (the SubRaise
/// catch in the dispatch loop), the unwinder truncates the operand stack to
/// the handler's setup depth and pushes the exception value.
/// [`reconcile_vstack_at_boundary`] cannot model this NON-SEQUENTIAL
/// transition — it explains a depth change via the previous opcode's normal
/// stack effect — so without this hook the mirror latches `vstack_valid =
/// false` at handler entry and every kept-stack guard inside the handler
/// declines.  Re-seed the mirror at the handler-entry coordinate instead:
/// place the caught `exc` box on the new TOS and source the surviving slots
/// below it from the virtualizable shadow (the unwind only truncates ABOVE
/// the handler depth, so those slots are unchanged at the raise point).
/// Subsequent in-handler exception opcodes reconcile via
/// [`VstackOpClass::ShadowReseed`], re-reading the shadow the lowering keeps
/// current.  A survivor slot the shadow cannot source stays a NONE hole and
/// the guard's `mirror_covers_kept` declines for it (safe fallback).
///
/// `handler_jit_pc` is the catch target: an OUTER full-body jitcode pc, or a
/// CALLEE coordinate when the catch happens inside an inline sub-walk.  No-op
/// when the walk has no jitcode to map the pc through.
pub(crate) fn vstack_enter_exception_handler<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    handler_jit_pc: usize,
    exc: OpRef,
) {
    if ctx.fbw_mode.inline_subwalk {
        vstack_enter_exception_handler_callee(ctx, handler_jit_pc, exc);
        return;
    }
    // The handler-entry operand stack is a FRESH reconstruction from the
    // authoritative virtualizable shadow plus the caught `exc` — it does NOT
    // depend on the pre-raise mirror being valid (the unwind discards the
    // operand stack above the handler depth, and the surviving slots below
    // are read from the shadow).  So REVIVE the mirror here even when the
    // pre-raise walk invalidated it (e.g. at a `LOAD_GLOBAL` NULL-sentinel on
    // the `raise` expression).  Only a missing full-body sym is unrecoverable.
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return;
    }
    // SAFETY: pointer live for the full-body walk; read-only layout fields.
    let sym = unsafe { &*full_body_sym };
    if sym.jitcode().is_null() {
        ctx.vstack_valid = false;
        return;
    }
    let (floor_py, code_ptr, twin_depth, twin_populated) = unsafe {
        let jc = &*sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            ctx.vstack_valid = false;
            return;
        }
        (
            vstack_containing_py_pc(&jc.payload.metadata, handler_jit_pc),
            jc.payload.code_ptr,
            jc.payload.depth_containing_for_jitcode_pc(handler_jit_pc),
            jc.payload.depth_containing_populated(),
        )
    };
    let handler_py = handler_entry_py_pc(code_ptr, ctx.vstack_cur_pypc, floor_py);
    // The twin is a py_pc-keyed static-liveness read behind a JitCode offset,
    // so it can only answer for the floor segment; on the exception-table
    // coordinate it would report the landing block's segment depth.  Read the
    // handler's own py depth there, and on the unpopulated-twin fallback
    // (skeleton / fixture).
    let handler_depth = if twin_populated && handler_py == floor_py {
        let depth = twin_depth.unwrap_or(0) as usize;
        if pcmap_containing_audit_enabled() {
            assert_eq!(
                depth,
                py_stack_depth(code_ptr, floor_py),
                "PYRE_PCMAP_CONTAINING_AUDIT: enter-handler containing-depth twin diverged at jit_pc {handler_jit_pc} (py {floor_py})"
            );
        }
        depth
    } else {
        py_stack_depth(code_ptr, handler_py)
    };
    ctx.frame_state.borrow_mut().vstack_boxes.clear();
    ctx.frame_state
        .borrow_mut()
        .vstack_boxes
        .resize(handler_depth, OpRef::NONE);
    disarm_vstack_reorder_region(ctx);
    // The unwinder pushes the caught exception onto the new TOS.
    if handler_depth >= 1 && exc != OpRef::NONE {
        ctx.frame_state.borrow_mut().vstack_boxes[handler_depth - 1] = exc;
    }
    vstack_handler_diag(
        "outer",
        handler_jit_pc,
        ctx.vstack_cur_pypc,
        handler_py,
        floor_py,
        handler_depth,
    );
    ctx.vstack_cur_pypc = handler_py;
    ctx.vstack_depth = handler_depth;
    ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
    ctx.vstack_handler_landing_py = (floor_py != handler_py).then_some(floor_py);
    // Revive: the handler-entry state is shadow-sourced, independent of the
    // pre-raise mirror.
    ctx.vstack_valid = true;
    // Fill the surviving slots below the pushed exc from the shadow; reseed
    // skips the already-set exc slot (non-NONE).  Leaves un-sourceable slots
    // NONE (per-slot decline) rather than latching the whole mirror invalid.
    let _ = reseed_vstack_from_shadow(ctx, handler_depth);
}

/// Handler-entry re-seed for a catch that happens INSIDE an inline sub-walk.
///
/// Two things differ from the full-body path, and both follow from the sub-walk
/// carrying a CALLEE-local mirror ([`seed_callee_vstack_mirror`]):
///
/// * The catch target is a callee coordinate, so the handler py_pc and depth
///   come from the active resume frame's own metadata rather than the outer
///   full-body tables.
/// * The virtualizable shadow belongs to the OUTER frame and cannot name a
///   callee operand slot, so the surviving slots below the pushed exception are
///   sourced from the callee-local mirror itself — the same premise that makes
///   `reconcile_vstack_at_boundary` skip the shadow hole-fill in a sub-walk.
///   The unwind only truncates ABOVE the handler depth, so those slots still
///   hold what the sub-walk tracked at the raise point.
///
/// Without a shadow to fall back on there is no way to REVIVE an already-invalid
/// mirror here, so an invalidated sub-walk stays invalid.
fn vstack_enter_exception_handler_callee<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    handler_jit_pc: usize,
    exc: OpRef,
) {
    if !ctx.vstack_valid {
        ctx.vstack_valid = false;
        return;
    }
    let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) else {
        ctx.vstack_valid = false;
        return;
    };
    let Some((floor_py, code_ptr, floor_depth)) =
        frame.vstack_coordinate_for_jitcode_pc(handler_jit_pc)
    else {
        ctx.vstack_valid = false;
        return;
    };
    // Same out-of-line catch target as on the full-body arm, resolved against
    // the callee's own code object.
    let handler_py = handler_entry_py_pc(code_ptr, ctx.vstack_cur_pypc, floor_py);
    let handler_depth = if handler_py == floor_py {
        floor_depth
    } else {
        py_stack_depth(code_ptr, handler_py)
    };
    // Truncate to the handler's setup depth, keeping the tracked survivors; a
    // mirror shallower than the handler depth pads with NONE holes, which
    // `mirror_covers_kept` declines per slot rather than latching invalid.
    ctx.frame_state
        .borrow_mut()
        .vstack_boxes
        .resize(handler_depth, OpRef::NONE);
    disarm_vstack_reorder_region(ctx);
    // The unwinder pushes the caught exception onto the new TOS.
    if handler_depth >= 1 && exc != OpRef::NONE {
        ctx.frame_state.borrow_mut().vstack_boxes[handler_depth - 1] = exc;
    }
    vstack_handler_diag(
        "callee",
        handler_jit_pc,
        ctx.vstack_cur_pypc,
        handler_py,
        floor_py,
        handler_depth,
    );
    ctx.vstack_cur_pypc = handler_py;
    ctx.vstack_depth = handler_depth;
    ctx.frame_state.borrow_mut().vstack_last_ref = OpRef::NONE;
    ctx.vstack_handler_landing_py = (floor_py != handler_py).then_some(floor_py);
}
