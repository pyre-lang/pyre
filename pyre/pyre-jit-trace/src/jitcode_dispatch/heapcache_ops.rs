//! Heapcache-aware field and array-item recording.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s field / array
//! `_opimpl_*` consulting `heapcache.py` (implementation mirror
//! `majit-metainterp/heapcache.rs`).
//!
//! getfield / setfield / getarrayitem / setarrayitem recorded through
//! the heapcache (`pyjitpl.py` `_opimpl_*field*` / `_do_*arrayitem_gc`):
//! a cache hit returns the cached OpRef without emitting IR; a miss
//! records the op and writes the result back into the cache.

use super::*;

/// The live pointer a ref operand carries, or `None` where the walk holds no
/// executable one.
///
/// RPython's MIFrame register contains the FrontendOp itself, whose `.value`
/// is the concrete pointer `executor.do_getfield_gc_*` reads.  Pyre also has a
/// typed register shadow, because some canonical and inlined helper arguments
/// are OpRefs allocated outside the active recorder and cannot be stamped
/// there; that shadow holds the same MIFrame box value, so it answers after
/// the ordinary OpRef carrier rather than instead of it.
///
/// A null and an all-ones word are both rejected: null is no object, and
/// all-ones is the `vable_setfield` storage placeholder, which says "no
/// concrete known" rather than naming one.
fn concrete_ref_operand_ptr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    obj: OpRef,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<i64> {
    ctx.trace_ctx
        .box_value(obj)
        .and_then(|value| match value {
            majit_ir::Value::Ref(reference) => Some(reference.0 as i64),
            _ => None,
        })
        .or_else(
            || match read_ref_reg_concrete(code, op, operand_offset, ctx) {
                ConcreteValue::Ref(reference) => Some(reference as usize as i64),
                _ => None,
            },
        )
        .filter(|&ptr| ptr != 0 && ptr != usize::MAX as i64)
}

/// The live value of a store's value operand, or `None` where the walk holds
/// no executable one.  Same carrier order as [`concrete_ref_operand_ptr`]: the
/// box's own value, then the typed register shadow.
fn concrete_store_value_operand<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    value_bank: char,
    valuebox: OpRef,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<majit_ir::Value> {
    let known = |value: &majit_ir::Value| match value {
        majit_ir::Value::Ref(r) => *r != majit_ir::GcRef::NO_CONCRETE && r.0 != usize::MAX,
        majit_ir::Value::Void => false,
        _ => true,
    };
    valuebox
        .inline_const_to_value()
        .or_else(|| ctx.trace_ctx.box_value(valuebox))
        .filter(known)
        .or_else(|| {
            let shadow = match value_bank {
                'i' => read_int_reg_concrete(code, op, operand_offset, ctx),
                'r' => read_ref_reg_concrete(code, op, operand_offset, ctx),
                'f' => read_float_reg_concrete(code, op, operand_offset, ctx),
                _ => ConcreteValue::Null,
            };
            match shadow {
                ConcreteValue::Int(v) => Some(majit_ir::Value::Int(v)),
                ConcreteValue::Bool(v) => Some(majit_ir::Value::Int(i64::from(v))),
                ConcreteValue::Ref(r) => Some(majit_ir::Value::Ref(majit_ir::GcRef(r as usize))),
                ConcreteValue::Float(v) => Some(majit_ir::Value::Float(v)),
                ConcreteValue::Null => None,
            }
            .filter(known)
        })
}

/// The executing half of `execute_and_record` (`pyjitpl.py`) for a
/// `SETFIELD_GC` / `SETARRAYITEM_GC` into an object this walk did not
/// allocate: `executor.execute` -> `cpu.bh_setfield_gc_*` /
/// `bh_setarrayitem_gc_*` performs the store while the op is recorded, so
/// the heap the rest of the walk executes against — the residual calls it
/// runs, the fields it reads back — is the one the trace describes.
///
/// `obj_ptr + offset` is the `size`-byte word, `ty` its bank, `before` what
/// it holds.  The displaced word is journaled ([`fbw_gc_store_journal_push`])
/// because a walk that does not commit hands its region to a replay that
/// re-executes it against the pre-walk heap.  A store the walk cannot execute
/// — no live receiver or value, or a word the value's bank cannot fill — is
/// one only that replay applies, which is what
/// [`fbw_mark_unjournaled_effect`] records.
#[allow(clippy::too_many_arguments)]
fn walker_execute_gc_store<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj_ptr: Option<i64>,
    offset: usize,
    size: usize,
    ty: majit_ir::Type,
    before: Option<majit_ir::Value>,
    value: Option<majit_ir::Value>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !ctx.trace_ctx.has_cpu() {
        return Ok(());
    }
    // One word holds either bank: a pointer the jitcode carries in an `i`
    // register is stored into a ref field as the same bits, and back.
    let value = value.and_then(|value| match (ty, value) {
        (majit_ir::Type::Int, majit_ir::Value::Ref(r)) => Some(majit_ir::Value::Int(r.0 as i64)),
        (majit_ir::Type::Ref, majit_ir::Value::Int(v)) => {
            Some(majit_ir::Value::Ref(majit_ir::GcRef(v as usize)))
        }
        (majit_ir::Type::Int, majit_ir::Value::Int(_))
        | (majit_ir::Type::Ref, majit_ir::Value::Ref(_))
        | (majit_ir::Type::Float, majit_ir::Value::Float(_)) => Some(value),
        _ => None,
    });
    let (Some(obj_ptr), Some(before), Some(value)) = (obj_ptr, before, value) else {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-gc-store] not executed pc={pc} obj={obj_ptr:?} offset={offset} before={before:?} value={value:?}"
            );
        }
        return walker_gc_store_not_executed(ctx, pc);
    };
    let obj = obj_ptr as usize as pyre_object::PyObjectRef;
    let managed = pyre_object::gc_hook::try_gc_owns_object(obj as pyre_object::gc_hook::GCREF);
    // SAFETY: `obj_ptr` is the live receiver the walk is executing over and
    // `offset` / `size` come from the op's own descr.
    if unsafe { fbw_gc_store_word(obj, offset, size, value, managed) } {
        ctx.trace_ctx.set_cut_observer(fbw_gc_store_journal_cut);
        // `execute_and_record` executes, then records: the op this store
        // belongs to is the next one, and a cut back past it undoes the store.
        let op_count = ctx.trace_ctx.get_trace_position()._count + 1;
        fbw_gc_store_journal_push(obj, offset, size, before, op_count);
        Ok(())
    } else {
        walker_gc_store_not_executed(ctx, pc)
    }
}

/// A recorded store — or a recorded `COND_CALL` whose condition held — the
/// walk could not execute.
///
/// Inside a helper descent the body goes on to read what it wrote, so the
/// walk does not continue past the lost write (the `setfield_raw_i` rule):
/// the descent declines, the cut undoes the stores it did execute
/// ([`fbw_gc_store_journal_cut`]) and the call runs as a residual.  The
/// top-level walk has no call to hand the region to; there the store is one
/// only the replay applies.
pub(crate) fn walker_gc_store_not_executed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic: 0 });
    }
    fbw_mark_unjournaled_effect(ResidualDecline::Symbolic);
    Ok(())
}

/// Live `0 <= index < len`.  A Ref bit-pattern in an Int register is
/// almost never a valid index; on wasm32 it still fits `i32`, so a
/// magnitude check is not enough.
fn index_in_array_bounds<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    array_ptr: i64,
    index_value: i64,
    descr: &majit_ir::DescrRef,
) -> bool {
    if index_value < 0 {
        return false;
    }
    match ctx.trace_ctx.arraylen_sanity_load(array_ptr, descr) {
        Some(majit_ir::Value::Int(len)) => index_value < len,
        _ => false,
    }
}

/// Constant index into an array that carries no length word.
///
/// `index_in_array_bounds` reads `arraylen_sanity_load`, which declines a
/// descr with no `lendescr` because `bh_arraylen_gc` has no offset to read.
/// A fat pointer's data (`Box<[T]>`) is that shape: the length is the
/// metadata word, and the bytes themselves have no header. The bounds
/// proof then fails closed, `GETARRAYITEM_GC` is recorded with no value,
/// and `opimpl_goto_if_not` has no int.
///
/// The length check exists to refuse a Ref bit-pattern in an Int register
/// (`bh_getarrayitem_gc_r` SIGBUS, `test.test_dict` `items ^ items`). A
/// constant index is not that bit-pattern, so `execute_with_descr` still
/// runs the load.
fn headerless_const_index(index: OpRef, index_value: i64, descr: &majit_ir::DescrRef) -> bool {
    index_value >= 0
        && index.is_constant()
        && descr
            .as_array_descr()
            .is_some_and(|array| array.len_descr().is_none())
}

/// `getarrayitem_gc_<i|r|f>/rid>X` handler. Operand layout `rid>X`:
/// 1B r-reg(array) + 1B i-reg(index) + 2B descr + 1B X-dst.
///
/// RPython parity: `pyjitpl.py _do_getarrayitem_gc_any`:
///
///   tobox = heapcache.getarrayitem(arraybox, indexbox, arraydescr)
///   if tobox: return tobox        # cache hit, no IR (recording-only)
///   resop = self.execute_with_descr(op, arraydescr, arraybox, indexbox)
///   heapcache.getarrayitem_now_known(arraybox, indexbox, resop, arraydescr)
///   return resop
///
/// `opcode` is one of `GetarrayitemGc{I,R,F}`; `dst_bank` selects the
/// result bank (`'i'`/`'r'`/`'f'`) the walker writes back into.  The
/// index operand is always int-classified, so it is decoded from the
/// `i` register bank.
pub(crate) fn getarrayitem_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    // `_opimpl_getarrayitem_gc_pure_any`: directly-constant array and
    // index operands bypass the heapcache completely — execute the load
    // now and substitute a `Const`, with no profiler counts (the bypass
    // calls `executor.execute`, not `execute_and_record`).
    let is_pure = matches!(
        opcode,
        OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemGcPureF
    );
    if is_pure && array.is_constant() && index.is_constant() {
        let load_type = match opcode {
            OpCode::GetarrayitemGcPureI => majit_ir::Type::Int,
            OpCode::GetarrayitemGcPureR => majit_ir::Type::Ref,
            _ => majit_ir::Type::Float,
        };
        if let (Some(majit_ir::Value::Ref(array_ref)), Some(majit_ir::Value::Int(index_value))) = (
            ctx.trace_ctx.box_value(array),
            ctx.trace_ctx.box_value(index),
        ) {
            let array_ptr = array_ref.0 as i64;
            if array_ptr != 0
                && array_ptr != usize::MAX as i64
                && (index_in_array_bounds(ctx, array_ptr, index_value, &descr)
                    || headerless_const_index(index, index_value, &descr))
            {
                let folded =
                    match ctx
                        .trace_ctx
                        .array_sanity_load(array_ptr, index_value, &descr, load_type)
                    {
                        Some(majit_ir::Value::Int(n)) => Some(ctx.trace_ctx.const_int(n)),
                        Some(majit_ir::Value::Ref(r)) => Some(ctx.trace_ctx.const_ref(r.0 as i64)),
                        Some(majit_ir::Value::Float(f)) => {
                            Some(ctx.trace_ctx.const_float(f.to_bits() as i64))
                        }
                        Some(majit_ir::Value::Void) | None => None,
                    };
                if let Some(folded) = folded {
                    let dst = code[op.pc + 5] as usize;
                    let concrete = concrete_from_recorded_opref(ctx, folded);
                    match dst_bank {
                        'i' => write_int_reg(ctx, op.pc, dst, folded, concrete)?,
                        'r' => write_ref_reg(ctx, op.pc, dst, folded, concrete)?,
                        _ => {
                            let len = ctx.registers_f.len();
                            let _ = ctx.registers_f.get(dst).ok_or(
                                DispatchError::RegisterOutOfRange {
                                    pc: op.pc,
                                    reg: dst,
                                    len,
                                    bank: "f",
                                },
                            )?;
                            ctx.registers_f.set(dst, folded);
                        }
                    }
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    let result = if let Some(cached) =
        ctx.trace_ctx
            .heapcache_getarrayitem(array, index, descr_index)
    {
        // pyjitpl.py `_do_getarrayitem_gc_any` cache hit:
        //   tobox = heapcache.getarrayitem(...)
        //   if tobox:
        //       profiler.count_ops(rop.GETARRAYITEM_GC_I, HEAPCACHED_OPS)
        //       return tobox
        // RPython hardcodes `GETARRAYITEM_GC_I` regardless of the
        // recorded `typ` ('i' / 'r' / 'f'); pyre matches the hardcode
        // for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetarrayitemGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        ctx.trace_ctx
            .profiler()
            .count_ops(opcode, majit_metainterp::counters::OPS);
        ctx.trace_ctx
            .profiler()
            .count_ops(opcode, majit_metainterp::counters::RECORDED_OPS);
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[array, index], descr.clone());
        // Box.value parity: `box_value` exposes the resolution chain
        // PyPy reads off `arraybox.getref_base()` / `indexbox.getint()`
        // (`rpython/jit/metainterp/executor.py`).  Any operand
        // whose Box.value is known unblocks `array_sanity_load`, not
        // just Const-pool entries (`pyjitpl.py resbox =
        // execute_with_descr(...); getarrayitem_now_known(...)`
        // parity).
        let load_type = match opcode {
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        // `executor.py do_getarrayitem_gc_*` reads `arraybox.getref_base()`
        // and `indexbox.getint()`. A bridge or inlined helper can hold those
        // on the MIFrame register shadow when the OpRef itself was not
        // stamped in this recorder (`concrete_ref_operand_ptr`).
        let array_ptr = concrete_ref_operand_ptr(code, op, 0, array, ctx);
        let index_value = match ctx.trace_ctx.box_value(index) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => match read_int_reg_concrete(code, op, 1, ctx) {
                ConcreteValue::Int(n) => Some(n),
                _ => None,
            },
        };
        let live_value = if let (Some(ty), Some(array_ptr), Some(index_value)) =
            (load_type, array_ptr, index_value)
        {
            // A helper walk can put a Ref bit-pattern in an Int index
            // register.  `bh_getarrayitem_gc_r` then SIGBUS
            // (`test.test_dict` `items ^ items`).  On wasm32 that
            // bit-pattern still fits `i32`, so prove `0 <= index < len`
            // instead of a magnitude heuristic. A headerless array has no
            // length word to prove against; `headerless_const_index`
            // still allows the constant-index load `execute_with_descr`
            // would have run.
            if index_in_array_bounds(ctx, array_ptr, index_value, &descr)
                || headerless_const_index(index, index_value, &descr)
            {
                ctx.trace_ctx
                    .array_sanity_load(array_ptr, index_value, &descr, ty)
            } else {
                None
            }
        } else {
            None
        };
        // Stamp the loaded value as Box.value of the recorded result
        // (RPython `Box(value)` constructor analog) so subsequent
        // consumers see the runtime concrete instead of the
        // GcRef(usize::MAX) sentinel.
        if let Some(live_value) = live_value {
            ctx.trace_ctx.set_opref_concrete(resbox, live_value);
        }
        ctx.trace_ctx
            .heapcache_getarrayitem_now_known(array, index, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 5] as usize;
    // concrete_of_opref derivation: derive shadow concrete from the recorded result's
    // `concrete_of_opref` entry instead of inventing Null.  Constant
    // arraybox + constant index hits land in `constants.get_value`;
    // virtualizable hits surface via `standard_virtualizable_box`;
    // `set_opref_concrete` stamps from upstream `binop_int_record`
    // flow back here too.  Null fallback preserves the prior contract.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setarrayitem_gc_<i|r|f>/ri{i,r,f}d` handler. Operand layout per
/// `bhimpl_setarrayitem_gc_{i,r,f}(cpu, array, index, newvalue,
/// arraydescr)` (`blackhole.py`):
/// 1B r-reg(array) + 1B i-reg(index) + 1B {i,r,f}-reg(newvalue) + 2B descr.
///
/// RPython parity: `pyjitpl.py _opimpl_setarrayitem_gc_any`
/// dispatches through `metainterp.execute_setarrayitem_gc(arraydescr,
/// arraybox, indexbox, itembox)` — RPython's wrapper records
/// `rop.SETARRAYITEM_GC` and updates the heapcache via
/// `setarrayitem`.
///
/// No skip-on-redundant short-circuit (matches RPython —
/// `_opimpl_setarrayitem_gc_any` has no `if cached == value: return`,
/// because `heapcache.setarrayitem` already handles aliasing
/// invalidation at the right granularity).
///
/// `value_bank` selects the newvalue register source: `'i'` /
/// `'r'` / `'f'`.
pub(crate) fn setarrayitem_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    // Operand layout `<r><i><v>d`: r-reg(array) + i-reg(index) + v(value)
    // + 2B descr-index.  For the `c`-coded short form
    // (`setarrayitem_gc_i/ricd`) the value byte is an inline signed
    // constant (`signedord`, `blackhole.py`) read as a `ConstInt`
    // box instead of an `i`-register slot, mirroring `setfield_gc_i/rcd`.
    let value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        'c' => OpRef::ConstInt(code[op.pc + 3] as i8 as i64),
        _ => unreachable!("value_bank must be 'i', 'r', 'f' or 'c'"),
    };
    let descr = read_descr(code, op, 3, ctx)?;
    let descr_index = descr.index();

    // `execute_setarrayitem_gc`: `execute_and_record` runs the store, then
    // records it, and `heapcache.setarrayitem` comes last.
    walker_execute_setarrayitem_gc(code, op, ctx, value_bank, array, index, value, &descr)?;
    ctx.trace_ctx
        .profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.trace_ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[array, index, value],
        descr.clone(),
    );
    // `upd.setarrayitem(valuebox)` (heapcache.py) parity — the
    // cache stores the Box identity (`value` OpRef); cache-hit
    // readers fetch the intrinsic value via `box_value(cached)` at
    // hit time.
    ctx.trace_ctx
        .heapcache_setarrayitem(array, index, descr_index, value);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `execute_setarrayitem_gc`'s store (`pyjitpl.py`) into an array this walk
/// did not allocate; [`walker_fill_materialized_array`] is the same store for
/// one it did.  Operand layout as [`setarrayitem_gc_via_heapcache`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn walker_execute_setarrayitem_gc<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
    array: OpRef,
    index: OpRef,
    value: OpRef,
    descr: &DescrRef,
) -> Result<(), DispatchError> {
    let Some(ad) = descr.as_array_descr() else {
        return Ok(());
    };
    // A block the walk allocated is held by nothing outside it, so its
    // stores need no undo entry.  The ref-items fill has its own path;
    // a `Signed` / `Float` block takes the typed store below.
    let fresh = ctx.trace_ctx.heap_cache().saw_allocation(array)
        || matches!(
            ctx.trace_ctx.opcode_of(array),
            Some(OpCode::NewArray | OpCode::NewArrayClear)
        );
    if fresh && ad.is_array_of_pointers() && !ad.is_array_of_structs() {
        walker_fill_materialized_array(ctx, array, index, value);
        return Ok(());
    }
    if array.is_constant() {
        return Ok(());
    }
    let (base_size, item_size, ty) = (ad.base_size(), ad.item_size(), ad.item_type());
    let array_ptr = concrete_ref_operand_ptr(code, op, 0, array, ctx);
    let index_value = index
        .inline_const_to_value()
        .or_else(|| ctx.trace_ctx.box_value(index))
        .and_then(|v| match v {
            majit_ir::Value::Int(i) => Some(i),
            _ => None,
        })
        .or_else(|| {
            if index.is_constant() {
                return None;
            }
            match read_int_reg_concrete(code, op, 1, ctx) {
                ConcreteValue::Int(i) => Some(i),
                _ => None,
            }
        });
    // The bounds proof `getarrayitem_gc_via_heapcache` asks before its load:
    // a Ref bit-pattern in the index register must not become a store.
    // An array with no length word has no bound to read; what is left to
    // refuse there is the bit-pattern itself, which is no index.
    let headerless_index = |i: i64| {
        ad.len_descr().is_none()
            && if cfg!(target_pointer_width = "64") {
                (0..=i64::from(u32::MAX)).contains(&i)
            } else {
                majit_ir::ptr_info::reasonable_array_index(i)
            }
    };
    let slot = array_ptr.zip(index_value).filter(|&(ptr, i)| {
        index_in_array_bounds(ctx, ptr, i, descr)
            || headerless_const_index(index, i, descr)
            || headerless_index(i)
    });
    if slot.is_none() && fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-gc-store] no slot pc={} array={array:?} array_op={:?} array_ptr={array_ptr:?} index={index:?} index_value={index_value:?} len={:?} has_len_descr={} base_size={base_size} item_size={item_size} ty={ty:?} fresh={fresh}",
            op.pc,
            ctx.trace_ctx.opcode_of(array),
            array_ptr.and_then(|ptr| ctx.trace_ctx.arraylen_sanity_load(ptr, descr)),
            ad.len_descr().is_some(),
        );
    }
    let before = slot.and_then(|(ptr, i)| ctx.trace_ctx.array_sanity_load(ptr, i, descr, ty));
    let offset = slot.map_or(0, |(_, i)| base_size + i as usize * item_size);
    let value = concrete_store_value_operand(code, op, 2, value_bank, value, ctx);
    if fresh {
        // `walker_fill_materialized_array`'s rule for a store it cannot
        // execute: the block is incomplete, so stop naming it as the
        // array's value.
        let stored = match (slot, value) {
            (Some((ptr, _)), Some(value)) if ctx.trace_ctx.has_cpu() => {
                let block = ptr as usize as pyre_object::PyObjectRef;
                // SAFETY: `slot` passed the bounds proof against the live block.
                unsafe { fbw_gc_store_word(block, offset, item_size, value, false) }
            }
            _ => false,
        };
        if !stored {
            ctx.trace_ctx
                .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
        }
        return Ok(());
    }
    walker_execute_gc_store(
        ctx,
        slot.map(|(ptr, _)| ptr),
        offset,
        item_size,
        ty,
        before,
        value,
        op.pc,
    )
}

/// Walker virtual-force fill — companion to the `NEW_ARRAY_CLEAR`
/// materialization in the `new_array_clear` handler (module-global
/// fresh-container off-by-one fix). When `array` is a block the walker
/// materialized for a `new_array` / `new_array_clear` recorded in this
/// trace, write the concrete element into it so a later BUILD_LIST /
/// BUILD_TUPLE residual reads a complete block during the walk. If the
/// element value (not a ref) or the index has no known concrete, the
/// block cannot be completed, so revert the array to the no-concrete
/// sentinel (`Ref(usize::MAX)`): the residual then declines and the void
/// store aborts, exactly as without materialization.
///
/// A real heap array store — whose `array` operand is a `GetfieldGcR`
/// load of a pre-existing container's items block — is left untouched,
/// so the interpreter's own SETARRAYITEM is never duplicated eagerly
/// here. The reload of a block from a container this walk allocated is
/// the walk's own object and is filled like a recorded NewArray.
pub(crate) fn walker_fill_materialized_array<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    // pyjitpl.py execute_setarrayitem_gc always stores into the object
    // execute_new_array_clear / execute_new_array allocated. The walker
    // stamps that block as the recorded NewArray / NewArrayClear's
    // concrete value, so a write here is the walk executing the store —
    // whether or not heapcache.new_array marked the box unescaped
    // (Const length only). Any other array is a real heap object the
    // interpreter mutates.
    if array.is_constant() {
        return;
    }
    match ctx.trace_ctx.opcode_of(array) {
        Some(OpCode::NewArray | OpCode::NewArrayClear) => {}
        // The block a container this walk allocated points at now. A
        // `conditional_call` the walk executed (`_ll_list_resize_ge` ->
        // `_ll_list_resize_hint_really`) replaces the recorded NewArray
        // with a block the callee allocated, and the reload of `l.items`
        // is the only box naming it. `_opimpl_setfield_gc_any` already
        // executes `l.length = newsize` on that container; without the
        // `ll_setitem_fast` store the list holds a null slot.
        Some(OpCode::GetfieldGcR) => {
            let owned = ctx.is_authoritative_executor
                && ctx
                    .trace_ctx
                    .ref_getfield_gc_r(array)
                    .is_some_and(|(_, obj)| ctx.trace_ctx.heap_cache().saw_allocation(obj));
            if !owned {
                return;
            }
        }
        _ => return,
    }
    let block = match ctx.trace_ctx.box_value(array) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE && r.as_usize() != 0 => {
            r.as_usize() as *mut pyre_object::object_array::ItemsBlock
        }
        // No stamped concrete → not materialized (gate off / non-ref / non-const).
        _ => return,
    };
    // Confirm it is one of our GC-managed materialization blocks.
    if !pyre_object::gc_hook::try_gc_owns_object(block as pyre_object::gc_hook::GCREF) {
        return;
    }
    // `new_array_clear` of `GcArray(Signed)` stamps a `TypedItemsBlock`.
    // A later `setarrayitem_gc_i` has to write the traced word into that
    // block: reverting the pointer (the ref-element arm below) would make
    // the next `getarrayitem_gc_i` unreadable, and `opimpl_goto_if_not_int_is_zero`
    // would abort on `index == FREE`.
    let int_tid = pyre_object::gc_int_array_gc_type_id();
    // SAFETY: `try_gc_owns_object` just accepted `block`, so a `GcHeader`
    // sits immediately in front of the payload.
    let tid = unsafe { (*majit_gc::header::header_of(block as usize)).type_id() };
    if tid == int_tid && int_tid != majit_rlib::lltypesystem::rlist::UNSET_GC_TYPE_ID {
        fill_materialized_signed_gcarray(ctx, array, block, index, value);
        return;
    }
    let idx = match ctx.trace_ctx.box_value(index) {
        Some(majit_ir::Value::Int(i)) if i >= 0 => i as usize,
        _ => {
            ctx.trace_ctx
                .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
            return;
        }
    };
    let cap = unsafe { pyre_object::object_array::items_block_capacity(block) };
    if idx >= cap {
        ctx.trace_ctx
            .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
        return;
    }
    let elem = match ctx.trace_ctx.box_value(value) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            r.as_usize() as pyre_object::PyObjectRef
        }
        _ => {
            ctx.trace_ctx
                .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
            return;
        }
    };
    unsafe {
        let base = pyre_object::object_array::items_block_items_base(block);
        *base.add(idx) = elem;
    }
    // Old→young barrier: the materialization block may have been promoted to
    // old-gen while `elem` is still young (the construction-barrier gap). A
    // nursery block carries no TRACK_YOUNG_PTRS so the barrier is a no-op.
    pyre_object::gc_hook::try_gc_write_barrier(block as pyre_object::gc_hook::GCREF);
}

/// Traced-iteration fill of a cleared `GcArray(Signed)` block from
/// `new_array_clear`'s `alloc_typed_items_block_nursery` plus
/// `typed_items_block_clear` (`llmodel.py` `bh_new_array_clear`).
/// Scalar words, so there is no write barrier. An index or value with
/// no `Box.value` cannot be replayed; drop the pointer the same way the
/// ref arm does, and the next load declines.
fn fill_materialized_signed_gcarray<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    array: OpRef,
    block: *mut pyre_object::object_array::ItemsBlock,
    index: OpRef,
    value: OpRef,
) {
    let revert = |ctx: &mut WalkContext<'_, '_, Sym>| {
        ctx.trace_ctx
            .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
    };
    let idx = match ctx.trace_ctx.box_value(index) {
        Some(majit_ir::Value::Int(i)) if i >= 0 => i as usize,
        _ => {
            revert(ctx);
            return;
        }
    };
    let block = block as *mut pyre_object::TypedItemsBlock;
    let cap = unsafe { pyre_object::typed_items_block_capacity(block) };
    if idx >= cap {
        revert(ctx);
        return;
    }
    let Some(majit_ir::Value::Int(word)) = ctx.trace_ctx.box_value(value) else {
        revert(ctx);
        return;
    };
    unsafe {
        *pyre_object::typed_items_block_items_base(block)
            .cast::<i64>()
            .add(idx) = word;
    }
}

/// Recording-time `bh_new_array_clear` for a constant-or-known-length
/// array of structs (`ordereddict.malloc_i64_entries` → `new_array_clear`).
///
/// The compiled op stays `NEW_ARRAY_CLEAR`. The pointer is the traced
/// iteration's `Op.value`, zeroed the way `_ll_malloc_entries` zeros a
/// `DICTENTRYARRAY`, so `bh_getinteriorfield_gc_i` of an unwritten slot
/// reads 0. `tid` is `ArrayDescr.tid` (`gc.py` `init_array_descr`).
/// `None` when no collector is installed or `item_size` is 0.
/// An unresolved tid with a collector installed is a bug
/// (`llmodel.py` `bh_new_array_clear` never declines on a descr).
pub(crate) fn materialize_cleared_struct_gcarray(
    cap: usize,
    tid: u32,
    item_size: usize,
    items_base: usize,
) -> Option<*mut u8> {
    if item_size == 0 || !majit_gc::gc_allocator_installed() {
        return None;
    }
    majit_ir::descr::assert_array_tid_for_malloc(tid, "materialize_cleared_struct_gcarray");
    let items_bytes = cap.checked_mul(item_size)?;
    let payload = items_base.checked_add(items_bytes)?;
    let raw = pyre_object::gc_hook::try_gc_alloc_stable(tid, payload)?;
    if raw.is_null() {
        return None;
    }
    // GCREF is Ptr(GcOpaqueType); the byte-addressed clear is
    // llmemory.Address (`*mut u8`). `write_bytes` on GCREFOpaque (ZST)
    // would write zero bytes.
    let bytes = raw.cast::<u8>();
    // The walker only calls this when `base_size` is
    // `TYPED_ITEMS_BLOCK_ITEMS_OFFSET`. Clearing from
    // `GcTypedArray.items` instead leaves the tail dirty on a 32-bit
    // target: that flat offset is 4, and an 8-aligned `Entry` starts at 8.
    unsafe {
        std::ptr::write_bytes(bytes, 0, payload);
        let block = bytes.cast::<pyre_object::GcTypedArray>();
        (*block).len = cap;
    }
    Some(bytes)
}

/// `setinteriorfield`'s type code: 0 = ref, 1 = int, 2 = float.
struct InteriorFieldAccess {
    items_base: usize,
    field_offset: usize,
    field_size: usize,
    item_size: usize,
    field_type: u8,
    signed: bool,
}

fn interior_field_access(descr: &majit_ir::DescrRef) -> Option<InteriorFieldAccess> {
    let ifd = descr.as_interior_field_descr()?;
    let field = ifd.field_descr();
    let array = ifd.array_descr();
    let field_size = field.field_size();
    let item_size = array.item_size();
    // `llmodel.py bh_setinteriorfield_gc_i` adds `arraydescr.basesize`.
    // That equals `GC_TYPED_ARRAY_ITEMS_OFFSET` only when the element
    // needs no padding past the length word. `GcEntries<i64, *mut PyObject>`
    // aligns items to 8, so wasm32 rejects the store if the flat offset
    // is required here.
    if field_size == 0 || item_size == 0 {
        return None;
    }
    let field_type = if field.is_pointer_field() {
        0
    } else if field.is_float_field() {
        2
    } else {
        1
    };
    Some(InteriorFieldAccess {
        items_base: array.base_size(),
        field_offset: field.offset(),
        field_size,
        item_size,
        field_type,
        signed: field.is_field_signed(),
    })
}

/// Box.value of an int operand, else the int-register shadow.
///
/// `from_register` is false for a `c`-argcode operand: that byte is the
/// constant itself, and reading it as a register index aliases another slot.
pub(crate) fn known_int_operand<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    value: OpRef,
    ctx: &WalkContext<'_, '_, Sym>,
    from_register: bool,
) -> Option<i64> {
    if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(value) {
        return Some(n);
    }
    if !from_register {
        return None;
    }
    match read_int_reg_concrete(code, op, operand_offset, ctx) {
        ConcreteValue::Int(n) => Some(n),
        ConcreteValue::Bool(bit) => Some(i64::from(bit)),
        _ => None,
    }
}

/// Payload address of one interior field, or `None` when the pointer is
/// not a collector object or the index does not fit the length word.
///
/// `arraylen_sanity_load` declines an interior descr (it is not an array
/// descr). The length word is the array descr's, at offset 0 on the
/// `GcTypedArray` / `DICTENTRYARRAY` header this materializer allocates.
fn interior_field_addr(
    array_ptr: i64,
    index: i64,
    access: &InteriorFieldAccess,
) -> Option<*mut u8> {
    if array_ptr == 0
        || array_ptr == usize::MAX as i64
        || !majit_ir::ptr_info::reasonable_array_index(index)
    {
        return None;
    }
    let block = array_ptr as usize as *mut u8;
    if !pyre_object::gc_hook::try_gc_owns_object(block as pyre_object::gc_hook::GCREF) {
        return None;
    }
    let len = unsafe { block.cast::<usize>().read_unaligned() };
    let index = index as usize;
    if index >= len {
        return None;
    }
    let byte_offset = index
        .checked_mul(access.item_size)?
        .checked_add(access.field_offset)?;
    let end = byte_offset.checked_add(access.field_size)?;
    let total = len.checked_mul(access.item_size)?;
    if end > total {
        return None;
    }
    let abs = access.items_base.checked_add(byte_offset)?;
    Some(unsafe { block.add(abs) })
}

/// `llmodel.py bh_getinteriorfield_gc_i` `read_int_at_mem`.
fn read_interior_int(addr: *const u8, size: usize, signed: bool) -> Option<i64> {
    unsafe {
        Some(match (size, signed) {
            (1, true) => addr.cast::<i8>().read_unaligned() as i64,
            (1, false) => addr.cast::<u8>().read_unaligned() as i64,
            (2, true) => addr.cast::<i16>().read_unaligned() as i64,
            (2, false) => addr.cast::<u16>().read_unaligned() as i64,
            (4, true) => addr.cast::<i32>().read_unaligned() as i64,
            (4, false) => addr.cast::<u32>().read_unaligned() as i64,
            (8, _) => addr.cast::<i64>().read_unaligned(),
            _ => return None,
        })
    }
}

fn load_interior_value(
    array_ptr: i64,
    index: i64,
    access: &InteriorFieldAccess,
) -> Option<majit_ir::Value> {
    let addr = interior_field_addr(array_ptr, index, access)?;
    match access.field_type {
        0 => {
            if access.field_size != std::mem::size_of::<usize>() {
                return None;
            }
            let raw = unsafe { addr.cast::<usize>().read_unaligned() };
            Some(majit_ir::Value::Ref(majit_ir::GcRef(raw)))
        }
        2 => {
            if access.field_size != std::mem::size_of::<u64>() {
                return None;
            }
            let bits = unsafe { addr.cast::<u64>().read_unaligned() };
            Some(majit_ir::Value::Float(f64::from_bits(bits)))
        }
        _ => read_interior_int(addr, access.field_size, access.signed).map(majit_ir::Value::Int),
    }
}

fn concrete_interior_store_bits<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    value: OpRef,
    value_bank: char,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<i64> {
    match value_bank {
        'i' => known_int_operand(code, op, 2, value, ctx, true),
        'r' => {
            if let Some(majit_ir::Value::Ref(reference)) = ctx.trace_ctx.box_value(value)
                && reference != majit_ir::GcRef::NO_CONCRETE
            {
                return Some(reference.as_usize() as i64);
            }
            match read_ref_reg_concrete(code, op, 2, ctx) {
                ConcreteValue::Ref(pointer) => Some(pointer as usize as i64),
                _ => None,
            }
        }
        'f' => {
            if let Some(majit_ir::Value::Float(bits)) = ctx.trace_ctx.box_value(value) {
                return Some(bits.to_bits() as i64);
            }
            match read_float_reg_concrete(code, op, 2, ctx) {
                ConcreteValue::Float(bits) => Some(bits.to_bits() as i64),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Concrete replay of `bh_setinteriorfield_gc_{i,r,f}`.
///
/// Not [`walker_fill_materialized_array`]: that helper writes one whole
/// element and reverts the array pointer when the value is not a ref.
/// An interior store writes `field_size` bytes (`f_valid` is one unsigned
/// byte) through `setinteriorfield`, and a ref store write-barriers the
/// array the way `bh_setinteriorfield_gc_r` does.
fn replay_setinteriorfield<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
    array: OpRef,
    index: OpRef,
    value: OpRef,
    descr: &majit_ir::DescrRef,
    value_bank: char,
) {
    let Some(access) = interior_field_access(descr) else {
        return;
    };
    let Some(array_ptr) = concrete_ref_operand_ptr(code, op, 0, array, ctx) else {
        return;
    };
    let Some(index) = known_int_operand(code, op, 1, index, ctx, true) else {
        return;
    };
    let Some(bits) = concrete_interior_store_bits(code, op, value, value_bank, ctx) else {
        return;
    };
    if interior_field_addr(array_ptr, index, &access).is_none() {
        return;
    }
    let array_raw = array_ptr as usize as *mut u8;
    if access.field_type == 0 {
        majit_gc::gc_write_barrier(majit_ir::GcRef(array_ptr as usize));
    }
    pyre_object::setinteriorfield(
        array_raw.cast::<pyre_object::GcTypedArray>(),
        access.items_base,
        index as usize,
        access.field_offset,
        access.field_size,
        access.item_size,
        access.field_type,
        bits,
    );
    if access.field_type == 0 {
        pyre_object::gc_hook::try_gc_write_barrier(array_raw as pyre_object::gc_hook::GCREF);
    }
}

/// `getinteriorfield_gc_<i|r|f>/rid>X`.
///
/// `pyjitpl.py _opimpl_getinteriorfield_gc_any`: the heapcache key is the
/// interior field descr, via the same `getarrayitem` methods as an array
/// element. A miss records `GETINTERIORFIELD_GC_*` and, when the block is
/// the traced allocation, stamps `bh_getinteriorfield_gc_*` so
/// `opimpl_goto_if_not_int_is_zero` can take `box.getint()`.
pub(crate) fn getinteriorfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();
    let result = if let Some(cached) =
        ctx.trace_ctx
            .heapcache_getarrayitem(array, index, descr_index)
    {
        ctx.trace_ctx
            .profiler()
            .count_ops(opcode, majit_metainterp::counters::HEAPCACHED_OPS);
        cached
    } else {
        ctx.trace_ctx
            .profiler()
            .count_ops(opcode, majit_metainterp::counters::OPS);
        ctx.trace_ctx
            .profiler()
            .count_ops(opcode, majit_metainterp::counters::RECORDED_OPS);
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[array, index], descr.clone());
        if let (Some(access), Some(array_ptr), Some(index_value)) = (
            interior_field_access(&descr),
            concrete_ref_operand_ptr(code, op, 0, array, ctx),
            known_int_operand(code, op, 1, index, ctx, true),
        ) && let Some(live) = load_interior_value(array_ptr, index_value, &access)
        {
            ctx.trace_ctx.set_opref_concrete(resbox, live);
        }
        ctx.trace_ctx
            .heapcache_getarrayitem_now_known(array, index, descr_index, resbox);
        resbox
    };
    let dst = code[op.pc + 5] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
        'r' => write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setinteriorfield_gc_<i|r|f>/ri{i,r,f}d`.
///
/// `pyjitpl.py execute_setinteriorfield_gc`: record `SETINTERIORFIELD_GC`
/// and `heapcache.setarrayitem` keyed by the interior descr. The concrete
/// store is `setinteriorfield`, not a whole-element array fill.
pub(crate) fn setinteriorfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    let value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 3, ctx)?;
    let descr_index = descr.index();
    ctx.trace_ctx
        .profiler()
        .count_ops(OpCode::SetinteriorfieldGc, majit_metainterp::counters::OPS);
    ctx.trace_ctx.profiler().count_ops(
        OpCode::SetinteriorfieldGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetinteriorfieldGc,
        &[array, index, value],
        descr.clone(),
    );
    replay_setinteriorfield(code, op, ctx, array, index, value, &descr, value_bank);
    ctx.trace_ctx
        .heapcache_setarrayitem(array, index, descr_index, value);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setfield_gc_<i|r>/<rid|rrd>` handler: read box (r-reg), valuebox
/// (i or r reg per `value_bank`), descr operand, then either skip
/// the IR emission (cache says the same value is already there) or
/// record `OpCode::SetfieldGc` and write through to the heapcache.
///
/// RPython parity: `pyjitpl.py _opimpl_setfield_gc_any`:
///
///   upd = heapcache.get_field_updater(box, fielddescr)
///   if upd.currfieldbox is valuebox:
///       return                       # cache hit, no IR
///   self.metainterp.execute_and_record(rop.SETFIELD_GC, fielddescr,
///                                       box, valuebox)
///   upd.setfield(valuebox)
///
/// **Alias-clearing writeback**: goes through
/// `HeapCache::setfield_cached` instead of `getfield_now_known`. The
/// difference is the alias-clearing semantic that RPython's
/// `FieldUpdater.setfield()` carries (heapcache.py routes to
/// `CacheEntry.do_write_with_aliasing`):
///
///   `_clear_cache_on_write(seen_alloc)` (heapcache.py) wipes
///   `cache_anything` unconditionally and additionally wipes
///   `cache_seen_allocation` when the write target itself is not
///   seen-allocated.  This conservatively kills any cached entry whose
///   source-box might alias the SETFIELD target.
///
/// `getfield_now_known` only inserts the new (obj, field, value) tuple
/// — it does NOT clear sibling entries.  Using it here meant a
/// subsequent `getfield_gc(other_obj, same_field)` could return a
/// stale value cached from before the SETFIELD.  Switching to
/// `setfield_cached` matches `do_write_with_aliasing` exactly.
///
/// `value_bank` selects the valuebox source: `'i'` reads
/// `registers_i[v]`, `'r'` reads `registers_r[v]`.
pub(crate) fn setfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `<r><v>d`: 1B r-reg(box) + 1B v(value) + 2B descr-index.
    // For the `c`-coded short form (`setfield_gc_i/rcd`) the value byte is
    // an inline signed constant (`signedord`, `blackhole.py`) read as a
    // `ConstInt` box instead of an `i`-register slot; obj and descr keep the
    // `rid` byte positions.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let valuebox = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        'c' => OpRef::ConstInt(code[op.pc + 2] as i8 as i64),
        _ => unreachable!("value_bank must be 'i', 'r', 'f' or 'c'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    // Cache hit: if the heapcache already records `valuebox` as the
    // current value of `(obj, descr)`, the SETFIELD_GC is redundant —
    // skip recording. RPython pyjitpl.py _opimpl_setfield_gc_any:
    //   if upd.currfieldbox is valuebox:
    //       self.metainterp.staticdata.profiler.count_ops(rop.SETFIELD_GC, Counters.HEAPCACHED_OPS)
    //       return
    let is_redundant = ctx
        .trace_ctx
        .heapcache_getfield_cached(obj, descr_index)
        .map(|b| b)
        == Some(valuebox);
    if is_redundant {
        ctx.trace_ctx.profiler().count_ops(
            OpCode::SetfieldGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
    } else {
        // Authoritative-executor eager store, the same posture as the
        // module-global cell fold in `mod.rs`: `_opimpl_setfield_gc_any`
        // reaches `executor.execute` → `cpu.bh_setfield_gc_*` through
        // `execute_and_record` (`pyjitpl.py`), so the store really
        // happens while the op is recorded.  Recording alone leaves the
        // concrete object holding its pre-store bytes while the trace
        // heapcache carries `valuebox`, and the next `getfield_gc_*` on the
        // same field hits the cache and trips its `executor.execute`
        // sanity check (`pyjitpl.py`) on the divergence.
        //
        // A box this walk allocated (`heapcache.new`'s HF_SEEN_ALLOCATION,
        // set by `new/d>r` and `new_with_vtable/d>r` right after
        // `execute_new_allocation` hands back a real, zeroed object) is held
        // by nothing outside the walk, so the write needs no journal entry
        // to survive a non-commit rollback — the abandoned allocation goes
        // with it.  A store into a pre-existing object is observable, so
        // [`walker_execute_gc_store`] journals what it displaces.
        if ctx.trace_ctx.heap_cache().saw_allocation(obj) {
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.trace_ctx.box_value(obj)
                && let Some(value) = ctx.trace_ctx.box_value(valuebox)
            {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    ctx.trace_ctx.field_store(struct_ptr, &descr, value);
                }
            }
        } else if ctx.trace_ctx.standard_virtualizable_box() != Some(obj)
            && let Some(fd) = descr.as_field_descr()
        {
            // The standard virtualizable's box carries the trace-stepping
            // heap copy rather than the live frame
            // ([`fbw_publish_exit_last_instr`]), so its stores keep their
            // own concrete counterparts.
            let (offset, size, ty) = (fd.offset(), fd.field_size(), fd.field_type());
            let obj_ptr = concrete_ref_operand_ptr(code, op, 0, obj, ctx);
            let value = concrete_store_value_operand(code, op, 1, value_bank, valuebox, ctx);
            let before = obj_ptr.and_then(|p| ctx.trace_ctx.field_sanity_load(p, &descr, ty));
            walker_execute_gc_store(ctx, obj_ptr, offset, size, ty, before, value, op.pc)?;
        }
        ctx.trace_ctx
            .profiler()
            .count_ops(OpCode::SetfieldGc, majit_metainterp::counters::OPS);
        ctx.trace_ctx
            .profiler()
            .count_ops(OpCode::SetfieldGc, majit_metainterp::counters::RECORDED_OPS);
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, valuebox], descr.clone());
        // Write-through with alias-clearing semantics
        // (`heapcache.py do_write_with_aliasing`).  Mirrors
        // `upd.setfield(valuebox)` (heapcache.py).
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, descr_index, valuebox);
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `getfield_gc_<i|r>/rd>X` handler: read a Ref-bank source register
/// + descr operand, consult the heapcache, and either return the
/// cached field box (no IR op recorded) or record the appropriate
/// `OpCode::GetfieldGc<I|R>` op and update the cache.
///
/// RPython parity: `opimpl_getfield_gc_<i|r>` →
/// `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py`).
/// RPython has a ConstPtr+is_always_pure() fast path
/// that fires `executor.execute(cpu, metainterp, opnum, fielddescr,
/// box)` and returns `ConstInt/ConstFloat/ConstPtr(resvalue)` —
/// recording NO trace op (the value is directly substituted as a Const
/// literal). The walker's `executor.execute` counterpart is
/// `field_sanity_load`, so the fast path is implemented: a constant
/// source register through an always-pure descr folds to the loaded
/// value as a Const literal with no recorded op.
///
/// Walker behaviour mirrors `_opimpl_getfield_gc_any_pureornot`
/// uniformly: heapcache hit returns the cached box (no IR op);
/// heapcache miss records the opcode the dispatch selected +
/// writes through. RPython aliases the `_pure` jitcode spelling to
/// the plain opimpl (`pyjitpl.py`), so both spellings record the
/// plain opnum here; the optimizer re-derives purity from
/// `descr.is_always_pure()` (`heap.py:641` const fold +
/// invalidation-exempt field cache). There is no post-trace rewrite
/// to the Pure opcodes.
///
/// `dst_bank` selects the result bank: `'i'` writes `registers_i[dst]`,
/// `'r'` writes `registers_r[dst]`.
pub(crate) fn getfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `rd>X`: 1B r-reg + 2B descr-index + 1B dst.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let descr_index = descr.index();
    let concrete_obj_ptr = concrete_ref_operand_ptr(code, op, 0, obj, ctx);

    // ConstPtr + always-pure fast path (pyjitpl.py): a constant
    // source through an immutable descr loads the field now and
    // substitutes the value as a Const literal, recording no op.
    let const_pure_result = if obj.is_constant() && descr.is_always_pure() {
        let load_type = match opcode {
            OpCode::GetfieldGcI => Some(majit_ir::Type::Int),
            OpCode::GetfieldGcR => Some(majit_ir::Type::Ref),
            OpCode::GetfieldGcF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let struct_ptr = concrete_obj_ptr;
        match (load_type, struct_ptr) {
            (Some(ty), Some(p)) => {
                ctx.trace_ctx
                    .field_sanity_load(p, &descr, ty)
                    .map(|v| match v {
                        majit_ir::Value::Int(n) => ctx.trace_ctx.const_int(n),
                        majit_ir::Value::Ref(r) => ctx.trace_ctx.const_ref(r.0 as i64),
                        majit_ir::Value::Float(f) => ctx.trace_ctx.const_float(f.to_bits() as i64),
                        _ => unreachable!("field_sanity_load returns Int/Ref/Float only"),
                    })
            }
            _ => None,
        }
    } else {
        None
    };

    // heaptracker.py special-cases the `typeptr` field: once a GUARD_CLASS
    // has pinned an object's class, reading its typeptr yields the known
    // class constant.  Inside an inline sub-walk the receiver's concrete
    // pointer often lives only in the register shadow (not the box value), so
    // the const-pure path above misses; fold the typeptr read straight from
    // the heapcache's known class instead.  This lets inlined type predicates
    // (`is_int`/`is_bool`, which read the typeptr and compare it against a
    // type address) fold during the walk.
    let is_typeptr_field = descr
        .as_field_descr()
        .is_some_and(|fd| fd.offset() == pyre_object::pyobject::OB_TYPE_OFFSET);
    let typeptr_const = if ctx.fbw_mode.inline_subwalk
        && !obj.is_constant()
        && is_typeptr_field
        && ctx.trace_ctx.heap_cache().is_class_known(obj)
    {
        match (concrete_obj_ptr, opcode) {
            (Some(p), OpCode::GetfieldGcI) => {
                let cls =
                    unsafe { (*(p as *const pyre_object::pyobject::PyObject)).ob_type as i64 };
                Some(ctx.trace_ctx.const_int(cls))
            }
            (Some(p), OpCode::GetfieldGcR) => {
                let cls =
                    unsafe { (*(p as *const pyre_object::pyobject::PyObject)).ob_type as i64 };
                Some(ctx.trace_ctx.const_ref(cls))
            }
            _ => None,
        }
    } else {
        None
    };

    let result = if let Some(folded) = typeptr_const {
        folded
    } else if let Some(constant) = const_pure_result {
        constant
    } else if let Some(cached) = ctx.trace_ctx.heapcache_getfield_cached(obj, descr_index) {
        // Cache hit (RPython _opimpl_getfield_gc_any_pureornot):
        //   if upd.currfieldbox is not None:
        //       self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //       return upd.currfieldbox
        // RPython hardcodes `GETFIELD_GC_I` for the count regardless of
        // the actual rop variant (`_i` / `_r` / `_f`); match the
        // hardcode for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        // Recording a `getfield_gc_*` whose FieldDescr lacks a parent_descr
        // backreference would later crash the optimizer's
        // `ensure_ptr_info_arg0` (`optimizer.py`).  Inside a sub-walk
        // abort gracefully instead, so the trace falls back to the interpreter
        // rather than carrying an op the optimizer cannot lower.  The fold
        // paths above (typeptr / const-pure / cache-hit) record nothing, so
        // they are unaffected; production sub-walks never reach this (they
        // would already panic).
        if ctx.fbw_mode.inline_subwalk
            && matches!(
                opcode,
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF
            )
            && let Some(fd) = descr.as_field_descr()
            && fd.get_parent_descr().is_none()
            && !fd.is_w_class()
        {
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!(
                    "[fbw-abort] FieldDescrMissingParentDescr field={} offset={} pc={}",
                    fd.field_name(),
                    fd.offset(),
                    op.pc,
                );
            }
            return Err(DispatchError::FieldDescrMissingParentDescr { pc: op.pc });
        }
        // Cache miss — record op + write through.  `box_value`
        // resolves the Box.value chain PyPy reads off
        // `box.getref_base()` in `executor.do_getfield_gc_*`
        // (`executor.py`); the sanity load fires whenever the
        // struct pointer is known (Const, vable shadow, or stamped),
        // mirroring `pyjitpl.py resbox = execute_with_descr(...);
        // upd.getfield_now_known(resbox)`.
        // The immutable jitcode spellings alias `opimpl_getfield_gc_*`
        // upstream, so the profiler sees the plain opnum.
        let profiled_opcode = opcode;
        ctx.trace_ctx
            .profiler()
            .count_ops(profiled_opcode, majit_metainterp::counters::OPS);
        ctx.trace_ctx
            .profiler()
            .count_ops(profiled_opcode, majit_metainterp::counters::RECORDED_OPS);
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[obj], descr.clone());
        ctx.trace_ctx
            .heapcache_getfield_now_known(obj, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref` so a
    // constant-folded predecessor (e.g. `binop_int_record` having
    // stamped this OpRef in OpRef concrete stamping) propagates through.  RPython
    // `Box.value` parity: `pyjitpl.py:executor.py` per-opcode LLOp
    // stamps `box.value` post-exec; pyre's `concrete_of_opref` reads
    // that channel.  Null fallback preserves the prior unknown-result
    // behaviour for cache-miss recorded ops.
    let mut concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    if matches!(concrete_for_shadow, ConcreteValue::Null) {
        // A heapcache hit returns the original FrontendOp.  In RPython that
        // object still carries the concrete `.value` installed when the field
        // op executed.  Pyre can encounter an equivalent OpRef allocated
        // outside the active recorder (canonical helper splice), where its
        // typed register shadow survives but the recorder-local value carrier
        // does not.  Re-read the observed field through the concrete receiver,
        // stamp when the OpRef belongs to this recorder, and always propagate
        // the value into this MIFrame register shadow.
        let load_type = match opcode {
            OpCode::GetfieldGcI => Some(majit_ir::Type::Int),
            OpCode::GetfieldGcR => Some(majit_ir::Type::Ref),
            OpCode::GetfieldGcF => Some(majit_ir::Type::Float),
            _ => None,
        };
        if let (Some(ty), Some(struct_ptr)) = (load_type, concrete_obj_ptr)
            && let Some(live_value) = ctx.trace_ctx.field_sanity_load(struct_ptr, &descr, ty)
        {
            ctx.trace_ctx.try_set_opref_concrete(result, live_value);
            concrete_for_shadow = match live_value {
                majit_ir::Value::Int(value) => ConcreteValue::Int(value),
                majit_ir::Value::Ref(reference) if reference != majit_ir::GcRef::NO_CONCRETE => {
                    ConcreteValue::Ref(reference.as_usize() as pyre_object::PyObjectRef)
                }
                majit_ir::Value::Float(value) => ConcreteValue::Float(value),
                majit_ir::Value::Ref(_) | majit_ir::Value::Void => ConcreteValue::Null,
            };
        }
    }
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let _ = ctx
                .registers_f
                .get(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            ctx.registers_f.set(dst, result);
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `virtualizable_gen.rs` pyre PyFrame static-field order
/// `[last_instr, pycode, valuestackdepth, debugdata]`.
pub(crate) const VABLE_CODE_FIELD_IDX: usize = 1;

/// `pyjitpl.py opimpl_guard_class` for the `guard_class/r>X` op
/// `jtransform.rs rewrite_op_getfield` emits in place of a read of the
/// header's class word.  The receiver's class is pinned with a `GuardClass`
/// unless the heapcache already knows it, and the op's result is that class
/// as a constant, in the bank (`dst_bank`) the replaced read was allocated
/// to.  A constant receiver records no guard, as `generate_guard` does not.
///
/// The class is read off the live receiver: the walker executes as it
/// records, so the concrete pointer is in the box value or the register
/// shadow.  A receiver whose pointer neither carries is one this walk cannot
/// execute a header read for, which [`getfield_gc_via_heapcache`] would
/// have recorded symbolically; the class cannot be pinned symbolically, so
/// the walk declines the op.
pub(crate) fn guard_class_record<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let dst = code[op.pc + 2] as usize;
    let concrete_obj_ptr = concrete_ref_operand_ptr(code, op, 0, obj, ctx);
    let Some(obj_ptr) = concrete_obj_ptr else {
        if fbw_debug_abort_enabled() {
            eprintln!("[fbw-abort] GuardClassReceiverNotConcrete pc={}", op.pc);
        }
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "guard_class (receiver not concrete)",
        });
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && pyre_object::tagged_int::is_tagged_int(obj_ptr as pyre_object::PyObjectRef)
    {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "guard_class (tagged receiver)",
        });
    }
    // SAFETY: `obj_ptr` is a live heap object the walk is executing over;
    // tagged immediates are declined above. The class word is the first
    // word of every `PyObject` header.
    let cls = unsafe { (*(obj_ptr as *const pyre_object::PyObject)).ob_type } as i64;
    ctx.trace_ctx
        .profiler()
        .count_ops(OpCode::GuardClass, majit_metainterp::counters::OPS);
    if !obj.is_constant() {
        walker_guard_class(ctx, op.pc, obj, cls)?;
    }
    match dst_bank {
        'i' => {
            let result = ctx.trace_ctx.const_int(cls);
            write_int_reg(ctx, op.pc, dst, result, ConcreteValue::Int(cls))?;
        }
        _ => {
            let result = ctx.trace_ctx.const_ref(cls);
            write_ref_reg(
                ctx,
                op.pc,
                dst,
                result,
                ConcreteValue::Ref(cls as usize as pyre_object::PyObjectRef),
            )?;
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// The box and live value of the int operand at `operand_offset`.
///
/// An `i` argcode reads an i-bank register; `c` is the `USE_C_FORM`
/// sibling (`assembler.py`) whose value is one inline signed byte
/// (`signedord`, `blackhole.py`).
fn int_operand<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Result<(OpRef, Option<i64>), DispatchError> {
    if op.argcodes.as_bytes().get(operand_offset) == Some(&b'c') {
        let n = code[op.pc + 1 + operand_offset] as i8 as i64;
        return Ok((OpRef::ConstInt(n), Some(n)));
    }
    let boxed = read_int_reg(code, op, operand_offset, ctx)?;
    let value = match boxed.inline_const_to_value() {
        Some(majit_ir::Value::Int(n)) => Some(n),
        _ => ctx
            .trace_ctx
            .box_value(boxed)
            .and_then(|value| match value {
                majit_ir::Value::Int(n) => Some(n),
                _ => None,
            }),
    }
    .or_else(
        || match read_int_reg_concrete(code, op, operand_offset, ctx) {
            ConcreteValue::Int(n) => Some(n),
            _ => None,
        },
    );
    Ok((boxed, value))
}

/// `pyjitpl.py opimpl_strlen` — `return self.execute(rop.STRLEN, strbox)`.
///
/// Operand layout `r>i`: 1B r-reg(string) + 1B i-reg(dst).  The live
/// length is `cpu.bh_strlen` (`llmodel.py` / `pyre_cpu`), stamped so a
/// later `goto_if_not` can take the observed branch.
pub(crate) fn opimpl_strlen<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let string = read_ref_reg(code, op, 0, ctx)?;
    let cpu = crate::pyre_cpu::shared();
    let resvalue = concrete_ref_operand_ptr(code, op, 0, string, ctx)
        .and_then(|ptr| cpu.bh_strlen(majit_ir::GcRef(ptr as usize)))
        .map(majit_ir::Value::Int);
    let result = ctx.trace_ctx.execute_and_record(
        Some(cpu.as_ref()),
        OpCode::Strlen,
        None,
        &[string],
        resvalue,
        0,
    );
    let dst = code[op.pc + 2] as usize;
    let concrete = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py opimpl_strgetitem` —
/// `return self.execute(rop.STRGETITEM, strbox, indexbox)`.
///
/// Operand layout `ri>i`: 1B r-reg(string) + 1B i-reg(index) + 1B i-dst.
/// `rc>i` replaces the index register with one signed immediate byte.
pub(crate) fn opimpl_strgetitem<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let string = read_ref_reg(code, op, 0, ctx)?;
    let (index, index_value) = int_operand(code, op, 1, ctx)?;
    let cpu = crate::pyre_cpu::shared();
    let resvalue = concrete_ref_operand_ptr(code, op, 0, string, ctx)
        .zip(index_value)
        .and_then(|(ptr, index_value)| {
            cpu.bh_strgetitem(majit_ir::GcRef(ptr as usize), index_value)
        })
        .map(majit_ir::Value::Int);
    let result = ctx.trace_ctx.execute_and_record(
        Some(cpu.as_ref()),
        OpCode::Strgetitem,
        None,
        &[string, index],
        resvalue,
        0,
    );
    let dst = code[op.pc + 3] as usize;
    let concrete = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Whether `[start, start + length)` lies inside the live STR at `string`,
/// the range `llmodel.py bh_strsetitem` / `bh_copystrcontent` assume.
fn str_range_in_bounds(string: i64, start: i64, length: i64) -> bool {
    let len = pyre_object::lowlevel_string::bh_lowlevel_string_len(string) as i64;
    start >= 0 && length >= 0 && start.checked_add(length).is_some_and(|stop| stop <= len)
}

/// `pyjitpl.py opimpl_newstr` — `return self.execute(rop.NEWSTR, lengthbox)`.
///
/// Operand layout `i>r` / `c>r`: 1B length (i-reg or signed immediate) +
/// 1B r-dst.  `executor.execute` allocates the STR through
/// `cpu.bh_newstr`; the recorded op's value cell roots it.
pub(crate) fn opimpl_newstr<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let (length, length_value) = int_operand(code, op, 0, ctx)?;
    let cpu = crate::pyre_cpu::shared();
    let resvalue = length_value.and_then(|n| ctx.trace_ctx.execute_newstr(n));
    ctx.trace_ctx
        .heapcache_invalidate_caches(OpCode::Newstr, &[length]);
    let result = ctx.trace_ctx.execute_and_record(
        Some(cpu.as_ref()),
        OpCode::Newstr,
        None,
        &[length],
        resvalue,
        0,
    );
    let dst = code[op.pc + 2] as usize;
    let concrete = match resvalue {
        Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) => {
            ConcreteValue::Ref(ptr as pyre_object::PyObjectRef)
        }
        _ => ConcreteValue::Null,
    };
    write_ref_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py opimpl_strsetitem` —
/// `return self.execute(rop.STRSETITEM, strbox, indexbox, newcharbox)`.
///
/// Operand layout `rii` and its `c` forms: 1B r-reg(string) + index +
/// newchar.  Only a buffer still being filled is ever stored into
/// (`rstr.py` strings are immutable once built), and the store is
/// idempotent, so the interpreter repeating it after the walk writes the
/// same byte.  `execute` runs the store before the op is recorded; a store
/// the walk cannot run (an operand without a concrete value, or a range
/// outside the buffer) declines instead of recording a mutation the walk's
/// own later reads would not see.
pub(crate) fn opimpl_strsetitem<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let string = read_ref_reg(code, op, 0, ctx)?;
    let (index, index_value) = int_operand(code, op, 1, ctx)?;
    let (newchar, newchar_value) = int_operand(code, op, 2, ctx)?;
    let Some(ptr) = concrete_ref_operand_ptr(code, op, 0, string, ctx) else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "strsetitem (string not concrete)",
        });
    };
    let (Some(index_value), Some(newchar_value)) = (index_value, newchar_value) else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "strsetitem (index or char not concrete)",
        });
    };
    if !str_range_in_bounds(ptr, index_value, 1) || !(0..=0xff).contains(&newchar_value) {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "strsetitem (out of range)",
        });
    }
    ctx.trace_ctx
        .execute_strsetitem(ptr, index_value, newchar_value);
    let args = [string, index, newchar];
    ctx.trace_ctx
        .heapcache_invalidate_caches(OpCode::Strsetitem, &args);
    let cpu = crate::pyre_cpu::shared();
    ctx.trace_ctx
        .execute_and_record(Some(cpu.as_ref()), OpCode::Strsetitem, None, &args, None, 0);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py opimpl_copystrcontent` — `return self.execute(
/// rop.COPYSTRCONTENT, srcbox, dstbox, srcstartbox, dststartbox,
/// lengthbox)`.
///
/// Operand layout `rriii` and its `c` forms.  The copy runs, or the walk
/// declines, under the same rule as [`opimpl_strsetitem`].
pub(crate) fn opimpl_copystrcontent<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let src = read_ref_reg(code, op, 0, ctx)?;
    let dst = read_ref_reg(code, op, 1, ctx)?;
    let (srcstart, srcstart_value) = int_operand(code, op, 2, ctx)?;
    let (dststart, dststart_value) = int_operand(code, op, 3, ctx)?;
    let (length, length_value) = int_operand(code, op, 4, ctx)?;
    let (Some(src_ptr), Some(dst_ptr)) = (
        concrete_ref_operand_ptr(code, op, 0, src, ctx),
        concrete_ref_operand_ptr(code, op, 1, dst, ctx),
    ) else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "copystrcontent (string not concrete)",
        });
    };
    let (Some(srcstart_value), Some(dststart_value), Some(length_value)) =
        (srcstart_value, dststart_value, length_value)
    else {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "copystrcontent (bound not concrete)",
        });
    };
    if !str_range_in_bounds(src_ptr, srcstart_value, length_value)
        || !str_range_in_bounds(dst_ptr, dststart_value, length_value)
    {
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "copystrcontent (out of range)",
        });
    }
    ctx.trace_ctx.execute_copystrcontent(
        src_ptr,
        dst_ptr,
        srcstart_value,
        dststart_value,
        length_value,
    );
    let args = [src, dst, srcstart, dststart, length];
    ctx.trace_ctx
        .heapcache_invalidate_caches(OpCode::Copystrcontent, &args);
    let cpu = crate::pyre_cpu::shared();
    ctx.trace_ctx.execute_and_record(
        Some(cpu.as_ref()),
        OpCode::Copystrcontent,
        None,
        &args,
        None,
        0,
    );
    Ok((DispatchOutcome::Continue, op.next_pc))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::interior_field_access;

    /// The flat `GcTypedArray` header and an 8-aligned `GcEntries` item
    /// base differ on wasm32 and match on a 64-bit host. A descr base
    /// past `GC_TYPED_ARRAY_ITEMS_OFFSET` is the same split.
    #[test]
    fn interior_field_access_follows_descr_base_past_the_flat_header() {
        let items_base = pyre_object::GC_TYPED_ARRAY_ITEMS_OFFSET + 16;
        let array: Arc<dyn majit_ir::descr::ArrayDescr> =
            Arc::new(majit_ir::descr::SimpleArrayDescr::with_flag(
                1,
                items_base,
                32,
                7,
                majit_ir::value::Type::Ref,
                majit_ir::descr::ArrayFlag::Struct,
            ));
        let field: Arc<dyn majit_ir::descr::FieldDescr> = Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(2, 0, 1, majit_ir::value::Type::Int, false),
        );
        let descr: majit_ir::DescrRef = Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
            3, array, field,
        ));
        let access = interior_field_access(&descr).expect("aligned items base is addressable");
        assert_eq!(access.items_base, items_base);
    }

    /// Recording-time `NEW_ARRAY_CLEAR` allocates with `ArrayDescr.tid`
    /// when a collector is installed (`gc.py` `init_array_descr`).
    #[test]
    fn materialize_cleared_struct_gcarray_uses_the_published_tid() {
        type K = i64;
        type V = pyre_object::PyObjectRef;
        let items_base = std::mem::offset_of!(pyre_object::rordereddict::GcEntries<K, V>, items);
        let item_size = std::mem::size_of::<pyre_object::rordereddict::Entry<K, V>>();
        assert_eq!(items_base, pyre_object::TYPED_ITEMS_BLOCK_ITEMS_OFFSET);
        if !majit_gc::gc_allocator_installed() {
            assert!(
                super::materialize_cleared_struct_gcarray(8, 1, item_size, items_base).is_none()
            );
            return;
        }
        let published = pyre_object::rordereddict::i64_pyobject_entries_gc_type_id();
        assert!(
            !majit_ir::descr::array_tid_is_unresolved(published),
            "host must have registered DICTENTRYARRAY before this test"
        );
        assert!(
            super::materialize_cleared_struct_gcarray(8, published, item_size, items_base)
                .is_some()
        );
    }
}
