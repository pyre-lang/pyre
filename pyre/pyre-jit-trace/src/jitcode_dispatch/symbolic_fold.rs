//! Trace-time folds for registered symbolic residual-call targets.
//!
//! A symbolic function address is not callable by the generic executor. This
//! registry admits only helpers whose constant-argument execution is safe to
//! materialize directly as a trace constant. Every failed precondition leaves
//! the residual-call dispatcher unchanged.

use super::{DispatchError, WalkContext, WalkSym};
use majit_ir::{GcRef, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

type Execute = fn(&[Value]) -> Option<Value>;

#[derive(Clone, Copy)]
struct Entry {
    arity: usize,
    execute: Execute,
}

fn execute_box_str_constant(args: &[Value]) -> Option<Value> {
    let Value::Ref(GcRef(ptr)) = args[0] else {
        return None;
    };
    // `box_str_constant`'s source is `&Wtf8` — rstr `Ptr(STR)`
    // (`OpKind::ConstStr`, `StrConstDescriptor.as_unicode_object = false`).
    // Intern by chars (`WeakValueDictRepr.ll_get`). A wrapper-kind operand
    // (`OpKind::ConstInternedStr`) is `interned_str_from_typed_const` with
    // `BoxStrConstKind::InternedUnicode`; this residual's argument is STR.
    let result = pyre_object::unicodeobject::interned_str_from_typed_const(
        ptr,
        pyre_object::unicodeobject::BoxStrConstKind::Str,
    )?;
    Some(Value::Ref(GcRef(result as usize)))
}

fn registry() -> &'static HashMap<i64, Entry> {
    static REGISTRY: OnceLock<HashMap<i64, Entry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let symbolic = majit_jitcode::codewriter::call::symbolic_fnaddr_for_segments([
            "pyre_object",
            "unicodeobject",
            "box_str_constant",
        ]);
        HashMap::from([(
            symbolic,
            Entry {
                arity: 1,
                execute: execute_box_str_constant,
            },
        )])
    })
}

/// Attempt a registered symbolic residual fold before the dispatcher records
/// the call. `Ok(false)` is a strict fall-through: no trace or register state
/// has changed, so the ordinary symbolic recording and abort gates still run.
pub(crate) fn try_fold_registered_symbolic_residual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    allboxes: &[majit_ir::OpRef],
    pc: usize,
    dst: usize,
    dst_bank: char,
) -> Result<bool, DispatchError> {
    if !ctx.is_authoritative_executor {
        return Ok(false);
    }
    let Some(&funcbox) = allboxes.first() else {
        return Ok(false);
    };
    if !funcbox.is_constant() {
        return Ok(false);
    }
    let Some(Value::Int(symbolic)) = ctx.trace_ctx.box_value(funcbox) else {
        return Ok(false);
    };
    if !majit_jitcode::codewriter::call::is_symbolic_fnaddr(symbolic) {
        return Ok(false);
    }
    let Some(entry) = registry().get(&symbolic) else {
        return Ok(false);
    };
    let args = &allboxes[1..];
    if args.len() != entry.arity || args.iter().any(|arg| !arg.is_constant()) {
        return Ok(false);
    }
    let Some(values) = args
        .iter()
        .map(|&arg| ctx.trace_ctx.box_value(arg))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(false);
    };
    let Some(result) = (entry.execute)(&values) else {
        return Ok(false);
    };
    let const_box = match (dst_bank, result) {
        ('i', Value::Int(value)) => ctx.trace_ctx.const_int(value),
        ('r', Value::Ref(value)) => ctx.trace_ctx.const_ref(value.as_usize() as i64),
        ('f', Value::Float(value)) => ctx.trace_ctx.const_float(value.to_bits() as i64),
        _ => return Ok(false),
    };
    super::residual_call::write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, const_box)?;
    Ok(true)
}
