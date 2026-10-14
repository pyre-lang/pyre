//! _opcode module — PyPy: `pypy/module/_opcode/`.
//!
//! Opcode metadata used by `opcode.py` and `dis.py`.

use pyre_object::*;
use rustpython_compiler_core::bytecode::{AnyOpcode, oparg};

fn try_opcode(raw: i64) -> Option<AnyOpcode> {
    u16::try_from(raw).ok()?.try_into().ok()
}

fn opcode_predicate(
    args: &[PyObjectRef],
    predicate: impl FnOnce(AnyOpcode) -> bool,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let opcode = args
        .first()
        .copied()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("opcode argument is required"))?;
    let raw = pyre_interpreter::baseobjspace::int_w(opcode)?;
    Ok(w_bool_from(try_opcode(raw).is_some_and(predicate)))
}

fn is_valid(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |_| true)
}

fn has_arg(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_arg())
}

fn has_const(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_const())
}

fn has_name(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_name())
}

fn has_jump(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_jump())
}

fn has_free(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_free())
}

fn has_local(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_local())
}

fn has_exc(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.is_block_push())
}

fn stack_effect(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // Bound scope: `opcode`, `oparg`, kw-only `jump` (`PY_NULL` omitted).
    if args.first().copied().filter(|o| !o.is_null()).is_none() {
        return Err(pyre_interpreter::PyError::type_error(
            "stack_effect() missing opcode",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let n = args.len();
    let base = roots.pin_roots(args);
    let raw = pyre_interpreter::baseobjspace::int_w(roots.get(base))?;
    let opcode = try_opcode(raw)
        .filter(|op| op.real().is_none_or(|real| real.deopt().is_none()))
        .ok_or_else(|| pyre_interpreter::PyError::value_error("invalid opcode or oparg"))?;
    let oparg_obj = if n > 1 {
        let w = roots.get(base + 1);
        if w.is_null() { None } else { Some(w) }
    } else {
        None
    };
    let oparg = match oparg_obj {
        Some(value) if unsafe { !is_none(value) } => pyre_interpreter::baseobjspace::int_w(value)?,
        _ => 0,
    };
    let oparg = u32::try_from(oparg)
        .map_err(|_| pyre_interpreter::PyError::value_error("invalid opcode or oparg"))?;
    let jump = if n > 2 {
        let w = roots.get(base + 2);
        if w.is_null() { None } else { Some(w) }
    } else {
        None
    };
    let effect = match jump {
        Some(value) if unsafe { !is_none(value) } => {
            if pyre_interpreter::baseobjspace::is_true(value)? {
                opcode.stack_effect_jump(oparg)
            } else {
                opcode.stack_effect(oparg)
            }
        }
        _ => opcode
            .stack_effect(oparg)
            .max(opcode.stack_effect_jump(oparg)),
    };
    Ok(w_int_new(effect as i64))
}

// Both enums already carry `Invalid = 0`, whose `desc()` is the
// `INTRINSIC_*_INVALID` entry the table starts with, so `iter()` alone produces
// the whole table. Prepending it again shifted every real name one slot up and
// mislabelled every intrinsic in `dis` output.
fn get_intrinsic1_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::IntrinsicFunction1::iter() {
        items.push(w_str_new_managed(value.desc()));
    }
    Ok(w_list_new(items.take()))
}

fn get_intrinsic2_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::IntrinsicFunction2::iter() {
        items.push(w_str_new_managed(value.desc()));
    }
    Ok(w_list_new(items.take()))
}

fn get_nb_ops(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // Each name string and each pair tuple is a nursery object, so they
    // all sit on one bracket. Reload the pair inputs from their slots
    // before `w_tuple_new` so a later mint cannot hand a stale local.
    let _roots = pyre_object::gc_roots::push_roots();
    let mut row_slots = Vec::new();
    for value in oparg::BinaryOperator::iter() {
        let desc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(value.desc()));
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(&value.to_string()));
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(desc_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        ]));
        row_slots.push(slot);
    }
    Ok(w_list_new(
        row_slots
            .into_iter()
            .map(pyre_object::gc_roots::shadow_stack_get)
            .collect(),
    ))
}

fn special_method_names_impl() -> PyObjectRef {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::SpecialMethod::iter() {
        items.push(w_str_new_managed(&value.to_string()));
    }
    w_list_new(items.take())
}

pyre_interpreter::py_module! {
    "_opcode",
    inline_functions: {
        fn get_opname(opcode: i64) -> String {
            format!("<{opcode}>")
        }
        fn get_special_method_names() -> PyObjectRef {
            crate::module::_opcode::special_method_names_impl()
        }
    },
    functions: {
        // `stack_effect(opcode, oparg=None, *, jump=None)` — the optional
        // tail and the keyword-only `jump` leave no single natural arity, so
        // the body enforces the count itself.
        "stack_effect"             / * = stack_effect; crate::Signature::new(vec!["opcode", "oparg", "jump"], None, None, 1, 0),
        "get_executor"             / 2 = |_| Ok(w_none()),
        "get_specialization_stats" / 0 = |_| Ok(w_none()),
        "get_intrinsic1_descs"     / 0 = get_intrinsic1_descs,
        "get_intrinsic2_descs"     / 0 = get_intrinsic2_descs,
        "get_nb_ops"               / 0 = get_nb_ops,
        "get_executor_count"       / 0 = |_| Ok(w_int_new(0)),
        "get_hot_code"             / 0 = |_| Ok(w_list_new(vec![])),
    },
    extra_init: |ns| {
        // `Python/bytecodes.c` exposes `ENABLE_SPECIALIZATION`; pyre has no
        // CPython-style adaptive specialization, so it reads False — tests
        // gated on `@requires_specialization` then skip.
        crate::__pyre_store!(ns, "ENABLE_SPECIALIZATION", w_bool_from(false));
        crate::__pyre_store!(ns, "ENABLE_SPECIALIZATION_FT", w_bool_from(false));
        for (name, function) in [
            ("is_valid", is_valid as pyre_interpreter::BuiltinCodeFn),
            ("has_arg", has_arg as pyre_interpreter::BuiltinCodeFn),
            ("has_const", has_const as pyre_interpreter::BuiltinCodeFn),
            ("has_name", has_name as pyre_interpreter::BuiltinCodeFn),
            ("has_jump", has_jump as pyre_interpreter::BuiltinCodeFn),
            ("has_free", has_free as pyre_interpreter::BuiltinCodeFn),
            ("has_local", has_local as pyre_interpreter::BuiltinCodeFn),
            ("has_exc", has_exc as pyre_interpreter::BuiltinCodeFn),
        ] {
            crate::__pyre_store!(ns, name, pyre_interpreter::make_builtin_function_with_arity(name, function, 1));
        }
    }
}
