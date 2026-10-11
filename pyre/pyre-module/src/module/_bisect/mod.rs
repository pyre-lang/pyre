//! `_bisect` accelerator module.
//!
//! PyPy keeps the algorithms app-level in `lib-python/3/bisect.py`; this
//! interpreter-level surface follows those loops while providing the optional
//! accelerator imported by that module. RustPython's corresponding owner is
//! `crates/stdlib/src/bisect.rs`.

use pyre_object::gc_roots::{pin_root, shadow_stack_get, shadow_stack_len};
use pyre_object::*;

/// The operands a search keeps live, held as shadow-stack slots rather than
/// plain Rust locals: `__index__`, `__len__`, `__getitem__`, `key` and
/// `__lt__` all run arbitrary Python, and a minor collection relocates a
/// young object behind a raw `PyObjectRef`. The shape `builtin_any` uses.
struct BisectArgs {
    a: usize,
    x: usize,
    lo: i64,
    hi: i64,
    key: Option<usize>,
}

impl BisectArgs {
    fn a(&self) -> PyObjectRef {
        shadow_stack_get(self.a)
    }

    fn x(&self) -> PyObjectRef {
        shadow_stack_get(self.x)
    }

    fn key(&self) -> Option<PyObjectRef> {
        self.key.map(shadow_stack_get)
    }
}

/// Pin `value` for the enclosing `push_roots` scope and return the slot that
/// owns it from here on — the pin queries the collector, so the slot, not
/// this argument, is what holds the forwarded object afterwards.
fn pin(value: PyObjectRef) -> usize {
    let slot = shadow_stack_len();
    let _ = pin_root(value);
    slot
}

fn argument(
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    index: usize,
    name: &str,
    function: &str,
) -> Result<Option<PyObjectRef>, pyre_interpreter::PyError> {
    let positional_value = positional.get(index).copied();
    let keyword_value = pyre_interpreter::builtins::kwarg_get(kwargs, name);
    if positional_value.is_some() && keyword_value.is_some() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{function}() got multiple values for argument '{name}'"
        )));
    }
    Ok(positional_value.or(keyword_value))
}

fn index_value(value: PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
    let index = pyre_interpreter::baseobjspace::space_index(value)?;
    pyre_interpreter::baseobjspace::int_w(index)
}

fn parse_args(
    args: &[PyObjectRef],
    function: &str,
) -> Result<BisectArgs, pyre_interpreter::PyError> {
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    pyre_interpreter::builtins::kwarg_reject_unknown(
        kwargs,
        &["a", "x", "lo", "hi", "key"],
        function,
    )?;
    if positional.len() > 4 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{function}() takes at most 4 positional arguments ({} given)",
            positional.len()
        )));
    }

    let a = argument(positional, kwargs, 0, "a", function)?.ok_or_else(|| {
        pyre_interpreter::PyError::type_error(format!(
            "{function}() missing required argument 'a' (pos 1)"
        ))
    })?;
    let x = argument(positional, kwargs, 1, "x", function)?.ok_or_else(|| {
        pyre_interpreter::PyError::type_error(format!(
            "{function}() missing required argument 'x' (pos 2)"
        ))
    })?;
    let lo_arg = argument(positional, kwargs, 2, "lo", function)?;
    let hi_arg = argument(positional, kwargs, 3, "hi", function)?
        .filter(|value| !unsafe { is_none(*value) });
    let key_arg = pyre_interpreter::builtins::kwarg_get(kwargs, "key")
        .filter(|value| !unsafe { is_none(*value) });

    // Every operand is rooted before the first callback runs: `__index__` and
    // `__len__` below already execute Python, and `args` is a plain slice a
    // collection does not rewrite.
    let a = pin(a);
    let x = pin(x);
    let lo_arg = lo_arg.map(pin);
    let hi_arg = hi_arg.map(pin);
    let key = key_arg.map(pin);

    let lo = match lo_arg {
        Some(slot) => index_value(shadow_stack_get(slot))?,
        None => 0,
    };
    if lo < 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "lo must be non-negative",
        ));
    }
    let hi = match hi_arg {
        Some(slot) => index_value(shadow_stack_get(slot))?,
        None => pyre_interpreter::baseobjspace::len_w(shadow_stack_get(a))?,
    };
    Ok(BisectArgs { a, x, lo, hi, key })
}

fn call_one(
    callable: PyObjectRef,
    arg: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::call::call_function_impl_result(callable, &[arg])
}

fn less_than(left: PyObjectRef, right: PyObjectRef) -> Result<bool, pyre_interpreter::PyError> {
    let result = pyre_interpreter::objspace::descroperation::compare(
        left,
        right,
        pyre_interpreter::objspace::descroperation::CompareOp::Lt,
    )?;
    pyre_interpreter::baseobjspace::is_true(result)
}

fn bisect(parsed: &mut BisectArgs, right: bool) -> Result<i64, pyre_interpreter::PyError> {
    while parsed.lo < parsed.hi {
        // Written this way instead of `(lo + hi) / 2` so a search spanning
        // `sys.maxsize` cannot overflow, matching `_bisectmodule.c`.
        let mid = parsed.lo + (parsed.hi - parsed.lo) / 2;
        // The index is allocated BEFORE the sequence is read back: an
        // allocation can collect, and a slot read after it is the forwarded
        // one. `w_int_new` is itself non-moving, so it survives the read.
        let index = w_int_new(mid);
        let mut item = pyre_interpreter::baseobjspace::getitem(parsed.a(), index)?;
        if let Some(key) = parsed.key() {
            item = call_one(key, item)?;
        }
        let is_less = if right {
            less_than(parsed.x(), item)?
        } else {
            less_than(item, parsed.x())?
        };
        if is_less == right {
            parsed.hi = mid;
        } else {
            parsed.lo = mid + 1;
        }
    }
    Ok(parsed.lo)
}

fn search(
    args: &[PyObjectRef],
    right: bool,
    function: &str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut parsed = parse_args(args, function)?;
    Ok(w_int_new(bisect(&mut parsed, right)?))
}

fn bisect_left(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    search(args, false, "bisect_left")
}

fn bisect_right(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    search(args, true, "bisect_right")
}

fn insort(
    args: &[PyObjectRef],
    right: bool,
    function: &str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut parsed = parse_args(args, function)?;
    // The search compares against the KEYED value; the insertion still stores
    // the original, so the two need separate roots.
    let original_x = parsed.x;
    if let Some(key) = parsed.key() {
        parsed.x = pin(call_one(key, shadow_stack_get(original_x))?);
    }
    let index = bisect(&mut parsed, right)?;
    // The `insert` lookup runs the attribute protocol and can relocate a young
    // object, so root the boxed index before it.
    let index = pin(w_int_new(index));
    let insert = pyre_interpreter::baseobjspace::getattr_str(parsed.a(), "insert")?;
    pyre_interpreter::call::call_function_impl_result(
        insert,
        &[shadow_stack_get(index), shadow_stack_get(original_x)],
    )?;
    Ok(w_none())
}

fn insort_left(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    insort(args, false, "insort_left")
}

fn insort_right(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    insort(args, true, "insort_right")
}

pub fn init(ns: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let left = pyre_interpreter::gateway::with_module(
        "_bisect",
        pyre_interpreter::make_module_builtin_function("bisect_left", bisect_left),
    );
    // Each function is pinned before the next allocation. The stores below
    // reload those original slots: a second `pin_root` of the Rust local
    // would publish the pre-forward word (`RootScope::pin_roots`).
    let left_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(left);
    let right = pyre_interpreter::gateway::with_module(
        "_bisect",
        pyre_interpreter::make_module_builtin_function("bisect_right", bisect_right),
    );
    let right_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(right);
    let insert_left = pyre_interpreter::gateway::with_module(
        "_bisect",
        pyre_interpreter::make_module_builtin_function("insort_left", insort_left),
    );
    let insert_left_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(insert_left);
    let insert_right = pyre_interpreter::gateway::with_module(
        "_bisect",
        pyre_interpreter::make_module_builtin_function("insort_right", insort_right),
    );
    let insert_right_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(insert_right);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "bisect_left",
            pyre_object::gc_roots::shadow_stack_get(left_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "bisect_right",
            pyre_object::gc_roots::shadow_stack_get(right_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "bisect",
            pyre_object::gc_roots::shadow_stack_get(right_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "insort_left",
            pyre_object::gc_roots::shadow_stack_get(insert_left_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "insort_right",
            pyre_object::gc_roots::shadow_stack_get(insert_right_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "insort",
            pyre_object::gc_roots::shadow_stack_get(insert_right_slot),
        );
    }
    Ok(())
}
