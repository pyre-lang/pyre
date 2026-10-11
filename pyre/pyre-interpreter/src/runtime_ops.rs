use std::slice;
use std::sync::OnceLock;

use crate::bytecode::{BinaryOperator, ComparisonOperator, ConvertValueOparg};
use pyre_object::{
    PY_NULL, PyObjectRef, W_ListIterObject, W_SeqIterObject, W_TupleIterObject, is_instance,
    is_list, is_list_iter, is_range_iter, is_seq_iter, is_str, is_tuple, is_tuple_iter, w_dict_new,
    w_dict_store_checked, w_int_get_value, w_int_new, w_list_getitem, w_list_len, w_list_new,
    w_range_iter_has_next, w_range_iter_next, w_str_from_wtf8, w_str_get_wtf8, w_str_len,
    w_tuple_getitem, w_tuple_len, w_tuple_new, w_tuple_new_from_slice,
};
use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::{
    PyError, PyErrorKind, builtin_code_get, function_get_code, is_builtin_code, is_function,
    w_code_get_ptr,
};

pub use majit_rlib::rffi::SandboxStub;

/// `pypy/interpreter/pyopcode.py MAKE_FUNCTION` stamps the new
/// function's `w_func_globals = self.w_globals` directly from the
/// running frame's dict object.  This entry point accepts the canonical
/// PyObjectRef (`w_globals`) so the freshly-created function inherits
/// the frame's exact `__globals__` identity.
pub fn make_function_from_code_obj_with_globals_obj(
    code_obj: PyObjectRef,
    w_globals: PyObjectRef,
) -> PyObjectRef {
    // `function.py __init__`:
    //   self.name = forcename or code.co_name
    //   self.qualname = qualname or self.name
    // so `name` is the bare `co_name` and `qualname` is the dotted
    // `co_qualname`.  `pyopcode.py MAKE_FUNCTION` then stamps the
    // qualified name from `codeobj.co_qualname`, which is why a later
    // `__code__ = new_code` assignment does NOT change `__qualname__`.  Both
    // are read straight off the code object, so neither allocates.
    crate::function::function_new_from_code(code_obj, w_globals)
}

fn decode_name(name_ptr: *const u8, name_len: i64) -> Option<&'static str> {
    if name_ptr.is_null() || name_len < 0 {
        return None;
    }
    let bytes = unsafe { slice::from_raw_parts(name_ptr, name_len as usize) };
    std::str::from_utf8(bytes).ok()
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_make_function_from_globals(
    w_globals: PyObjectRef,
    code_obj: PyObjectRef,
) -> PyObjectRef {
    // `w_globals` is the globals OBJECT (the JIT threads the vable
    // `w_globals` slot).  Capture it directly; the raw `*mut DictStorage`
    // is recovered from the object wherever a frame still needs it.
    make_function_from_code_obj_with_globals_obj(code_obj, w_globals)
}

/// SET_FUNCTION_ATTRIBUTE residual: stamp one attribute on `func` per the
/// compile-time `flag` discriminant, then return `func` so the caller pushes
/// the same function back onto the stack (net stack effect -1).  `flag` is the
/// `MakeFunctionFlag` bit-position discriminant (`oparg.rs`
/// `MakeFunctionFlag`): `Defaults = 0`, `KwOnlyDefaults = 1`,
/// `Annotations = 2`, `Closure = 3`, `Annotate = 4`, `TypeParams = 5`.  The
/// match must stay in sync with `eval.rs`
/// `set_function_attribute_with_flag` and the codewriter's
/// `SetFunctionAttribute` arm.  Sets a typed field on the function; runs no
/// user code and never raises → `Plain`.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_set_function_attribute(
    func: PyObjectRef,
    attr: PyObjectRef,
    flag: i64,
) -> i64 {
    match flag {
        0 => unsafe { crate::function_set_defaults(func, attr) },
        1 => unsafe { crate::function::function_set_kwdefaults_from_definition(func, attr) },
        2 => {
            // `function.py fset_func_annotations` — the eager
            // annotations dict is stamped on the typed `Function.w_ann`
            // slot via `function_write_barrier`, so `f.__annotations__ is
            // f.__annotations__` holds through the getattr arm.
            unsafe { crate::function::function_set_annotations(func, attr) };
        }
        3 => unsafe { crate::function_set_closure(func, attr) },
        4 => {
            // PEP 649: `attr` is the `__annotate__` callable the runtime
            // `__annotations__` getter evaluates lazily; stored on the
            // typed `w_annotate` slot.
            unsafe { crate::function::function_set_annotate_unchecked(func, attr) };
        }
        // `TypeParams = 5` names a PEP 695 type-parameter tuple, but the
        // codegen never sets that bit: a generic `def` stamps its parameters
        // through `CALL_INTRINSIC_2 INTRINSIC_SET_FUNCTION_TYPE_PARAMS`
        // instead, which writes `Function.w_typeparams`.  Accept the operand
        // rather than panic on a bytecode stream that does carry it.
        _ => {}
    }
    func as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_load_name_from_namespace(
    frame_ptr: *const crate::pyframe::PyFrame,
    mut w_globals: PyObjectRef,
    name_ptr: *const u8,
    name_len: i64,
) -> PyObjectRef {
    let Some(name) = decode_name(name_ptr, name_len) else {
        return PY_NULL;
    };
    // A nursery-born frame moves under `finditem_str`. The raw operand is the
    // abandoned copy after that call; read the builtin module back through the
    // anchor.
    let frame_anchor =
        unsafe { crate::eval::FrameAnchor::from_raw(frame_ptr as *mut crate::pyframe::PyFrame) };
    // `pyopcode.py _load_global`: `space.finditem_str(w_globals,
    // varname)`.  finditem_str takes the borrowed-string fast path for real
    // dict layouts and dispatches a dict subclass through the general mapping
    // object, so a raising key `__eq__` propagates through the traced path
    // instead of being swallowed as a miss.
    if !w_globals.is_null() {
        let value = match pyre_object::with_roots!(w_globals =>
            crate::baseobjspace::finditem_str(w_globals, name)
        ) {
            Ok(value) => value,
            Err(mut error) => {
                jit_publish_exception(error.to_exc_object());
                return PY_NULL;
            }
        };
        if let Some(v) = value {
            return v;
        }
    }
    // Globals miss: `_load_global` (pyopcode.py) falls back to
    // `self.get_builtin().getdictvalue(space, varname)`.  Go through the
    // accessor, never the raw slot: the constructor (pyframe.py) assigns
    // `self.builtin` only under `honor__builtins__`, so with the option off
    // (`baseobjspace::HONOR_BUILTINS`) the field stays unset and
    // `get_builtin` answers `space.builtin` instead.  A
    // frame the JIT materialized from a virtual never wrote the slot, so a
    // raw read there resolves no builtin at all and every builtin name
    // raises NameError.  The accessor still prefers the picked module when a
    // frame does carry one, so a mid-execution `__builtins__` rebind is
    // honoured exactly as before.
    let frame = frame_anchor.live();
    let w_builtin = if !frame.is_null() {
        unsafe { (*frame).get_builtin() }
    } else {
        std::ptr::null_mut()
    };
    if !w_globals.is_null() && unsafe { pyre_object::dictmultiobject::is_module_dict(w_globals) } {
        if let Some(v) =
            unsafe { crate::eval::load_global_via_cache_extern(w_globals, w_builtin, name) }
        {
            return v;
        }
    } else if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
        // `_load_global` builtin fallback also fires on non-module-dict
        // globals (e.g. exec/eval with a plain `dict` for globals).
        let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if !w_builtin_dict.is_null()
            && let Ok(Some(v)) = crate::baseobjspace::finditem_str(w_builtin_dict, name)
        {
            return v;
        }
    }
    PY_NULL
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_store_name_to_namespace(
    w_globals: PyObjectRef,
    name_ptr: *const u8,
    name_len: i64,
    value: PyObjectRef,
) -> i64 {
    let Some(name) = decode_name(name_ptr, name_len) else {
        return 0;
    };
    // `celldict.py STORE_GLOBAL_cached` (jitted): `space.setitem_str(
    // get_w_globals(), varname, w_newvalue)`.  As in the interpreter path,
    // only actual dict layouts may use the raw strategy call; a dict subclass
    // remains the receiver of generic mapping assignment.
    if !w_globals.is_null() {
        if unsafe { pyre_object::is_dict(w_globals) } {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(w_globals, name, value);
            }
        } else {
            let _key_roots = pyre_object::gc_roots::push_roots();
            let live = pyre_object::gc_roots::pin_roots(&[w_globals, value]);
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
            if let Err(mut error) = crate::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(live),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(live + 1),
            ) {
                jit_publish_exception(error.to_exc_object());
            }
        }
    }
    0
}

type JitFunctionCaller = extern "C" fn(
    frame_ptr: *const crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: *const PyObjectRef,
    nargs: i64,
) -> i64;

static JIT_FUNCTION_CALLER: OnceLock<JitFunctionCaller> = OnceLock::new();

pub fn register_jit_function_caller(caller: JitFunctionCaller) {
    let _ = JIT_FUNCTION_CALLER.set(caller);
}

type JitExcRaiser = extern "C" fn(value: i64);

static JIT_EXC_RAISER: OnceLock<JitExcRaiser> = OnceLock::new();

pub fn register_jit_exc_raiser(raiser: JitExcRaiser) {
    let _ = JIT_EXC_RAISER.set(raiser);
}

type JitExcClearer = extern "C" fn();

static JIT_EXC_CLEARER: OnceLock<JitExcClearer> = OnceLock::new();

pub fn register_jit_exc_clearer(clearer: JitExcClearer) {
    let _ = JIT_EXC_CLEARER.set(clearer);
}

#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
fn publish_sandbox_stub(fnname: &'static str) {
    let _ = jit_publish_residual_error(crate::host_seam::stub(fnname));
}

/// Install the residual publisher `sandbox_stub_publish` calls
/// (`jit_publish_residual_error` of `host_seam::stub`).
pub fn register_sandbox_stub_publisher() {
    #[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
    {
        majit_rlib::rffi::register_sandbox_stub_publisher(publish_sandbox_stub);
    }
}

/// Drop `BH_LAST_EXC_VALUE` and the backend pos_exception / pos_exc_value cells
/// that residual helpers fill.  Upstream has no separate clearing routine:
/// `grab_exc_value` consumes the exception on the guard exit.  Pyre's
/// interpreter-side `except` handler and its compiled-loop entry do not pass
/// through that consumer, so they clear both carriers here instead, or a
/// previous raise would be re-delivered by the next `GUARD_NO_EXCEPTION`.
#[inline]
pub(crate) fn jit_clear_published_exception() {
    if let Some(clearer) = JIT_EXC_CLEARER.get() {
        clearer();
    }
}

/// llmodel.py _store_exception: publish `exc_obj` into the
/// backend's pos_exception/pos_exc_value cells so the residual call's
/// GuardNoException sees it and side-exits into the handler. Mirrors
/// `jit_call_user_function_from_frame` (call_jit.rs). MUST NOT use
/// a side-channel TLS slot — that path is drained before the guard
/// machinery runs and would bypass try/except. The call helpers return
/// garbage on Err; resume data hands control to the except block.
#[inline]
pub(crate) fn jit_publish_exception(exc_obj: PyObjectRef) {
    if exc_obj != PY_NULL
        && let Some(raiser) = JIT_EXC_RAISER.get()
    {
        raiser(exc_obj as i64);
    }
}

/// Publish a residual-call error through both exception channels and return
/// the one-word sentinel required by the residual-call ABI.
#[inline]
pub fn jit_publish_residual_error(mut error: PyError) -> i64 {
    let exc_obj = error.to_exc_object();
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|cell| cell.set(exc_obj as i64));
    jit_publish_exception(exc_obj);
    0
}

/// [`jit_publish_residual_error`] for a residual whose result is an object:
/// the null reference is the garbage result the following
/// `GuardNoException` never reads.
pub fn jit_publish_residual_error_ref(error: PyError) -> PyObjectRef {
    jit_publish_residual_error(error);
    PY_NULL
}

impl majit_ir::helper_fnaddr::ResidualError for PyError {
    #[inline]
    fn publish_residual(self) {
        let _ = jit_publish_residual_error(self);
    }
}

/// Widest `jit_call_known_builtin_N` published by [`known_builtin_call_helper`],
/// plus the leading code slot.
const MAX_KNOWN_BUILTIN_ARGS: usize = 9;

fn call_builtin_with_args(callable: PyObjectRef, args: &[PyObjectRef]) -> i64 {
    debug_assert!(args.len() <= MAX_KNOWN_BUILTIN_ARGS);
    let code = unsafe { crate::getcode(callable) } as PyObjectRef;
    // `args` is the compiled call's own argument array; no root walker updates
    // it, and `builtin_code_call` roots nothing of its own — the direct call
    // sites hand it an already-live slice.  Publish `[code, args...]` for the
    // duration of the body and read the call arguments back from it, the same
    // frame `call_builtin_code_positional` builds for the interpreter entry.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = _roots.base();
    let _ = _roots.pin_root(code);
    for &arg in args {
        let _ = _roots.pin_root(arg);
    }
    let mut rooted = [PY_NULL; MAX_KNOWN_BUILTIN_ARGS];
    for (index, slot) in rooted[..args.len()].iter_mut().enumerate() {
        *slot = _roots.get(root_base + 1 + index);
    }
    match unsafe { crate::builtin_code_call(_roots.get(root_base), &rooted[..args.len()]) } {
        Ok(result) => result as i64,
        Err(mut e) => {
            jit_publish_exception(e.to_exc_object());
            0 // garbage — GuardNoException will fire
        }
    }
}

fn jit_call_user_function_with_args(
    frame_ptr: *const crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> i64 {
    let Some(caller) = JIT_FUNCTION_CALLER.get().copied() else {
        let code_ptr = unsafe { function_get_code(callable) };
        panic!("jit function caller bridge is not installed for code_ptr={code_ptr:p}");
    };
    caller(frame_ptr, callable, args.as_ptr(), args.len() as i64)
}

fn call_callable_with_args(
    frame_ptr: *const crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> i64 {
    let _ = frame_ptr;
    match crate::call::call_function_impl_result(callable, args) {
        Ok(result) => result as i64,
        Err(mut err) => {
            jit_publish_exception(err.to_exc_object());
            0 // garbage — GuardNoException will fire
        }
    }
}

macro_rules! define_callable_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(
            frame_ptr: *const crate::pyframe::PyFrame,
            callable: PyObjectRef
            $(, $arg: PyObjectRef)*
        ) -> i64 {
            call_callable_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_builtin_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(callable: PyObjectRef $(, $arg: PyObjectRef)*) -> i64 {
            call_builtin_with_args(callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_function_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(
            frame_ptr: *const crate::pyframe::PyFrame,
            callable: PyObjectRef
            $(, $arg: PyObjectRef)*
        ) -> i64 {
            jit_call_user_function_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_flat_ref_helper {
    ($inner:ident, $name:ident $(, $arg:ident)*) => {
        pub extern "C" fn $name($($arg: PyObjectRef),*) -> PyObjectRef {
            $inner(&[$($arg),*])
        }
    };
}

define_callable_call_helper!(jit_call_callable_0);
define_callable_call_helper!(jit_call_callable_1, arg0);
define_callable_call_helper!(jit_call_callable_2, arg0, arg1);
define_callable_call_helper!(jit_call_callable_3, arg0, arg1, arg2);
define_callable_call_helper!(jit_call_callable_4, arg0, arg1, arg2, arg3);
define_callable_call_helper!(jit_call_callable_5, arg0, arg1, arg2, arg3, arg4);
define_callable_call_helper!(jit_call_callable_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_callable_call_helper!(
    jit_call_callable_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_callable_call_helper!(
    jit_call_callable_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_builtin_call_helper!(jit_call_known_builtin_0);
define_known_builtin_call_helper!(jit_call_known_builtin_1, arg0);
define_known_builtin_call_helper!(jit_call_known_builtin_2, arg0, arg1);
define_known_builtin_call_helper!(jit_call_known_builtin_3, arg0, arg1, arg2);
define_known_builtin_call_helper!(jit_call_known_builtin_4, arg0, arg1, arg2, arg3);
define_known_builtin_call_helper!(jit_call_known_builtin_5, arg0, arg1, arg2, arg3, arg4);
define_known_builtin_call_helper!(jit_call_known_builtin_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_known_builtin_call_helper!(
    jit_call_known_builtin_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_builtin_call_helper!(
    jit_call_known_builtin_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_function_call_helper!(jit_call_known_function_0);
define_known_function_call_helper!(jit_call_known_function_1, arg0);
define_known_function_call_helper!(jit_call_known_function_2, arg0, arg1);
define_known_function_call_helper!(jit_call_known_function_3, arg0, arg1, arg2);
define_known_function_call_helper!(jit_call_known_function_4, arg0, arg1, arg2, arg3);
define_known_function_call_helper!(jit_call_known_function_5, arg0, arg1, arg2, arg3, arg4);
define_known_function_call_helper!(
    jit_call_known_function_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_known_function_call_helper!(
    jit_call_known_function_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_function_call_helper!(
    jit_call_known_function_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

/// Which arm a callable dispatches through (`baseobjspace.py`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallableKind {
    /// A `BuiltinCode` carrier — `call_args_and_c_profile` / native code.
    Builtin,
    /// A `PyCode` carrier — an interpreted user function.
    User,
}

/// Name the arm `callable` dispatches through, or report it as not
/// callable.
///
/// The caller branches on the answer with the two arms written out
/// (`baseobjspace.py`); handing them in as closures instead puts an
/// `FnOnce::call_once` in front of every dispatch, and a closure has no
/// lifted counterpart — RPython spells this as a plain conditional.
pub fn classify_callable(mut callable: PyObjectRef) -> Result<CallableKind, PyError> {
    // Re-raise an error `park_jit_pending_error` parked (`unpackiterable_driver`)
    // before dispatching.  A fresh stack check is performed only when a Python
    // frame is entered (`PyFrame.execute_frame.insert_stack_check_here`);
    // builtin dispatch itself is not a recursive frame entry.
    // `drain_jit_pending_exception` materialises a GC `PyError` handle
    // (`from_exc_object` / `make_pyerror`); pin `callable` across it the way
    // `PyFrame.execute_frame` pins `operr` across `stack_check`.
    pyre_object::with_roots!(callable => crate::stack_check::drain_jit_pending_exception())?;
    unsafe {
        if crate::is_function_carrier(callable) {
            // All callables are Function objects. Check code type to distinguish
            // builtins (BuiltinCode) from user functions (PyCode).
            let code = crate::getcode(callable);
            if is_builtin_code(code as PyObjectRef) {
                Ok(CallableKind::Builtin)
            } else {
                Ok(CallableKind::User)
            }
        } else {
            let message = if callable.is_null() {
                "<NULL> object is not callable (interpreter received a null callable)".to_string()
            } else {
                let type_name = crate::typedef::r#type(callable)
                    .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()) })
                    .unwrap_or_else(|| unsafe { (*(*callable).ob_type).name });
                format!("'{type_name}' object is not callable")
            };
            Err(PyError::type_error(message))
        }
    }
}

pub fn binary_op_tag(op: BinaryOperator) -> Option<i64> {
    Some(match op {
        BinaryOperator::Add => 0,
        BinaryOperator::Subtract => 1,
        BinaryOperator::Multiply => 2,
        BinaryOperator::FloorDivide => 3,
        BinaryOperator::Remainder => 4,
        BinaryOperator::TrueDivide => 5,
        BinaryOperator::Subscr => 6,
        BinaryOperator::Power => 7,
        BinaryOperator::Lshift => 8,
        BinaryOperator::Rshift => 9,
        BinaryOperator::And => 10,
        BinaryOperator::Or => 11,
        BinaryOperator::Xor => 12,
        // In-place variants get distinct tags (13-24) so the residual
        // dispatch consults the in-place special (`__iadd__` etc.) instead
        // of collapsing to the plain binary op.
        BinaryOperator::InplaceAdd => 13,
        BinaryOperator::InplaceSubtract => 14,
        BinaryOperator::InplaceMultiply => 15,
        BinaryOperator::InplaceFloorDivide => 16,
        BinaryOperator::InplaceRemainder => 17,
        BinaryOperator::InplaceTrueDivide => 18,
        BinaryOperator::InplacePower => 19,
        BinaryOperator::InplaceLshift => 20,
        BinaryOperator::InplaceRshift => 21,
        BinaryOperator::InplaceAnd => 22,
        BinaryOperator::InplaceOr => 23,
        BinaryOperator::InplaceXor => 24,
        BinaryOperator::MatrixMultiply => 25,
        BinaryOperator::InplaceMatrixMultiply => 26,
    })
}

/// Reverse of binary_op_tag: tag (0-24) → BinaryOperator.
/// The blackhole interpreter receives the compact tag from the codewriter
/// and needs to recover the original operator for binary_value dispatch.
pub fn binary_op_from_tag(tag: i64) -> Option<BinaryOperator> {
    Some(match tag {
        0 => BinaryOperator::Add,
        1 => BinaryOperator::Subtract,
        2 => BinaryOperator::Multiply,
        3 => BinaryOperator::FloorDivide,
        4 => BinaryOperator::Remainder,
        5 => BinaryOperator::TrueDivide,
        6 => BinaryOperator::Subscr,
        7 => BinaryOperator::Power,
        8 => BinaryOperator::Lshift,
        9 => BinaryOperator::Rshift,
        10 => BinaryOperator::And,
        11 => BinaryOperator::Or,
        12 => BinaryOperator::Xor,
        13 => BinaryOperator::InplaceAdd,
        14 => BinaryOperator::InplaceSubtract,
        15 => BinaryOperator::InplaceMultiply,
        16 => BinaryOperator::InplaceFloorDivide,
        17 => BinaryOperator::InplaceRemainder,
        18 => BinaryOperator::InplaceTrueDivide,
        19 => BinaryOperator::InplacePower,
        20 => BinaryOperator::InplaceLshift,
        21 => BinaryOperator::InplaceRshift,
        22 => BinaryOperator::InplaceAnd,
        23 => BinaryOperator::InplaceOr,
        24 => BinaryOperator::InplaceXor,
        25 => BinaryOperator::MatrixMultiply,
        26 => BinaryOperator::InplaceMatrixMultiply,
        _ => return None,
    })
}

/// True for the augmented-assignment (`NB_INPLACE_*`) binary-op tags (13..=24).
/// A mutable receiver (`list`/`bytearray`/`set`/`dict`/`array`) is mutated in
/// place by these and returns `self`; an immutable one (`int`/`float`/`str`/
/// `tuple`) returns a fresh object. Tags 0..=12 are the non-in-place operators.
pub fn binary_op_tag_is_inplace(tag: i64) -> bool {
    (13..=24).contains(&tag)
}

/// COMPARE helper tags outside the 0..=5 rich-comparison range
/// [`compare_op_from_tag`] decodes.  `contains` / `not_contains` are
/// CONTAINS_OP; `is` / `is_not` are IS_OP (`space.is_w`);
/// `ISINSTANCE_OP_TAG` is CHECK_EXC_MATCH.
pub const COMPARE_OP_CONTAINS: i64 = 6;
pub const COMPARE_OP_NOT_CONTAINS: i64 = 7;
pub const COMPARE_OP_IS: i64 = 8;
pub const COMPARE_OP_IS_NOT: i64 = 9;

/// The `compare_fn` tag `CHECK_EXC_MATCH` passes for its
/// `isinstance(exc, match_type)` check.  It is outside the 0..=5 range
/// [`compare_op_from_tag`] decodes, so a `CompareOp` residual carrying it is
/// the exception-match shape and nothing else.
pub const ISINSTANCE_OP_TAG: i64 = 10;

pub fn compare_op_tag_is_identity(tag: i64) -> bool {
    tag == COMPARE_OP_IS || tag == COMPARE_OP_IS_NOT
}

/// `pyopcode.py IS_OP`: `space.is_w(w_1, w_2)`, inverted when `invert`
/// is non-zero, returned as `space.w_True` / `space.w_False`.
///
/// IS_OP lowers to an `inline_call` of this body, so the JIT traces
/// `is_w`'s per-type identity rather than a may-force residual.
#[inline(never)]
pub fn is_op(w_1: PyObjectRef, w_2: PyObjectRef, invert: i64) -> PyObjectRef {
    let res = crate::baseobjspace::is_w(w_1, w_2);
    pyre_object::w_bool_from(if invert != 0 { !res } else { res })
}

pub fn compare_op_tag_is_contains(tag: i64) -> bool {
    tag == COMPARE_OP_CONTAINS || tag == COMPARE_OP_NOT_CONTAINS
}

pub fn compare_op_tag(op: ComparisonOperator) -> i64 {
    match op {
        ComparisonOperator::Less => 0,
        ComparisonOperator::LessOrEqual => 1,
        ComparisonOperator::Greater => 2,
        ComparisonOperator::GreaterOrEqual => 3,
        ComparisonOperator::Equal => 4,
        ComparisonOperator::NotEqual => 5,
    }
}

/// Reverse of compare_op_tag: tag (0-5) → ComparisonOperator.
pub fn compare_op_from_tag(tag: i64) -> Option<ComparisonOperator> {
    Some(match tag {
        0 => ComparisonOperator::Less,
        1 => ComparisonOperator::LessOrEqual,
        2 => ComparisonOperator::Greater,
        3 => ComparisonOperator::GreaterOrEqual,
        4 => ComparisonOperator::Equal,
        5 => ComparisonOperator::NotEqual,
        _ => return None,
    })
}

/// One-word residual ABI for [`build_list_from_refs`].
///
/// The items argument is a length-prefixed `GcTypedArray` (`bh_newlist_from_array`).
#[majit_macros::dont_look_inside]
pub extern "C" fn build_list_from_refs_jit_abi(
    array: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let items = pyre_object::gc_roots::gcarray_ref_items(array);
    build_list_from_refs(&items)
}

pub fn build_list_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    // BUILD_LIST pops the elements first, so they live only in this slice
    // while `w_list_new` allocates. Pin them and hand the constructor the
    // rewritten slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(items);
    let live: Vec<PyObjectRef> = (0..items.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(base + i))
        .collect();
    w_list_new(live)
}

/// One-word residual ABI for [`build_tuple_from_refs`].
///
/// Same length-prefixed array word as [`build_list_from_refs_jit_abi`].
#[majit_macros::dont_look_inside]
pub extern "C" fn build_tuple_from_refs_jit_abi(
    array: *const pyre_object::object_array::GcTypedArray,
) -> PyObjectRef {
    let items = pyre_object::gc_roots::gcarray_ref_items(array);
    build_tuple_from_refs(&items)
}

pub fn build_tuple_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(items);
    // Fixed-arity helpers top out at 8; longer tuples fall back to a Vec.
    const STACK_CAP: usize = 8;
    let len = items.len();
    let mut stack_buf = [PY_NULL; STACK_CAP];
    let heap_buf: Vec<PyObjectRef>;
    let live: &[PyObjectRef] = if len <= STACK_CAP {
        for i in 0..len {
            stack_buf[i] = pyre_object::gc_roots::shadow_stack_get(base + i);
        }
        &stack_buf[..len]
    } else {
        heap_buf = (0..len)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(base + i))
            .collect();
        &heap_buf
    };
    w_tuple_new_from_slice(live)
}

/// BUILD_MAP evaluation, shared by the interpreter (`build_map`) and the JIT
/// residual (`bh_build_map_from_array`).  Stores each `[key, value]` pair
/// through the checked dict setitem, which hashes the key (may run user
/// `__hash__` / `__eq__`); an unhashable key raises, so — like
/// `build_set_from_refs` — this is fallible.
/// `pyopcode.py BUILD_MAP` unrolls the pair-store loop.
#[majit_macros::unroll_safe]
pub fn build_map_from_refs(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Each store hashes its key through `__hash__`, so every turn is a
    // collection point.  The fresh dict and `items` — for the JIT entries a
    // copy of the compiled call's argument registers — are native locals no
    // root walker updates, so publish the whole set and read every operand
    // back.  `build_set_from_refs` reaches the same bracket through
    // `builtin_set_add_items_impl`.
    let _roots = pyre_object::gc_roots::push_roots();
    let items_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in items {
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict_new());
    let dict = || pyre_object::gc_roots::shadow_stack_get(dict_slot);
    for index in (0..items.len() - items.len() % 2).step_by(2) {
        let key = pyre_object::gc_roots::shadow_stack_get(items_base + index);
        let value = pyre_object::gc_roots::shadow_stack_get(items_base + index + 1);
        unsafe {
            w_dict_store_checked(dict(), key, value).map_err(|_| {
                crate::baseobjspace::take_pending_dict_key_error(
                    pyre_object::gc_roots::shadow_stack_get(items_base + index),
                )
            })?;
        }
    }
    Ok(dict())
}

/// BUILD_SET evaluation, shared by the JIT residual (`bh_build_set_from_array`).
/// Builds a set from the forced element array; element hashing may run user
/// `__hash__` / `__eq__` and a non-hashable element raises `TypeError`, so —
/// unlike `build_map_from_refs` / `build_tuple_from_refs` — this is fallible.
pub fn build_set_from_refs(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::builtins::builtin_set_from_items(items)
}

/// BUILD_STRING residual (`bh_build_string_from_array`): concatenates `parts`
/// (f-string fragments, in bottom-to-top / source order) into a single `str`
/// via `Utf8StringBuilder`.  `pyopcode.py BUILD_STRING` reads each fragment
/// through `space.utf8_len_w`.  A non-`str` (exact `str` or a `str` subclass,
/// the `isinstance_str_w` check `expect_str` uses) raises
/// `TypeError("sequence item {i}: expected str instance, {typename} found")`
/// at the first such fragment; `i` is its 0-based position in `parts`.  The
/// `str` arm is the `Utf8StringBuilder.append_utf8` + `newutf8` path and runs
/// no user code.
pub fn build_string_from_refs(parts: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut builder = pyre_object::rutf8::Utf8StringBuilder::new(0);
    for (i, part) in parts.iter().enumerate() {
        unsafe {
            if !crate::baseobjspace::isinstance_str_w(*part) {
                return Err(crate::PyError::type_error(format!(
                    "sequence item {i}: expected str instance, {} found",
                    crate::baseobjspace::object_functionstr_type_name(*part)
                )));
            }
            builder.append_utf8(
                pyre_object::unicodeobject::w_str_storage(*part),
                pyre_object::unicodeobject::w_str_len(*part) as i64,
            );
        }
    }
    Ok(pyre_object::unicodeobject::w_str_from_storage_and_length(
        builder.build(),
        builder.getlength() as usize,
    ))
}

/// CONVERT_VALUE conversion code, shared by the interpreter and the JIT
/// codewriter so the `convert_value` residual carries a stable integer the
/// C ABI can pass (`ConvertValueOparg` can't cross the residual boundary).
/// `0 = Str`, `1 = Repr`, `2 = Ascii`, `3 = None` (`None` behaves as `Str`).
pub fn convert_value_code(conv: ConvertValueOparg) -> i64 {
    match conv {
        ConvertValueOparg::Str => 0,
        ConvertValueOparg::Repr => 1,
        ConvertValueOparg::Ascii => 2,
        ConvertValueOparg::None => 3,
    }
}

/// CONVERT_VALUE evaluation, shared by the interpreter (`convert_value`) and
/// the JIT residual (`bh_convert_value_fn`).  `conv` is a
/// [`convert_value_code`] integer.
///
/// An exact `str` under `Str` / `None` returns itself (`descr_str`). An
/// exact builtin `int` goes through `descr_str` for every code: its
/// `descr_repr` is the same body, and `ascii_from_object` returns that
/// text. An instance under `Str` / `None` returns the app-level `__str__`
/// object (`DescrOperation.str`), or the `__repr__` object when that
/// lookup is `object`'s (`descr__str__`). An instance under `Repr` returns
/// the app-level `__repr__` object. `Ascii` and every other shape go
/// through [`convert_value_slow`], which keeps a lone surrogate in WTF-8.
/// A user `__str__` / `__repr__` may run Python → fallible.
///
/// CONVERT_VALUE lowers to an `inline_call` of this body, so the JIT traces
/// the exact `str` / `int` arms and keeps every other shape behind the
/// `dont_look_inside` [`convert_value_slow`].
pub fn convert_value(value: PyObjectRef, conv: i64) -> Result<PyObjectRef, crate::PyError> {
    // `descr_str` (unicodeobject.py) returns `self` for an exact `str`.
    if (conv == 0 || conv == 3)
        && unsafe { pyre_object::is_exact_type(value, &pyre_object::STR_TYPE) }
    {
        return Ok(value);
    }
    // `intobject.py descr_repr` is also its `descr_str`, and its digits are
    // ASCII, so `ascii_from_object` (unicodeobject.py) returns the repr as is.
    // Machine int is the storage `ob_type`; a subclass keeps that `ob_type`
    // and may override `__str__` / `__repr__`.
    if unsafe {
        pyre_object::py_type_check(value, &pyre_object::INT_TYPE)
            && pyre_object::is_exact_builtin_instance(value)
    } {
        return Ok(unsafe { pyre_object::descr_str(value) });
    }
    convert_value_slow(value, conv)
}

/// [`convert_value`] for every shape its exact `str` / `int` arms do not
/// answer.
#[majit_macros::dont_look_inside]
fn convert_value_slow(mut value: PyObjectRef, conv: i64) -> Result<PyObjectRef, crate::PyError> {
    // `DescrOperation.str` returns the app-level `__str__` object, and
    // `descr__str__` calls `__repr__` and returns that object. A tagged
    // immediate has no `w_class`; `py_str_wtf8` formats it below.
    let tagged =
        pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(value);
    if !tagged && !value.is_null() && unsafe { pyre_object::is_instance(value) } {
        let str_conv = conv == 0 || conv == 3;
        if str_conv {
            if let Some(result) = pyre_object::with_roots!(value =>
                unsafe { crate::display::try_call_dunder_obj_above_object(value, "__str__") }
            )? {
                return Ok(result);
            }
        }
        if str_conv || conv == 1 {
            if let Some(result) = pyre_object::with_roots!(value =>
                unsafe { crate::display::try_call_dunder_obj(value, "__repr__") }
            )? {
                return Ok(result);
            }
        }
    }
    if conv == 0 || conv == 3 {
        // `descr_str` (unicodeobject.py) converts anything but an exact `str`
        // — a subclass included — to a fresh base `str`.
        let w = unsafe { crate::py_str_wtf8(value)? };
        return Ok(pyre_object::w_str_from_wtf8_managed(w));
    }
    if conv == 1 {
        return Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
            crate::py_repr_wtf8(value)?
        }));
    }
    let s = match conv {
        2 => rustpython_wtf8::Wtf8Buf::from_string(crate::builtins::py_ascii(value)?),
        _ => unsafe { crate::py_str_wtf8(value)? },
    };
    Ok(pyre_object::w_str_from_wtf8_managed(s))
}

/// FORMAT_SIMPLE / FORMAT_WITH_SPEC evaluation, shared by the interpreter
/// (`format_simple` / `format_with_spec`) and the JIT residuals
/// (`bh_format_simple_fn` / `bh_format_with_spec_fn`).  Formats `value`
/// through `format_value_dispatch` (user `__format__` may run Python →
/// fallible); a `PY_NULL` or non-`str` `spec` reads as the empty spec
/// (`str(value)`), matching `format_simple`.
pub fn format_value(value: PyObjectRef, spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    // The spec crosses to `__format__` as the object the opcode pushed, the
    // way `FORMAT_VALUE` hands `w_spec` to `space.format` (pyopcode.py).
    // Its fill may be any code point, and nothing here reads it, so
    // `f"{x:\ud800>4}"` keeps the lone surrogate it was given.
    crate::type_methods::format_w(value, spec)
}

/// STORE_SLICE evaluation (`obj[start:stop] = value`): builds a `slice`
/// object and dispatches through `setitem`, mirroring `PyObject_SetItem`
/// with a slice key. The dual of `binary_slice_values`; a `None` step.
pub fn store_slice_values(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    // `w_slice_new` allocates and holds neither the receiver nor the stored
    // value, so a `list` / `dict` in either relocates while the slice is built.
    // The bounds it does hold are pinned by the constructor itself; the fresh
    // slice reaches the heap only through `setitem`, so it takes one pin.
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    let value_slot = obj_slot + 1;
    roots.publish(&[obj, value]);
    roots.normalize(obj_slot, 2);
    let slice = pyre_object::w_slice_new(start, stop, pyre_object::w_none());
    let slice = roots.pin_root(slice);
    crate::baseobjspace::setitem(roots.get(obj_slot), slice, roots.get(value_slot))?;
    Ok(())
}

/// BINARY_SLICE evaluation, shared by the interpreter (`binary_slice`)
/// and the JIT residual (`bh_binary_slice_fn`): returns `obj[start:stop]`.
/// `list` / `str` / `tuple` slice on element (code-point for `str`)
/// boundaries; everything else (`bytes`, `bytearray`, instances with
/// `__getitem__`) falls back to a `slice` object dispatched through
/// `getitem`. A `None` start/stop defaults to `0` / `len`.
///
/// Residual wrapper around [`binary_slice_values_inner`]. The inner owns
/// the operand-root bracket. This extra scope pins the three operands
/// across the inner call so they are not live unbracketed `PyObjectRef`s
/// at that collecting hop.
#[inline(never)]
pub fn binary_slice_values(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::publish_roots(&[obj, start, stop]);
    pyre_object::gc_roots::normalize_roots(obj_slot, 3);
    binary_slice_values_inner(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        pyre_object::gc_roots::shadow_stack_get(obj_slot + 1),
        pyre_object::gc_roots::shadow_stack_get(obj_slot + 2),
    )
}

/// Body of [`binary_slice_values`]. The public function stays
/// `inline(never)` so this split has its own graph; the residual C ABI
/// wrapper calls this directly so compiled traces do not pay the
/// wrapper's hop. `inline(never)` is load-bearing: rustc otherwise
/// folds this body into the wrapper and the codewriter never mints the
/// graph a trace descends (`specialize.rs try_walker_orthodox_binary_slice`).
///
/// The `push_roots` bracket lives here so the native path publishes the
/// three operands as one livevar set before any safepoint (`pin_roots`).
/// `shadow_stack_erase` scalar-replaces that Open/Publish/Close
/// (`w_range_iter_one_arg_new`), so the descended jitcode has no residual
/// root calls. Integer-list copy then walks `ll_listslice_inner`; that
/// helper's `new_int_list` bracket is erased the same way, leaving
/// `new_array` + `OS_ARRAYCOPY` in the compiled loop.
#[inline(never)]
pub(crate) fn binary_slice_values_inner(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    unsafe {
        // `eval_slice_index` may run a user `__index__`, and every arm below
        // allocates: any of the three operands relocates across either.  They
        // are published as one livevar set, and each consumer that follows a
        // collection point reads its operand back off the root slot — the
        // receiver included, since its type says nothing about whether it moves.
        // Sequential `pin_root` would query after the first write and leave
        // the later words unpublished at that safepoint (`pin_roots`).
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let start_slot = obj_slot + 1;
        let stop_slot = obj_slot + 2;
        pyre_object::gc_roots::publish_roots(&[obj, start, stop]);
        pyre_object::gc_roots::normalize_roots(obj_slot, 3);
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let start = pyre_object::gc_roots::shadow_stack_get(start_slot);
        let stop = pyre_object::gc_roots::shadow_stack_get(stop_slot);
        if pyre_object::is_list(obj) {
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let stop = pyre_object::gc_roots::shadow_stack_get(stop_slot);
            let raw_e = if pyre_object::is_none(stop) {
                None
            } else {
                Some(crate::sliceobject::eval_slice_index(stop)?)
            };
            // The length comes after both bounds have been converted: a
            // bound's own `__index__` may have resized this very list, and it
            // is what the omitted `stop` defaults to and what a negative bound
            // folds against.  Read off the root slot, since the same call is
            // what may have moved the list.
            let len =
                pyre_object::w_list_len(pyre_object::gc_roots::shadow_stack_get(obj_slot)) as i64;
            let e = raw_e.unwrap_or(len);
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
            // Bounds are normalised here, outside the copy: a user
            // `__index__` may have resized this list, and the operand roots
            // above are what keep it alive across that call. The lock covers
            // only the copy, matching `list_iter_descr_next`: unpublished
            // `gil_ready` is lock-free; once threads exist, `ll_listslice`
            // acquires, copies, and releases as one residual.
            let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
            if pyre_object::gil_ready::gil_ready_word() != 0 {
                return Ok(pyre_object::listobject::ll_listslice(obj, s, e));
            }
            return Ok(pyre_object::listobject::ll_listslice_inner(obj, s, e));
        }
        if pyre_object::is_str(obj) {
            // Slice on code-point boundaries over the WTF-8 view, so a
            // surrogate-bearing or multi-byte string slices correctly.  The
            // bounds convert through the object's own index storage
            // (`_index_to_byte`, unicodeobject.py) and the count is the
            // stored `_length`: deriving either by walking the payload would
            // make a one-character slice cost the whole string.  A bound equal
            // to the count resolves to the end of the buffer, which is what
            // the `stop` default asks for.
            let len = pyre_object::w_str_len(obj) as i64;
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let stop = pyre_object::gc_roots::shadow_stack_get(stop_slot);
            let e = if pyre_object::is_none(stop) {
                len
            } else {
                crate::sliceobject::eval_slice_index(stop)?
            };
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = (if e < 0 { (len + e).max(0) } else { e.min(len) } as usize).max(s);
            // `_utf8[start:end]` through `ll_stringslice_startstop`
            // (`rstr.py`); wrap with `space.newutf8`. A whole-string slice
            // of an exact `str` shares the operand (`unicodeobject.py`
            // `_getitem_slice`). Take storage and index table after the last
            // conversion — off the receiver that conversion may have moved.
            let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
            if s == 0
                && e == pyre_object::w_str_len(obj)
                && pyre_object::pyobject::is_exact_type(obj, &pyre_object::pyobject::STR_TYPE)
            {
                return Ok(obj);
            }
            let start_b = pyre_object::w_str_index_to_byte(obj, s) as i64;
            let stop_b = pyre_object::w_str_index_to_byte(obj, e) as i64;
            let storage = unsafe { pyre_object::unicodeobject::w_str_storage(obj) };
            let piece =
                pyre_object::lowlevel_string::ll_stringslice_startstop(storage, start_b, stop_b);
            return Ok(pyre_object::unicodeobject::w_str_from_storage_and_length(
                piece,
                e - s,
            ));
        }
        if pyre_object::is_tuple(obj) {
            // A tuple's length is fixed, so it is read before the bounds
            // convert; the receiver word is not carried across one, because
            // `eval_slice_index` may run a user `__index__` whose allocation
            // moves the tuple.
            let len = pyre_object::w_tuple_len(obj) as i64;
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let stop = pyre_object::gc_roots::shadow_stack_get(stop_slot);
            let e = if pyre_object::is_none(stop) {
                len
            } else {
                crate::sliceobject::eval_slice_index(stop)?
            };
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
            // `ll_listslice_startstop`: `ll_newlist` plus `ll_arraycopy`
            // (`tuple_getslice_step1_block`). Not a host `Vec`.
            return Ok(unsafe {
                crate::baseobjspace::tuple_slice_startstop(
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    s,
                    e,
                )
            });
        }
        // Fall back to slice(start, stop) → getitem dispatch.
        // Handles bytes, bytearray, instances with __getitem__, etc.
        binary_slice_getitem_fallback(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(start_slot),
            pyre_object::gc_roots::shadow_stack_get(stop_slot),
        )
    }
}

/// [`binary_slice_values_inner`] for every type the list / str / tuple
/// arms do not slice. `space.getitem` over `space.newslice`.
///
/// `w_slice_new` brackets its malloc with `push_roots`
/// (`gct_fv_gc_malloc`). Residualising that RAII `RootScope` inside a
/// look-inside walk drops the shadow stack at the residual return, and
/// the inlined write of `W_SliceObject` then reads `start` / `stop` /
/// `step` from the wrong slots. Keep the whole fallback opaque, matching
/// `convert_value_slow`.
#[inline(never)]
#[majit_macros::dont_look_inside]
fn binary_slice_getitem_fallback(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let start = pyre_object::gc_roots::pin_root(start);
    let stop = pyre_object::gc_roots::pin_root(stop);
    let slice_obj = pyre_object::sliceobject::w_slice_new(start, stop, pyre_object::w_none());
    let slice_obj = pyre_object::gc_roots::pin_root(slice_obj);
    crate::baseobjspace::getitem(pyre_object::gc_roots::shadow_stack_get(obj_slot), slice_obj)
}

fn build_list_from_args(args: &[PyObjectRef]) -> PyObjectRef {
    build_list_from_refs(args)
}

fn build_tuple_from_args(args: &[PyObjectRef]) -> PyObjectRef {
    build_tuple_from_refs(args)
}

fn build_map_from_args(args: &[PyObjectRef]) -> PyObjectRef {
    // Legacy fixed-arity BUILD_MAP residual reached only on the blackhole /
    // deopt path (the codewriter lowers BUILD_MAP through the array-based
    // `bh_build_map_from_array`).  An unhashable key raises; signal it through
    // `BH_LAST_EXC_VALUE` and return PY_NULL, like the other blackhole-only
    // residuals.
    match build_map_from_refs(args) {
        Ok(dict) => dict,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            PY_NULL
        }
    }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_list_0() -> PyObjectRef {
    w_list_new(vec![])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_tuple_0() -> PyObjectRef {
    w_tuple_new(vec![])
}

define_flat_ref_helper!(build_list_from_args, jit_build_list_1, arg0);
define_flat_ref_helper!(build_list_from_args, jit_build_list_2, arg0, arg1);
define_flat_ref_helper!(build_list_from_args, jit_build_list_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_1, arg0);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_2, arg0, arg1);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_map_from_args, jit_build_map_0);
define_flat_ref_helper!(build_map_from_args, jit_build_map_1, arg0, arg1);
define_flat_ref_helper!(build_map_from_args, jit_build_map_2, arg0, arg1, arg2, arg3);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_3,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_4,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

/// Take a helper's residual-call address with its signature checked.
///
/// The arity-to-address accessors below erase the signature, so the checked
/// publishers in `jit_fnaddr` (`ResidualSlot` / `ResidualRet`) never see it and
/// cannot reject a helper the residual ABI is unable to describe.  Passing the
/// fn item through `residual_extern_addr_N`, whose parameters and result are
/// bounded by those same traits, restores that check at the one point where
/// the type still exists: a helper with a parameter or result the residual ABI
/// cannot describe fails to compile here instead of being published and then
/// called with the wrong number of registers.
///
/// The second argument is the helper's machine-argument count, which is not
/// the Python argument count the accessor matches on: a `jit_call_callable_N`
/// carries the frame and the callable ahead of its `N` arguments, a
/// `jit_call_known_builtin_N` carries the callable, and a `jit_build_map_N`
/// takes a key and a value per pair.  Getting that count wrong is the same
/// build error: writing `3` for `jit_call_callable_2` reports a mismatched
/// fn-pointer type naming `jit_call_callable_2` at the call site.
macro_rules! residual_extern_addr_fns {
    ($($name:ident($($a:ident),*);)*) => {$(
        #[inline]
        fn $name<$($a: crate::jit_fnaddr::ResidualSlot,)* R: crate::jit_fnaddr::ResidualRet>(
            f: extern "C" fn($($a),*) -> R,
        ) -> *const () {
            f as *const ()
        }
    )*};
}

residual_extern_addr_fns! {
    residual_extern_addr_0();
    residual_extern_addr_1(A1);
    residual_extern_addr_2(A1, A2);
    residual_extern_addr_3(A1, A2, A3);
    residual_extern_addr_4(A1, A2, A3, A4);
    residual_extern_addr_5(A1, A2, A3, A4, A5);
    residual_extern_addr_6(A1, A2, A3, A4, A5, A6);
    residual_extern_addr_7(A1, A2, A3, A4, A5, A6, A7);
    residual_extern_addr_8(A1, A2, A3, A4, A5, A6, A7, A8);
    residual_extern_addr_9(A1, A2, A3, A4, A5, A6, A7, A8, A9);
    residual_extern_addr_10(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
}

macro_rules! word_fn_addr {
    ($f:ident, 0) => {
        residual_extern_addr_0($f)
    };
    ($f:ident, 1) => {
        residual_extern_addr_1($f)
    };
    ($f:ident, 2) => {
        residual_extern_addr_2($f)
    };
    ($f:ident, 3) => {
        residual_extern_addr_3($f)
    };
    ($f:ident, 4) => {
        residual_extern_addr_4($f)
    };
    ($f:ident, 5) => {
        residual_extern_addr_5($f)
    };
    ($f:ident, 6) => {
        residual_extern_addr_6($f)
    };
    ($f:ident, 7) => {
        residual_extern_addr_7($f)
    };
    ($f:ident, 8) => {
        residual_extern_addr_8($f)
    };
    ($f:ident, 9) => {
        residual_extern_addr_9($f)
    };
    ($f:ident, 10) => {
        residual_extern_addr_10($f)
    };
}

pub fn callable_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => word_fn_addr!(jit_call_callable_0, 2),
        1 => word_fn_addr!(jit_call_callable_1, 3),
        2 => word_fn_addr!(jit_call_callable_2, 4),
        3 => word_fn_addr!(jit_call_callable_3, 5),
        4 => word_fn_addr!(jit_call_callable_4, 6),
        5 => word_fn_addr!(jit_call_callable_5, 7),
        6 => word_fn_addr!(jit_call_callable_6, 8),
        7 => word_fn_addr!(jit_call_callable_7, 9),
        8 => word_fn_addr!(jit_call_callable_8, 10),
        _ => return None,
    })
}

pub fn known_builtin_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => word_fn_addr!(jit_call_known_builtin_0, 1),
        1 => word_fn_addr!(jit_call_known_builtin_1, 2),
        2 => word_fn_addr!(jit_call_known_builtin_2, 3),
        3 => word_fn_addr!(jit_call_known_builtin_3, 4),
        4 => word_fn_addr!(jit_call_known_builtin_4, 5),
        5 => word_fn_addr!(jit_call_known_builtin_5, 6),
        6 => word_fn_addr!(jit_call_known_builtin_6, 7),
        7 => word_fn_addr!(jit_call_known_builtin_7, 8),
        8 => word_fn_addr!(jit_call_known_builtin_8, 9),
        _ => return None,
    })
}

pub fn known_function_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => word_fn_addr!(jit_call_known_function_0, 2),
        1 => word_fn_addr!(jit_call_known_function_1, 3),
        2 => word_fn_addr!(jit_call_known_function_2, 4),
        3 => word_fn_addr!(jit_call_known_function_3, 5),
        4 => word_fn_addr!(jit_call_known_function_4, 6),
        5 => word_fn_addr!(jit_call_known_function_5, 7),
        6 => word_fn_addr!(jit_call_known_function_6, 8),
        7 => word_fn_addr!(jit_call_known_function_7, 9),
        8 => word_fn_addr!(jit_call_known_function_8, 10),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
pub enum FlatBuildKind {
    List,
    Tuple,
    Map,
}

pub fn list_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => word_fn_addr!(jit_build_list_0, 0),
        1 => word_fn_addr!(jit_build_list_1, 1),
        2 => word_fn_addr!(jit_build_list_2, 2),
        3 => word_fn_addr!(jit_build_list_3, 3),
        4 => word_fn_addr!(jit_build_list_4, 4),
        5 => word_fn_addr!(jit_build_list_5, 5),
        6 => word_fn_addr!(jit_build_list_6, 6),
        7 => word_fn_addr!(jit_build_list_7, 7),
        8 => word_fn_addr!(jit_build_list_8, 8),
        _ => return None,
    })
}

pub fn tuple_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => word_fn_addr!(jit_build_tuple_0, 0),
        1 => word_fn_addr!(jit_build_tuple_1, 1),
        2 => word_fn_addr!(jit_build_tuple_2, 2),
        3 => word_fn_addr!(jit_build_tuple_3, 3),
        4 => word_fn_addr!(jit_build_tuple_4, 4),
        5 => word_fn_addr!(jit_build_tuple_5, 5),
        6 => word_fn_addr!(jit_build_tuple_6, 6),
        7 => word_fn_addr!(jit_build_tuple_7, 7),
        8 => word_fn_addr!(jit_build_tuple_8, 8),
        _ => return None,
    })
}

pub fn map_build_helper(pair_count: usize) -> Option<*const ()> {
    Some(match pair_count {
        0 => word_fn_addr!(jit_build_map_0, 0),
        1 => word_fn_addr!(jit_build_map_1, 2),
        2 => word_fn_addr!(jit_build_map_2, 4),
        3 => word_fn_addr!(jit_build_map_3, 6),
        4 => word_fn_addr!(jit_build_map_4, 8),
        _ => return None,
    })
}

pub fn flat_build_helper(kind: FlatBuildKind, count: usize) -> Option<*const ()> {
    match kind {
        FlatBuildKind::List => list_build_helper(count),
        FlatBuildKind::Tuple => tuple_build_helper(count),
        FlatBuildKind::Map => map_build_helper(count),
    }
}

/// Look up an entry in a proxy-free GC module/namespace dict.
pub fn module_ns_get(ns: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, name) }
}

/// Store into a proxy-free GC module/namespace dict (successor to
/// `dict_storage_store` for migrated namespaces).
///
/// Both words already exist. `ShadowStackFrameworkGCTransformer.push_roots`
/// publishes that live set once; `pop_roots` reloads each local before setitem.
pub fn module_ns_store(ns: PyObjectRef, name: &str, value: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[ns, value]);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            pyre_object::gc_roots::shadow_stack_get(base),
            name,
            pyre_object::gc_roots::shadow_stack_get(base + 1),
        );
    }
}

pub fn module_ns_store_wtf8(ns: PyObjectRef, name: &Wtf8, value: PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(ns, name, value) }
}

pub fn module_ns_delete(ns: PyObjectRef, name: &str) -> bool {
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, name) }
}

/// WTF-8 keyed delete from a proxy-free GC module/namespace dict.
pub fn module_ns_delete_wtf8(ns: PyObjectRef, name: &Wtf8) -> bool {
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_wtf8_no_proxy(ns, name) }
}

/// Run and store `f()` only when `name` is absent from a GC namespace dict.
///
/// The value does not exist until `f` returns, so the namespace is pinned
/// first and the newborn after. `pop_roots` reloads both words before setitem.
pub fn module_ns_get_or_insert_with(ns: PyObjectRef, name: &str, f: impl FnOnce() -> PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        if pyre_object::dictmultiobject::w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            name,
        )
        .is_none()
        {
            let value = f();
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(value);
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                pyre_object::gc_roots::shadow_stack_get(ns_slot),
                name,
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
    }
}

fn type_dict_ptr(cls: PyObjectRef) -> pyre_object::gc_hook::GCREF {
    unsafe { pyre_object::w_type_get_dict_ptr(cls) }
}

pub fn type_dict_has_storage(cls: PyObjectRef) -> bool {
    !type_dict_ptr(cls).is_null()
}

/// Raw namespace read (`_getdictvalue_no_unwrapping`): the stored object,
/// which may be a `MutableCell`.  Callers that run after type creation and
/// hand the result to Python use [`type_dict_lookup`] instead.
pub fn type_dict_lookup_no_unwrapping(cls: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return None;
    }
    let value = unsafe { pyre_object::w_dict_getitem_str(dict as PyObjectRef, name)? };
    if value.is_null() { None } else { Some(value) }
}

pub fn type_dict_lookup_wtf8_no_unwrapping(cls: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return None;
    }
    let value = unsafe { pyre_object::w_dict_getitem_wtf8(dict as PyObjectRef, name)? };
    if value.is_null() { None } else { Some(value) }
}

/// `W_TypeObject.getdictvalue`: the namespace entry with a `MutableCell`
/// unwrapped.  The raw form is [`type_dict_lookup_no_unwrapping`].
pub fn type_dict_lookup(cls: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let value = type_dict_lookup_no_unwrapping(cls, name)?;
    let value = unsafe { pyre_object::celldict::unwrap_cell(value) };
    if value.is_null() { None } else { Some(value) }
}

pub fn type_dict_lookup_wtf8(cls: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    let value = type_dict_lookup_wtf8_no_unwrapping(cls, name)?;
    let value = unsafe { pyre_object::celldict::unwrap_cell(value) };
    if value.is_null() { None } else { Some(value) }
}

pub fn type_dict_contains(cls: PyObjectRef, name: &str) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_getitem_str(dict as PyObjectRef, name).is_some() }
}

pub fn type_dict_store(cls: PyObjectRef, name: &str, value: PyObjectRef) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_setitem_str_no_proxy(dict as PyObjectRef, name, value) };
    true
}

pub fn type_dict_store_wtf8(cls: PyObjectRef, name: &Wtf8, value: PyObjectRef) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_setitem_wtf8_no_proxy(dict as PyObjectRef, name, value) };
    true
}

pub fn type_dict_delete(cls: PyObjectRef, name: &str) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_delitem_str_no_proxy(dict as PyObjectRef, name) }
}

pub fn type_dict_delete_wtf8(cls: PyObjectRef, name: &Wtf8) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_delitem_wtf8_no_proxy(dict as PyObjectRef, name) }
}

pub fn sequence_len(mut seq: PyObjectRef) -> Result<usize, PyError> {
    unsafe {
        if is_tuple(seq) {
            return Ok(w_tuple_len(seq));
        }
        if is_list(seq) {
            return Ok(w_list_len(seq));
        }
        if is_str(seq) {
            // Cached code point count — surrogate-safe (avoids w_str_get_value).
            return Ok(w_str_len(seq));
        }
        // Try __len__ on instances
        if is_instance(seq)
            && let Ok(len_val) = pyre_object::with_roots!(seq => crate::baseobjspace::len(seq))
        {
            return Ok(w_int_get_value(len_val) as usize);
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            crate::error::type_name_of(seq)
        )))
    }
}

pub fn sequence_getitem(seq: PyObjectRef, index: usize) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_tuple(seq) {
            return w_tuple_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("tuple index out of range"));
        }
        if is_list(seq) {
            return w_list_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("list index out of range"));
        }
        if is_str(seq) {
            // Read the code point through the WTF-8 view so a surrogate-bearing
            // string yields its lone surrogate instead of panicking.
            return pyre_object::w_str_codepoint_at(seq, index)
                .map(|c| {
                    let mut one = Wtf8Buf::new();
                    one.push(c);
                    pyre_object::w_str_from_wtf8_managed(one)
                })
                .ok_or_else(|| PyError::type_error("string index out of range"));
        }
        // Try getitem for instances
        if is_instance(seq) {
            let _roots = pyre_object::gc_roots::push_roots();
            let seq_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(seq);
            let idx_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_int_new(index as i64));
            return crate::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(seq_slot),
                pyre_object::gc_roots::shadow_stack_get(idx_slot),
            );
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            crate::error::type_name_of(seq)
        )))
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_sequence_getitem(seq: PyObjectRef, index: i64) -> PyObjectRef {
    match sequence_getitem(seq, index as usize) {
        Ok(value) => value,
        // Return PY_NULL on out-of-bounds — the guard after this call
        // will detect the null and side-exit to the interpreter.
        // RPython: residual calls that fail trigger guard failure, not crash.
        Err(_) => pyre_object::PY_NULL,
    }
}

pub fn unpack_sequence_exact(seq: PyObjectRef, count: usize) -> Result<Vec<PyObjectRef>, PyError> {
    unpack_sequence_collected(seq, count, false)
}

/// Same collection as [`unpack_sequence_exact`], but the returned vec is in
/// TOS order: the last element is first, so a forward push leaves the first
/// element on top.
pub fn unpack_sequence_exact_tos(
    seq: PyObjectRef,
    count: usize,
) -> Result<Vec<PyObjectRef>, PyError> {
    unpack_sequence_collected(seq, count, true)
}

fn unpack_sequence_collected(
    mut seq: PyObjectRef,
    count: usize,
    tos_order: bool,
) -> Result<Vec<PyObjectRef>, PyError> {
    // Fast path only for exact built-in sequence types. Subclasses and other
    // instances may define custom `__iter__` that must be honored.
    // CPython 3.14 `ceval.c:_PyEval_UnpackIterableStackRef` reports the total
    // for exact lists, tuples, and dicts after finding one excess item.  Dicts
    // still need the iterator path to produce keys, so their exact length is
    // handled in the excess-item arm below rather than by `sequence_getitem`.
    // A str is unpacked by the generic iterator loop, which stops one item past
    // `count` and so has no total to report.
    let (exact_sequence_len, reports_total) = unsafe {
        if pyre_object::is_exact_tuple(seq) {
            (Some(w_tuple_len(seq)), true)
        } else if pyre_object::is_exact_list(seq) {
            (Some(w_list_len(seq)), true)
        } else if pyre_object::is_exact_type(seq, &pyre_object::STR_TYPE) {
            (Some(w_str_len(seq)), false)
        } else {
            (None, false)
        }
    };
    if let Some(len) = exact_sequence_len {
        if len != count {
            // `baseobjspace.py _unpackiterable_known_length_jitlook`
            // raises ValueError on length mismatch.
            let msg = if len < count {
                format!("not enough values to unpack (expected {count}, got {len})")
            } else if reports_total {
                format!("too many values to unpack (expected {count}, got {len})")
            } else {
                format!("too many values to unpack (expected {count})")
            };
            return Err(PyError::value_error(msg));
        }
        // A fetch off an unboxed `list` strategy boxes the element, and a `str`
        // fetch builds the one-character string, so every turn allocates: the
        // receiver relocates and a plain accumulator is scanned by nothing.
        // Publish both and rebuild the flat return from the slots, as the
        // iterator loop below does.
        let _roots = pyre_object::gc_roots::push_roots();
        let seq_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(seq);
        let seq = || pyre_object::gc_roots::shadow_stack_get(seq_slot);
        let items_base = pyre_object::gc_roots::shadow_stack_len();
        for idx in 0..count {
            let _ = pyre_object::gc_roots::pin_root(sequence_getitem(seq(), idx)?);
        }
        let mut items = Vec::with_capacity(count);
        push_shadow_slots(&mut items, items_base, count, tos_order);
        return Ok(items);
    }
    // Fallback: iteration protocol (handles type objects with metaclass __iter__, etc.)
    // baseobjspace.py _unpackiterable_known_length_jitlook.  pyopcode.py STORE_NAME
    // UNPACK_SEQUENCE wraps the whole `fixedview_unroll` (iter + known-length
    // loop) in a TypeError → "cannot unpack non-iterable %T object" remap.
    let iter_result = pyre_object::with_roots!(seq => crate::baseobjspace::iter(seq));
    let non_iterable = || {
        PyError::type_error(format!(
            "cannot unpack non-iterable {} object",
            crate::baseobjspace::object_functionstr_type_name(seq)
        ))
    };
    let iter = match iter_result {
        Ok(it) => it,
        Err(e) if e.kind == PyErrorKind::TypeError => return Err(non_iterable()),
        Err(e) => return Err(e),
    };
    // `next` runs the iterator's `__next__` on every turn, so the accumulator
    // cannot be a plain `Vec` — nothing scans it, which would leave each item
    // already pulled unreachable while the next one is produced, and `iter` is
    // a native copy the collector does not update.  Publish the whole set and
    // rebuild the flat return from it, as `_unpackiterable_known_length_jitlook`
    // does for the same loop.
    let _roots = pyre_object::gc_roots::push_roots();
    let seq_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(seq);
    let _ = pyre_object::gc_roots::pin_root(iter);
    let seq = || pyre_object::gc_roots::shadow_stack_get(seq_slot);
    let iter = || pyre_object::gc_roots::shadow_stack_get(seq_slot + 1);
    let root_base = seq_slot + 1;
    let mut pulled = 0usize;
    loop {
        match crate::baseobjspace::next(iter()) {
            Ok(val) => {
                if pulled == count {
                    // CPython 3.14 ceval.c:2313-2320 checks the original
                    // object's exact builtin kind only after `PyIter_Next`
                    // found the first excess item.  Lists and tuples took the
                    // fast path above; exact dicts reach this iterator path.
                    // Keep dict subclasses generic so their `__len__` is not
                    // observed and the message has no total.
                    if unsafe { pyre_object::is_exact_type(seq(), &pyre_object::DICT_TYPE) } {
                        let len = unsafe { pyre_object::w_dict_len(seq()) };
                        if len > count {
                            return Err(PyError::value_error(format!(
                                "too many values to unpack (expected {count}, got {len})"
                            )));
                        }
                    }
                    return Err(PyError::value_error(format!(
                        "too many values to unpack (expected {count})"
                    )));
                }
                let _ = pyre_object::gc_roots::pin_root(val);
                pulled += 1;
            }
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                if e.kind == PyErrorKind::TypeError {
                    return Err(non_iterable());
                }
                return Err(e);
            }
        }
    }
    if pulled < count {
        return Err(PyError::value_error(format!(
            "not enough values to unpack (expected {count}, got {pulled})"
        )));
    }
    let mut items = Vec::with_capacity(pulled);
    push_shadow_slots(&mut items, root_base + 1, pulled, tos_order);
    Ok(items)
}

fn push_shadow_slots(items: &mut Vec<PyObjectRef>, base: usize, count: usize, tos_order: bool) {
    if tos_order {
        let mut index = count;
        while index > 0 {
            index -= 1;
            items.push(pyre_object::gc_roots::shadow_stack_get(base + index));
        }
    } else {
        for index in 0..count {
            items.push(pyre_object::gc_roots::shadow_stack_get(base + index));
        }
    }
}

/// UNPACK_EX — split `value` for `a, *b, c = value` into `before` head
/// items, a starred middle list, and `after` tail items, returning the
/// `before + 1 + after` slots in TOS order ([head items], middle list,
/// [tail items]).  Shared by the interpreter's `unpack_ex` handler and the
/// JIT residual `bh_unpack_ex_fn`; the portal reads each slot back out with
/// `bh_unpack_item_fn`, exactly as `unpack_sequence_exact`.  Raises
/// ValueError when fewer than `before + after` values are available.
pub fn unpack_ex_slots(
    before: usize,
    after: usize,
    value: PyObjectRef,
) -> Result<Vec<PyObjectRef>, PyError> {
    // `collect_iterable` runs the iterator's `__next__` on every turn and the
    // starred middle `list` is allocated part-way through the split, so neither
    // `value` nor the element words may stay in plain natives across them: a
    // `list` / `dict` among them relocates and the native copy names the
    // pre-move address.  Publish them and read every operand back out of the
    // slots, as `unpack_sequence_exact` does for the same split.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(value);
    let value = || pyre_object::gc_roots::shadow_stack_get(value_slot);
    let elements: Vec<PyObjectRef> = unsafe {
        if is_tuple(value()) {
            pyre_object::w_tuple_items_copy_as_vec(value())
        } else if is_list(value()) {
            pyre_object::w_list_items_copy_as_vec_mode(
                value(),
                majit_metainterp::jit::we_are_jitted(),
            )
        } else {
            // pyopcode.py UNPACK_EX wraps `fixedview` in a
            // TypeError → "cannot unpack non-iterable %T object" remap.
            // `collect_iterable` is the `fixedview` analog (iter + next
            // loop), so any TypeError it raises is remapped here.
            match crate::builtins::collect_iterable(value()) {
                Ok(items) => items,
                Err(e) if e.kind == PyErrorKind::TypeError => {
                    return Err(PyError::type_error(format!(
                        "cannot unpack non-iterable {} object",
                        crate::baseobjspace::object_functionstr_type_name(value())
                    )));
                }
                Err(e) => return Err(e),
            }
        }
    };
    let min_expected = before + after;
    if elements.len() < min_expected {
        return Err(PyError::value_error(format!(
            "not enough values to unpack (expected at least {}, got {})",
            min_expected,
            elements.len()
        )));
    }
    let elements_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in &elements {
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let element = |index: usize| pyre_object::gc_roots::shadow_stack_get(elements_base + index);
    let middle_len = elements.len() - min_expected;
    let mut middle = Vec::with_capacity(middle_len);
    for index in before..before + middle_len {
        middle.push(element(index));
    }
    let middle_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_list_new(middle));
    // Head and tail are read back only after the middle list exists, so the
    // returned slots all carry post-move words.
    let mut slots = Vec::with_capacity(before + 1 + after);
    slots.extend((0..before).map(element));
    slots.push(pyre_object::gc_roots::shadow_stack_get(middle_slot));
    slots.extend((before + middle_len..elements.len()).map(element));
    Ok(slots)
}

pub fn ensure_range_iter(iter: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if is_range_iter(iter) || is_seq_iter(iter) || is_list_iter(iter) || is_tuple_iter(iter) {
            return Ok(());
        }
        // Convert list/tuple to seq iterator
        if is_list(iter) {
            // Replace TOS with a seq iterator wrapping the list
            // This is called on TOS after GET_ITER pops and pushes.
            // But ensure_iter is called ON the iter — it can't replace stack.
            // So we need to create iter before calling ensure.
            // Actually ensure_range_iter is called AFTER get_iter pushes.
            // The problem: GET_ITER calls ensure_iter_value on the pushed value.
            // For list, we need to push a seq_iter instead of the list itself.
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        crate::baseobjspace::object_functionstr_type_name(iter)
    )))
}

/// Current length of the sequence a `W_SeqIterObject` walks. `None` for a
/// payload outside list/tuple/str/array, leaving callers to fall back to
/// the length captured at iterator creation.  The covered set mirrors the
/// item-fetch arms of `baseobjspace::next` / `range_iter_next_or_null`, so
/// `range_iter_continues` and the next-value helper stay consistent (a
/// payload one can fetch from is a payload one can length).
///
/// # Safety
/// `seq` must be a valid object pointer.
unsafe fn seq_iter_current_len(seq: PyObjectRef) -> Option<i64> {
    unsafe {
        if is_list(seq) {
            Some(w_list_len(seq) as i64)
        } else if is_tuple(seq) {
            Some(w_tuple_len(seq) as i64)
        } else if is_str(seq) {
            // Code-point count, not byte count (matches the str seq-iter
            // seed in baseobjspace::iter).
            Some(w_str_len(seq) as i64)
        } else if pyre_object::bytesobject::is_bytes_like(seq) {
            Some(pyre_object::bytesobject::bytes_like_len(seq) as i64)
        } else if pyre_object::interp_array::is_array(seq) {
            Some(pyre_object::interp_array::w_array_len(seq) as i64)
        } else {
            None
        }
    }
}

pub fn range_iter_continues(iter: PyObjectRef) -> Result<bool, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_has_next(iter));
        }
        if pyre_object::is_long_range_iter(iter) {
            return Ok(pyre_object::w_long_range_iter_has_next(iter));
        }
        if is_seq_iter(iter) {
            let si = &*(iter as *const W_SeqIterObject);
            // A cleared sequence (exhausted cursor, iterobject.py:90-98) has no
            // more items — a re-iteration of an exhausted iterator stops here
            // without dereferencing the freed sequence.  A negative cursor is
            // `bytearray_iterator`'s exhausted sentinel and keeps its source.
            if si.seq.is_null() || si.index < 0 {
                return Ok(false);
            }
            // sequenceiterator re-reads the live sequence length and PyPy's
            // W_FastListIterObject.descr_next ends on a getitem IndexError;
            // neither snapshots a length. Read the CURRENT sequence length so
            // an element appended during iteration is observed (and a removed
            // tail ends the loop) rather than the length captured at iterator
            // creation.
            let len = seq_iter_current_len(si.seq).unwrap_or(si.length);
            // `length` is the creation-time snapshot (still `i64`). The live
            // cursor is a machine word; widen it, which drops no bits.
            return Ok(pyre_object::seq_index_to_i64(si.index) < len);
        }
        if is_list_iter(iter) {
            let si = &*(iter as *const W_ListIterObject);
            // A negative cursor is the `__setstate__` exhausted sentinel.
            // Compare the `Signed` cursor with the live length in word width.
            // Casting the cursor to `usize` would record a narrowing mask.
            return Ok(!si.seq.is_null()
                && si.index >= 0
                && si.index < w_list_len(si.seq) as isize);
        }
        if is_tuple_iter(iter) {
            let si = &*(iter as *const W_TupleIterObject);
            return Ok(!si.seq.is_null() && si.index < w_tuple_len(si.seq) as isize);
        }
    }
    Err(PyError::type_error("not an iterator"))
}

pub fn range_iter_next_or_null(iter: PyObjectRef) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_next(iter).unwrap_or(PY_NULL));
        }
        if pyre_object::is_long_range_iter(iter) {
            return Ok(pyre_object::w_long_range_iter_next(iter).unwrap_or(PY_NULL));
        }
        if is_seq_iter(iter) {
            let si = &mut *(iter as *mut W_SeqIterObject);
            // An exhausted cursor cleared its sequence (iterobject.py:90-98
            // `self.w_seq = None`); a re-entry stays exhausted without
            // dereferencing the freed sequence.  A negative cursor is
            // `bytearray_iterator`'s exhausted sentinel and keeps its source.
            if si.seq.is_null() || si.index < 0 {
                return Ok(PY_NULL);
            }
            let idx = si.index;
            // Bounds come from the sequence's CURRENT state (see
            // range_iter_continues): getitem returns None past the live
            // length, so an element appended during iteration is yielded and a
            // removed tail ends the loop. Mirrors baseobjspace::next.
            let idx_i64 = pyre_object::seq_index_to_i64(idx);
            let item = if is_list(si.seq) {
                w_list_getitem(si.seq, idx_i64)
            } else if is_tuple(si.seq) {
                w_tuple_getitem(si.seq, idx_i64)
            } else if is_str(si.seq) {
                // Box the idx-th code point as a one-character str, reading
                // the WTF-8 view so a lone surrogate is yielded instead of
                // panicking. Only `iter(str)` through the builtin reaches here
                // with a str payload (the GET_ITER opcode materialises a char
                // list); without this arm `seq_iter_current_len` would say
                // "more" while this helper returned PY_NULL forever, hanging
                // the loop. Mirrors baseobjspace::next.
                pyre_object::w_str_codepoint_at(si.seq, idx as usize).map(|cp| {
                    let mut one = Wtf8Buf::new();
                    one.push(cp);
                    pyre_object::w_str_from_wtf8_managed(one)
                })
            } else if pyre_object::bytesobject::is_bytes_like(si.seq) {
                // Each item is the byte's ordinal, read from the live buffer so
                // a bytearray resized mid-iteration is observed.
                if idx < pyre_object::bytesobject::bytes_like_len(si.seq) as isize {
                    Some(w_int_new(pyre_object::bytesobject::bytes_like_getitem(
                        si.seq,
                        idx as usize,
                    ) as i64))
                } else {
                    None
                }
            } else if pyre_object::interp_array::is_array(si.seq) {
                if let Some(hooks) = crate::importing::optional_module_hooks()
                    && idx < pyre_object::interp_array::w_array_len(si.seq) as isize
                {
                    // [3.14-spec] PyPy's generic `W_SeqIterObject.descr_next`
                    // advances only after `getitem` succeeds (and clears the
                    // source on any error). v3.14.6 `arrayiter_next` passes
                    // `it->index++` to the item getter, so an invalid unicode
                    // slot raises once and the following slot remains next.
                    si.index += 1;
                    return (hooks.array_w_getitem)(si.seq, idx as usize, false);
                }
                None
            } else {
                None
            };
            if let Some(v) = item {
                si.index += 1;
                return Ok(v);
            }
            // iterobject.py:96-98 — clear the sequence ref on exhaustion so a
            // held iterator's `__reduce__`/`__length_hint__` report it as an
            // exhausted cursor (matches the generic-`__getitem__` arm of
            // `baseobjspace::next`).
            si.seq = PY_NULL;
            return Ok(PY_NULL);
        }
        if is_list_iter(iter) {
            let si = &mut *(iter as *mut W_ListIterObject);
            // The `__setstate__` sentinel keeps the sequence so a later
            // in-range state can revive the iterator (mirrors
            // `baseobjspace::next`).
            if si.seq.is_null() || si.index < 0 {
                return Ok(PY_NULL);
            }
            if let Some(item) = w_list_getitem(si.seq, pyre_object::seq_index_to_i64(si.index)) {
                si.index += 1;
                return Ok(item);
            }
            si.seq = PY_NULL;
            return Ok(PY_NULL);
        }
        if is_tuple_iter(iter) {
            return Ok(crate::baseobjspace::tuple_iter_descr_next(iter));
        }
    }
    Err(PyError::type_error("not an iterator"))
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_range_iter_next_or_null(iter: PyObjectRef) -> PyObjectRef {
    match range_iter_next_or_null(iter) {
        Ok(value) => value,
        Err(err) => panic!("range iter next failed in JIT: {err}"),
    }
}

/// A sequence iterator whose payload is not one of the inline fast types
/// (list/tuple/str/array, see `seq_iter_current_len`) is a generic
/// sequence-protocol iterator: `next` fetches each item through
/// `space.getitem` (iterobject.py W_SeqIterObject.descr_next), so it is
/// driven by `baseobjspace::next` rather than the inline range helper.  An
/// exhausted cursor (null seq) routes here too so the trailing StopIteration
/// is raised by `next`.
///
/// # Safety
/// `iter` must be a valid object pointer.
unsafe fn is_generic_seq_iter(iter: PyObjectRef) -> bool {
    unsafe {
        is_seq_iter(iter) && {
            let seq = (*(iter as *const W_SeqIterObject)).seq;
            seq.is_null() || seq_iter_current_len(seq).is_none()
        }
    }
}

/// FOR_ITER iterator kinds that advance through `space.next` rather than
/// the inline range helper: generic sequence-protocol iterators, generators,
/// itertools/enumerate/reversed/filter/map/zip/set/dictview/sre, callable
/// iterators, and user `__next__` instances. Shared so the interpreter
/// `iter_next` and the tracer agree on which iterators take the `space.next`
/// leg.
pub fn via_space_next(iter: PyObjectRef) -> bool {
    unsafe {
        is_generic_seq_iter(iter)
            || is_instance(iter)
            || pyre_object::generator::is_generator(iter)
            || pyre_object::interp_itertools::is_repeat(iter)
            || pyre_object::interp_itertools::is_count(iter)
            || pyre_object::interp_itertools::is_takewhile(iter)
            || pyre_object::interp_itertools::is_dropwhile(iter)
            || pyre_object::interp_itertools::is_filterfalse(iter)
            || pyre_object::interp_itertools::is_pairwise(iter)
            || pyre_object::interp_itertools::is_cycle(iter)
            || pyre_object::interp_itertools::is_chain(iter)
            || pyre_object::functional::is_enumerate(iter)
            || pyre_object::functional::is_reversed(iter)
            || pyre_object::functional::is_filter(iter)
            || pyre_object::functional::is_map(iter)
            || pyre_object::functional::is_zip(iter)
            || pyre_object::operation::is_callable_iterator(iter)
            // setobject.py `W_SetIterObject.next` delegates to its live
            // `IteratorImplementation.next`, whose length guard must run on
            // every FOR_ITER step.  It is not one of the scalar range/seq
            // layouts handled by `range_iter_continues`.
            || pyre_object::setobject::is_set_iterator(iter)
            || pyre_object::dictmultiobject::is_dict_view_iterator(iter)
            || pyre_object::interp_sre::is_sre_scanner(iter)
            || crate::module::r#struct::is_unpack_iter(iter)
    }
}

/// Residual FOR_ITER `space.next` for the `via_space_next` iterator kinds.
/// Exhaustion is signalled the same way as the range fast path: a null
/// return that the trailing for-iter GuardNonnull catches, side-exiting to
/// the interpreter which re-runs FOR_ITER and ends the loop (eval.rs
/// `iter_next` MRO-matches StopIteration to exhausted). A *real* exception is
/// published into BOTH the backend exception cells (so the compiled trace /
/// blackhole GuardNoException side-exits) AND `BH_LAST_EXC_VALUE` (so the
/// full-body walk's `execute_residual_call` returns Err and records the
/// can-raise path instead of mistaking the null for exhaustion). Keeping the
/// two seams in sync mirrors `call_jit::publish_residual_call_exception`, the
/// dual-publish every other MayForce residual uses.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_next(iter: PyObjectRef) -> PyObjectRef {
    let next_value = crate::baseobjspace::next(iter);
    // StopIteration is not a frame-level exception for FOR_ITER; return
    // null so the GuardNonnull (not GuardNoException) fires. Pin the handle
    // across `matches_stop_iteration`, which can collect.
    let mut err = match next_value {
        Ok(value) => return value,
        Err(err) => err,
    };
    let (stop, mut err) = err.matches_stop_iteration_keep();
    if stop {
        return PY_NULL;
    }
    let exc_obj = err.to_exc_object();
    if exc_obj != PY_NULL {
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
    }
    jit_publish_exception(exc_obj);
    PY_NULL
}

/// `s.add(value)` written as a bound-method call, for the walker arm that
/// recognises it (`try_walker_specialize_set_add_method`).
///
/// The arm pins the callable to the `set.add` builtin and guards the receiver
/// in the same spelling `require_set_receiver` uses, so the two checks the
/// builtin opens with — its receiver predicate and one positional argument —
/// are answered by those guards, and what remains is the store
/// [`crate::opcode_ops::set_add_value`] already performs for `SET_ADD`. The
/// element `__hash__` can run Python, so this is a MayForce residual like the
/// accumulator it shares its body with.
///
/// Returns `None`. A raising `__hash__` publishes through both exception
/// channels and answers PY_NULL, the residual-call ABI [`jit_next`] documents.
pub extern "C" fn jit_set_add_method(set: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    match crate::opcode_ops::set_add_value(set, value) {
        Ok(()) => pyre_object::w_none(),
        Err(err) => jit_publish_residual_error_ref(err),
    }
}

/// `1` when [`pyre_object::plain_int_already_in_int_set`] is true.
///
/// Recorded as a cannot-collect call when the traced `set.add` is that hit.
/// It does not insert, so a user `__hash__` never runs here. `0` fails the
/// guard and the interpreter performs the real add.
pub extern "C" fn jit_int_set_add_already_present(set: PyObjectRef, value: PyObjectRef) -> i64 {
    pyre_object::plain_int_already_in_int_set(set, value) as i64
}

/// `pyopcode.py` FOR_ITER exception discrimination:
/// `e.match(space, space.w_StopIteration)`. The caught object and match class
/// are Python-level objects, so use the same MRO-aware helper as
/// CHECK_EXC_MATCH. This is infallible and deliberately never publishes a
/// backend or blackhole exception.
pub extern "C" fn jit_exception_match(exc: PyObjectRef, match_class: PyObjectRef) -> i64 {
    crate::eval::check_exc_match_against(exc, match_class) as i64
}

/// Ref-returning bridge for the `next(w_iterator)` residual call in
/// `unpackiterable_portal` (the `unpackiterable_driver` portal).
///
/// Unlike [`jit_next`], StopIteration is published as an ordinary exception
/// rather than collapsed to a null sentinel: the unpack loop body matches on
/// it (`e.match(space.w_StopIteration) → break`) to terminate, so the trace's
/// `GuardNoException` + exception-match ops must observe the raised
/// StopIteration.  `jit_next`'s null-for-StopIteration convention exists for
/// `FOR_ITER`'s `GuardNonnull` and is wrong here.
#[majit_macros::jit_may_force]
pub extern "C" fn bh_next(iter: PyObjectRef) -> PyObjectRef {
    match crate::baseobjspace::next(iter) {
        Ok(value) => value,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            if exc_obj != PY_NULL {
                majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            }
            jit_publish_exception(exc_obj);
            PY_NULL
        }
    }
}

/// Residual PyPy `GET_ITER` operation.  `space.iter` may execute a user
/// `__iter__`, so the trace treats this as may-force and consumes the
/// published exception through the ordinary guards.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_get_iter(iterable: PyObjectRef) -> PyObjectRef {
    match crate::baseobjspace::iter(iterable) {
        Ok(iterator) => iterator,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            if exc_obj != PY_NULL {
                majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            }
            jit_publish_exception(exc_obj);
            PY_NULL
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyExecutionContext;
    use pyre_object::{w_int_get_value, w_int_new};

    #[test]
    fn test_classify_callable_names_the_builtin_arm() {
        let ctx = PyExecutionContext::default();
        let abs = ctx.lookup_builtin("abs").expect("abs builtin must exist");
        assert_eq!(
            classify_callable(abs).expect("builtin dispatch should succeed"),
            CallableKind::Builtin
        );

        // Builtins are now Function objects; extract code then func pointer.
        let code = unsafe { crate::getcode(abs) };
        let func = unsafe { crate::builtin_code_get(code as PyObjectRef) };
        let result = func(&[w_int_new(-9)]).expect("abs(-9) should succeed");

        unsafe {
            assert_eq!(w_int_get_value(result), 9);
        }
    }

    #[test]
    fn test_classify_callable_rejects_non_callable() {
        let err = classify_callable(w_int_new(3)).expect_err("non-callable dispatch should fail");

        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message_text().contains("not callable"));
    }

    #[test]
    fn test_range_iter_helpers_share_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 2, 1);
        assert!(range_iter_continues(iter).unwrap());
        let first = range_iter_next_or_null(iter).unwrap();
        let second = range_iter_next_or_null(iter).unwrap();
        let done = range_iter_next_or_null(iter).unwrap();
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_range_iter_helper_shares_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 2, 1);
        let first = jit_range_iter_next_or_null(iter);
        let second = jit_range_iter_next_or_null(iter);
        let done = jit_range_iter_next_or_null(iter);
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_sequence_getitem_shares_runtime_sequence_semantics() {
        let tuple = pyre_object::w_tuple_new(vec![w_int_new(3), w_int_new(5)]);
        let item = jit_sequence_getitem(tuple, 1);
        unsafe {
            assert_eq!(w_int_get_value(item), 5);
        }
    }

    #[test]
    fn seq_iter_observes_list_growth_during_iteration() {
        // listiterator re-reads the live list size each step, so an append
        // during iteration is observed. range_iter_continues /
        // range_iter_next_or_null must read the CURRENT length, not the length
        // captured at iterator creation — otherwise `for x in xs:
        // xs.append(...)` stops at the initial length.
        let list = pyre_object::w_list_new(vec![w_int_new(0)]);
        let iter = pyre_object::w_seq_iter_new(list, 1);
        let mut seen = Vec::new();
        while range_iter_continues(iter).unwrap() {
            let v = range_iter_next_or_null(iter).unwrap();
            assert!(!v.is_null());
            let x = unsafe { w_int_get_value(v) };
            seen.push(x);
            if unsafe { pyre_object::w_list_len(list) } < 5 {
                unsafe { pyre_object::w_list_append(list, w_int_new(x + 1)) };
            }
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn seq_iter_over_str_payload_yields_codepoints_and_terminates() {
        // `iter(str)` through the builtin (baseobjspace::iter) makes a seq-iter
        // with a STR payload (the GET_ITER opcode instead materialises a char
        // list). range_iter_continues / range_iter_next_or_null must fetch and
        // advance for that payload too; otherwise `continues` stays true while
        // `next` returns PY_NULL forever, hanging `for c in iter(s)`.
        let s = pyre_object::unicodeobject::box_str_constant(Wtf8::new("abcde"));
        let iter = pyre_object::w_seq_iter_new(s, 5);
        let mut count = 0;
        while range_iter_continues(iter).unwrap() {
            let v = range_iter_next_or_null(iter).unwrap();
            assert!(
                !v.is_null(),
                "str seq-iter yielded NULL while continues==true"
            );
            count += 1;
            assert!(
                count <= 5,
                "str seq-iter did not terminate (infinite-loop regression)"
            );
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn set_iter_uses_space_next_for_live_size_guard() {
        let set = pyre_object::w_set_new();
        let iter = pyre_object::w_set_iter_new(set);
        assert!(via_space_next(iter));
        assert!(range_iter_continues(iter).is_err());
    }
}
