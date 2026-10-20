//! The object protocol -- PyPy `cpyext/object.py` and `cpyext/mapping.py`.

use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use super::typeobject::{CPyTypeObject, CPyVarObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int, c_void};

/// Call `callable` with `arguments` and the keywords `names` spells.
///
/// The keywords form an `Arguments(..., keyword_names_w=names)`, which needs
/// the running frame: an entry point reached with none is being called from
/// outside any interpreter and reports that rather than guessing one.
pub(super) fn call_named(
    callable: PyObjectRef,
    arguments: &[PyObjectRef],
    names: &[(rustpython_wtf8::Wtf8Buf, PyObjectRef)],
) -> Result<PyObjectRef, crate::PyError> {
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "a cpyext keyword call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, callable, arguments, names)
    })
}

/// [`call_named`] for the single keyword an entry point names itself.
pub(super) fn call_keyword(
    callable: PyObjectRef,
    arguments: &[PyObjectRef],
    name: &str,
    value: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    call_named(
        callable,
        arguments,
        &[(
            rustpython_wtf8::Wtf8Buf::from_string(name.to_string()),
            value,
        )],
    )
}

/// The interpreter object behind an argument, or a recorded `SystemError`.
///
/// Upstream's generated wrappers reject a NULL argument with
/// `PyErr_BadInternalCall`; this is that check in one place.
pub(super) fn argument(raw: *mut CPyObject) -> Option<PyObjectRef> {
    let value = unsafe { pyobject::from_ref(raw) };
    if value.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(value)
}

/// [`argument`] for an entry point taking several of them.
///
/// Converting a mirror can [`pyobject::realize`] it, and realizing allocates.
/// A mirror never moves, but an interpreter object an earlier conversion
/// answered with does — so every mirror is realized first, and only then read.
pub(super) fn arguments<const N: usize>(raws: [*mut CPyObject; N]) -> Option<[PyObjectRef; N]> {
    for raw in raws {
        unsafe { pyobject::realize(raw) };
    }
    let mut values = [pyre_object::PY_NULL; N];
    for (value, raw) in values.iter_mut().zip(raws) {
        *value = unsafe { pyobject::from_ref(raw) };
        if value.is_null() {
            unsafe { super::pyerrors::PyErr_BadInternalCall() };
            return None;
        }
    }
    Some(values)
}

/// The realizing half of [`arguments`], for an entry point that converts its
/// mirrors through something other than one call to it — a NULL that means
/// "delete", a typed check, a C string beside them.
///
/// Stands at the head of such an entry point: once every mirror it was handed
/// is realized, no conversion below can allocate, whatever order they run in.
/// A null or already-realized mirror is nothing to do.
pub(super) fn realize_all<const N: usize>(raws: [*mut CPyObject; N]) {
    for raw in raws {
        unsafe { pyobject::realize(raw) };
    }
}

/// Whether `object`'s user-visible class is exactly the one `tp` describes.
///
/// The layout an object carries is shared with its subclasses — an instance of
/// a `str` subclass is a `W_UnicodeObject` under `STR_TYPE`, and `bool` is laid
/// out as `int` — so a `*_CheckExact` spelling has to compare the class the
/// interpreter reports, not the layout.
pub(super) fn is_exactly(object: PyObjectRef, tp: *const pyre_object::PyType) -> bool {
    match (
        crate::typedef::r#type(object),
        crate::typedef::gettypefor(tp),
    ) {
        (Some(actual), Some(exact)) => actual == exact,
        _ => false,
    }
}

/// `size` slots holding `fill`, or `None` with `MemoryError` recorded.
///
/// A sequence constructor takes its length from C, so the request can be one
/// no allocator can satisfy.  `tuple_alloc` answers that with `PyErr_NoMemory`,
/// where the `vec![fill; size]` this replaces panics — an unwind out of an
/// `extern "C"` frame, which aborts the process.
pub(super) fn item_slots(size: isize, fill: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    let size = size.max(0) as usize;
    let mut items: Vec<PyObjectRef> = Vec::new();
    if size > isize::MAX as usize / std::mem::size_of::<PyObjectRef>()
        || items.try_reserve_exact(size).is_err()
    {
        unsafe { super::pyerrors::PyErr_NoMemory() };
        return None;
    }
    items.resize(size, fill);
    Some(items)
}

/// A new reference to the result of an interpreter operation, or NULL with the
/// error recorded.
pub(super) fn result(value: Result<PyObjectRef, crate::PyError>) -> *mut CPyObject {
    match trap(value) {
        Some(value) => pyobject::make_ref(value),
        None => std::ptr::null_mut(),
    }
}

/// `object.name(*arguments)`.
///
/// The attribute lookup can collect, and an argument that is a list or a dict
/// moves when it does, so everything is pinned across it and read back after.
pub(super) fn call_method(
    object: PyObjectRef,
    name: &str,
    arguments: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(object);
    let first_argument = pyre_object::gc_roots::shadow_stack_len();
    for &argument in arguments {
        let _ = roots.pin_root(argument);
    }
    let method = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(object_slot),
        name,
    )?;
    let method_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(method);
    let reloaded: Vec<PyObjectRef> = (0..arguments.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(first_argument + index))
        .collect();
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(method_slot),
        &reloaded,
    )
}

/// The attribute name an entry point was handed, as the `str` it must be.
fn attribute_name(name: *mut CPyObject) -> Option<PyObjectRef> {
    let name = argument(name)?;
    if !unsafe { pyre_object::is_str(name) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "attribute name must be string, not '{}'",
            crate::type_methods::arg_type_name(name)
        )));
        return None;
    }
    Some(name)
}

fn name_of(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetAttrString(
    object: *mut CPyObject,
    name: *const c_char,
) -> *mut CPyObject {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getattr_str(object, &name))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetAttrString(
    object: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    realize_all([object, value]);
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return -1;
    };
    let value = unsafe { pyobject::from_ref(value) };
    let outcome = if value.is_null() {
        crate::baseobjspace::delattr_str(object, &name).map(|_| pyre_object::w_none())
    } else {
        crate::baseobjspace::setattr_str(object, &name, value)
    };
    if trap(outcome).is_none() { -1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HasAttrString(
    object: *mut CPyObject,
    name: *const c_char,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    match crate::baseobjspace::getattr_str(object, &name) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
) -> *mut CPyObject {
    let Some([object, name]) = arguments([object, name]) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getattr(object, name))
}

/// A NULL `value` is the deletion spelling, which is what
/// [`PyObject_SetAttrString`] already reads it as.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    realize_all([object, name, value]);
    let Some([object, name]) = arguments([object, name]) else {
        return -1;
    };
    let value = unsafe { pyobject::from_ref(value) };
    let outcome = if value.is_null() {
        crate::baseobjspace::delattr(object, name).map(|_| pyre_object::w_none())
    } else {
        crate::baseobjspace::setattr(object, name, value)
    };
    if trap(outcome).is_none() {
        return -1;
    }
    0
}

// ── the generic attribute protocol ────────────────────────────────────────
//
// `tp_getattro` and `tp_setattro` default to these, and an extension that
// overrides either one routinely calls back into them for the cases it does not
// handle itself.  They are the `object.__getattribute__` / `__setattr__`
// terminals — the descriptor protocol without the receiver type's own override
// (`object.py:286-312`).

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GenericGetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([object, name]);
    let (Some(object), Some(w_name)) = (argument(object), attribute_name(name)) else {
        return std::ptr::null_mut();
    };
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    result(match name.as_str() {
        Ok(name) => crate::baseobjspace::object_getattribute(object, name),
        Err(_) => unsafe {
            crate::baseobjspace::object_getattribute_surrogate(object, w_name, name)
        },
    })
}

/// `PyObject_GenericSetAttr(object, name, value)` — a NULL `value` is the
/// deletion spelling, which is `object.__delattr__` (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GenericSetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    realize_all([object, name, value]);
    let (Some(object), Some(w_name)) = (argument(object), attribute_name(name)) else {
        return -1;
    };
    let value = unsafe { pyobject::from_ref(value) };
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    let outcome = match (name.as_str(), value.is_null()) {
        (Ok(name), false) => crate::baseobjspace::object_setattr(object, name, value),
        (Ok(name), true) => crate::baseobjspace::object_delattr(object, name),
        (Err(_), false) => unsafe {
            crate::baseobjspace::object_setattr_surrogate(object, w_name, name, value)
        },
        (Err(_), true) => unsafe {
            crate::baseobjspace::object_delattr_surrogate(object, w_name, name)
        },
    };
    if trap(outcome).is_none() { -1 } else { 0 }
}

/// `PyObject_GenericGetDict(object, context)` — the `__dict__` descriptor's
/// getter (`typedef.py descr_get_dict`).  The `context` a getset
/// closure would carry has nothing to say here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GenericGetDict(
    object: *mut CPyObject,
    _context: *mut c_void,
) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getdict(object).and_then(|w_dict| {
        if w_dict.is_null() {
            return Err(crate::PyError::type_error(format!(
                "descriptor '__dict__' doesn't apply to '{}' objects",
                crate::type_methods::arg_type_name(object)
            )));
        }
        Ok(w_dict)
    }))
}

/// `PyObject_GenericSetDict(object, value, context)` — `typedef.py:549-550
/// descr_set_dict`.  A NULL `value` asks for a deletion this descriptor does
/// not offer (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GenericSetDict(
    object: *mut CPyObject,
    value: *mut CPyObject,
    _context: *mut c_void,
) -> c_int {
    realize_all([object, value]);
    let Some(object) = argument(object) else {
        return -1;
    };
    let value = unsafe { pyobject::from_ref(value) };
    if value.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error("cannot delete __dict__"));
        return -1;
    }
    if trap(crate::baseobjspace::setdict(object, value)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Str(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(unsafe { crate::display::py_str_wtf8(object) }.map(pyre_object::w_str_from_wtf8))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Repr(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(unsafe { crate::display::py_repr_wtf8(object) }.map(pyre_object::w_str_from_wtf8))
}

/// `PyObject_ASCII(object)` — `ascii(object)` (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_ASCII(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::builtins::py_ascii_obj(object))
}

/// `PyObject_Bytes(object)` — `bytes(object)` through `__bytes__` and, failing
/// that, the buffer/iterable conversion; `b"<NULL>"` for a NULL one
/// (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Bytes(object: *mut CPyObject) -> *mut CPyObject {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(b"<NULL>"));
    }
    result(bytes_of_object(object))
}

/// `object.py:197-202` — the exact-`bytes` identity, then `invoke_bytes_method`,
/// then `PyBytes_FromObject`.
fn bytes_of_object(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if unsafe {
        pyre_object::pyobject::is_exact_type(object, &pyre_object::bytesobject::BYTES_TYPE)
    } {
        return Ok(object);
    }
    // `bytesobject.py invoke_bytes_method` — the override's answer is
    // returned unmodified, a `bytes` subclass included.
    if let Some(method) = unsafe { crate::baseobjspace::lookup(object, "__bytes__") } {
        let w_type = crate::typedef::r#type(object).map_or(pyre_object::PY_NULL, |t| t.as_ptr());
        let answer =
            unsafe { crate::baseobjspace::get_and_call_function(method, object, w_type, &[]) }?;
        if !unsafe { pyre_object::bytesobject::is_bytes(answer) } {
            return Err(crate::PyError::type_error(format!(
                "__bytes__ returned non-bytes (type {})",
                crate::type_methods::arg_type_name(answer)
            )));
        }
        return Ok(answer);
    }
    super::bytesobject::bytes_of(object)
}

/// `PyObject_Format(object, spec)` — `format(object, spec)`, with a NULL spec
/// standing for the empty one (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Format(
    object: *mut CPyObject,
    spec: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([object, spec]);
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let spec = unsafe { pyobject::from_ref(spec) };
    let spec = if spec.is_null() {
        Ok(pyre_object::PY_NULL)
    } else {
        crate::type_methods::check_format_spec(spec, "PyObject_Format() argument 2")
    };
    result(spec.and_then(|spec| crate::type_methods::format_w(object, spec)))
}

/// `PyObject_Dir(object)` — `dir(object)`, and the caller's own scope for a
/// NULL one (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Dir(object: *mut CPyObject) -> *mut CPyObject {
    let object = unsafe { pyobject::from_ref(object) };
    let arguments: &[PyObjectRef] = if object.is_null() {
        &[]
    } else {
        std::slice::from_ref(&object)
    };
    result(crate::builtins::builtin_dir(arguments))
}

// ── comparison and hashing ────────────────────────────────────────────────

/// The six `Py_LT`..`Py_GE` identifiers, in the order `object.py:251-256` tests
/// them.
fn comparison_of(opid: c_int) -> Option<crate::bytecode::ComparisonOperator> {
    use crate::bytecode::ComparisonOperator;
    Some(match opid {
        0 => ComparisonOperator::Less,
        1 => ComparisonOperator::LessOrEqual,
        2 => ComparisonOperator::Equal,
        3 => ComparisonOperator::NotEqual,
        4 => ComparisonOperator::Greater,
        5 => ComparisonOperator::GreaterOrEqual,
        _ => return None,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_RichCompare(
    left: *mut CPyObject,
    right: *mut CPyObject,
    opid: c_int,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    let Some(op) = comparison_of(opid) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::compare_value(left, right, op))
}

/// `PyObject_RichCompareBool(left, right, opid)` — the truth of the comparison,
/// with the identity shortcut that makes an object equal to itself without
/// calling `__eq__` (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_RichCompareBool(
    left: *mut CPyObject,
    right: *mut CPyObject,
    opid: c_int,
) -> c_int {
    let Some([w_left, w_right]) = arguments([left, right]) else {
        return -1;
    };
    if crate::baseobjspace::is_w(w_left, w_right) {
        match opid {
            2 => return 1,
            3 => return 0,
            _ => {}
        }
    }
    let outcome = unsafe { PyObject_RichCompare(left, right, opid) };
    if outcome.is_null() {
        return -1;
    }
    let truth = unsafe { PyObject_IsTrue(outcome) };
    unsafe { pyobject::decref(outcome) };
    truth
}

/// `PyObject_Hash(object)` — `hash(object)`, or -1 with a `TypeError` for an
/// unhashable one (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Hash(object: *mut CPyObject) -> isize {
    let Some(object) = argument(object) else {
        return -1;
    };
    match trap(crate::baseobjspace::hash_w_strict(object)) {
        // -1 is the failure answer, so no successful hash may be it; the value
        // that would collide is reported as -2.
        Some(-1) => -2,
        Some(hash) => hash as isize,
        None => -1,
    }
}

/// `PyObject_HashNotImplemented(object)` — what a type puts in `tp_hash` to say
/// it has none (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HashNotImplemented(object: *mut CPyObject) -> isize {
    let object = unsafe { pyobject::from_ref(object) };
    let name = if object.is_null() {
        "object".to_string()
    } else {
        crate::type_methods::arg_type_name(object)
    };
    super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
        "unhashable type: '{name}'"
    )));
    -1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_IsTrue(object: *mut CPyObject) -> c_int {
    let Some(object) = argument(object) else {
        return -1;
    };
    match trap(crate::baseobjspace::is_true(object)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Not(object: *mut CPyObject) -> c_int {
    match unsafe { PyObject_IsTrue(object) } {
        -1 => -1,
        0 => 1,
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Size(object: *mut CPyObject) -> isize {
    let Some(object) = argument(object) else {
        return -1;
    };
    let length =
        trap(crate::baseobjspace::len(object).and_then(|value| {
            crate::baseobjspace::gateway_int_w(value).map(|length| length as isize)
        }));
    length.unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    let Some([object, key]) = arguments([object, key]) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(object, key))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    realize_all([object, key, value]);
    let Some([object, key]) = arguments([object, key]) else {
        return -1;
    };
    let Some(value) = argument(value) else {
        return -1;
    };
    if trap(crate::baseobjspace::setitem(object, key, value)).is_none() {
        return -1;
    }
    0
}

/// `PyObject_Length` — the other spelling of [`PyObject_Size`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Length(object: *mut CPyObject) -> isize {
    unsafe { PyObject_Size(object) }
}

// ── deletion and optional lookup ──────────────────────────────────────────

/// `PyObject_DelItem(object, key)` — `del object[key]` (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_DelItem(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    let Some([object, key]) = arguments([object, key]) else {
        return -1;
    };
    if trap(crate::baseobjspace::delitem(object, key)).is_none() {
        return -1;
    }
    0
}

/// `PyObject_DelItemString(object, key)` — `del object[key]` for a C string
/// key.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_DelItemString(
    object: *mut CPyObject,
    key: *const c_char,
) -> c_int {
    let (Some(object), Some(key)) = (argument(object), name_of(key)) else {
        return -1;
    };
    // Building the key is an allocation, so the receiver is read back from the
    // shadow stack rather than from the pointer taken before it.
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(object);
    let w_key = pyre_object::w_str_new_managed(&key);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_key);
    let outcome = crate::baseobjspace::delitem(
        pyre_object::gc_roots::shadow_stack_get(object_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    );
    if trap(outcome).is_none() { -1 } else { 0 }
}

/// `PyObject_DelAttr(object, name)` — `del object.name`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_DelAttr(object: *mut CPyObject, name: *mut CPyObject) -> c_int {
    let Some([object, name]) = arguments([object, name]) else {
        return -1;
    };
    if trap(crate::baseobjspace::delattr(object, name)).is_none() {
        return -1;
    }
    0
}

/// `PyObject_DelAttrString(object, name)` — `del object.name` for a C string
/// name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_DelAttrString(
    object: *mut CPyObject,
    name: *const c_char,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return -1;
    };
    if trap(crate::baseobjspace::delattr_str(object, &name)).is_none() {
        return -1;
    }
    0
}

/// The lookup the optional-attribute entry points share: `Some` for a hit,
/// `None` for a miss that raised nothing, and `Err` with the error recorded.
///
/// A miss reaches here as an `AttributeError`, which is precisely what the
/// caller of these asked not to be told about.
fn optional_attr(
    attribute: Result<PyObjectRef, crate::PyError>,
) -> Result<Option<PyObjectRef>, ()> {
    match attribute {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(None),
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            Err(())
        }
    }
}

/// Report an optional lookup through the `(int, PyObject **)` pair these use.
///
/// `out` is non-NULL: each entry point rejects a NULL one and clears it before
/// anything else can fail.
fn report_optional(found: Result<Option<PyObjectRef>, ()>, out: *mut *mut CPyObject) -> c_int {
    match found {
        Ok(Some(value)) => {
            unsafe { *out = pyobject::make_ref(value) };
            1
        }
        Ok(None) => 0,
        Err(()) => -1,
    }
}

/// `PyObject_GetOptionalAttr(object, name, out)` — 1 with `*out` a new
/// reference, 0 with `*out` NULL for an absent attribute, -1 for an error.
///
/// # Safety
/// `out` must be a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetOptionalAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
    out: *mut *mut CPyObject,
) -> c_int {
    realize_all([object, name]);
    if out.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe { *out = std::ptr::null_mut() };
    let (Some(object), Some(name)) = (argument(object), attribute_name(name)) else {
        return -1;
    };
    report_optional(
        optional_attr(crate::baseobjspace::lookup_attr(object, name)),
        out,
    )
}

/// `PyObject_GetOptionalAttrString(object, name, out)` —
/// [`PyObject_GetOptionalAttr`] for a C string name.
///
/// # Safety
/// See [`PyObject_GetOptionalAttr`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetOptionalAttrString(
    object: *mut CPyObject,
    name: *const c_char,
    out: *mut *mut CPyObject,
) -> c_int {
    if out.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe { *out = std::ptr::null_mut() };
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return -1;
    };
    report_optional(
        optional_attr(crate::baseobjspace::getattr_str(object, &name)),
        out,
    )
}

/// `PyObject_HasAttrWithError(object, name)` — 1, 0, or -1 with the error the
/// lookup raised left standing, which is what separates it from
/// [`PyObject_HasAttr`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HasAttrWithError(
    object: *mut CPyObject,
    name: *mut CPyObject,
) -> c_int {
    realize_all([object, name]);
    let (Some(object), Some(name)) = (argument(object), attribute_name(name)) else {
        return -1;
    };
    match optional_attr(crate::baseobjspace::lookup_attr(object, name)) {
        Ok(found) => found.is_some() as c_int,
        Err(()) => -1,
    }
}

/// `PyObject_HasAttrStringWithError(object, name)` —
/// [`PyObject_HasAttrWithError`] for a C string name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HasAttrStringWithError(
    object: *mut CPyObject,
    name: *const c_char,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return -1;
    };
    match optional_attr(crate::baseobjspace::getattr_str(object, &name)) {
        Ok(found) => found.is_some() as c_int,
        Err(()) => -1,
    }
}

/// `PyObject_HasAttr(object, name)` — the error-swallowing spelling
/// (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HasAttr(object: *mut CPyObject, name: *mut CPyObject) -> c_int {
    let found = unsafe { PyObject_HasAttrWithError(object, name) };
    if found < 0 {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    }
    found
}

/// `PyObject_Call(callable, args, kwargs)` — `args` must be a tuple and
/// `kwargs` a dict or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Call(
    callable: *mut CPyObject,
    args: *mut CPyObject,
    keywords: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([callable, args, keywords]);
    let Some([callable, args]) = arguments([callable, args]) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::is_tuple(args) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "argument list must be a tuple",
        ));
        return std::ptr::null_mut();
    }
    let keywords = unsafe { pyobject::from_ref(keywords) };
    // A `dict` subclass is accepted, and what goes on is its mapping: the
    // unpack below reads dict storage directly, and a subclass instance is an
    // object holding a dict rather than being one.
    let kwargs = if keywords.is_null() {
        keywords
    } else if unsafe { crate::baseobjspace::isinstance_dict_w(keywords) } {
        crate::type_methods::resolve_dict_backing(keywords)
    } else {
        pyre_object::PY_NULL
    };
    if kwargs.is_null() && !keywords.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "keyword arguments must be a dict",
        ));
        return std::ptr::null_mut();
    }
    result(call_with_keywords(callable, args, kwargs))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallObject(
    callable: *mut CPyObject,
    args: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([callable, args]);
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    let args = unsafe { pyobject::from_ref(args) };
    if args.is_null() {
        return result(crate::call::call_function_impl_result(callable, &[]));
    }
    unsafe {
        PyObject_Call(
            pyobject::as_pyobj(callable),
            pyobject::as_pyobj(args),
            std::ptr::null_mut(),
        )
    }
}

fn call_with_keywords(
    callable: PyObjectRef,
    args: PyObjectRef,
    kwargs: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(callable);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args);
    let kwargs_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(kwargs);
    let arguments = unsafe {
        pyre_object::tupleobject::w_tuple_items_copy_as_vec(
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        )
    };
    let kwargs = pyre_object::gc_roots::shadow_stack_get(kwargs_slot);
    if kwargs.is_null() {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &arguments,
        );
    }
    let entries = unsafe { pyre_object::w_dict_str_entries(kwargs) };
    if entries.is_empty() {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &arguments,
        );
    }
    let names: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = entries
        .iter()
        .map(|(name, value)| (rustpython_wtf8::Wtf8Buf::from_string(name.clone()), *value))
        .collect();
    call_named(
        pyre_object::gc_roots::shadow_stack_get(callable_slot),
        &arguments,
        &names,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCallable_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    crate::baseobjspace::callable_w(object) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_IsInstance(
    object: *mut CPyObject,
    class: *mut CPyObject,
) -> c_int {
    let Some([object, class]) = arguments([object, class]) else {
        return -1;
    };
    match trap(crate::baseobjspace::isinstance(object, class)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

/// `PyObject_IsSubclass(derived, class)` — `issubclass(derived, class)`
/// (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_IsSubclass(
    derived: *mut CPyObject,
    class: *mut CPyObject,
) -> c_int {
    let Some([derived, class]) = arguments([derived, class]) else {
        return -1;
    };
    match trap(crate::baseobjspace::issubclass(derived, class)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

/// `PyObject_AsFileDescriptor(object)` — the object's own `int` value, or what
/// its `fileno()` reports (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsFileDescriptor(object: *mut CPyObject) -> c_int {
    let Some(object) = argument(object) else {
        return -1;
    };
    trap(file_descriptor_of(object)).unwrap_or(-1)
}

fn file_descriptor_of(object: PyObjectRef) -> Result<c_int, crate::PyError> {
    let descriptor = match crate::baseobjspace::int_w(object) {
        Ok(descriptor) => descriptor,
        Err(_) => {
            let roots = pyre_object::gc_roots::push_roots();
            let Ok(fileno) = crate::baseobjspace::getattr_str(object, "fileno") else {
                return Err(crate::PyError::type_error(
                    "argument must be an int, or have a fileno() method.",
                ));
            };
            let fileno_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(fileno);
            let answer = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(fileno_slot),
                &[],
            )?;
            crate::baseobjspace::int_w(answer)?
        }
    };
    if descriptor < 0 {
        return Err(crate::PyError::value_error(
            "file descriptor cannot be a negative integer",
        ));
    }
    Ok(descriptor as c_int)
}

/// `PyObject_InitVar(object, tp, size)` — [`super::typeobject::PyObject_Init`]
/// for a type that counts items.
///
/// # Safety
/// `object` must be a block of at least `tp->tp_basicsize + size *
/// tp->tp_itemsize` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_InitVar(
    object: *mut CPyVarObject,
    tp: *mut CPyTypeObject,
    size: isize,
) -> *mut CPyVarObject {
    if !object.is_null() {
        unsafe {
            (*object).ob_size = size;
            super::typeobject::PyObject_Init(object as *mut CPyObject, tp);
        }
    }
    object
}

/// `PyObject_GC_IsTracked(object)` — every object pyre holds is reachable by
/// its collector, so this is the constant 1 (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GC_IsTracked(_object: *mut CPyObject) -> c_int {
    1
}

/// `PyObject_GC_IsFinalized(object)` — whether `tp_finalize` has already run
/// for this object (`object.py`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GC_IsFinalized(object: *mut CPyObject) -> c_int {
    (!object.is_null() && super::gc::is_finalized(object as usize)) as c_int
}

/// `PyObject_CallFinalizerFromDealloc(object)` — run the type's
/// `tp_finalize` from a deallocator that is about to free the block.
///
/// The block is at zero when a deallocator reaches here, and a finalizer may
/// hand `self` out, so it holds one reference of its own while it runs: a
/// count above that afterwards is something else having kept the object, and
/// the caller is told to leave the block alone.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallFinalizerFromDealloc(object: *mut CPyObject) -> c_int {
    if object.is_null() {
        return 0;
    }
    let tp = unsafe { (*object).ob_type };
    if tp.is_null() {
        return 0;
    }
    let finalize = unsafe { (*tp).tp_finalize };
    if finalize.is_null() {
        return 0;
    }
    unsafe { (*object).ob_refcnt = 1 };
    unsafe { PyObject_CallFinalizer(object) };
    debug_assert!(
        unsafe { (*object).ob_refcnt } > 0,
        "a finalizer dropped the reference the caller was holding for it"
    );
    // Undo the temporary reference by hand rather than through `decref`,
    // which at zero would re-enter the deallocator this is called from.
    // Whatever is left above zero is something the finalizer handed `self`
    // to, and the caller is told to leave the block alone.
    unsafe { (*object).ob_refcnt -= 1 };
    if unsafe { (*object).ob_refcnt } != 0 {
        -1
    } else {
        0
    }
}

/// `object.py:501 PyObject_CallFinalizer(object)` — run the type's
/// `tp_finalize`, once.
///
/// A collected type's finalizer runs at most once over the object's life: a
/// deallocator chain reaches this from every level that defines one, and a
/// second call would release the same resources twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallFinalizer(object: *mut CPyObject) {
    if object.is_null() {
        return;
    }
    let tp = unsafe { (*object).ob_type };
    if tp.is_null() {
        return;
    }
    let finalize = unsafe { (*tp).tp_finalize };
    if finalize.is_null() {
        return;
    }
    let collected = super::gc::has_gc(tp);
    if collected && super::gc::is_finalized(object as usize) {
        return;
    }
    let finalize: unsafe extern "C" fn(*mut CPyObject) = unsafe { std::mem::transmute(finalize) };
    unsafe { finalize(object) };
    if collected {
        super::gc::mark_finalized(object as usize);
    }
}

/// `PyObject_VisitManagedDict(object, visit, arg)` — nothing to visit.
///
/// A traversal reports the references the C block holds, and the instance
/// namespace is not one of them: it hangs off the interpreter object, which
/// the collector reaches without going through this layer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_VisitManagedDict(
    _object: *mut CPyObject,
    _visit: *const c_void,
    _arg: *mut c_void,
) -> c_int {
    0
}

/// `PyObject_ClearManagedDict(object)` — empty the instance namespace.
///
/// `dictobject.c:7488-7491` states when it is called: "when the object is
/// being freed or cleared by the GC and therefore known to have no
/// references".  A block reached from `clear_garbage` has no interpreter
/// object left to hold a namespace, and this reports nothing -- it returns
/// void, so an error recorded here would be found by whoever calls next.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_ClearManagedDict(object: *mut CPyObject) {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return;
    }
    let dict = crate::baseobjspace::getdict_native(object);
    if !dict.is_null() {
        unsafe { pyre_object::dictmultiobject::w_dict_clear(dict) };
    }
}

/// `PyObject_VectorcallDict(callable, args, nargsf, keywords)` — the vector
/// call whose keywords arrive as a mapping rather than as names beside the
/// values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_VectorcallDict(
    callable: *mut CPyObject,
    args: *const *mut CPyObject,
    nargsf: usize,
    keywords: *mut CPyObject,
) -> *mut CPyObject {
    if keywords.is_null() {
        return unsafe { PyObject_Vectorcall(callable, args, nargsf, std::ptr::null_mut()) };
    }
    let nargs = vectorcall_nargs(nargsf);
    let positional = unsafe { tuple_from_vector(args, nargs) };
    if positional.is_null() {
        return std::ptr::null_mut();
    }
    let answer = unsafe { PyObject_Call(callable, positional, keywords) };
    unsafe { pyobject::decref(positional) };
    answer
}

/// The `tuple` a vector's first `count` entries make, as a new reference.
pub(super) unsafe fn tuple_from_vector(
    args: *const *mut CPyObject,
    count: usize,
) -> *mut CPyObject {
    let tuple = unsafe { super::tupleobject::PyTuple_New(count as isize) };
    if tuple.is_null() {
        return std::ptr::null_mut();
    }
    for index in 0..count {
        let item = unsafe { *args.add(index) };
        // A NULL here would be stored as an empty slot -- `PyTuple_New` fills
        // the block with those already, so nothing downstream would notice --
        // and the hole reaches the interpreter as a value.  The vectorcall
        // path refuses it, and this is the same vector.
        if item.is_null() {
            unsafe { pyobject::decref(tuple) };
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "vectorcall argument vector holds a NULL",
            ));
            return std::ptr::null_mut();
        }
        unsafe { pyobject::incref(item) };
        if unsafe { super::tupleobject::PyTuple_SetItem(tuple, index as isize, item) } != 0 {
            unsafe { pyobject::decref(tuple) };
            return std::ptr::null_mut();
        }
    }
    tuple
}

// ── the object allocator ──────────────────────────────────────────────────
//
// `PyObject_Free` is the deallocator for all three of these, and for a block
// `tp_alloc` handed out; upstream reaches the same raw allocator from both
// directions (`object.py:18-42`, `src/object.c:103-105`).

/// `PyObject_Malloc(size)` — uninitialized, as `malloc` is (`object.py`).
#[unsafe(no_mangle)]
pub extern "C" fn PyObject_Malloc(size: usize) -> *mut c_void {
    pyobject::allocate_raw(size, false)
}

/// `PyObject_Calloc(count, element)` — `count * element` zeroed bytes,
/// refusing a product that would wrap (`object.py`).
#[unsafe(no_mangle)]
pub extern "C" fn PyObject_Calloc(count: usize, element: usize) -> *mut c_void {
    if element != 0 && count > (isize::MAX as usize) / element {
        return std::ptr::null_mut();
    }
    pyobject::allocate_raw(count * element, true)
}

/// `PyObject_Realloc(block, size)` — a NULL block is an allocation
/// (`object.py`).
///
/// # Safety
/// `block` must be NULL or a live block from [`PyObject_Malloc`],
/// [`PyObject_Calloc`] or this function, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Realloc(block: *mut c_void, size: usize) -> *mut c_void {
    if block.is_null() {
        return PyObject_Malloc(size);
    }
    unsafe { pyobject::reallocate_raw(block, size) }
}

/// `Py_TYPE(object)` as a call, which is what an extension reaches when it
/// declares the prototype itself instead of expanding the header's macro.
///
/// The answer is the field, not [`crate::typedef::r#type`]: the macro reads
/// `ob_type` and hands the block back borrowed, so a mirror whose type is not
/// linked yet must answer NULL here exactly as it does there.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_TYPE(object: *mut CPyObject) -> *mut CPyTypeObject {
    if object.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*object).ob_type }
}

/// `Py_IS_TYPE(object, tp)` as a call, for the reason [`Py_TYPE`] is one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IS_TYPE(object: *mut CPyObject, tp: *mut CPyTypeObject) -> c_int {
    if object.is_null() {
        return 0;
    }
    (unsafe { (*object).ob_type } == tp) as c_int
}

/// `Py_TYPE(object)`, borrowed — the mirror a type mirror is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Type(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let Some(class) = crate::typedef::r#type(object) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(class.as_ptr())
}

// ── the constants an extension asks for by identifier ───────────────────

/// The identifiers `Py_GetConstant` takes, under the header's own names.
///
/// An identifier is an index rather than a flag -- a caller asks for exactly
/// one -- so these stay numbers.  Declaring each once mints the count
/// [`CONSTANT_MIRRORS`] is sized by and the table
/// `every_constant_identifier_is_the_number_the_header_gives_it` walks, so an
/// identifier added here is compared against the header without anyone
/// remembering to list it.
macro_rules! constant_identifiers {
    ($($name:ident = $value:expr,)*) => {
        $(const $name: usize = $value;)*

        /// One past the last identifier the header declares.
        const CONSTANT_COUNT: usize = [$($value,)*].len();

        #[cfg(test)]
        const ALL_CONSTANT_IDENTIFIERS: &[(&str, usize)] = &[$((stringify!($name), $value),)*];
    };
}

constant_identifiers! {
    PY_CONSTANT_NONE = 0,
    PY_CONSTANT_FALSE = 1,
    PY_CONSTANT_TRUE = 2,
    PY_CONSTANT_ELLIPSIS = 3,
    PY_CONSTANT_NOT_IMPLEMENTED = 4,
    PY_CONSTANT_ZERO = 5,
    PY_CONSTANT_ONE = 6,
    PY_CONSTANT_EMPTY_STR = 7,
    PY_CONSTANT_EMPTY_BYTES = 8,
    PY_CONSTANT_EMPTY_TUPLE = 9,
}

/// The object an identifier names.
fn constant_object(identifier: usize) -> PyObjectRef {
    match identifier {
        PY_CONSTANT_NONE => pyre_object::w_none(),
        PY_CONSTANT_FALSE => pyre_object::boolobject::w_bool_from(false),
        PY_CONSTANT_TRUE => pyre_object::boolobject::w_bool_from(true),
        PY_CONSTANT_ELLIPSIS => pyre_object::w_ellipsis(),
        PY_CONSTANT_NOT_IMPLEMENTED => pyre_object::w_not_implemented(),
        PY_CONSTANT_ZERO => pyre_object::w_int_new(0),
        PY_CONSTANT_ONE => pyre_object::w_int_new(1),
        PY_CONSTANT_EMPTY_STR => pyre_object::w_str_new(""),
        PY_CONSTANT_EMPTY_BYTES => pyre_object::bytesobject::w_bytes_from_bytes(b""),
        PY_CONSTANT_EMPTY_TUPLE => pyre_object::tupleobject::w_tuple_new(Vec::new()),
        _ => unreachable!("the caller indexed CONSTANT_MIRRORS first"),
    }
}

/// One mirror per constant, minted on first ask.
///
/// The mirror rather than the object: `Py_GetConstantBorrowed` hands it out
/// without a reference, so it has to outlive every caller, and two asks for
/// the same identifier have to answer the same pointer — which for `0` and
/// `1` and the empty containers they would not, each `w_int_new` being its own
/// object.
static CONSTANT_MIRRORS: [std::sync::OnceLock<usize>; CONSTANT_COUNT] =
    [const { std::sync::OnceLock::new() }; CONSTANT_COUNT];

fn constant_mirror(identifier: std::ffi::c_uint) -> Option<*mut CPyObject> {
    let index = identifier as usize;
    let slot = CONSTANT_MIRRORS.get(index)?;
    let held = *slot.get_or_init(|| {
        let raw = pyobject::make_ref(constant_object(index));
        unsafe { (*raw).ob_refcnt = pyobject::REFCNT_IMMORTAL };
        raw as usize
    });
    Some(held as *mut CPyObject)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_GetConstant(identifier: std::ffi::c_uint) -> *mut CPyObject {
    let Some(raw) = constant_mirror(identifier) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    unsafe { pyobject::incref(raw) };
    raw
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_GetConstantBorrowed(identifier: std::ffi::c_uint) -> *mut CPyObject {
    let Some(raw) = constant_mirror(identifier) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    raw
}

/// `Py_ReprEnter(obj)` — 0 when the caller may go on to render `obj`, 1 when
/// it is already being rendered on this thread and the caller should emit its
/// own elision instead.
///
/// The set is the interpreter's own, so a container reached from a C `tp_repr`
/// and one reached from Python see the same recursion.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_ReprEnter(object: *mut CPyObject) -> c_int {
    let Some(object) = argument(object) else {
        // Nothing to record the entry against, which is answered the way a
        // missing thread state is: the caller may go ahead.
        return 0;
    };
    (!crate::display::repr_enter(object)) as c_int
}

/// `Py_ReprLeave(obj)` — undo the entry [`Py_ReprEnter`] recorded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_ReprLeave(object: *mut CPyObject) {
    let object = unsafe { super::pyobject::from_ref(object) };
    if !object.is_null() {
        crate::display::repr_leave(object);
    }
}

/// `Py_HashBuffer(ptr, len)` — the hash `bytes` of the same content has.
///
/// # Safety
/// `pointer` must address `length` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_HashBuffer(pointer: *const std::ffi::c_void, length: isize) -> isize {
    if pointer.is_null() || length < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(pointer as *const u8, length as usize) };
    crate::builtins::hash_buffer(bytes) as isize
}

/// `pyobject.py _Py_HashPointer` — the hash an object's address gives.
///
/// Upstream answers with the address itself.  The rotation here is what the
/// same call does for CPython, and it is what an extension keying a table by
/// address wants: an allocator hands out addresses that agree in their low
/// bits, and a table indexed by the low bits of the hash would put all of them
/// in one chain.
///
/// -1 is what an error is reported by, so the one value that would be mistaken
/// for one is moved.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_HashPointer(pointer: *const std::ffi::c_void) -> isize {
    let bits = pointer as usize;
    let hash = ((bits >> 4) | (bits << (usize::BITS - 4))) as isize;
    match hash {
        -1 => -2,
        hash => hash,
    }
}

/// The name [`Py_HashPointer`] was published under before 3.13.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Py_HashPointer(pointer: *const std::ffi::c_void) -> isize {
    unsafe { Py_HashPointer(pointer) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(Py_HashPointer as *const ());
    std::hint::black_box(_Py_HashPointer as *const ());
    std::hint::black_box(Py_ReprEnter as *const ());
    std::hint::black_box(Py_ReprLeave as *const ());
    std::hint::black_box(Py_HashBuffer as *const ());
    std::hint::black_box(Py_GetConstant as *const ());
    std::hint::black_box(Py_GetConstantBorrowed as *const ());
    std::hint::black_box(PyObject_GetAttrString as *const ());
    std::hint::black_box(PyObject_SetAttrString as *const ());
    std::hint::black_box(PyObject_HasAttrString as *const ());
    std::hint::black_box(PyObject_GetAttr as *const ());
    std::hint::black_box(PyObject_SetAttr as *const ());
    std::hint::black_box(PyObject_Str as *const ());
    std::hint::black_box(PyObject_Repr as *const ());
    std::hint::black_box(PyObject_IsTrue as *const ());
    std::hint::black_box(PyObject_Not as *const ());
    std::hint::black_box(PyObject_Size as *const ());
    std::hint::black_box(PyObject_GetItem as *const ());
    std::hint::black_box(PyObject_SetItem as *const ());
    std::hint::black_box(PyObject_Call as *const ());
    std::hint::black_box(PyObject_CallObject as *const ());
    std::hint::black_box(PyCallable_Check as *const ());
    std::hint::black_box(PyObject_IsInstance as *const ());
    std::hint::black_box(PyObject_Type as *const ());
    std::hint::black_box(Py_TYPE as *const ());
    std::hint::black_box(PyObject_GenericGetAttr as *const ());
    std::hint::black_box(PyObject_GenericSetAttr as *const ());
    std::hint::black_box(PyObject_GenericGetDict as *const ());
    std::hint::black_box(PyObject_GenericSetDict as *const ());
    std::hint::black_box(PyObject_ASCII as *const ());
    std::hint::black_box(PyObject_Bytes as *const ());
    std::hint::black_box(PyObject_Format as *const ());
    std::hint::black_box(PyObject_Dir as *const ());
    std::hint::black_box(PyObject_RichCompare as *const ());
    std::hint::black_box(PyObject_RichCompareBool as *const ());
    std::hint::black_box(PyObject_Hash as *const ());
    std::hint::black_box(PyObject_HashNotImplemented as *const ());
    std::hint::black_box(PyObject_Length as *const ());
    std::hint::black_box(PyObject_DelItem as *const ());
    std::hint::black_box(PyObject_DelItemString as *const ());
    std::hint::black_box(PyObject_DelAttr as *const ());
    std::hint::black_box(PyObject_DelAttrString as *const ());
    std::hint::black_box(PyObject_GetOptionalAttr as *const ());
    std::hint::black_box(PyObject_GetOptionalAttrString as *const ());
    std::hint::black_box(PyObject_HasAttr as *const ());
    std::hint::black_box(PyObject_HasAttrWithError as *const ());
    std::hint::black_box(PyObject_HasAttrStringWithError as *const ());
    std::hint::black_box(PyObject_IsSubclass as *const ());
    std::hint::black_box(PyObject_AsFileDescriptor as *const ());
    std::hint::black_box(PyObject_InitVar as *const ());
    std::hint::black_box(PyObject_GC_IsTracked as *const ());
    std::hint::black_box(PyObject_GC_IsFinalized as *const ());
    std::hint::black_box(PyObject_Malloc as *const ());
    std::hint::black_box(PyObject_Calloc as *const ());
    std::hint::black_box(PyObject_Realloc as *const ());
}

// ── The vectorcall protocol ───────────────────────────────────────────────
//
// A vectorcall passes its arguments as a flat C array rather than a tuple and
// a dict: `args[..nargs]` are positional and `args[nargs..]` are the values
// for the names in `kwnames`, in order.  Pyre answers these by unpacking that
// array and going through its own call path -- the type's `tp_vectorcall` slot
// is an optimisation the caller may not assume was taken, which is why
// `PyVectorcall_Call` falls back to `tp_call` when a type declares no offset
// (`cpyext/src/call.c:120-129`).

/// The bit a caller sets in `nargsf` to say it left a spare slot before
/// `args[0]` that a callee may overwrite with its own receiver.
///
/// Pyre reads the array and never writes it, so the bit only has to be
/// stripped before the count is used.
const VECTORCALL_ARGUMENTS_OFFSET: usize = 1 << (usize::BITS - 1);

/// The positional count carried by an `nargsf`.
fn vectorcall_nargs(nargsf: usize) -> usize {
    nargsf & !VECTORCALL_ARGUMENTS_OFFSET
}

/// Call `callable` with a vectorcall argument vector.
///
/// # Safety
/// `args` must name `vectorcall_nargs(nargsf) + len(kwnames)` readable
/// pointers, and `kwnames` must be NULL or a tuple of `str`.
unsafe fn call_vector(
    callable: PyObjectRef,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    let nargs = vectorcall_nargs(nargsf);

    // Every incoming value is pinned before anything below can collect, and
    // read back from the shadow stack afterwards.  The callable is pinned
    // first: converting a mirror below can realize it, which allocates.
    let roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(callable);
    let kwnames = unsafe { pyobject::from_ref(kwnames) };
    if !kwnames.is_null() && !unsafe { pyre_object::is_tuple(kwnames) } {
        return Err(crate::PyError::type_error(
            "vectorcall keyword names must be a tuple",
        ));
    }
    let kwnames_slot = pyre_object::gc_roots::shadow_stack_len();
    let kwnames = roots.pin_root(kwnames);
    let named = if kwnames.is_null() {
        0
    } else {
        unsafe { pyre_object::tupleobject::w_tuple_len(kwnames) }
    };
    let value_base = pyre_object::gc_roots::shadow_stack_len();
    for index in 0..nargs + named {
        let value = unsafe { pyobject::from_ref(*args.add(index)) };
        if value.is_null() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "vectorcall argument vector holds a NULL",
            ));
        }
        let _ = roots.pin_root(value);
    }
    let value_at = |index: usize| pyre_object::gc_roots::shadow_stack_get(value_base + index);

    // The names come out before the values are read back, so that the
    // allocation each one makes cannot move an address already copied.
    let mut names: Vec<rustpython_wtf8::Wtf8Buf> = Vec::with_capacity(named);
    if named != 0 {
        let items = unsafe {
            pyre_object::tupleobject::w_tuple_items_copy_as_vec(
                pyre_object::gc_roots::shadow_stack_get(kwnames_slot),
            )
        };
        for name in items {
            if !unsafe { pyre_object::is_str(name) } {
                return Err(crate::PyError::type_error(
                    "vectorcall keyword name is not a string",
                ));
            }
            // `call_named` takes the keyword as WTF-8. Copy the raw
            // buffer: `str_utf8_w` rejects a lone surrogate the tuple is
            // still allowed to hold.
            let mut keyword = rustpython_wtf8::Wtf8Buf::new();
            keyword.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(name) });
            names.push(keyword);
        }
    }

    let positional: Vec<PyObjectRef> = (0..nargs).map(value_at).collect();
    if named == 0 {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &positional,
        );
    }
    let keywords: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = names
        .into_iter()
        .enumerate()
        .map(|(index, name)| (name, value_at(nargs + index)))
        .collect();
    call_named(
        pyre_object::gc_roots::shadow_stack_get(callable_slot),
        &positional,
        &keywords,
    )
}

/// `PyObject_Vectorcall(callable, args, nargsf, kwnames)`.
///
/// # Safety
/// See [`call_vector`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Vectorcall(
    callable: *mut CPyObject,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    result(unsafe { call_vector(callable, args, nargsf, kwnames) })
}

/// `PyObject_VectorcallMethod(name, args, nargsf, kwnames)` — `args[0]` is the
/// object the method is looked up on, and stays the call's first argument.
///
/// # Safety
/// See [`call_vector`]; `args` must additionally hold at least one entry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_VectorcallMethod(
    name: *mut CPyObject,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> *mut CPyObject {
    if vectorcall_nargs(nargsf) < 1 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    // `argument` reports a NULL itself, so the arity is the only condition
    // left to raise for.
    realize_all([name, unsafe { *args }]);
    let (Some(name), Some(receiver)) = (argument(name), argument(unsafe { *args })) else {
        return std::ptr::null_mut();
    };
    // The bound method carries the receiver, so the remaining vector is the
    // argument list and the offset bit no longer describes it.
    let Some(method) = trap(crate::baseobjspace::getattr(receiver, name)) else {
        return std::ptr::null_mut();
    };
    let nargs = vectorcall_nargs(nargsf) - 1;
    result(unsafe { call_vector(method, args.add(1), nargs, kwnames) })
}

/// The C signature a `tp_vectorcall_offset` names.
type VectorcallFunc = unsafe extern "C" fn(
    *mut CPyObject,
    *const *mut CPyObject,
    usize,
    *mut CPyObject,
) -> *mut CPyObject;

/// `_PyVectorcall_FunctionInline` -- the function a callable's type lends it,
/// read out of the callable itself at the offset the type names.
///
/// The flag and the offset are one fact in two places: `type_ready_pre_checks`
/// refuses a type that sets the bit without filling `tp_call` and a positive
/// offset, so the bit alone answers whether the read is meant.  Upstream
/// asserts the offset instead of testing it, which is a wild read in a build
/// with assertions off; the test costs a compare and cannot be the thing that
/// goes wrong.
///
/// A null stored at the offset is not an error -- `_PyObject_VectorcallTstate`
/// answers `NULL` for it and its caller falls back to `tp_call`.
///
/// # Safety
/// `raw` must be a live mirror, and its type's `tp_vectorcall_offset` must
/// describe that mirror's own storage -- which is what the flag asserts.
pub(super) unsafe fn vectorcall_function(raw: *mut CPyObject) -> Option<VectorcallFunc> {
    if raw.is_null() {
        return None;
    }
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null()
        || !unsafe {
            (*tp)
                .tp_flags
                .contains(super::typeobject::TpFlags::PY_TPFLAGS_HAVE_VECTORCALL)
        }
    {
        return None;
    }
    let offset = unsafe { (*tp).tp_vectorcall_offset };
    if offset <= 0 {
        return None;
    }
    unsafe { *((raw as *mut u8).offset(offset) as *const Option<VectorcallFunc>) }
}

/// `PyVectorcall_Call(callable, tuple, dict)` — the tuple/dict spelling of a
/// vectorcall, which is what a type's `tp_call` is set to when it declares one
/// (`src/call.c PyVectorcall_Call`).
///
/// The function is read out of the instance at its type's
/// `tp_vectorcall_offset` and called.  Answering with `tp_call` in every case
/// cannot work: a type that declares an offset is one whose `tp_call` is this
/// very entry point, so the two call each other without end.  Cython's
/// function objects are exactly that shape, and calling one recursed until the
/// stack ran out.
///
/// # Safety
/// `tuple` must be a tuple and `dict` NULL or a dict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyVectorcall_Call(
    callable: *mut CPyObject,
    tuple: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([callable, tuple, dict]);
    if callable.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    // `_PyVectorcall_Call` asserts both of these before it unpacks them.  An
    // assertion is a crash in one build and a wild read in every other, so a
    // caller that broke the contract is told so instead.
    let mismatched = |object: *mut CPyObject, wanted_tuple: bool| unsafe {
        let value = pyobject::from_ref(object);
        !value.is_null()
            && !match wanted_tuple {
                true => pyre_object::is_tuple(value),
                false => pyre_object::is_dict(value),
            }
    };
    if mismatched(tuple, true) || mismatched(dict, false) {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let tp = unsafe { (*callable).ob_type };
    let offset = match tp.is_null() {
        true => 0,
        false => unsafe { (*tp).tp_vectorcall_offset },
    };
    // `src/call.c`: an object whose type declares no offset -- which is every
    // object this runtime owns, since a mirror carries none -- is called
    // through `tp_call` as it stands.
    if offset == 0 && !tp.is_null() && !unsafe { (*tp).tp_call }.is_null() {
        return unsafe { PyObject_Call(callable, tuple, dict) };
    }
    let function = match offset > 0 {
        false => None,
        true => unsafe { *((callable as *mut u8).offset(offset) as *const Option<VectorcallFunc>) },
    };
    let Some(function) = function else {
        let name = crate::type_methods::arg_type_name(unsafe { pyobject::from_ref(callable) });
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "'{name}' object does not support vectorcall"
        )));
        return std::ptr::null_mut();
    };
    unsafe { call_vectorcall_slot(function, callable, tuple, dict) }
}

/// `_PyStack_UnpackDict` and the call it feeds: the positional values first,
/// then one value per name in `kwnames`, in that order.
///
/// Each value is handed over as an owned reference and released afterwards,
/// which is what the unpacking upstream does -- the callee may drop the tuple
/// or the mapping the values came from.
///
/// # Safety
/// `tuple` must be a tuple and `dict` NULL or a dict.
unsafe fn call_vectorcall_slot(
    function: VectorcallFunc,
    callable: *mut CPyObject,
    tuple: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let w_tuple = unsafe { pyobject::from_ref(tuple) };
    let _ = roots.pin_root(w_tuple);
    let w_dict = unsafe { pyobject::from_ref(dict) };
    let _ = roots.pin_root(w_dict);
    let at = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);

    let positional: Vec<PyObjectRef> = match at(0).is_null() {
        true => Vec::new(),
        false => (0..unsafe { pyre_object::w_tuple_len(at(0)) })
            .filter_map(|index| unsafe { pyre_object::w_tuple_getitem(at(0), index as i64) })
            .collect(),
    };
    let nargs = positional.len();
    let keywords: Vec<(PyObjectRef, PyObjectRef)> = match at(1).is_null() {
        true => Vec::new(),
        false => unsafe { pyre_object::dictmultiobject::w_dict_items(at(1)) },
    };

    // The names go over as one tuple, which allocates, so every value is on
    // the shadow stack before it is built and all of them are read back.
    let values = base + 2;
    for value in positional
        .into_iter()
        .chain(keywords.iter().map(|&(_, value)| value))
    {
        let _ = roots.pin_root(value);
    }
    let names = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(match keywords.is_empty() {
        true => pyre_object::PY_NULL,
        false => pyre_object::w_tuple_new(keywords.iter().map(|&(name, _)| name).collect()),
    });

    let count = nargs + keywords.len();
    let vector: Vec<*mut CPyObject> = (0..count)
        .map(|index| pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(values + index)))
        .collect();
    let kwnames = match pyre_object::gc_roots::shadow_stack_get(names).is_null() {
        true => std::ptr::null_mut(),
        false => pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(names)),
    };
    let answer = unsafe { function(callable, vector.as_ptr(), nargs, kwnames) };
    for held in vector {
        unsafe { pyobject::decref(held) };
    }
    unsafe { pyobject::decref(kwnames) };
    answer
}

/// `PyObject_CallNoArgs(callable)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallNoArgs(callable: *mut CPyObject) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    result(crate::call::call_function_impl_result(callable, &[]))
}

/// `PyObject_CallOneArg(callable, arg)`.
///
/// # Safety
/// `arg` must be a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallOneArg(
    callable: *mut CPyObject,
    arg: *mut CPyObject,
) -> *mut CPyObject {
    realize_all([callable, arg]);
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    let argument_vector = [arg];
    result(unsafe { call_vector(callable, argument_vector.as_ptr(), 1, std::ptr::null_mut()) })
}

#[cfg(test)]
mod tests {
    /// `Py_GetConstant` takes the number an extension compiled against
    /// `include/pyre3.15t/object.h` wrote, so the identifiers are one table in
    /// two places.  This walks the header and rejects any identifier whose
    /// number the Rust side spells differently, or does not spell at all.
    #[test]
    fn every_constant_identifier_is_the_number_the_header_gives_it() {
        const HEADER: &str = include_str!("../../../../include/pyre3.15t/object.h");

        let mut header: Vec<(&str, usize)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_CONSTANT_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let Ok(value) = body.trim().parse::<usize>() else {
                continue;
            };
            header.push((name, value));
        }

        assert_eq!(
            header.len(),
            super::ALL_CONSTANT_IDENTIFIERS.len(),
            "object.h declares {} identifiers and this file declares {}",
            header.len(),
            super::ALL_CONSTANT_IDENTIFIERS.len()
        );

        for (name, theirs) in &header {
            let found = super::ALL_CONSTANT_IDENTIFIERS
                .iter()
                .find(|(known, _)| known.strip_prefix("PY_CONSTANT_") == Some(name));
            let Some((_, ours)) = found else {
                panic!(
                    "object.h defines Py_CONSTANT_{name} = {theirs}, \
                     and this file has no PY_CONSTANT_{name}"
                );
            };
            assert_eq!(
                ours, theirs,
                "Py_CONSTANT_{name} is {theirs} in object.h and {ours} here"
            );
        }
    }
}
