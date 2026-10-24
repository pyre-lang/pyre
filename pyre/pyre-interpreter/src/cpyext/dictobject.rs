//! `dict` -- PyPy `cpyext/dictobject.py`.

use super::object::{argument, arguments};
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8;
use std::ffi::{CStr, c_char, c_int};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_New() -> *mut CPyObject {
    pyobject::make_ref(pyre_object::dictmultiobject::w_dict_new())
}

/// The mapping a `PyDict_*` entry point operates on.
///
/// `PyDict_Check` is the gate every one of these applies, so a `dict` subclass
/// is accepted -- and the operation reaches the concrete mapping rather than
/// dispatching the subclass's Python-level override, which is what makes
/// `PyDict_GetItem` on a `defaultdict` miss instead of calling
/// `__missing__`.  In pyre a `dict` subclass instance is not a dict but an
/// object holding one, so that concrete mapping has to be resolved rather than
/// used directly.
///
/// A `mappingproxy` is not a `dict` subclass and is refused here, even though
/// it also wraps a mapping `resolve_dict_backing` would hand back.
fn dict_argument(object: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let value = argument(object)?;
    match dict_backing(value) {
        Some(backing) => Some(backing),
        None => {
            super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
                "{function}(): dict expected"
            )));
            None
        }
    }
}

/// The concrete mapping behind a value that passes `PyDict_Check`, with
/// nothing recorded when it does not -- the caller says what a refusal is.
fn dict_backing(value: PyObjectRef) -> Option<PyObjectRef> {
    if !unsafe { crate::baseobjspace::isinstance_dict_w(value) } {
        return None;
    }
    let backing = crate::type_methods::resolve_dict_backing(value);
    (!backing.is_null()).then_some(backing)
}

/// A `dict`, or the `SystemError` `PyErr_BadInternalCall` records -- the
/// refusal the entry points that grew a `PyObject **` out-parameter make,
/// rather than the `TypeError` the older ones make.
fn internal_dict(object: *mut CPyObject) -> Option<PyObjectRef> {
    let value = argument(object)?;
    let backing = dict_backing(value);
    if backing.is_none() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
    }
    backing
}

fn key_name(name: *const c_char) -> Option<String> {
    if name.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, key, value]);
    let Some(dict) = dict_argument(object, "PyDict_SetItem") else {
        return -1;
    };
    let Some([key, value]) = arguments([key, value]) else {
        return -1;
    };
    if trap(crate::baseobjspace::setitem(dict, key, value)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetItemString(
    object: *mut CPyObject,
    key: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, value]);
    let Some(dict) = dict_argument(object, "PyDict_SetItemString") else {
        return -1;
    };
    let (Some(key), Some(value)) = (key_name(key), argument(value)) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(value);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            &key,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )
    };
    0
}

/// Borrowed, and — as upstream and CPython both specify — it does not set an
/// exception when the key is missing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([object, key]);
    let Some(dict) = dict_argument(object, "PyDict_GetItem") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    let Some(key) = argument(key) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => pyobject::borrow_from(object, value),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemString(
    object: *mut CPyObject,
    key: *const c_char,
) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, "PyDict_GetItemString") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    let Some(key) = key_name(key) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return std::ptr::null_mut();
    };
    match unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, &key) } {
        Some(value) => pyobject::borrow_from(object, value),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_DelItem(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    super::object::realize_all([object, key]);
    let Some(dict) = dict_argument(object, "PyDict_DelItem") else {
        return -1;
    };
    let Some(key) = argument(key) else {
        return -1;
    };
    if trap(crate::baseobjspace::delitem(dict, key)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Size(object: *mut CPyObject) -> isize {
    let Some(dict) = dict_argument(object, "PyDict_Size") else {
        return -1;
    };
    unsafe { pyre_object::dictmultiobject::w_dict_len(dict) as isize }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Contains(object: *mut CPyObject, key: *mut CPyObject) -> c_int {
    super::object::realize_all([object, key]);
    let Some(dict) = dict_argument(object, "PyDict_Contains") else {
        return -1;
    };
    let Some(key) = argument(key) else {
        return -1;
    };
    match trap(crate::baseobjspace::contains(dict, key)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { crate::baseobjspace::isinstance_dict_w(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::DICT_TYPE)) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyDict_New as *const ());
    std::hint::black_box(PyDict_SetItem as *const ());
    std::hint::black_box(PyDict_SetItemString as *const ());
    std::hint::black_box(PyDict_GetItem as *const ());
    std::hint::black_box(PyDict_GetItemString as *const ());
    std::hint::black_box(PyDict_DelItem as *const ());
    std::hint::black_box(PyDict_Size as *const ());
    std::hint::black_box(PyDict_Contains as *const ());
    std::hint::black_box(PyDict_Pop as *const ());
    std::hint::black_box(PyDict_PopString as *const ());
    std::hint::black_box(PyDictProxy_New as *const ());
    std::hint::black_box(PyDict_Check as *const ());
    std::hint::black_box(PyDict_CheckExact as *const ());
}

/// `obj.keys()`.
fn call_keys(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let method = crate::baseobjspace::getattr_str(object, "keys")?;
    crate::call::call_function_impl_result(method, &[])
}

/// Every item an iterable yields.
fn unpack_all(object: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    crate::baseobjspace::unpackiterable(object, -1)
}

/// `PyDict_Clear(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Clear(object: *mut CPyObject) {
    let Some(dict) = dict_argument(object, "PyDict_Clear") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return;
    };
    unsafe { pyre_object::dictmultiobject::w_dict_clear(dict) };
}

/// `PyDict_Copy(dict)` — a new `dict` with the same items.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Copy(object: *mut CPyObject) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, "PyDict_Copy") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(unsafe { pyre_object::dictmultiobject::w_dict_copy(dict) })
}

/// `PyDict_DelItemString(dict, key)`.
///
/// # Safety
/// `key` must be a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_DelItemString(object: *mut CPyObject, key: *const c_char) -> c_int {
    let (Some(dict), Some(name)) = (dict_argument(object, "PyDict_DelItemString"), key_name(key))
    else {
        return -1;
    };
    let w_key = pyre_object::w_str_new_managed(&name);
    match trap(crate::baseobjspace::delitem(dict, w_key)) {
        Some(()) => 0,
        None => -1,
    }
}

/// `PyDict_GetItemWithError(dict, key)` — a borrowed reference, or NULL with
/// no exception set when the key is simply absent (`dictobject.py`).
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemWithError(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([object, key]);
    let (Some(dict), Some(key)) = (
        dict_argument(object, "PyDict_GetItemWithError"),
        argument(key),
    ) else {
        return std::ptr::null_mut();
    };
    match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => pyobject::borrow_from(object, value),
        Err(error) if error.kind == crate::PyErrorKind::KeyError => std::ptr::null_mut(),
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            std::ptr::null_mut()
        }
    }
}

/// Fill `*out` with a *strong* reference to `dict[key]`, answering 1 when the
/// key was there, 0 when it was not and -1 on error.
///
/// # Safety
/// `out` must be null or a writable `PyObject *`.
pub(super) unsafe fn get_item_ref(
    found: Result<Option<PyObjectRef>, crate::PyError>,
    out: *mut *mut CPyObject,
) -> c_int {
    if !out.is_null() {
        unsafe { *out = std::ptr::null_mut() };
    }
    match found {
        Ok(None) => 0,
        Ok(Some(value)) => {
            if !out.is_null() {
                unsafe { *out = pyobject::make_ref(value) };
            }
            1
        }
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

/// `PyDict_GetItemRef(dict, key, &result)`.
///
/// # Safety
/// `result` must be null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemRef(
    object: *mut CPyObject,
    key: *mut CPyObject,
    result: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, key]);
    let (Some(dict), Some(key)) = (dict_argument(object, "PyDict_GetItemRef"), argument(key))
    else {
        return -1;
    };
    let found = match crate::baseobjspace::getitem(dict, key) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(error) => Err(error),
    };
    unsafe { get_item_ref(found, result) }
}

/// `PyDict_SetDefault(dict, key, default)` — the value `key` already had, or
/// `default` after storing it, borrowed from the dict either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetDefault(
    object: *mut CPyObject,
    key: *mut CPyObject,
    default_value: *mut CPyObject,
) -> *mut CPyObject {
    let mut value = std::ptr::null_mut();
    if unsafe { PyDict_SetDefaultRef(object, key, default_value, &mut value) } < 0 {
        return std::ptr::null_mut();
    }
    // The strong reference the `Ref` spelling answers with is what makes this
    // one a borrow: the dict holds the value, and the caller is handed the
    // dict's reference rather than one of its own.
    let borrowed = pyobject::borrow_from(object, unsafe { pyobject::from_ref(value) });
    unsafe { pyobject::decref(value) };
    borrowed
}

/// `PyDict_SetDefaultRef(dict, key, default, &result)` — 1 when the key was
/// already there, 0 when the default was inserted, -1 on failure.
///
/// The reference handed back is a new one either way, which is what separates
/// it from the borrowing `PyDict_SetDefault` beside it.  A NULL
/// `result` is the caller saying it wants the insertion and not the value.
///
/// # Safety
/// `result` must be null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_SetDefaultRef(
    object: *mut CPyObject,
    key: *mut CPyObject,
    default_value: *mut CPyObject,
    result: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, key, default_value]);
    // Nothing is handed back unless the whole call works, so the failure
    // arms below can just leave this as it is.
    if !result.is_null() {
        unsafe { *result = std::ptr::null_mut() };
    }
    let (Some(dict), Some(key), Some(default_value)) = (
        dict_argument(object, "PyDict_SetDefaultRef"),
        argument(key),
        argument(default_value),
    ) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(key);
    let default_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(default_value);
    // The key is looked for once, so which of the two things happened is read
    // off the size either side of it.  Asking the dict whether it holds the key
    // would run the key's `__hash__` and `__eq__` a second time, and a key that
    // counts them would see this call as two.
    let before = unsafe {
        pyre_object::dictmultiobject::w_dict_len(pyre_object::gc_roots::shadow_stack_get(dict_slot))
    };
    let key = pyre_object::gc_roots::shadow_stack_get(key_slot);
    let found = unsafe {
        pyre_object::dictmultiobject::w_dict_setdefault_checked(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(default_slot),
        )
    }
    .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(key));
    let Some(found) = trap(found) else {
        return -1;
    };
    let present = unsafe {
        pyre_object::dictmultiobject::w_dict_len(pyre_object::gc_roots::shadow_stack_get(dict_slot))
    } == before;
    if !result.is_null() {
        unsafe { *result = pyobject::make_ref(found) };
    }
    present as c_int
}

/// `PyDict_GetItemStringRef(dict, key, &result)`.
///
/// # Safety
/// `key` must be NUL-terminated and `result` null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_GetItemStringRef(
    object: *mut CPyObject,
    key: *const c_char,
    result: *mut *mut CPyObject,
) -> c_int {
    let (Some(dict), Some(name)) = (
        dict_argument(object, "PyDict_GetItemStringRef"),
        key_name(key),
    ) else {
        return -1;
    };
    let found = Ok(unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, &name) });
    unsafe { get_item_ref(found, result) }
}

/// Take `key` out of `dict`, answering 1 when it was there, 0 when it was
/// not and -1 on failure.
///
/// # Safety
/// `result` must be null or a writable `PyObject *`.
unsafe fn pop_from(dict: PyObjectRef, key: PyObjectRef, result: *mut *mut CPyObject) -> c_int {
    // An empty dictionary is answered without hashing the key, so a key that
    // cannot be hashed is a miss there rather than a `TypeError`.
    if unsafe { pyre_object::dictmultiobject::w_dict_len(dict) } == 0 {
        return unsafe { get_item_ref(Ok(None), result) };
    }
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let key = roots.pin_root(key);
    let found = unsafe {
        pyre_object::dictmultiobject::w_dict_pop_checked(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
        )
    }
    .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(key));
    unsafe { get_item_ref(found, result) }
}

/// `PyDict_Pop(dict, key, &result)` — 1 when the key was there and has been
/// removed, 0 when it was not, -1 on failure.
///
/// A NULL `result` throws the value away, which is how a caller spells
/// "remove it if it is there".
///
/// # Safety
/// `result` must be null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Pop(
    object: *mut CPyObject,
    key: *mut CPyObject,
    result: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([object, key]);
    let (Some(dict), Some(key)) = (internal_dict(object), argument(key)) else {
        if !result.is_null() {
            unsafe { *result = std::ptr::null_mut() };
        }
        return -1;
    };
    unsafe { pop_from(dict, key, result) }
}

/// `PyDict_PopString(dict, key, &result)`.
///
/// # Safety
/// `key` must be NUL-terminated and `result` null or a writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_PopString(
    object: *mut CPyObject,
    key: *const c_char,
    result: *mut *mut CPyObject,
) -> c_int {
    super::object::realize_all([object]);
    let (Some(name), Some(dict)) = (key_name(key), internal_dict(object)) else {
        if !result.is_null() {
            unsafe { *result = std::ptr::null_mut() };
        }
        return -1;
    };
    unsafe { pop_from(dict, pyre_object::w_str_new_managed(&name), result) }
}

/// `PyDictProxy_New(mapping)` — the read-only view `types.MappingProxyType`
/// builds, over anything with `__getitem__` that is not a list or a tuple.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDictProxy_New(object: *mut CPyObject) -> *mut CPyObject {
    let Some(mapping) = argument(object) else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::typedef::mappingproxy_from_mapping(mapping))
}

/// Build one of the three view lists.  Upstream answers a `list` rather than a
/// view object (`dictobject.py:180-206`), the C caller expecting something it
/// can index.
fn view_list(
    object: *mut CPyObject,
    name: &str,
    part: impl FnOnce(PyObjectRef) -> PyObjectRef,
) -> *mut CPyObject {
    let Some(dict) = dict_argument(object, name) else {
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    let dict = roots.pin_root(dict);
    pyobject::make_ref(part(dict))
}

/// `PyDict_Keys(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Keys(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Keys", |dict| {
        pyre_object::listobject::w_list_new(
            unsafe { pyre_object::dictmultiobject::w_dict_items(dict) }
                .into_iter()
                .map(|(key, _)| key)
                .collect(),
        )
    })
}

/// `PyDict_Values(dict)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Values(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Values", |dict| {
        pyre_object::listobject::w_list_new(
            unsafe { pyre_object::dictmultiobject::w_dict_items(dict) }
                .into_iter()
                .map(|(_, value)| value)
                .collect(),
        )
    })
}

/// `PyDict_Items(dict)` — a list of `(key, value)` pairs.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Items(object: *mut CPyObject) -> *mut CPyObject {
    view_list(object, "PyDict_Items", |dict| {
        let mut items = pyre_object::gc_roots::RootedItems::new();
        for (key, value) in unsafe { pyre_object::dictmultiobject::w_dict_items(dict) } {
            items.push(pyre_object::tupleobject::w_tuple_new(vec![key, value]));
        }
        pyre_object::listobject::w_list_new(items.take())
    })
}

/// Whether `object` still iterates the way `dict` does — `Py_TYPE(b)->tp_iter
/// == dict_iter`, the second half of the condition [`PyDict_Merge`] reads its
/// source under.
///
/// An object whose type cannot be read answers `true`: it is then an exact dict
/// as far as anything here can tell, and the direct read is what that deserves.
fn iterates_as_dict(object: PyObjectRef) -> bool {
    let Some(own_type) = crate::typedef::r#type(object) else {
        return true;
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
    unsafe {
        match (
            crate::baseobjspace::lookup_in_type(
                own_type.as_ptr(),
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__iter__")),
            ),
            crate::baseobjspace::lookup_in_type(
                dict_type,
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__iter__")),
            ),
        ) {
            (Some(own), Some(canonical)) => std::ptr::eq(own, canonical),
            _ => true,
        }
    }
}

/// `PyDict_Merge(a, b, override)` — copy `b`'s items into `a`, replacing an
/// existing key only when `over` is non-zero.
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Merge(
    into: *mut CPyObject,
    from: *mut CPyObject,
    over: c_int,
) -> c_int {
    super::object::realize_all([into, from]);
    let (Some(target), Some(source)) = (dict_argument(into, "PyDict_Merge"), argument(from)) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[target, source]);
    let target = || pyre_object::gc_roots::shadow_stack_get(base);
    let source_now = pyre_object::gc_roots::shadow_stack_get(base + 1);

    // The pairs go on the shadow stack rather than staying in the vector they
    // arrive in: `contains` and `setitem` below run the target's own
    // `__contains__` / `__setitem__` and re-hash the key, so either can
    // collect, and a `Vec` is not somewhere the collector looks.  The
    // containers are already re-read per iteration for exactly that reason.
    //
    // A mapping that is not a dict is read through `keys()`, as upstream does
    // for the non-dict case (`dictobject.py:214-231`).
    //
    // `dict_merge` takes the entry-by-entry path only for a source whose
    // `tp_iter` is still `dict`'s: a subclass that iterates its own way is read
    // through `keys()` so that the override is what decides the order and the
    // membership.  One that does not override it is read directly, so a `keys()`
    // override alone does not change what a merge sees.
    let source_mapping = if unsafe { crate::baseobjspace::isinstance_dict_w(source_now) }
        && iterates_as_dict(source_now)
    {
        crate::type_methods::resolve_dict_backing(source_now)
    } else {
        pyre_object::PY_NULL
    };
    let (pairs, count) = if !source_mapping.is_null() {
        let items: Vec<(PyObjectRef, PyObjectRef)> =
            unsafe { pyre_object::dictmultiobject::w_dict_items(source_mapping) };
        let flat: Vec<PyObjectRef> = items
            .iter()
            .flat_map(|&(key, value)| [key, value])
            .collect();
        (pyre_object::gc_roots::pin_roots(&flat), items.len())
    } else {
        let keys = match trap(call_keys(source_now)) {
            Some(keys) => keys,
            None => return -1,
        };
        let keys_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(keys);
        let keys = match trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
            keys_slot,
        ))) {
            Some(keys) => keys,
            None => return -1,
        };
        // `getitem` runs the mapping's own `__getitem__`, so the keys are read
        // back per iteration and each pair is published as it is produced.
        let unpacked = pyre_object::gc_roots::pin_roots(&keys);
        let pairs = pyre_object::gc_roots::shadow_stack_len();
        for index in 0..keys.len() {
            let value = match trap(crate::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                pyre_object::gc_roots::shadow_stack_get(unpacked + index),
            )) {
                Some(value) => value,
                None => return -1,
            };
            let _ = roots.pin_root(pyre_object::gc_roots::shadow_stack_get(unpacked + index));
            let _ = roots.pin_root(value);
        }
        (pairs, keys.len())
    };

    for index in 0..count {
        if over == 0 {
            match crate::baseobjspace::contains(
                target(),
                pyre_object::gc_roots::shadow_stack_get(pairs + index * 2),
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    super::pyerrors::set_pending_error(error);
                    return -1;
                }
            }
        }
        let key = pyre_object::gc_roots::shadow_stack_get(pairs + index * 2);
        let value = pyre_object::gc_roots::shadow_stack_get(pairs + index * 2 + 1);
        if trap(crate::baseobjspace::setitem(target(), key, value)).is_none() {
            return -1;
        }
    }
    0
}

/// `PyDict_Update(a, b)` — `PyDict_Merge(a, b, 1)` (`dictobject.py`).
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Update(into: *mut CPyObject, from: *mut CPyObject) -> c_int {
    unsafe { PyDict_Merge(into, from, 1) }
}

/// `PyDict_MergeFromSeq2(a, seq2, override)` — `seq2` yields `(key, value)`
/// pairs.
///
/// # Safety
/// Both arguments must be null or live references.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_MergeFromSeq2(
    into: *mut CPyObject,
    seq2: *mut CPyObject,
    over: c_int,
) -> c_int {
    super::object::realize_all([into, seq2]);
    let (Some(target), Some(sequence)) =
        (dict_argument(into, "PyDict_MergeFromSeq2"), argument(seq2))
    else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[target, sequence]);
    let sequence = roots.get(base + 1);
    let target = || pyre_object::gc_roots::shadow_stack_get(base);

    let Some(pairs) = trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
        base + 1,
    ))) else {
        return -1;
    };
    // The unpacked sequence and each pair's two elements are read back out of
    // the shadow stack, not carried in their vectors: `unpack_all`, `contains`
    // and `setitem` all run user code, and a `Vec` is not somewhere the
    // collector looks.  Each iteration opens its own scope so the pushes it
    // makes are popped again rather than growing with the sequence.
    let unpacked = pyre_object::gc_roots::pin_roots(&pairs);
    for index in 0..pairs.len() {
        let pair_roots = pyre_object::gc_roots::push_roots();
        let pair_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pair_roots.pin_root(pyre_object::gc_roots::shadow_stack_get(unpacked + index));
        let Some(items) = trap(unpack_all(pyre_object::gc_roots::shadow_stack_get(
            pair_slot,
        ))) else {
            return -1;
        };
        if items.len() != 2 {
            super::pyerrors::set_pending_error(crate::PyError::value_error(format!(
                "dictionary update sequence element #{index} has length {}; 2 is required",
                items.len()
            )));
            return -1;
        }
        let element = pyre_object::gc_roots::pin_roots(&items);
        if over == 0 {
            match crate::baseobjspace::contains(
                target(),
                pyre_object::gc_roots::shadow_stack_get(element),
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    super::pyerrors::set_pending_error(error);
                    return -1;
                }
            }
        }
        let key = pyre_object::gc_roots::shadow_stack_get(element);
        let value = pyre_object::gc_roots::shadow_stack_get(element + 1);
        if trap(crate::baseobjspace::setitem(target(), key, value)).is_none() {
            return -1;
        }
    }
    0
}

/// The key snapshot `PyDict_Next` walks, one per dictionary being iterated.
///
/// Upstream keeps it in the mirror's own `_tmpkeys` field, holding a strong
/// reference so the keys stay alive and the collector keeps forwarding them
/// (`dictobject.py`).  Pyre's mirror has no such field, so the
/// reference is held here instead, keyed by the mirror's address; it is the
/// reference, not this table, that roots the list.
type SnapshotMap = super::address_table::HeldMap<usize>;
use super::address_table::{AddressTable, hold};

static ITERATION_KEYS: AddressTable<SnapshotMap> = AddressTable::new(SnapshotMap::with_hasher(
    std::hash::BuildHasherDefault::new(),
));

/// The lock word can be held by a thread the child does not have; the
/// snapshots the payload names are still mapped, because `fork` copies the
/// address space, and each is still owed the release its walk ends with.
pub(super) unsafe fn after_fork_child() {
    unsafe { ITERATION_KEYS.reinit_after_fork() };
}

/// `PyDict_Next(dict, &pos, &key, &value)` — 1 for each pair, 0 once they are
/// all reported.  `pos` starts at 0 and is only ever handed back
/// (`dictobject.py`).
///
/// The keys are snapshotted on the first call, so a value reassigned during
/// the walk is seen but a new key is not -- which is the documented contract.
///
/// # Safety
/// `pos` must be a writable `Py_ssize_t` the caller initialised to 0, and
/// `key` / `value` must be null or writable `PyObject *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDict_Next(
    object: *mut CPyObject,
    pos: *mut isize,
    key: *mut *mut CPyObject,
    value: *mut *mut CPyObject,
) -> c_int {
    let Some(dict) = dict_argument(object, "PyDict_Next") else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    if pos.is_null() {
        return 0;
    }
    let index = unsafe { *pos };

    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);

    let snapshot = if index == 0 {
        let keys: Vec<PyObjectRef> = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                dict_slot,
            ))
        }
        .into_iter()
        .map(|(key, _)| key)
        .collect();
        let held = pyobject::make_ref(pyre_object::listobject::w_list_new(keys));
        let previous = ITERATION_KEYS
            .lock()
            .insert(hold(object as usize), held as usize);
        if let Some(previous) = previous {
            unsafe { pyobject::decref(previous as *mut CPyObject) };
        }
        held
    } else {
        match ITERATION_KEYS.lock().get(&(object as usize)) {
            // `pos` should have been 0; there is nothing to walk.
            None => return 0,
            Some(&held) => held as *mut CPyObject,
        }
    };

    let w_keys = unsafe { pyobject::from_ref(snapshot) };
    let length = unsafe { pyre_object::listobject::w_list_len(w_keys) } as isize;
    if index >= length {
        if let Some(held) = ITERATION_KEYS.lock().remove(&(object as usize)) {
            unsafe { pyobject::decref(held as *mut CPyObject) };
        }
        return 0;
    }

    let Some(w_key) = (unsafe { pyre_object::listobject::w_list_getitem(w_keys, index as i64) })
    else {
        return 0;
    };
    // The lookup below can collect, so the key is pinned first and both it and
    // the value are read back from the shadow stack afterwards.
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_key);
    let Some(w_value) = trap(crate::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )) else {
        return 0;
    };
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_value);
    unsafe { *pos = index + 1 };
    if !key.is_null() {
        unsafe {
            *key = pyobject::borrow_from(object, pyre_object::gc_roots::shadow_stack_get(key_slot))
        };
    }
    if !value.is_null() {
        unsafe {
            *value =
                pyobject::borrow_from(object, pyre_object::gc_roots::shadow_stack_get(value_slot))
        };
    }
    1
}

/// Drop the snapshot a `PyDict_Next` walk left behind.
///
/// A caller that stops walking before the end never reaches the arm that frees
/// it, so the mirror's own deallocation is the other end of that lifetime.
pub(super) fn forget_iteration(mirror: usize) {
    if let Some(held) = ITERATION_KEYS.take(mirror) {
        unsafe { pyobject::decref(held as *mut CPyObject) };
    }
}
