//! The `PyCFunction` carrier -- PyPy `cpyext/methodobject.py`.
//!
//! Upstream's `W_PyCFunctionObject` is an interp-level class holding the
//! `PyMethodDef` pointer, the bound `self` and the defining module.  Pyre has
//! no typed-payload class for it yet, so the carrier is an instance of a native
//! type whose reserved dict keys hold the same three values — the shape
//! `_ctypes`'s `CFuncPtr` already uses for a native function pointer
//! (`module/_ctypes/funcptr.rs`).  Replacing it with a `#[pyre_class]` payload
//! is part of the C-defined-types slice, where a typed mirror has to exist
//! anyway.

use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::hash::BuildHasherDefault;

bitflags::bitflags! {
    /// The `methodobject.h` flags a `PyMethodDef` row carries in `ml_flags`.
    ///
    /// Two places spell this table: the header an extension compiles against
    /// and this declaration.  `every_meth_flag_is_the_bit_the_header_gives_it`
    /// compares the two, walking `Flags::FLAGS` rather than a list somebody
    /// has to remember to extend.
    ///
    /// Four of them name a calling convention rather than a property --
    /// `METH_VARARGS`, `METH_NOARGS`, `METH_O`, `METH_FASTCALL` -- and a row
    /// may name exactly one; the rest modify it.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct MethFlags: c_int {
        const METH_VARARGS = 0x0001;
        const METH_KEYWORDS = 0x0002;
        const METH_NOARGS = 0x0004;
        const METH_O = 0x0008;
        const METH_CLASS = 0x0010;
        const METH_STATIC = 0x0020;
        const METH_COEXIST = 0x0040;
        const METH_FASTCALL = 0x0080;
        /// The method is handed the class defining it beside its receiver, and
        /// so carries a `PyCMethodObject` rather than a `PyCFunctionObject`.
        const METH_METHOD = 0x0200;
    }
}

/// C writes `ml_flags` through its own `int` declaration, so the field the
/// mirror hands it has to be that word and nothing wider or narrower.
const _: () = assert!(size_of::<MethFlags>() == size_of::<c_int>());
const _: () = assert!(align_of::<MethFlags>() == align_of::<c_int>());

/// One row of a `PyMethodDef` table.
#[repr(C)]
pub struct CPyMethodDef {
    pub ml_name: *const c_char,
    pub ml_meth: *const c_void,
    pub ml_flags: MethFlags,
    pub ml_doc: *const c_char,
}

/// Reserved carrier keys.  Namespaced so a C module cannot collide with them
/// through `PyModule_AddObject`.
const ML_KEY: &str = "__pyre_ml__";
const SELF_KEY: &str = "__self__";
const NAME_KEY: &str = "__name__";
const QUALNAME_KEY: &str = "__qualname__";
const DOC_KEY: &str = "__doc__";
const MODULE_KEY: &str = "__module__";
/// The defining class a `METH_METHOD` row is handed beside its receiver.
const CLASS_KEY: &str = "__pyre_mm_class__";

/// The carrier keys a store from Python must not reach.
///
/// Everything the carrier knows lives in its instance namespace, which is a
/// mapping like any other; a store into one of these would retarget the
/// receiver a C callable casts to its own struct, or name a different
/// `PyMethodDef` row for it to call.  `__module__` is the exception, and only
/// on a function carrier -- `store_attribute` reads the receiver, not the name
/// alone, because a descriptor has no attribute of that name to write.
/// The `tp_methods` / `tp_getset` / `tp_members` descriptor carriers in
/// `typeobject` share this fence and these keys; `__pyre_def__` there holds a
/// definition's address outright, so a store into it is the same hole in a
/// straighter line.
const RESERVED_KEYS: &[&str] = &[
    ML_KEY,
    SELF_KEY,
    NAME_KEY,
    QUALNAME_KEY,
    DOC_KEY,
    CLASS_KEY,
    "__pyre_def__",
    "__objclass__",
];

/// Every `PyMethodDef` row a carrier may name.
///
/// The carrier records an index into this table rather than the definition's
/// address.  Its namespace is reachable from Python — `__dict__` is a mapping
/// like any other — so whatever the carrier holds is under the writer's
/// control; an index naming no row is refused, where an address would be
/// transmuted to a function pointer and called.
///
/// The definitions live in the loaded extension's static storage, which the
/// module cache keeps mapped, so the table only ever grows by one row per
/// distinct method row converted.
#[derive(Default)]
struct MethodDefTable {
    rows: Vec<usize>,
    index_of: super::address_table::AddressMap<usize>,
}

static METHOD_DEFS: super::ForkMutex<MethodDefTable> = super::ForkMutex::new(MethodDefTable {
    rows: Vec::new(),
    index_of: HashMap::with_hasher(BuildHasherDefault::new()),
});

pub(super) unsafe fn after_fork_child() {
    unsafe { METHOD_DEFS.reinit_after_fork() };
}

/// The index `method` is filed under, entering it if it is new.
fn intern_method_def(method: *mut CPyMethodDef) -> i64 {
    let address = method as usize;
    let mut table = METHOD_DEFS.lock();
    if let Some(&index) = table.index_of.get(&address) {
        return index as i64;
    }
    let index = table.rows.len();
    table.rows.push(address);
    table.index_of.insert(address, index);
    index as i64
}

/// The definition an index names, or `None` for one that names no row.
fn method_def_at(index: i64) -> Option<*mut CPyMethodDef> {
    let table = METHOD_DEFS.lock();
    let address = *table.rows.get(usize::try_from(index).ok()?)?;
    Some(address as *mut CPyMethodDef)
}

static PYCFUNCTION_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static PYCMETHOD_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `__setattr__` / `__delattr__` for a carrier type.
///
/// `value` is `None` for the delete spelling.
fn store_attribute(
    carrier: PyObjectRef,
    name: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::unicodeobject::is_str(name) } {
        return Err(crate::PyError::type_error("attribute name must be string"));
    }
    let attribute = unsafe { pyre_object::w_str_get_wtf8(name) }.to_string();
    // `__module__` is the one writable slot, and only on a function carrier:
    // `meth_members` gives `builtin_function_or_method` a settable member of
    // that name, while a descriptor carries no such attribute at all.
    if !(attribute == MODULE_KEY && is_carrier(carrier)) {
        let type_name = unsafe { pyre_object::w_type_get_name((*carrier).w_class) };
        let message = if RESERVED_KEYS.contains(&attribute.as_str()) {
            format!("attribute '{attribute}' of '{type_name}' objects is not writable")
        } else {
            format!(
                "'{type_name}' object has no attribute '{attribute}' \
                 and no __dict__ for setting new attributes"
            )
        };
        return Err(crate::PyError::attribute_error(message));
    }
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        // A deleted member slot reads back as `None` rather than going away,
        // so the delete spelling stores that and stays repeatable.
        let value = value.unwrap_or_else(pyre_object::w_none);
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(dict, MODULE_KEY, value) };
    }
    Ok(pyre_object::w_none())
}

fn descr_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "__setattr__ requires name and value",
        ));
    }
    store_attribute(args[0], args[1], Some(args[2]))
}

fn descr_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("__delattr__ requires a name"));
    }
    store_attribute(args[0], args[1], None)
}

/// Install the stores every carrier type refuses, so only a function
/// carrier's `__module__` can be written into a namespace from Python.
pub(super) fn install_attribute_fence(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__setattr__",
            crate::make_builtin_function_with_arity("__setattr__", descr_setattr, 3)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__delattr__",
            crate::make_builtin_function_with_arity("__delattr__", descr_delattr, 2)
        );
    }
}

/// The carrier type, named as upstream names its typedef.
///
/// This is what `PyCFunction_Type` names, and so what `PyCFunction_Check`
/// answers for.  An interpreter builtin such as `len` is a *different*
/// `builtin_function_or_method` and answers no: it carries no `PyMethodDef`,
/// so an entry point that agreed it was one would then have nothing to hand
/// back from `PyCFunction_GetFunction` but an error, which a caller spelling
/// the two as one expression does not look for.
pub fn pycfunction_type() -> PyObjectRef {
    PYCFUNCTION_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("builtin_function_or_method", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__call__",
                    crate::make_builtin_function("__call__", descr_call)
                );
                crate::__pyre_put_new!(
                    ns_slot,
                    "__repr__",
                    crate::make_builtin_function_with_arity("__repr__", descr_repr, 1)
                );
            };
            install_attribute_fence(pyre_object::gc_roots::shadow_stack_get(ns_slot));
        });
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

/// `methodobject.py W_PyCMethodObject` — the carrier a `METH_METHOD` row
/// gets, which is everything `pycfunction_type` carries plus the class the
/// definition was declared in.
///
/// This is what `PyCMethod_Type` names, and it derives from
/// `PyCFunction_Type`, so `PyCFunction_Check` answers yes for one.
pub fn pycmethod_type() -> PyObjectRef {
    PYCMETHOD_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "builtin_method",
            |_ns| {},
            pycfunction_type(),
        );
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

/// Whether `carrier` is one of this module's own, rather than something that
/// merely has the reserved keys in its namespace.
fn is_carrier(carrier: PyObjectRef) -> bool {
    !carrier.is_null()
        && unsafe { crate::baseobjspace::issubtype_w((*carrier).w_class, pycfunction_type()) }
}

/// The `PyMethodDef` behind `object`, or NULL with a `SystemError` when
/// nothing is.
///
/// The type is checked before the namespace is read: `space.interp_w(
/// W_PyCFunctionObject, w_obj)` is what upstream's accessors open with, and
/// without it any object carrying a plausible `__pyre_ml__` would hand a
/// caller a C function pointer.
fn checked_method_def(object: *mut CPyObject) -> Option<*mut CPyMethodDef> {
    let object = unsafe { pyobject::from_ref(object) };
    let definition = is_carrier(object).then(|| method_def(object)).flatten();
    if definition.is_none() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "bad argument to internal function",
        ));
    }
    definition
}

/// `methodobject.py PyCFunction_GetFunction(object)` — the C function the
/// carrier calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetFunction(object: *mut CPyObject) -> *mut c_void {
    match checked_method_def(object) {
        Some(definition) => (unsafe { (*definition).ml_meth }) as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// `PyCFunction_GetFlags(object)` — the `METH_*` set the definition carries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetFlags(object: *mut CPyObject) -> c_int {
    match checked_method_def(object) {
        Some(definition) => unsafe { (*definition).ml_flags }.bits(),
        None => -1,
    }
}

/// `PyCFunction_GetSelf(object)` — the receiver the carrier was bound to,
/// borrowed, and NULL for a definition declared `METH_STATIC`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetSelf(object: *mut CPyObject) -> *mut CPyObject {
    let Some(definition) = checked_method_def(object) else {
        return std::ptr::null_mut();
    };
    if unsafe { (*definition).ml_flags }.contains(MethFlags::METH_STATIC) {
        return std::ptr::null_mut();
    }
    let carrier = unsafe { pyobject::from_ref(object) };
    match carrier_get(carrier, SELF_KEY) {
        Some(receiver) => pyobject::borrow_from(object, receiver),
        None => std::ptr::null_mut(),
    }
}

/// `PyMethodDescr_Check(op)` — a descriptor built from a `PyMethodDef` row.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethodDescr_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null()
        && unsafe {
            crate::baseobjspace::issubtype_w(
                (*object).w_class,
                super::typeobject::method_descriptor_type(),
            )
        }) as c_int
}

/// `PyMethodDescr_CheckExact(op)`.
///
/// # Safety
/// `object` must be null or a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethodDescr_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null()
        && unsafe { (*object).w_class } == super::typeobject::method_descriptor_type()) as c_int
}

/// `methodobject.py PyDescr_NewMethod(type, method)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDescr_NewMethod(
    tp: *mut super::typeobject::CPyTypeObject,
    method: *mut CPyMethodDef,
) -> *mut CPyObject {
    unsafe {
        descriptor_over(
            super::typeobject::method_descriptor_type(),
            tp,
            method,
            "PyDescr_NewMethod",
        )
    }
}

/// `methodobject.py PyDescr_NewClassMethod(type, method)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDescr_NewClassMethod(
    tp: *mut super::typeobject::CPyTypeObject,
    method: *mut CPyMethodDef,
) -> *mut CPyObject {
    unsafe {
        descriptor_over(
            super::typeobject::classmethod_descriptor_type(),
            tp,
            method,
            "PyDescr_NewClassMethod",
        )
    }
}

/// The descriptor `carrier_type` makes of `method` declared in `tp`, which is
/// everything the two entry points above share.
///
/// # Safety
/// `tp` must be null or a live type mirror, and `method` null or a live row.
unsafe fn descriptor_over(
    carrier_type: PyObjectRef,
    tp: *mut super::typeobject::CPyTypeObject,
    method: *mut CPyMethodDef,
    entry_point: &str,
) -> *mut CPyObject {
    let w_type = unsafe { pyobject::from_ref(tp as *mut CPyObject) };
    if method.is_null() || w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("bad argument to {entry_point}"),
        ));
        return std::ptr::null_mut();
    }
    pyobject::make_ref(super::typeobject::new_method_descriptor(
        carrier_type,
        w_type,
        method,
    ))
}

/// `methodobject.py PyClassMethod_New(f)` — `classmethod(f)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyClassMethod_New(function: *mut CPyObject) -> *mut CPyObject {
    unsafe { wrapped_in(function, &pyre_object::function::CLASSMETHOD_TYPE) }
}

/// `methodobject.py PyStaticMethod_New(f)` — `staticmethod(f)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyStaticMethod_New(function: *mut CPyObject) -> *mut CPyObject {
    unsafe { wrapped_in(function, &pyre_object::function::STATICMETHOD_TYPE) }
}

/// `wrapper(function)` — the two above differ only in which class wraps.
///
/// # Safety
/// `function` must be null or a live reference.
unsafe fn wrapped_in(
    function: *mut CPyObject,
    wrapper: &'static pyre_object::pyobject::PyType,
) -> *mut CPyObject {
    let Some(function) = super::object::argument(function) else {
        return std::ptr::null_mut();
    };
    let wrapper = crate::typedef::gettypeobject(wrapper);
    super::object::result(crate::call::call_function_impl_result(wrapper, &[function]))
}

/// `PyCFunction_New(ml, self)` — the callable a `PyMethodDef` row describes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_New(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { PyCFunction_NewEx(method, receiver, std::ptr::null_mut()) }
}

/// `PyCFunction_NewEx(ml, self, module)` — the same, naming the module the
/// definition came from.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_NewEx(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
    module: *mut CPyObject,
) -> *mut CPyObject {
    if method.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    unsafe { PyCMethod_New(method, receiver, module, std::ptr::null_mut()) }
}

/// `methodobject.py PyCMethod_New(ml, self, module, cls)` — the same,
/// naming the class a `METH_METHOD` definition was declared in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCMethod_New(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
    module: *mut CPyObject,
    class: *mut super::typeobject::CPyTypeObject,
) -> *mut CPyObject {
    if method.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let class = class as *mut CPyObject;
    super::object::realize_all([receiver, module, class]);
    let receiver = unsafe { pyobject::from_ref(receiver) };
    let module = unsafe { pyobject::from_ref(module) };
    let class = unsafe { pyobject::from_ref(class) };
    if !class.is_null() && !unsafe { pyre_object::is_type(class) } {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "bad argument to PyCMethod_New",
        ));
        return std::ptr::null_mut();
    }
    let module = if module.is_null() {
        pyre_object::w_none()
    } else {
        module
    };
    let class = (!class.is_null()).then_some(class);
    super::object::result(new_pycfunction_in_class(method, receiver, module, class))
}

/// `methodobject.py call_noargs PyCMethod_GetClass(op)` — the class a `METH_METHOD`
/// definition was declared in, borrowed, and NULL for one that is not.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCMethod_GetClass(
    object: *mut CPyObject,
) -> *mut super::typeobject::CPyTypeObject {
    let Some(definition) = checked_method_def(object) else {
        return std::ptr::null_mut();
    };
    if !unsafe { (*definition).ml_flags }.contains(MethFlags::METH_METHOD) {
        return std::ptr::null_mut();
    }
    let carrier = unsafe { pyobject::from_ref(object) };
    match carrier_class(carrier) {
        Some(class) => {
            pyobject::borrow_from(object, class) as *mut super::typeobject::CPyTypeObject
        }
        None => std::ptr::null_mut(),
    }
}

/// The class a `METH_METHOD` carrier was built with.
///
/// Whatever answers the key is cast to a `PyTypeObject *` and handed to the C
/// function, so a value that is not a type is not one of ours.
fn carrier_class(carrier: PyObjectRef) -> Option<PyObjectRef> {
    carrier_get(carrier, CLASS_KEY)
        .filter(|&class| !class.is_null() && unsafe { pyre_object::is_type(class) })
}

fn carrier_get(carrier: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) }
}

fn carrier_set(carrier: PyObjectRef, key: &str, value: PyObjectRef) {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(dict, key, value) };
    }
}

fn text_or_none(pointer: *const c_char) -> PyObjectRef {
    if pointer.is_null() {
        return pyre_object::w_none();
    }
    pyre_object::w_str_new_managed(&unsafe { CStr::from_ptr(pointer) }.to_string_lossy())
}

fn text_or_empty(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// `methodobject.py:W_PyCFunctionObject.__init__`.
///
/// `w_self` is the bound receiver a module-level function is called with —
/// the module object itself — and `w_module` the `__module__` string.
pub fn new_pycfunction(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    w_module: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    new_pycfunction_in_class(method, w_self, w_module, None)
}

/// [`new_pycfunction`] naming the class a `METH_METHOD` definition was
/// declared in, whose carrier is `pycmethod_type` rather than
/// `pycfunction_type`.
pub fn new_pycfunction_in_class(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    w_module: PyObjectRef,
    w_class: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    // The flag and the class have to agree, and every carrier is built here:
    // a module table reaches this with no class, which is what makes a
    // `METH_METHOD` row in one an error at module creation.
    let message = match (
        unsafe { (*method).ml_flags }.contains(MethFlags::METH_METHOD),
        w_class.is_some(),
    ) {
        (true, false) => {
            Some("attempting to create PyCMethod with a METH_METHOD flag but no class")
        }
        (false, true) => {
            Some("attempting to create PyCFunction with class but no METH_METHOD flag")
        }
        _ => None,
    };
    if let Some(message) = message {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            message,
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_module);
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_class.unwrap_or_else(pyre_object::w_none));
    let carrier_type = if w_class.is_some() {
        pycmethod_type()
    } else {
        pycfunction_type()
    };
    let carrier = pyre_object::w_instance_new(carrier_type);
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier);

    let name = text_or_none(unsafe { (*method).ml_name });
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        NAME_KEY,
        name,
    );
    let name = carrier_get(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        NAME_KEY,
    )
    .unwrap_or_else(pyre_object::w_none);
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        QUALNAME_KEY,
        name,
    );
    let doc = text_or_none(unsafe { (*method).ml_doc });
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        DOC_KEY,
        doc,
    );
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        SELF_KEY,
        pyre_object::gc_roots::shadow_stack_get(self_slot),
    );
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        MODULE_KEY,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    );
    let ml = pyre_object::w_int_new(intern_method_def(method));
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        ML_KEY,
        ml,
    );
    if w_class.is_some() {
        carrier_set(
            pyre_object::gc_roots::shadow_stack_get(carrier_slot),
            CLASS_KEY,
            pyre_object::gc_roots::shadow_stack_get(class_slot),
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(carrier_slot))
}

/// C-visible `PyCFunctionObject`, the twin of the struct in
/// `include/pyre3.14t/methodobject.h`.
///
/// cffi reads `m_ml` and `m_module` off the block rather than through an entry
/// point (`lib_obj.c _cpyextfunc_get`), and the module read is an identity
/// test against the reference the caller passed to `PyCFunction_NewEx`.
#[repr(C)]
pub struct CPyCFunctionObject {
    pub ob_base: CPyObject,
    pub m_ml: *mut CPyMethodDef,
    pub m_self: *mut CPyObject,
    pub m_module: *mut CPyObject,
    pub m_weakreflist: *mut CPyObject,
    pub vectorcall: *mut c_void,
}

/// C-visible `PyCMethodObject` — a `METH_METHOD` carrier, which holds the
/// defining class past everything the function carrier holds.
#[repr(C)]
pub struct CPyCMethodObject {
    pub func: CPyCFunctionObject,
    pub mm_class: *mut super::typeobject::CPyTypeObject,
}

/// The layouts are written out twice -- here and in the header -- so a field
/// added to one without the other stops compiling rather than writing
/// somewhere unclaimed.
const _: () = {
    assert!(std::mem::offset_of!(CPyCFunctionObject, ob_base) == 0);
    assert!(std::mem::offset_of!(CPyCFunctionObject, m_ml) == size_of::<CPyObject>());
    assert!(
        std::mem::offset_of!(CPyCFunctionObject, m_self)
            == size_of::<CPyObject>() + size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyCFunctionObject, m_module)
            == size_of::<CPyObject>() + 2 * size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyCFunctionObject, m_weakreflist)
            == size_of::<CPyObject>() + 3 * size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyCFunctionObject, vectorcall)
            == size_of::<CPyObject>() + 4 * size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyCMethodObject, mm_class)
            == size_of::<CPyObject>() + 5 * size_of::<usize>()
    );
};

/// A carrier type that has been built, and null before it has.
///
/// [`pycfunction_type`] mints its type on first call, and this runs from
/// inside the mint of some other type's mirror; a type that does not exist yet
/// has no instances, so the answer for one is the same either way.
fn built_carrier_type(cell: &pyre_object::gc_roots::RootedOnceRef) -> PyObjectRef {
    cell.get().unwrap_or(std::ptr::null_mut())
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries --
/// `methodobject.py basestruct=PyCFunctionObject.TO` for a function carrier
/// and `PyCMethodObject.TO` for a `METH_METHOD` one, and 0 for every other
/// type, which asks for the plain header.
pub(super) fn basicsize(w_type: PyObjectRef) -> isize {
    if w_type.is_null() {
        return 0;
    }
    let derived_from = |carrier: PyObjectRef| {
        !carrier.is_null() && unsafe { crate::baseobjspace::issubtype_w(w_type, carrier) }
    };
    if derived_from(built_carrier_type(&PYCMETHOD_TYPE_OBJ)) {
        return size_of::<CPyCMethodObject>() as isize;
    }
    match derived_from(built_carrier_type(&PYCFUNCTION_TYPE_OBJ)) {
        true => size_of::<CPyCFunctionObject>() as isize,
        false => 0,
    }
}

/// Fill a freshly allocated mirror -- `methodobject.py cfunction_attach`, and
/// `cmethod_attach` for the one field past it.
pub(super) fn attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    // `is_carrier` would mint the carrier type, and minting allocates: this
    // runs inside another mirror's fill, whose caller is holding raw reads of
    // objects a collection would move.
    let carrier = built_carrier_type(&PYCFUNCTION_TYPE_OBJ);
    if w_obj.is_null()
        || carrier.is_null()
        || !unsafe { crate::baseobjspace::issubtype_w((*w_obj).w_class, carrier) }
    {
        return;
    }
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null() || unsafe { (*tp).tp_basicsize } < size_of::<CPyCFunctionObject>() as isize {
        return;
    }
    // Every value is read out of the namespace before the first `make_ref`,
    // which allocates and may move the rest.
    let ml = method_def(w_obj).unwrap_or(std::ptr::null_mut());
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier_get(w_obj, SELF_KEY).unwrap_or_else(pyre_object::w_none));
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier_get(w_obj, MODULE_KEY).unwrap_or_else(pyre_object::w_none));
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier_class(w_obj).unwrap_or(pyre_object::PY_NULL));
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);
    let m_self = pyobject::make_ref(reload(self_slot));
    let m_module = pyobject::make_ref(reload(module_slot));
    let block = raw as *mut CPyCFunctionObject;
    unsafe {
        (*block).m_ml = ml;
        (*block).m_self = m_self;
        (*block).m_module = m_module;
    }
    if unsafe { (*tp).tp_basicsize } < size_of::<CPyCMethodObject>() as isize {
        return;
    }
    let mm_class = pyobject::make_ref(reload(class_slot)) as *mut super::typeobject::CPyTypeObject;
    unsafe { (*(raw as *mut CPyCMethodObject)).mm_class = mm_class };
}

/// Release the references a carrier mirror owns -- `methodobject.py
/// cfunction_dealloc` and `cmethod_dealloc`.
pub(super) fn forget_block(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    let tp = unsafe { (*raw).ob_type } as usize;
    // `as_pyobj` rather than a realizing read: this runs while a mirror is
    // being deallocated, where nothing may build an interpreter object.  The
    // type decides it rather than the size, as it does for a slice mirror: a
    // C-defined type whose own storage reaches this far fills and frees these
    // words itself.
    let carrier = |cell| pyobject::as_pyobj(built_carrier_type(cell)) as usize;
    let function = carrier(&PYCFUNCTION_TYPE_OBJ);
    let method = carrier(&PYCMETHOD_TYPE_OBJ);
    if tp == 0 || (tp != function && tp != method) {
        return;
    }
    let block = raw as *mut CPyCFunctionObject;
    unsafe {
        pyobject::decref((*block).m_self);
        pyobject::decref((*block).m_module);
        (*block).m_self = std::ptr::null_mut();
        (*block).m_module = std::ptr::null_mut();
        (*block).m_ml = std::ptr::null_mut();
    }
    if tp != method {
        return;
    }
    let block = raw as *mut CPyCMethodObject;
    unsafe {
        pyobject::decref((*block).mm_class as *mut CPyObject);
        (*block).mm_class = std::ptr::null_mut();
    }
}

/// C references owned by `methodobject.py cfunction_attach` / `cmethod_attach`.
///
/// These carrier types do not declare `Py_TPFLAGS_HAVE_GC`, so `tp_traverse`
/// cannot report their `m_self`, `m_module`, and `mm_class` fields. They still
/// belong to the rawrefcount C graph: treating a short-lived bound method's
/// `m_self` as an outside reference keeps its receiver alive for an extra
/// collection, which is observable for `_cffi_backend.FFI` and its type cache.
pub(super) fn mirror_edges(edges: &mut Vec<(usize, Vec<usize>)>) {
    let carrier = |cell| pyobject::as_pyobj(built_carrier_type(cell)) as usize;
    let function = carrier(&PYCFUNCTION_TYPE_OBJ);
    let method = carrier(&PYCMETHOD_TYPE_OBJ);
    if function == 0 && method == 0 {
        return;
    }
    for address in pyobject::entered_blocks() {
        let raw = address as *mut CPyObject;
        let tp = unsafe { (*raw).ob_type } as usize;
        if tp != function && tp != method {
            continue;
        }
        let block = raw as *mut CPyCFunctionObject;
        let mut referents = Vec::with_capacity(if tp == method { 3 } else { 2 });
        for referent in unsafe { [(*block).m_self, (*block).m_module] } {
            if !referent.is_null() {
                referents.push(referent as usize);
            }
        }
        if tp == method {
            let class = unsafe { (*(raw as *mut CPyCMethodObject)).mm_class };
            if !class.is_null() {
                referents.push(class as usize);
            }
        }
        if !referents.is_empty() {
            edges.push((address, referents));
        }
    }
}

fn method_def(carrier: PyObjectRef) -> Option<*mut CPyMethodDef> {
    let ml = carrier_get(carrier, ML_KEY)?;
    if !unsafe { pyre_object::is_int(ml) } {
        return None;
    }
    method_def_at(unsafe { pyre_object::w_int_get_value(ml) })
}

fn method_name(carrier: PyObjectRef) -> String {
    carrier_get(carrier, NAME_KEY)
        .filter(|&name| unsafe { pyre_object::unicodeobject::is_str(name) })
        .map(|name| unsafe { pyre_object::w_str_get_wtf8(name) }.to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    Ok(pyre_object::w_str_new_managed(&format!(
        "<built-in function {}>",
        method_name(carrier)
    )))
}

/// `methodobject.py:W_PyCFunctionObject.descr_call`.
fn descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__call__ requires self"));
    }
    let carrier = args[0];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(method) = method_def(carrier) else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext function carrier lost its method definition",
        ));
    };
    let Some(w_self) = carrier_get(carrier, SELF_KEY) else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext function carrier lost its receiver",
        ));
    };
    call_method_def_in_class(method, w_self, carrier_class(carrier), positional, kwargs)
}

/// Validate one call against its `ml_flags` and hand it to the bridge, naming
/// the class a `METH_METHOD` row is handed beside its receiver.
///
/// Shared with the `tp_methods` descriptor, whose receiver is the instance the
/// attribute was read through rather than a bound `__self__`.
pub(super) fn call_method_def_in_class(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    defining_class: Option<PyObjectRef>,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let name = text_or_empty(unsafe { (*method).ml_name });
    let flags = unsafe { (*method).ml_flags }
        & !(MethFlags::METH_CLASS | MethFlags::METH_STATIC | MethFlags::METH_COEXIST);
    let keywords: Vec<(String, PyObjectRef)> = match kwargs {
        Some(dict) if crate::builtins::has_real_kwargs(kwargs) => unsafe {
            pyre_object::w_dict_str_entries(dict)
                .into_iter()
                .filter(|(key, _)| key != "__pyre_kw__")
                .collect()
        },
        _ => Vec::new(),
    };
    if !keywords.is_empty() && !flags.contains(MethFlags::METH_KEYWORDS) {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no keyword arguments"
        )));
    }
    if flags.contains(MethFlags::METH_NOARGS) && !positional.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no arguments ({} given)",
            positional.len()
        )));
    }
    if flags.contains(MethFlags::METH_O) && positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes exactly one argument ({} given)",
            positional.len()
        )));
    }
    // Exactly one, not at least one: the flags select which signature
    // `ml_meth` has, so a table declaring two of them names no signature at
    // all and would otherwise be dispatched through whichever this layer
    // happens to test for first.
    if (flags
        & (MethFlags::METH_NOARGS
            | MethFlags::METH_O
            | MethFlags::METH_VARARGS
            | MethFlags::METH_FASTCALL))
        .bits()
        .count_ones()
        != 1
    {
        return Err(crate::PyError::runtime_error(format!(
            "{name}() uses an unknown calling convention"
        )));
    }
    // `METH_METHOD` widens the fastcall-with-keywords signature by one
    // argument rather than naming a signature of its own, so it is the only
    // combination it may appear in; in any other it would select a transmute
    // whose arity the callee does not have.
    if flags.contains(MethFlags::METH_METHOD)
        && flags & (MethFlags::METH_FASTCALL | MethFlags::METH_KEYWORDS)
            != (MethFlags::METH_FASTCALL | MethFlags::METH_KEYWORDS)
    {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("{name}() method: bad call flags"),
        ));
    }
    let defining_class = match (flags.contains(MethFlags::METH_METHOD), defining_class) {
        (false, _) => None,
        (true, Some(class)) => Some(class),
        (true, None) => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{name}() method: no defining class"),
            ));
        }
    };
    super::call_cfunction_in_class(
        unsafe { (*method).ml_meth },
        flags,
        w_self,
        defining_class,
        positional,
        &keywords,
    )
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyMethodDescr_Check as *const ());
    std::hint::black_box(PyMethodDescr_CheckExact as *const ());
    std::hint::black_box(PyDescr_NewMethod as *const ());
    std::hint::black_box(PyDescr_NewClassMethod as *const ());
    std::hint::black_box(PyClassMethod_New as *const ());
    std::hint::black_box(PyStaticMethod_New as *const ());
}

#[cfg(test)]
mod tests {
    /// `ml_flags` is one word two places spell: an extension compiled against
    /// `include/pyre3.14t/methodobject.h` puts bits into it, and this file
    /// reads them.  Nothing tied the two together -- upstream has no such
    /// pair, because `api.py` reads each value out of the real header with
    /// `rffi_platform.ConstantInteger` -- so this walks the header and rejects
    /// any flag whose bit the Rust side spells differently.
    ///
    /// `MethFlags::FLAGS` is what makes the walk complete: it carries every
    /// constant the `bitflags!` block declares, so a flag added there is
    /// compared here without anyone remembering to list it.
    #[test]
    fn every_meth_flag_is_the_bit_the_header_gives_it() {
        use bitflags::Flags as _;

        const HEADER: &str = include_str!("../../../../include/pyre3.14t/methodobject.h");

        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define METH_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let Some(digits) = body.trim().strip_prefix("0x") else {
                continue;
            };
            let Ok(value) = std::ffi::c_int::from_str_radix(digits, 16) else {
                continue;
            };
            header.push((name, value));
        }

        // A parser that read nothing would agree with every flag below, so the
        // floor comes first.  The header declares nine.
        assert_eq!(
            header.len(),
            9,
            "methodobject.h declares {} METH_ flags, not the 9 this read",
            header.len()
        );

        for flag in super::MethFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("METH_")
                .expect("every flag is declared under the header's own name");
            let ours = flag.value().bits();
            let (_, theirs) = header
                .iter()
                .find(|(known, _)| *known == name)
                .unwrap_or_else(|| panic!("methodobject.h declares no METH_{name}"));
            assert_eq!(
                ours, *theirs,
                "METH_{name} is {theirs:#x} in methodobject.h and {ours:#x} here"
            );
        }
    }
}
