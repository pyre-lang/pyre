//! The bound-method and code-object entry points -- PyPy
//! `cpyext/funcobject.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use std::ffi::{c_char, c_int};

/// `funcobject.py PyMethod_New(func, self)` — bind `receiver` to
/// `function`, the way an attribute lookup on an instance does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_New(
    function: *mut CPyObject,
    receiver: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([function, receiver]);
    let (Some(function), Some(receiver)) = (argument(function), argument(receiver)) else {
        return std::ptr::null_mut();
    };
    result(Ok(pyre_object::w_method_new(
        function,
        receiver,
        pyre_object::PY_NULL,
    )))
}

/// The member `reader` names on `method`, borrowed, or NULL with a
/// `SystemError` when `method` is not a bound method.
///
/// Borrowed is what both readers answer with: the member is reachable through
/// the method for as long as the caller holds it.
unsafe fn method_member(
    method: *mut CPyObject,
    reader: unsafe fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
) -> *mut CPyObject {
    let Some(object) = argument(method) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::function::is_method(object) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::borrow_from(method, unsafe { reader(object) })
}

/// `funcobject.py PyMethod_Function(method)` — the callable the binding
/// wraps, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Function(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_func) }
}

/// `funcobject.py PyMethod_Self(method)` — the receiver the binding
/// carries, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Self(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_self) }
}

// ── code objects ────────────────────────────────────────────────────────

/// `funcobject.py PyCode_NewEmpty(filename, funcname, firstlineno)` — a
/// code object with the given source location and a body that raises
/// `AssertionError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCode_NewEmpty(
    filename: *const c_char,
    funcname: *const c_char,
    firstlineno: c_int,
) -> *mut CPyObject {
    // The two names are decoded as `codeobject.c` decodes them: the function
    // name as UTF-8, the path through the filesystem codec, so a path this
    // runtime cannot spell as UTF-8 still names a file.
    let name = unsafe { super::unicodeobject::PyUnicode_FromString(funcname) };
    let path = unsafe { super::unicodeobject::PyUnicode_DecodeFSDefault(filename) };
    let code = match (argument(name), argument(path)) {
        (Some(name), Some(path)) => result(crate::pycode::code_new_empty(
            path,
            name,
            firstlineno as i64,
        )),
        _ => std::ptr::null_mut(),
    };
    for reference in [name, path] {
        if !reference.is_null() {
            unsafe { pyobject::decref(reference) };
        }
    }
    code
}

/// `funcobject.py:161 PyUnstable_Code_NewWithPosOnlyArgs(...)` — a code object
/// built from every field, as `code.__new__` takes them.
///
/// The C argument order puts the free and cell names before the filename;
/// the interpreter constructor takes them last, and derives `nlocals` from
/// `varnames` rather than reading the one the caller passed, as
/// `codeobject.c` does.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnstable_Code_NewWithPosOnlyArgs(
    argcount: c_int,
    posonlyargcount: c_int,
    kwonlyargcount: c_int,
    _nlocals: c_int,
    stacksize: c_int,
    flags: c_int,
    code: *mut CPyObject,
    consts: *mut CPyObject,
    names: *mut CPyObject,
    varnames: *mut CPyObject,
    freevars: *mut CPyObject,
    cellvars: *mut CPyObject,
    filename: *mut CPyObject,
    name: *mut CPyObject,
    qualname: *mut CPyObject,
    firstlineno: c_int,
    linetable: *mut CPyObject,
    exceptiontable: *mut CPyObject,
) -> *mut CPyObject {
    let objects = [
        code,
        consts,
        names,
        varnames,
        freevars,
        cellvars,
        filename,
        name,
        qualname,
        linetable,
        exceptiontable,
    ];
    if super::object::arguments(objects).is_none() {
        return std::ptr::null_mut();
    }
    // Read back through the mirror each time rather than kept in a local: a
    // mirror's address does not move, and the interpreter object it names does.
    let object = |raw| unsafe { pyobject::from_ref(raw) };
    // A name list that is not a tuple is a caller error, which the C entry
    // point reports the way every other bad argument to one is reported.
    for list in [names, varnames, freevars, cellvars] {
        if !unsafe { pyre_object::is_tuple(object(list)) } {
            unsafe { super::pyerrors::PyErr_BadInternalCall() };
            return std::ptr::null_mut();
        }
    }
    // `nlocals` is derived from `varnames` rather than read from the argument
    // of that name, which `PyUnstable_Code_NewWithPosOnlyArgs` also ignores.
    let nlocals = unsafe { pyre_object::tupleobject::w_tuple_len(object(varnames)) };
    // Every int here allocates, so they are built and rooted before any of the
    // eleven objects is read: a collection while one is built moves the
    // storage the earlier ones name.
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for value in [
        argcount as i64,
        posonlyargcount as i64,
        kwonlyargcount as i64,
        nlocals as i64,
        stacksize as i64,
        flags as i64,
        firstlineno as i64,
    ] {
        let _ = roots.pin_root(pyre_object::w_int_new(value));
    }
    let count = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    result(unsafe {
        crate::pycode::code_new(&[
            pyre_object::PY_NULL,
            count(0),
            count(1),
            count(2),
            count(3),
            count(4),
            count(5),
            object(code),
            object(consts),
            object(names),
            object(varnames),
            object(filename),
            object(name),
            object(qualname),
            count(6),
            object(linetable),
            object(exceptiontable),
            object(freevars),
            object(cellvars),
        ])
    })
}

/// `funcobject.py:134 PyUnstable_Code_New(...)` — the same with no
/// positional-only arguments.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnstable_Code_New(
    argcount: c_int,
    kwonlyargcount: c_int,
    nlocals: c_int,
    stacksize: c_int,
    flags: c_int,
    code: *mut CPyObject,
    consts: *mut CPyObject,
    names: *mut CPyObject,
    varnames: *mut CPyObject,
    freevars: *mut CPyObject,
    cellvars: *mut CPyObject,
    filename: *mut CPyObject,
    name: *mut CPyObject,
    qualname: *mut CPyObject,
    firstlineno: c_int,
    linetable: *mut CPyObject,
    exceptiontable: *mut CPyObject,
) -> *mut CPyObject {
    unsafe {
        PyUnstable_Code_NewWithPosOnlyArgs(
            argcount,
            0,
            kwonlyargcount,
            nlocals,
            stacksize,
            flags,
            code,
            consts,
            names,
            varnames,
            freevars,
            cellvars,
            filename,
            name,
            qualname,
            firstlineno,
            linetable,
            exceptiontable,
        )
    }
}
