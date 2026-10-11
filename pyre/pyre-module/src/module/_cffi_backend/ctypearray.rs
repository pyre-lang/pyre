//! `_cffi_backend.__CData_iterator` — PyPy:
//! `pypy/module/_cffi_backend/ctypearray.py`'s `W_CDataIter`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj;
use super::ctypeobj;

/// `ctypearray.py W_CDataIter`.
#[pyre_interpreter::pyre_class("_cffi_backend.__CData_iterator")]
#[derive(Default)]
pub struct W_CDataIter {
    /// `W_CDataIter.ctitem`.
    pub ctitem: PyObjectRef,
    /// `W_CDataIter.cdata` — the array, which owns the memory being walked.
    pub cdata: PyObjectRef,
    /// `W_CDataIter._next`.
    pub next: *mut u8,
    /// `W_CDataIter._stop`.
    pub stop: *mut u8,
}

/// `W_CTypeArray.iter` — `W_CDataIter(space, self.ctitem, cdata)`.
pub fn new_cdata_iter(mut w_cdata: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdataobj::cdata_arg(w_cdata))?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    let item = super::ctypeptr::item_of(ct)?;
    let length = cdata.array_length()?;
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(w_cdata);
    let item_slot = cdata_slot + 1;
    let _ = roots.pin_root(item.as_object());
    let start = cdata.ptr;
    Ok(W_CDataIter::allocate_stable(W_CDataIter {
        ctitem: roots.get(item_slot),
        cdata: roots.get(cdata_slot),
        next: start as *mut u8,
        stop: unsafe { (start as *mut u8).offset((length * item.size) as isize) },
        ..Default::default()
    }))
}

static CDATA_ITER_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.__CData_iterator`.
pub fn cdata_iter_type() -> PyObjectRef {
    CDATA_ITER_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.__CData_iterator",
            init_cdata_iter_type,
            pyre_interpreter::typedef::w_object(),
            <W_CDataIter as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CDataIter as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

fn init_cdata_iter_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    for (name, f) in [
        (
            "__iter__",
            iter_w as pyre_interpreter::gateway::BuiltinCodeFn,
        ),
        ("__next__", next_w),
    ] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, f, 1)
        )
    }
}

/// `W_CDataIter.iter_w`.
fn iter_w(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(args[0])
}

/// `W_CDataIter.next_w`.
fn next_w(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let it = W_CDataIter::from_obj(args[0])
        .ok_or_else(|| PyError::type_error("expected a __CData_iterator"))?;
    if std::ptr::eq(it.next, it.stop) {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::StopIteration,
            "",
        ));
    }
    let item = ctypeobj::ctype_at(it.ctitem)
        .ok_or_else(|| PyError::system_error("iterator without an item type"))?;
    let result = it.next;
    it.next = unsafe { it.next.offset(item.size as isize) };
    unsafe { ctypeobj::convert_to_object(item, result as usize) }
}
