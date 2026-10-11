//! CFFI API-function wrappers — PyPy: `pypy/module/_cffi_backend/wrapper.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::W_CData;
use super::realize_c_type::W_RawFuncType;
use super::{allocator, cdataobj, ctypefunc, ctypeobj, ctypeptr};

/// `W_FunctionWrapper`.
#[pyre_interpreter::pyre_class("_cffi_backend.__FFIFunctionWrapper")]
// `W_FunctionWrapper._immutable_ = True`: every field is written once, by
// `new_function_wrapper`, and only read afterwards.
#[majit_macros::jit_immutable_fields(
    "w_ffi",
    "fnptr",
    "directfnptr",
    "w_rawfunctype",
    "w_fnname",
    "w_modulename"
)]
#[derive(Default)]
pub struct W_FunctionWrapper {
    pub w_ffi: PyObjectRef,
    /// `W_FunctionWrapper.fnptr` — the address of the C function, as
    /// `rffi.CCHARP`.  A raw-flavour pointer is integer-kind, so the address
    /// belongs in the integer register bank rather than among the references
    /// a collector traces and rewrites.
    pub fnptr: usize,
    /// `W_FunctionWrapper.directfnptr` — the address of the C function
    /// bypassing the API wrapper, zero when the module was built without one.
    pub directfnptr: usize,
    pub w_rawfunctype: PyObjectRef,
    pub w_fnname: PyObjectRef,
    pub w_modulename: PyObjectRef,
}

fn wrapper_arg(w_wrapper: PyObjectRef) -> Result<&'static mut W_FunctionWrapper, PyError> {
    W_FunctionWrapper::from_obj(w_wrapper)
        .ok_or_else(|| PyError::type_error("expected an FFI function wrapper"))
}

pub fn new_function_wrapper(
    w_ffi: PyObjectRef,
    fnptr: usize,
    directfnptr: usize,
    w_rawfunctype: PyObjectRef,
    fnname: &str,
    modulename: &str,
) -> Result<PyObjectRef, PyError> {
    let _ = function_wrapper_type();
    let raw = W_RawFuncType::from_obj(w_rawfunctype)
        .ok_or_else(|| PyError::system_error("expected a raw function type"))?;
    let ctype = ctypeobj::ctype_arg(raw.nostruct_ctype)?;
    assert_eq!(ctype.kind, ctypeobj::KIND_FUNC);
    assert!(ctype.cif_descr != 0);
    if !raw.nostruct_locs.is_null() {
        assert_eq!(
            ctypefunc::fargs_of(ctype).len(),
            unsafe { pyre_object::bytesobject::w_bytes_data(raw.nostruct_locs) }.len()
        );
    }

    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(w_ffi);
    let _ = roots.pin_root(w_rawfunctype);
    let _ = roots.pin_root(pyre_object::w_str_new(fnname));
    let _ = roots.pin_root(pyre_object::w_str_new(modulename));
    let obj = W_FunctionWrapper::allocate_stable(W_FunctionWrapper {
        fnptr,
        directfnptr,
        ..Default::default()
    });
    let wrapper = W_FunctionWrapper::from_obj(obj).expect("allocate_stable returns this layout");
    wrapper.w_ffi = roots.get(base);
    wrapper.w_rawfunctype = roots.get(base + 1);
    wrapper.w_fnname = roots.get(base + 2);
    wrapper.w_modulename = roots.get(base + 3);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    Ok(obj)
}

/// `W_FunctionWrapper.typeof`.
pub fn typeof_wrapper(w_wrapper: PyObjectRef, w_ffi: PyObjectRef) -> Result<PyObjectRef, PyError> {
    W_RawFuncType::unwrap_as_fnptr(wrapper_arg(w_wrapper)?.w_rawfunctype, w_ffi)
}

/// `prepare_args`.
pub fn prepare_args(
    w_rawfunctype: PyObjectRef,
    args_w: &mut [PyObjectRef],
    start_index: usize,
) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let raw_slot = roots.base();
    let _ = roots.pin_root(w_rawfunctype);
    let args_slot = raw_slot + 1;
    for &arg in args_w.iter() {
        let _ = roots.pin_root(arg);
    }
    let raw = W_RawFuncType::from_obj(roots.get(raw_slot))
        .ok_or_else(|| PyError::system_error("expected a raw function type"))?;
    let locs = unsafe { pyre_object::bytesobject::w_bytes_data(raw.nostruct_locs) }.to_vec();
    let fargs = ctypefunc::fargs_of(ctypeobj::ctype_arg(raw.nostruct_ctype)?);
    for i in start_index..locs.len() {
        if locs[i] != b'A' {
            continue;
        }
        let w_arg = roots.get(args_slot + i);
        let w_farg = fargs[i];
        let farg = ctypeobj::ctype_arg(w_farg)?;
        assert!(farg.is_ptr_or_array());
        let w_arg = if W_CData::from_obj(w_arg).is_some_and(|arg| arg.ctype == farg.ctitem) {
            cdataobj::new_cdata_ptr_to_struct(
                W_CData::from_obj(w_arg).expect("checked above").ptr,
                w_farg,
                w_arg,
            )
        } else if unsafe { pyre_object::pyobject::is_none(w_arg) } {
            continue;
        } else {
            ctypeobj::newp(w_farg, w_arg)?
        };
        roots.set(args_slot + i, w_arg);
    }
    for (i, arg) in args_w.iter_mut().enumerate() {
        *arg = roots.get(args_slot + i);
    }
    Ok(())
}

/// `W_FunctionWrapper.descr_call`.
fn wrapper_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let wrapper_slot = roots.base();
    for &arg in args {
        let _ = roots.pin_root(arg);
    }
    let wrapper = wrapper_arg(roots.get(wrapper_slot))?;
    // Everything about the type of the function is read as an immutable
    // attribute of the raw function type, which makes it the obvious thing to
    // promote in order to do the call: with it constant, `nostruct_ctype` and
    // the `cif_descr` behind it are constants too.
    let raw_type = majit_metainterp::jit::promote(wrapper.w_rawfunctype);
    let raw = W_RawFuncType::from_obj(raw_type)
        .ok_or_else(|| PyError::system_error("function wrapper lost its raw type"))?;
    let nargs_expected = raw.nostruct_nargs as usize;
    let nargs_given = args.len().saturating_sub(1);
    if nargs_given != nargs_expected {
        let fnname = pyre_interpreter::baseobjspace::text_w(wrapper.w_fnname)?;
        let message = match nargs_expected {
            0 => format!("{fnname}() takes no arguments ({nargs_given} given)"),
            1 => format!("{fnname}() takes exactly one argument ({nargs_given} given)"),
            _ => {
                format!("{fnname}() takes exactly {nargs_expected} arguments ({nargs_given} given)")
            }
        };
        return Err(PyError::type_error(message));
    }

    let mut call_args: Vec<PyObjectRef> = (1..args.len())
        .map(|i| roots.get(wrapper_slot + i))
        .collect();
    let nostruct_ctype = raw.nostruct_ctype;
    let type_base = roots.pin_roots(&[raw_type, nostruct_ctype]);
    let locs = if raw.nostruct_locs.is_null() {
        Vec::new()
    } else {
        unsafe { pyre_object::bytesobject::w_bytes_data(raw.nostruct_locs) }.to_vec()
    };
    let fnptr = wrapper.fnptr;
    if locs.first() == Some(&b'R') {
        let fargs = ctypefunc::fargs_of(ctypeobj::ctype_arg(nostruct_ctype)?);
        let mut nonzero_allocator = allocator::W_Allocator {
            should_clear_after_alloc: 0,
            ..Default::default()
        };
        let w_result_cdata = ctypeptr::pointer_newp_with_allocator(
            fargs[0],
            pyre_object::w_none(),
            Some(&mut nonzero_allocator),
        )?;
        let result_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(w_result_cdata);
        call_args.insert(0, roots.get(result_slot));
        prepare_args(roots.get(type_base), &mut call_args, 1)?;
        let _ = ctypefunc::call(
            ctypeobj::ctype_arg(roots.get(type_base + 1))?,
            fnptr,
            &call_args,
        )?;
        let result = cdataobj::cdata_arg(roots.get(result_slot))?;
        let result_ptr = ctypeobj::ctype_arg(result.ctype)?;
        assert_eq!(result_ptr.kind, ctypeobj::KIND_POINTER);
        unsafe { ctypeobj::convert_to_object(ctypeobj::ctype_arg(result_ptr.ctitem)?, result.ptr) }
    } else {
        if !locs.is_empty() {
            prepare_args(roots.get(type_base), &mut call_args, 0)?;
        }
        ctypefunc::call(
            ctypeobj::ctype_arg(roots.get(type_base + 1))?,
            fnptr,
            &call_args,
        )
    }
}

fn wrapper_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let wrapper = wrapper_arg(args[0])?;
    let raw = W_RawFuncType::from_obj(wrapper.w_rawfunctype)
        .ok_or_else(|| PyError::system_error("function wrapper lost its raw type"))?;
    let fnname = pyre_interpreter::baseobjspace::text_w(wrapper.w_fnname)?.to_string();
    let doc = raw.repr_fn_type(wrapper.w_ffi, &fnname)?;
    Ok(pyre_object::w_str_new_managed(&format!(
        "<FFIFunctionWrapper '{doc}'>"
    )))
}

fn wrapper_get_doc(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let wrapper = wrapper_arg(args[1])?;
    let raw = W_RawFuncType::from_obj(wrapper.w_rawfunctype)
        .ok_or_else(|| PyError::system_error("function wrapper lost its raw type"))?;
    let fnname = pyre_interpreter::baseobjspace::text_w(wrapper.w_fnname)?.to_string();
    let modulename = pyre_interpreter::baseobjspace::text_w(wrapper.w_modulename)?.to_string();
    let doc = raw.repr_fn_type(wrapper.w_ffi, &fnname)?;
    Ok(pyre_object::w_str_new_managed(&format!(
        "{doc};\n\nCFFI C function from {}.lib",
        modulename
    )))
}

fn wrapper_get_name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(wrapper_arg(args[1])?.w_fnname)
}

fn wrapper_get_module(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(wrapper_arg(args[1])?.w_modulename)
}

fn wrapper_get(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    pyre_interpreter::type_methods::arity_at_least(args, "__get__", 2)?;
    pyre_interpreter::type_methods::arity_at_most(args, "__get__", 3)?;
    let _ = wrapper_arg(args[0])?;
    Ok(args[0])
}

/// `W_FunctionWrapper.try_extract_direct_fnptr_as_cdata`.
pub fn try_extract_direct_fnptr_as_cdata(w_wrapper: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let wrapper = wrapper_arg(w_wrapper)?;
    if wrapper.directfnptr == 0 {
        return Ok(w_wrapper);
    }
    let w_ctype = typeof_wrapper(w_wrapper, wrapper.w_ffi)?;
    Ok(cdataobj::new_cdata(wrapper.directfnptr, w_ctype))
}

static FUNCTION_WRAPPER_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.__FFIFunctionWrapper`.
pub fn function_wrapper_type() -> PyObjectRef {
    FUNCTION_WRAPPER_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.__FFIFunctionWrapper",
            init_function_wrapper_type,
            pyre_interpreter::typedef::w_object(),
            <W_FunctionWrapper as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_FunctionWrapper as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

fn init_function_wrapper_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    store(
        "__repr__",
        pyre_interpreter::make_builtin_function_with_arity("__repr__", wrapper_repr, 1),
    );
    store(
        "__call__",
        pyre_interpreter::make_builtin_function("__call__", wrapper_call),
    );
    store(
        "__get__",
        pyre_interpreter::make_builtin_function("__get__", wrapper_get),
    );
    for (name, getter) in [
        (
            "__name__",
            wrapper_get_name as pyre_interpreter::gateway::BuiltinCodeFn,
        ),
        ("__module__", wrapper_get_module),
        ("__doc__", wrapper_get_doc),
    ] {
        let getter = pyre_interpreter::make_builtin_function_with_arity(name, getter, 2);
        store(
            name,
            pyre_interpreter::typedef::make_getset_property_named(
                getter,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                name,
            ),
        );
    }
}
