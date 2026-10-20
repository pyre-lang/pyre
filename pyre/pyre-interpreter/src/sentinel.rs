//! PEP 661 builtin `sentinel` type (`sentinelobject.c` `PySentinel_Type`).

use pyre_object::*;

use crate::gateway::Signature;
use crate::typedef::{
    check_user_subclass, make_getset_property_named, make_new_descr_with_signature,
};

fn caller_module() -> Result<PyObjectRef, crate::PyError> {
    // `executioncontext.py gettopframe_nohidden`, same walk as
    // `typeobject.py ensure_module_attr`.
    let ec = crate::call::getexecutioncontext();
    let frame = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec).gettopframe_nohidden() }
    };
    if frame.is_null() {
        return Ok(w_none());
    }
    let w_globals = unsafe { (*frame).get_w_globals() };
    if w_globals.is_null() || unsafe { is_none(w_globals) } {
        return Ok(w_none());
    }
    Ok(crate::baseobjspace::finditem_str(w_globals, "__name__")?.unwrap_or_else(w_none))
}

fn sentinel_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    if self_.is_null() || !unsafe { is_sentinel(self_) } {
        let received = if self_.is_null() {
            "NULL".to_string()
        } else {
            crate::type_methods::arg_type_name(self_)
        };
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'sentinel' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a 'sentinel' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(self_)
}

fn sentinel_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args
        .first()
        .copied()
        .filter(|o| !o.is_null())
        .ok_or_else(|| {
            crate::PyError::type_error("sentinel() missing required argument 'cls' (pos 1)")
        })?;
    let name = args
        .get(1)
        .copied()
        .filter(|o| !o.is_null())
        .ok_or_else(|| {
            crate::PyError::type_error("sentinel() missing required argument 'name' (pos 1)")
        })?;
    let repr = args
        .get(2)
        .copied()
        .filter(|o| !o.is_null())
        .unwrap_or(w_none());
    let _roots = gc_roots::push_roots();
    let cls_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(cls);
    let name_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(name);
    let repr_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(repr);
    let exact = crate::typedef::gettypeobject(&SENTINEL_TYPE);
    check_user_subclass(exact, gc_roots::shadow_stack_get(cls_slot))?;
    let name = gc_roots::shadow_stack_get(name_slot);
    if unsafe { !is_str(name) } {
        return Err(crate::PyError::type_error(format!(
            "sentinel() argument 1 must be str, not {}",
            crate::type_methods::clinic_arg_type_name(name)
        )));
    }
    let repr = gc_roots::shadow_stack_get(repr_slot);
    let stored_repr = if unsafe { is_none(repr) } {
        PY_NULL
    } else if unsafe { is_str(repr) } {
        repr
    } else {
        return Err(crate::PyError::type_error(format!(
            "sentinel() argument 'repr' must be str or None, not {}",
            crate::type_methods::clinic_arg_type_name(repr)
        )));
    };
    let repr_store_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(stored_repr);
    let module = caller_module()?;
    Ok(w_sentinel_new(
        gc_roots::shadow_stack_get(name_slot),
        module,
        gc_roots::shadow_stack_get(repr_store_slot),
    ))
}

fn sentinel_fget_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || !unsafe { is_sentinel(self_) } {
        return Err(crate::PyError::attribute_error("__name__"));
    }
    Ok(unsafe { w_sentinel_get_name(self_) })
}

fn sentinel_fset_name(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::attribute_error("readonly attribute"))
}

fn sentinel_fget_module(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || !unsafe { is_sentinel(self_) } {
        return Err(crate::PyError::attribute_error("__module__"));
    }
    let module = unsafe { w_sentinel_get_module(self_) };
    if module.is_null() {
        return Err(crate::PyError::attribute_error("__module__"));
    }
    Ok(module)
}

fn sentinel_fset_module(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || !unsafe { is_sentinel(self_) } {
        return Err(crate::PyError::attribute_error("__module__"));
    }
    let value = args.get(2).copied().unwrap_or(PY_NULL);
    unsafe { w_sentinel_set_module(self_, value) };
    Ok(w_none())
}

fn sentinel_fdel_module(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || !unsafe { is_sentinel(self_) } {
        return Err(crate::PyError::attribute_error("__module__"));
    }
    unsafe { w_sentinel_set_module(self_, PY_NULL) };
    Ok(w_none())
}

fn sentinel_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = sentinel_receiver(args, "__repr__", false)?;
    let repr = unsafe { w_sentinel_get_repr(self_) };
    if repr.is_null() {
        Ok(unsafe { w_sentinel_get_name(self_) })
    } else {
        Ok(repr)
    }
}

fn sentinel_descr_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    sentinel_receiver(args, "__copy__", true)
}

fn sentinel_descr_deepcopy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    sentinel_receiver(args, "__deepcopy__", true)
}

fn sentinel_descr_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = sentinel_receiver(args, "__reduce__", true)?;
    Ok(unsafe { w_sentinel_get_name(self_) })
}

pub(crate) fn init_sentinel_type(ns: PyObjectRef) {
    let _root_scope = gc_roots::push_roots();
    let ns_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            crate::function_set_kwonly_defaults!(
                make_new_descr_with_signature(
                    sentinel_descr_new,
                    Signature::new(vec!["cls", "name", "repr"], None, None, 1, 2),
                ),
                "repr" => w_none(),
            )
        )
    };
    unsafe {
        crate::function::fset_func_text_signature(
            w_dict_getitem_str(gc_roots::shadow_stack_get(ns_slot), "__new__")
                .expect("sentinel.__new__"),
            w_str_new("($type, name, /, *, repr=None)"),
        );
    }
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::gateway::make_slot_wrapper_with_arity_and_doc(
                "__repr__",
                sentinel_descr_repr,
                1,
                "Return repr(self).",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__copy__",
            crate::gateway::make_method_descriptor_with_arity("__copy__", sentinel_descr_copy, 1)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__deepcopy__",
            crate::gateway::make_method_descriptor_with_arity(
                "__deepcopy__",
                sentinel_descr_deepcopy,
                2,
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__reduce__",
            crate::gateway::make_method_descriptor_with_arity(
                "__reduce__",
                sentinel_descr_reduce,
                1,
            )
        )
    };
    let name_getter = crate::make_builtin_function_with_arity("__name__", sentinel_fget_name, 2);
    let name_getter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(name_getter);
    let name_setter = crate::make_builtin_function_with_arity("__name__", sentinel_fset_name, 3);
    let name_setter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(name_setter);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__name__",
            make_getset_property_named(
                gc_roots::shadow_stack_get(name_getter_slot),
                gc_roots::shadow_stack_get(name_setter_slot),
                gc_roots::shadow_stack_get(name_setter_slot),
                "__name__",
            )
        )
    };
    let module_getter =
        crate::make_builtin_function_with_arity("__module__", sentinel_fget_module, 2);
    let module_getter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(module_getter);
    let module_setter =
        crate::make_builtin_function_with_arity("__module__", sentinel_fset_module, 3);
    let module_setter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(module_setter);
    let module_deleter =
        crate::make_builtin_function_with_arity("__module__", sentinel_fdel_module, 2);
    let module_deleter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(module_deleter);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__module__",
            make_getset_property_named(
                gc_roots::shadow_stack_get(module_getter_slot),
                gc_roots::shadow_stack_get(module_setter_slot),
                gc_roots::shadow_stack_get(module_deleter_slot),
                "__module__",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__doc__",
            w_str_new("Create a unique sentinel object with the given name.")
        )
    };
}
