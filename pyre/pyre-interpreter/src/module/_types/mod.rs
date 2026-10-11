//! `_types` native type-object exports.

use pyre_object::*;

fn store(mut ns: PyObjectRef, name: &str, mut ty: PyObjectRef) {
    // Both words already exist. Sequential `pin_root(ns)` would be a
    // safepoint while `ty` is still unpublished (`RootScope::pin_roots`).
    crate::__pyre_store!(ns, name, ty);
}

#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
fn capsule_type() -> PyObjectRef {
    crate::cpyext::capsule::capsule_type()
}

#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
/// The build carries no capsules at all, so the name answers with a type that
/// can produce none: `PyCapsule_Type` has no `tp_new` and no
/// `Py_TPFLAGS_BASETYPE`, and a capsule only ever comes from `PyCapsule_New`.
fn capsule_type() -> PyObjectRef {
    static CAPSULE_TYPE: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    CAPSULE_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("PyCapsule", |ns| unsafe {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(
                ns_slot,
                "__new__",
                crate::typedef::make_new_descr(|_| {
                    Err(crate::PyError::type_error(
                        "cannot create 'PyCapsule' instances",
                    ))
                })
            );
        });
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

pub fn init(ns: PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let function_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(crate::typedef::gettypeobject(
        &crate::function::FUNCTION_TYPE,
    ));
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "AsyncGeneratorType",
        crate::typedef::gettypeobject(&pyre_object::generator::ASYNC_GENERATOR_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "BuiltinFunctionType",
        crate::typedef::gettypeobject(&crate::function::BUILTIN_FUNCTION_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "BuiltinMethodType",
        crate::typedef::gettypeobject(&crate::function::BUILTIN_FUNCTION_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "CapsuleType",
        capsule_type(),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "CellType",
        crate::typedef::gettypeobject(&pyre_object::nestedscope::CELL_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "ClassMethodDescriptorType",
        crate::typedef::gettypeobject(&crate::function::CLASSMETHOD_DESCRIPTOR_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "CodeType",
        crate::typedef::gettypeobject(&crate::pycode::CODE_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "CoroutineType",
        crate::typedef::gettypeobject(&pyre_object::generator::COROUTINE_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "EllipsisType",
        crate::typedef::gettypeobject(&pyre_object::ELLIPSIS_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "FrameType",
        crate::typedef::gettypeobject(&crate::pyframe::FRAME_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "FunctionType",
        pyre_object::gc_roots::shadow_stack_get(function_slot),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "GeneratorType",
        crate::typedef::gettypeobject(&pyre_object::generator::GENERATOR_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "GenericAlias",
        crate::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "GetSetDescriptorType",
        crate::typedef::gettypeobject(&pyre_object::typedef::GETSET_DESCRIPTOR_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "LambdaType",
        pyre_object::gc_roots::shadow_stack_get(function_slot),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "MappingProxyType",
        crate::typedef::gettypeobject(&pyre_object::MAPPING_PROXY_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "MemberDescriptorType",
        crate::typedef::gettypeobject(&pyre_object::typedef::MEMBER_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "MethodDescriptorType",
        crate::typedef::gettypeobject(&crate::function::METHOD_DESCRIPTOR_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "MethodType",
        crate::typedef::gettypeobject(&pyre_object::function::METHOD_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "MethodWrapperType",
        crate::typedef::gettypeobject(&crate::function::METHOD_WRAPPER_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "ModuleType",
        crate::typedef::gettypeobject(&pyre_object::MODULE_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "NoneType",
        crate::typedef::gettypeobject(&pyre_object::NONE_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "NotImplementedType",
        crate::typedef::gettypeobject(&pyre_object::NOTIMPLEMENTED_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "SimpleNamespace",
        crate::module::sys::vm::simple_namespace_type(),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "TracebackType",
        crate::typedef::gettypeobject(&crate::pytraceback::PYTRACEBACK_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "UnionType",
        crate::typedef::gettypeobject(&pyre_object::UNION_TYPE),
    );
    store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "WrapperDescriptorType",
        crate::typedef::gettypeobject(&crate::function::SLOT_WRAPPER_TYPE),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn capsule_type_publishes_null_tp_new() {
        crate::typedef::init_typeobjects();
        let capsule_type = super::capsule_type();
        assert!(unsafe { pyre_object::w_type_disallows_instantiation(capsule_type) });

        let flags = crate::baseobjspace::getattr_str(capsule_type, "__flags__")
            .expect("PyCapsule.__flags__ lookup failed");
        assert_ne!(unsafe { pyre_object::w_int_get_value(flags) } & (1 << 7), 0);
    }
}
