//! ctypes metaclasses, `CField` descriptors, and the `Structure`/`Union` bases.
//!
//! `PyCSimpleType` / `PyCStructType` / `PyCUnionType` / `PyCArrayType` /
//! `PyCPointerType` are the metaclasses of `_SimpleCData` / `Structure` /
//! `Union` / `Array` / `_Pointer`.  Their `__new__` builds the class and
//! computes its [`super::stginfo::StgInfoData`] (simple validation, the
//! struct/union layout in [`process_fields`], or the array/pointer element
//! metadata).  `CField` is the per-field data descriptor installed into a class
//! dict; its `__get__`/`__set__` read and write the instance buffer at the
//! field offset, aliasing nested aggregates as sub-views
//! (`super::cdata::make_subview`).
//!
//! `ctype * n` builds a cached `Array` subtype (`array_type_from_ctype`);
//! `POINTER(T)` builds a cached `_Pointer` subtype, memoised on `T`'s
//! `StgInfo.pointer_type` and read back through the `__pointer_type__` getset.

use super::cdata;
use super::stginfo::{self, ParamFunc, StgInfoData};
use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;

type PyResult = Result<PyObjectRef, pyre_interpreter::PyError>;

/// A ctypes metaclass derived from `type`, whose *instances* are therefore
/// type objects.
///
/// `make_builtin_type_with_base` would name the general instance layout
/// instead, which is the record `check_user_subclass` reads to decide whether
/// `type.__new__(mcls, …)` may allocate — and `type` itself is set up with
/// `TYPE_TYPE` for exactly that reason.  Upstream never faces the question:
/// its ctypes metaclasses are app-level (`class _CDataMeta(type)`,
/// basics.py) and inherit `type`'s instance layout from their base.
fn make_ctypes_metatype(name: &str, init: impl FnOnce(PyObjectRef)) -> PyObjectRef {
    let base = ctype_type();
    let layout = unsafe { pyre_object::w_type_get_layout_ptr(base) };
    let tp = pyre_interpreter::typedef::make_builtin_type_with_overridetypedef(
        name,
        init,
        base,
        unsafe { (*layout).typedef },
    );
    super::finish_cpython_type(tp, "_ctypes", true)
}

// ── cached type objects ────────────────────────────────────────────────

macro_rules! cached_type {
    ($cell:ident, $f:ident, $build:expr) => {
        static $cell: pyre_object::gc_roots::RootedOnceRef =
            pyre_object::gc_roots::RootedOnceRef::new();
        pub(super) fn $f() -> PyObjectRef {
            $cell.get_or_init($build)
        }
    };
}

// CPython 3.14 `_ctypes.CType_Type` and PyPy `_CDataMeta` are the common
// metaclass owner for every concrete ctypes metaclass.  Keeping the shared
// methods here avoids the former pyre-only duplication on every child.
cached_type!(CTYPE_TYPE, ctype_type, || {
    let base = pyre_interpreter::typedef::w_type();
    let layout = unsafe { pyre_object::w_type_get_layout_ptr(base) };
    let tp = pyre_interpreter::typedef::make_builtin_type_with_overridetypedef(
        "CType_Type",
        install_shared_meta,
        base,
        unsafe { (*layout).typedef },
    );
    super::finish_cpython_type(tp, "_ctypes", true)
});

cached_type!(PYCSIMPLETYPE, pycsimpletype_type, || {
    make_ctypes_metatype("PyCSimpleType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_new(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            csimpletype_new,
        );
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            csimpletype_init,
        );
    })
});

cached_type!(PYCSTRUCTTYPE, pycstructtype_type, || {
    make_ctypes_metatype("PyCStructType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_new(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cstructtype_new,
        );
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cstructtype_init,
        );
        install_fields_getset(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    })
});

cached_type!(PYCUNIONTYPE, pycuniontype_type, || {
    // The union metaclass's Python-visible name is `UnionType` (matching
    // `_ctypes.UnionType`), though its Rust identifier is PyCUnionType.
    make_ctypes_metatype("UnionType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_new(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cuniontype_new,
        );
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cuniontype_init,
        );
        install_fields_getset(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    })
});

cached_type!(PYCFUNCPTRTYPE, pycfuncptrtype_type, || {
    make_ctypes_metatype("PyCFuncPtrType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cfuncptrtype_init,
        );
    })
});

cached_type!(STRUCTURE, structure_type, || {
    let tp = pyre_interpreter::typedef::make_builtin_type_with_base(
        "Structure",
        init_aggregate_base,
        cdata::cdata_type(),
    );
    finish_aggregate_base(tp, pycstructtype_type(), "struct");
    super::finish_cpython_type(tp, "_ctypes", true)
});

cached_type!(UNION, union_type, || {
    let tp = pyre_interpreter::typedef::make_builtin_type_with_base(
        "Union",
        init_aggregate_base,
        cdata::cdata_type(),
    );
    finish_aggregate_base(tp, pycuniontype_type(), "union");
    super::finish_cpython_type(tp, "_ctypes", true)
});

cached_type!(CFIELD, cfield_type, || {
    let tp = pyre_interpreter::typedef::make_builtin_type("CField", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let mut ns = pyre_object::gc_roots::pin_root(ns);
        pyre_interpreter::__pyre_store!(
            ns,
            "__new__",
            pyre_interpreter::typedef::make_new_descr(cfield_new_internal)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "__get__",
            pyre_interpreter::make_builtin_function("__get__", cfield_get)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "__set__", // `cfield_set` reads its value out of the slice, so the arity is
            // declared rather than left to the body.
            pyre_interpreter::make_builtin_function_with_arity("__set__", cfield_set, 3)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "__delete__",
            pyre_interpreter::make_builtin_function("__delete__", |_args| {
                Err(pyre_interpreter::PyError::type_error(
                    "can't delete attribute",
                ))
            })
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "__repr__",
            pyre_interpreter::make_builtin_function("__repr__", cfield_repr)
        );
        pyre_interpreter::__pyre_store!(
            ns,
            "__setattr__",
            pyre_interpreter::make_builtin_function("__setattr__", |_args| {
                Err(pyre_interpreter::PyError::attribute_error(
                    "CField attributes are read-only",
                ))
            })
        );
    });
    unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
    let tp = super::finish_cpython_type(tp, "ctypes", true);
    // [3.14-spec] CPython's public `ctypes.CField` descriptor type omits
    // BASETYPE.  PyPy does not publish this private storage owner at all;
    // keep pyre's existing descriptor implementation but prevent a user
    // subclass from acquiring an unsupported layout.
    unsafe { pyre_object::w_type_suppress_cpython_basetype(tp) };
    tp
});

cached_type!(PYCARRAYTYPE, pycarraytype_type, || {
    make_ctypes_metatype("PyCArrayType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_new(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            carraytype_new,
        );
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            carraytype_init,
        );
    })
});

cached_type!(PYCPOINTERTYPE, pycpointertype_type, || {
    make_ctypes_metatype("PyCPointerType", |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        install_new(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cpointertype_new,
        );
        install_init(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            cpointertype_init,
        );
    })
});

cached_type!(ARRAY, array_type, || {
    let tp = pyre_interpreter::typedef::make_builtin_type_with_base(
        "Array",
        init_array_base,
        cdata::cdata_type(),
    );
    finish_element_base(tp, pycarraytype_type());
    super::finish_cpython_type(tp, "_ctypes", true)
});

cached_type!(POINTER_BASE, pointer_base_type, || {
    let tp = pyre_interpreter::typedef::make_builtin_type_with_base(
        "_Pointer",
        init_pointer_base,
        cdata::cdata_type(),
    );
    finish_element_base(tp, pycpointertype_type());
    super::finish_cpython_type(tp, "_ctypes", true)
});

/// The receiver test the `Array` descriptors are checked against —
/// `wrapperdescr_call`'s `_PyObject_RealIsSubclass(Py_TYPE(self),
/// PyDescr_TYPE(descr))` with `PyCArray_Type` as `d_type`.  Every element
/// type builds its own subtype, so the test has to accept subclasses;
/// `isinstance_w` reads the native `w_class` and walks the MRO, which is
/// what the class hierarchy discriminates here — every cdata shares one
/// Rust layout.
pub fn is_array_instance(obj: PyObjectRef) -> bool {
    unsafe { pyre_interpreter::baseobjspace::isinstance_w(obj, array_type()) }
}

/// The same test against `PyCPointer_Type` for the `_Pointer` descriptors.
pub fn is_pointer_instance(obj: PyObjectRef) -> bool {
    unsafe { pyre_interpreter::baseobjspace::isinstance_w(obj, pointer_base_type()) }
}

fn install_new(ns: PyObjectRef, f: pyre_interpreter::gateway::BuiltinCodeFn) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(ns, "__new__", pyre_interpreter::typedef::make_new_descr(f));
}

fn install_init(ns: PyObjectRef, f: pyre_interpreter::gateway::BuiltinCodeFn) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "__init__",
        pyre_interpreter::make_builtin_function("__init__", f)
    );
}

/// `__mul__`, `__pointer_type__`, and `from_param` — shared by all metaclasses.
fn install_shared_meta(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "from_address",
        pyre_interpreter::make_builtin_function("from_address", cdata::cdata_from_address)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "in_dll",
        pyre_interpreter::make_builtin_function("in_dll", cdata::cdata_in_dll)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__mul__",
        pyre_interpreter::make_builtin_function("__mul__", meta_mul)
    );
    let pointer_get =
        pyre_interpreter::make_builtin_function_with_arity("__pointer_type__", pointer_type_get, 2);
    let pointer_get_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pointer_get);
    let pointer_set =
        pyre_interpreter::make_builtin_function_with_arity("__pointer_type__", pointer_type_set, 3);
    let pointer_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pointer_set);
    let pointer_del =
        pyre_interpreter::make_builtin_function_with_arity("__pointer_type__", pointer_type_del, 2);
    let pointer_del_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pointer_del);
    // Later pins forward the earlier functions. The descriptor reads the
    // slots (`ShadowStackFrameworkGCTransformer.pop_roots`).
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "__pointer_type__",
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(pointer_get_slot),
            pyre_object::gc_roots::shadow_stack_get(pointer_set_slot),
            pyre_object::gc_roots::shadow_stack_get(pointer_del_slot),
            "__pointer_type__"
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "from_param",
        pyre_interpreter::make_builtin_function("from_param", meta_from_param)
    );
}

/// No `fdel`: upstream `_fields_` is an ordinary class attribute that
/// `_structunion_setattro` intercepts on the way through `tp_setattro`, so a
/// delete there re-runs `PyCStructUnionType_update_stginfo` and then removes
/// the class-dict entry.  Reproducing that needs the setattro hook this getset
/// stands in for, not a deleter.
fn install_fields_getset(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(ns, "_fields_", {
        let fields_get_fn =
            pyre_interpreter::make_builtin_function_with_arity("_fields_", fields_get, 2);
        let fields_get_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(fields_get_fn);
        let fields_set_fn =
            pyre_interpreter::make_builtin_function_with_arity("_fields_", fields_set, 3);
        let fields_set_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(fields_set_fn);
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(fields_get_slot),
            pyre_object::gc_roots::shadow_stack_get(fields_set_slot),
            pyre_object::PY_NULL,
            "_fields_",
        )
    });
}

fn init_aggregate_base(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(structure_new)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__init__",
        pyre_interpreter::make_builtin_function("__init__", structure_init)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__setattr__",
        pyre_interpreter::make_builtin_function("__setattr__", structure_setattr)
    );
}

fn structure_setattr(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 3 || !unsafe { pyre_object::is_str(args[1]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "attribute name must be a string",
        ));
    }
    let obj = args[0];
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(args[1]) }) else {
        // A lone surrogate names no field. `object_setattr_surrogate` stores
        // it the way `object.__setattr__` stores any other name.
        let w_name = args[1];
        let text = unsafe { pyre_object::w_str_get_wtf8(w_name) };
        return unsafe {
            pyre_interpreter::baseobjspace::object_setattr_surrogate(obj, w_name, text, args[2])
        };
    };
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let slots_empty = pyre_interpreter::type_dict_lookup(cls, "__slots__").is_some_and(|slots| {
        (unsafe { pyre_object::is_tuple(slots) } && unsafe { pyre_object::w_tuple_len(slots) } == 0)
            || (unsafe { pyre_object::is_list(slots) }
                && unsafe { pyre_object::w_list_len(slots) } == 0)
    });
    if slots_empty && unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, name) }.is_none()
    {
        return Err(pyre_interpreter::PyError::attribute_error(format!(
            "'{}' object has no attribute '{}'",
            type_name(cls),
            name,
        )));
    }
    pyre_interpreter::baseobjspace::object_setattr(obj, name, args[2])
}

fn finish_aggregate_base(tp: PyObjectRef, metaclass: PyObjectRef, paramfunc: &'static str) {
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(tp, true);
        (*tp).w_class = metaclass;
    }
    // Bare Structure/Union are abstract; the first real subtype gets the
    // default zero-sized storage state from its metaclass __init__.
    let _ = paramfunc;
}

/// Finish the `Array` / `_Pointer` base: hasdict, acceptable-as-base, stamp its
/// metaclass.  Unlike the aggregate bases these get **no** default `StgInfo`, so
/// a bare `Array()` / `_Pointer()` is abstract until a subtype supplies
/// `_type_` (`POINTER(T)`) or `_type_`+`_length_` (`T * n`).
fn finish_element_base(tp: PyObjectRef, metaclass: PyObjectRef) {
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(tp, true);
        pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, true);
        (*tp).w_class = metaclass;
    }
}

fn init_array_base(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    // CPython 3.14 Modules/_ctypes/_ctypes.c Array_methods —
    // Py_GenericAlias with METH_CLASS.
    pyre_interpreter::__pyre_store!(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(pyre_interpreter::make_builtin_function(
            "__class_getitem__",
            pyre_interpreter::_pypy_generic_alias::generic_alias_class_getitem,
        ))
    );
    install_new(pyre_object::gc_roots::shadow_stack_get(ns_slot), array_new);
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "__init__",
        pyre_interpreter::make_builtin_function("__init__", array_init)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__len__",
        pyre_interpreter::make_builtin_function("__len__", array_len)
    );
    // These two read their key and value straight out of the slice, so the
    // declared arity is what stands between a short call and an
    // out-of-bounds read: `(c_int * 3).__getitem__(1)` aborted the
    // interpreter rather than raising.
    pyre_interpreter::__pyre_store!(
        ns,
        "__getitem__",
        pyre_interpreter::make_builtin_function_with_arity("__getitem__", array_getitem, 2)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__setitem__",
        pyre_interpreter::make_builtin_function_with_arity("__setitem__", array_setitem, 3)
    );
}

fn init_pointer_base(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    install_new(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        pointer_new,
    );
    let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "__init__",
        pyre_interpreter::make_builtin_function("__init__", pointer_init)
    );
    // Indexed the same way `_ctypes.Array`'s pair is, and unguarded the same
    // way before the arity was declared.
    pyre_interpreter::__pyre_store!(
        ns,
        "__getitem__",
        pyre_interpreter::make_builtin_function_with_arity("__getitem__", pointer_getitem, 2)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__setitem__",
        pyre_interpreter::make_builtin_function_with_arity("__setitem__", pointer_setitem, 3)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__bool__",
        pyre_interpreter::make_builtin_function("__bool__", pointer_bool)
    );
    pyre_interpreter::__pyre_store!(ns, "contents", {
        let contents_get_fn =
            pyre_interpreter::make_builtin_function_with_arity("contents", contents_get, 2);
        let contents_get_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(contents_get_fn);
        let contents_set_fn =
            pyre_interpreter::make_builtin_function_with_arity("contents", contents_set, 3);
        let contents_set_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(contents_set_fn);
        let contents_del_fn = pyre_interpreter::make_builtin_function_with_arity(
            "contents",
            |_args| {
                Err(pyre_interpreter::PyError::type_error(
                    "Pointer does not support item deletion",
                ))
            },
            2,
        );
        let contents_del_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(contents_del_fn);
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(contents_get_slot),
            pyre_object::gc_roots::shadow_stack_get(contents_set_slot),
            pyre_object::gc_roots::shadow_stack_get(contents_del_slot),
            "contents",
        )
    });
    pyre_interpreter::__pyre_store!(
        ns,
        "set_type",
        pyre_object::function::w_classmethod_new(pyre_interpreter::make_builtin_function(
            "set_type",
            pointer_set_type,
        ))
    );
}

fn pointer_set_type(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 || !unsafe { pyre_object::is_type(args[0]) && pyre_object::is_type(args[1]) }
    {
        return Err(pyre_interpreter::PyError::type_error(
            "set_type() requires a ctypes type",
        ));
    }
    let mut pointer_cls = args[0];
    let mut proto = args[1];
    let mut data = StgInfoData::new(
        host_ctypes::pointer_size(),
        host_ctypes::simple_type_align("P").unwrap_or(host_ctypes::pointer_size()),
        ParamFunc::Pointer,
    );
    data.element_size =
        pyre_object::with_roots!(pointer_cls, proto => stginfo::field_size_of(proto)).unwrap_or(0);
    data.length = 1;
    data.proto = Some(proto);
    data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    let info = pyre_object::with_roots!(pointer_cls, proto => stginfo::stginfo_new(data));
    stginfo::stginfo_set(pointer_cls, info);
    set_type_attr(pointer_cls, "_type_", proto);
    if let Some(info) = stginfo::stginfo_of(proto) {
        stginfo::stginfo_set_pointer_type(info, pointer_cls);
    }
    Ok(pyre_object::w_none())
}

// ── metaclass `__new__` ────────────────────────────────────────────────

fn csimpletype_new(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::builtins::type_descr_new(args)
}

fn cstructtype_new(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::builtins::type_descr_new(args)
}

fn cuniontype_new(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::builtins::type_descr_new(args)
}

const META_INITIALIZED_KEY: &str = "_ctypes_stginfo_initialized";

fn metaclass_init(args: &[PyObjectRef], initialize: fn(PyObjectRef) -> PyResult) -> PyResult {
    if args.len() < 4 {
        return Err(pyre_interpreter::PyError::type_error(
            "metaclass __init__ expects name, bases and namespace",
        ));
    }
    let cls = args
        .first()
        .copied()
        .filter(|&obj| unsafe { pyre_object::is_type(obj) })
        .ok_or_else(|| {
            pyre_interpreter::PyError::type_error("metaclass __init__ requires a type")
        })?;
    if own_dict_get(cls, META_INITIALIZED_KEY).is_some() {
        return Err(pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::SystemError,
            "already initialized",
        ));
    }
    initialize(cls)?;
    set_type_attr(cls, META_INITIALIZED_KEY, pyre_object::w_bool_from(true));
    Ok(pyre_object::w_none())
}

fn csimpletype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, simple_init_stginfo)
}

fn cstructtype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, |cls| struct_union_init_stginfo(cls, false))
}

fn cfuncptrtype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, funcptr_init_stginfo)
}

/// `PyCFuncPtrType` layout: pointer-sized `StgInfo` with `ISPOINTER`.
///
/// What a call itself needs -- `_argtypes_`, `_restype_`, `_flags_` -- is read
/// off the class where it was set, so the carrier is here for the questions
/// every other ctypes type answers out of one: how wide a field of this type
/// is, and what a `POINTER` to it or an array of it is built over.
fn funcptr_init_stginfo(mut cls: PyObjectRef) -> PyResult {
    let psize = host_ctypes::pointer_size();
    // `paramfunc` names the shape a carrier is read as, and a function pointer
    // is none of the five this build reads -- an argument of one is settled by
    // `classify_argtype` before a `paramfunc` is asked for.
    let mut data = StgInfoData::new(psize, psize, ParamFunc::Other);
    data.length = 1;
    data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    data.format = Some("X{}".to_string());
    let info = pyre_object::with_roots!(cls => stginfo::stginfo_new(data));
    stginfo::stginfo_set(cls, info);
    Ok(pyre_object::w_none())
}

fn cuniontype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, |cls| struct_union_init_stginfo(cls, true))
}

/// `PyCSimpleType` layout: validate `_type_` and build a simple `StgInfo`.
fn simple_init_stginfo(mut cls: PyObjectRef) -> PyResult {
    // No `_type_` at all: `_SimpleCData` itself and abstract intermediates.
    let Some(declared) = cdata::declared_type_str(cls) else {
        return Ok(pyre_object::w_none());
    };
    // Judged on what the class declared, not on the code it was read as: a
    // declaration too long to be one reads back as no code, which is what an
    // abstract intermediate looks like too.
    if declared.chars().count() != 1 || !host_ctypes::simple_type_chars().contains(declared) {
        return Err(cdata::invalid_type_code_error());
    }
    let tc = cdata::TypeCode::new(declared).ok_or_else(cdata::invalid_type_code_error)?;
    let size = host_ctypes::simple_type_size(&tc).ok_or_else(cdata::invalid_type_code_error)?;
    let align = host_ctypes::simple_type_align(&tc).ok_or_else(cdata::invalid_type_code_error)?;
    let mut data = StgInfoData::new(size, align, ParamFunc::Simple);
    data.format = Some(tc.to_string());
    if host_ctypes::simple_type_is_pointer(&tc) {
        data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    }
    let info = pyre_object::with_roots!(cls => stginfo::stginfo_new(data));
    stginfo::stginfo_set(cls, info);

    let endian_capable = matches!(
        tc.as_str(),
        "b" | "B" | "h" | "H" | "i" | "I" | "l" | "L" | "q" | "Q" | "f" | "d" | "g"
    );
    if endian_capable && size == 1 {
        set_type_attr(cls, "__ctype_le__", cls);
        set_type_attr(cls, "__ctype_be__", cls);
    } else if endian_capable && own_dict_get(cls, "_ctypes_native_peer").is_none() {
        // The namespace `dict` moves: each insertion allocates the key string
        // and may grow the dict, and building the swapped class allocates
        // throughout, so the word is read back out of a root slot at every use.
        // The allocated values arrive already built — call arguments evaluate
        // left to right, so reading the slot inline with an allocating value
        // expression would hand the store a pre-move address.
        let roots = pyre_object::gc_roots::push_roots();
        let cls_slot = roots.base();
        let _ = roots.pin_root(cls);
        let ns_slot = cls_slot + 1;
        let _ = roots.pin_root(pyre_object::w_dict_new());
        let w_tc = pyre_object::w_str_new(&tc);
        unsafe {
            pyre_object::w_dict_setitem_str(roots.get(ns_slot), "_type_", w_tc);
            pyre_object::w_dict_setitem_str(
                roots.get(ns_slot),
                "_swappedbytes_",
                pyre_object::w_bool_from(true),
            );
            pyre_object::w_dict_setitem_str(
                roots.get(ns_slot),
                "_ctypes_native_peer",
                roots.get(cls_slot),
            );
        }
        let cls = roots.get(cls_slot);
        let bases = unsafe { pyre_object::typeobject::w_type_get_bases(cls) };
        let name = unsafe { pyre_object::typeobject::w_type_get_name(cls) };
        let w_name = pyre_object::w_str_new(name);
        let swapped = pyre_interpreter::call::type_call_instantiate(
            pycsimpletype_type(),
            &[w_name, bases, roots.get(ns_slot)],
        )?;
        let cls = roots.get(cls_slot);
        // The new class does not move, but nothing references it until the
        // first `set_type_attr` below completes and that store allocates its
        // key string, so it takes one pin for liveness.
        let swapped = roots.pin_root(swapped);
        if cfg!(target_endian = "little") {
            set_type_attr(cls, "__ctype_le__", cls);
            set_type_attr(cls, "__ctype_be__", swapped);
            set_type_attr(swapped, "__ctype_le__", cls);
            set_type_attr(swapped, "__ctype_be__", swapped);
        } else {
            set_type_attr(cls, "__ctype_le__", swapped);
            set_type_attr(cls, "__ctype_be__", cls);
            set_type_attr(swapped, "__ctype_le__", swapped);
            set_type_attr(swapped, "__ctype_be__", cls);
        }
    }
    Ok(pyre_object::w_none())
}

// ── struct/union layout ────────────────────────────────────────────────

/// MRO items of `cls`; empty while the type is still being initialised and
/// carries no mro yet.
fn mro_items<'a>(cls: PyObjectRef) -> &'a [PyObjectRef] {
    let mro = unsafe { pyre_object::typeobject::w_type_get_mro(cls) };
    if mro.is_null() {
        return &[];
    }
    unsafe { (*mro).as_slice() }
}

/// First base (in MRO order after `cls`) that carries a `StgInfo`.
fn first_base_stginfo(cls: PyObjectRef) -> Option<PyObjectRef> {
    mro_items(cls)
        .iter()
        .skip(1)
        .find_map(|&t| stginfo::stginfo_of(t))
}

fn usize_attr(cls: PyObjectRef, name: &str, default: usize) -> usize {
    match unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, name) } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            (unsafe { pyre_object::w_int_get_value(o) }).max(0) as usize
        }
        _ => default,
    }
}

fn align_attr(cls: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
    match unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_align_") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            let value = unsafe { pyre_object::w_int_get_value(o) };
            if value < 0 {
                Err(pyre_interpreter::PyError::value_error(
                    "_align_ must be a non-negative integer",
                ))
            } else {
                Ok((value as usize).max(1))
            }
        }
        Some(_) => Err(pyre_interpreter::PyError::value_error(
            "_align_ must be a non-negative integer",
        )),
        None => Ok(1),
    }
}

fn anonymous_names(cls: PyObjectRef) -> Result<Vec<String>, pyre_interpreter::PyError> {
    let Some(value) = pyre_interpreter::type_dict_lookup(cls, "_anonymous_") else {
        return Ok(Vec::new());
    };
    let items = seq_items(value)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("_anonymous_ must be a sequence"))?;
    items
        .into_iter()
        .map(|item| {
            if unsafe { pyre_object::is_str(item) } {
                Ok(pyre_interpreter::baseobjspace::str_utf8_w(item)?.to_string())
            } else {
                Err(pyre_interpreter::PyError::type_error(
                    "_anonymous_ items must be strings",
                ))
            }
        })
        .collect()
}

/// Install the flattened descriptors contributed by an anonymous aggregate.
/// Anonymous carrier fields are skipped while their contents are recursively
/// promoted, exactly as `PyCStructUnionType_update_stginfo` does.
fn promote_anonymous_fields(
    mut cls: PyObjectRef,
    mut proto: PyObjectRef,
    base_offset: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let anonymous = anonymous_names(proto)?;
    let fields = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(proto, "_fields_") }
        .ok_or_else(|| {
            pyre_interpreter::PyError::attribute_error("anonymous field has no _fields_")
        })?;
    for entry in pyre_object::with_roots!(cls, proto => field_entries(fields))? {
        let name = entry.name;
        let mut child_proto = entry.ty;
        let mut child = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(proto, &name) }
            .ok_or_else(|| {
                pyre_interpreter::PyError::attribute_error(format!(
                    "type has no attribute '{name}'"
                ))
            })?;
        let offset = base_offset
            + pyre_object::with_roots!(child, child_proto, cls, proto => cf_usize(child, "offset"));
        if anonymous.iter().any(|anon| anon == &name) {
            pyre_object::with_roots!(cls, proto => promote_anonymous_fields(cls, child_proto, offset))?;
        } else {
            let byte_size = pyre_object::with_roots!(child, child_proto, cls, proto => cf_usize(child, "byte_size"));
            let index = pyre_object::with_roots!(child, child_proto, cls, proto => cf_usize(child, "index"));
            let promoted = pyre_object::with_roots!(child, cls, proto => cfield_new(
                &name,
                child_proto,
                offset,
                byte_size,
                index,
            ));
            // The promoted field's dict moves and each copied entry allocates
            // the key string. `promoted`, `child`, `cls`, and `proto` stay
            // pinned across those copies; later reads use the slots.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[promoted, child, cls, proto]);
            let d_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyre_interpreter::baseobjspace::getdict_native(
                roots.get(base),
            ));
            unsafe {
                for key in ["size", "bit_size", "bit_offset", "is_bitfield"] {
                    if let Some(value) = pyre_object::w_dict_getitem_str(
                        pyre_interpreter::baseobjspace::getdict_native(roots.get(base + 1)),
                        key,
                    ) {
                        pyre_object::w_dict_setitem_str(roots.get(d_slot), key, value);
                    }
                }
            }
            set_type_attr(roots.get(base + 2), &name, roots.get(base));
            cls = roots.get(base + 2);
            proto = roots.get(base + 3);
        }
    }
    Ok(())
}

struct FieldEntry {
    name: String,
    ty: PyObjectRef,
    bits: Option<usize>,
}

/// Parse `_fields_` into the same `(name, ctype[, bits])` records consumed by
/// CPython/PyPy's aggregate layout builder.
fn field_entries(fields: PyObjectRef) -> Result<Vec<FieldEntry>, pyre_interpreter::PyError> {
    let items = seq_items(fields).ok_or_else(|| {
        pyre_interpreter::PyError::type_error("_fields_ must be a sequence of 2-tuples")
    })?;
    let mut out = Vec::with_capacity(items.len());
    for mut it in items {
        if !unsafe { pyre_object::is_tuple(it) } {
            return Err(pyre_interpreter::PyError::type_error(
                "_fields_ entries must be tuples",
            ));
        }
        let n = unsafe { pyre_object::w_tuple_len(it) };
        if n < 2 {
            return Err(pyre_interpreter::PyError::type_error(
                "_fields_ entries must be (name, type) pairs",
            ));
        }
        if n > 3 {
            return Err(pyre_interpreter::PyError::type_error(
                "_fields_ entries must be (name, type) or (name, type, bits)",
            ));
        }
        let name = unsafe { pyre_object::w_tuple_getitem(it, 0) }.unwrap_or(pyre_object::PY_NULL);
        let mut ty = unsafe { pyre_object::w_tuple_getitem(it, 1) }.unwrap_or(pyre_object::PY_NULL);
        if name.is_null() || !unsafe { pyre_object::is_str(name) } {
            return Err(pyre_interpreter::PyError::type_error(
                "field name must be a string",
            ));
        }
        if ty.is_null() || !unsafe { pyre_object::is_type(ty) } {
            return Err(pyre_interpreter::PyError::type_error(
                "field type must be a ctypes type",
            ));
        }
        let name =
            pyre_object::with_roots!(it, ty => pyre_interpreter::baseobjspace::str_utf8_w(name))?
                .to_string();
        let bits = if n == 3 {
            let mut value =
                unsafe { pyre_object::w_tuple_getitem(it, 2) }.unwrap_or(pyre_object::PY_NULL);
            if value.is_null()
                || !unsafe { pyre_object::is_int(value) || pyre_object::is_long(value) }
            {
                return Err(pyre_interpreter::PyError::type_error(
                    "bit width must be an integer",
                ));
            }
            let tc = cdata::type_code_of(ty).unwrap_or_default();
            if !matches!(
                tc.as_str(),
                "b" | "B" | "h" | "H" | "i" | "I" | "l" | "L" | "q" | "Q" | "?"
            ) {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "bit fields not allowed for type {}",
                    type_name(ty),
                )));
            }
            let size =
                pyre_object::with_roots!(ty, value => stginfo::field_size_of(ty)).unwrap_or(0);
            let width =
                pyre_object::with_roots!(ty => pyre_interpreter::baseobjspace::int_w(value))
                    .unwrap_or(-1);
            if width <= 0 || width as usize > size.saturating_mul(8) {
                return Err(pyre_interpreter::PyError::value_error(format!(
                    "number of bits invalid for bit field '{name}'",
                )));
            }
            Some(width as usize)
        } else {
            None
        };
        out.push(FieldEntry { name, ty, bits });
    }
    Ok(out)
}

fn seq_items(obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if unsafe { pyre_object::is_tuple(obj) } {
        let n = unsafe { pyre_object::w_tuple_len(obj) } as i64;
        Some(
            (0..n)
                .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i) })
                .collect(),
        )
    } else if unsafe { pyre_object::is_list(obj) } {
        let n = unsafe { pyre_object::w_list_len(obj) } as i64;
        Some(
            (0..n)
                .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i) })
                .collect(),
        )
    } else {
        None
    }
}

/// Mark a field type's `StgInfo` FINAL (creating a minimal one if absent), so
/// it cannot later gain `_fields_`.
fn mark_type_final(mut ty: PyObjectRef, size: usize, align: usize) {
    match stginfo::stginfo_of(ty) {
        Some(info) => stginfo::stginfo_mark_final(info),
        None => {
            let mut data = StgInfoData::new(size, align, ParamFunc::Simple);
            data.flags |= stginfo::DICTFLAG_FINAL;
            let info = pyre_object::with_roots!(ty => stginfo::stginfo_new(data));
            stginfo::stginfo_set(ty, info);
        }
    }
}

/// Compute the layout for a struct (`is_union=false`) or union, installing the
/// `CField` descriptors and the class `StgInfo`.  Port of `process_fields`.
fn struct_union_init_stginfo(mut cls: PyObjectRef, is_union: bool) -> PyResult {
    // `_fields_` directly in the new class dict → process it; else clone the
    // first base's StgInfo (or a default).
    let own_fields = pyre_interpreter::type_dict_lookup(cls, "_fields_")
        .filter(|&f| !unsafe { pyre_object::is_none(f) });
    match own_fields {
        Some(fields) => process_fields(cls, fields, is_union),
        None => {
            let paramfunc = if is_union {
                ParamFunc::Union
            } else {
                ParamFunc::Struct
            };
            match first_base_stginfo(cls) {
                Some(mut base_info) => {
                    let size = pyre_object::with_roots!(base_info, cls => stginfo::stginfo_size(base_info));
                    let align = pyre_object::with_roots!(base_info, cls => stginfo::stginfo_align(base_info));
                    let mut data = StgInfoData::new(size, align, paramfunc);
                    data.length = pyre_object::with_roots!(base_info, cls => stginfo::stginfo_length(base_info));
                    // Cleared FINAL / pointer_type on the clone; mark base FINAL.
                    let info =
                        pyre_object::with_roots!(base_info, cls => stginfo::stginfo_new(data));
                    stginfo::stginfo_set(cls, info);
                    stginfo::stginfo_mark_final(base_info);
                }
                None => {
                    let info = pyre_object::with_roots!(cls => stginfo::stginfo_new(StgInfoData::new(0, 1, paramfunc)));
                    stginfo::stginfo_set(cls, info)
                }
            }
            Ok(pyre_object::w_none())
        }
    }
}

fn process_fields(cls: PyObjectRef, fields: PyObjectRef, is_union: bool) -> PyResult {
    // `_ctypes.cfield.process_fields` replaces the layout of an incomplete
    // aggregate in place.  Keep the pointer type already cached on the old
    // StgInfo: POINTER(Complete) must remain the type made while Complete was
    // incomplete (including its frozen PEP 3118 "&B" format).
    let cached_pointer_type = stginfo::stginfo_of(cls).and_then(stginfo::stginfo_pointer_type);
    // `_fields_` is accepted as a `list` as well as a tuple, and a `list`
    // moves; the whole layout pass below allocates before the sequence is
    // stored back on the class, so it is pinned here and read back at that
    // store.
    let fields_roots = pyre_object::gc_roots::push_roots();
    let fields_slot = fields_roots.pin_roots(&[
        fields,
        cls,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        cached_pointer_type.unwrap_or(pyre_object::PY_NULL),
    ]);
    let cls_slot = fields_slot + 1;
    let bi_slot = fields_slot + 2;
    let ftype_slot = fields_slot + 3;
    let info_slot = fields_slot + 4;
    let pointer_type_slot = fields_slot + 5;
    let cls = || fields_roots.get(cls_slot);
    let entries = field_entries(fields_roots.get(fields_slot))?;
    let anonymous = anonymous_names(cls())?;

    for name in &anonymous {
        let Some(entry) = entries.iter().find(|entry| &entry.name == name) else {
            return Err(pyre_interpreter::PyError::attribute_error(format!(
                "'{name}' is specified in _anonymous_ but not in _fields_"
            )));
        };
        if !matches!(proto_kind(entry.ty), ParamFunc::Struct | ParamFunc::Union) {
            return Err(pyre_interpreter::PyError::type_error(
                "anonymous field must be a structure or union",
            ));
        }
    }

    let is_swapped =
        unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls(), "_swappedbytes_") }
            .is_some();
    let pack = usize_attr(cls(), "_pack_", 0);
    let forced = align_attr(cls())?;
    if pack > 0
        && !cfg!(windows)
        && unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls(), "_layout_") }.is_none()
    {
        pyre_interpreter::warn::warn_deprecation(
            "_pack_ without explicit _layout_ uses deprecated MSVC layout",
        )?;
    }

    // Reject before mutating the class: a type already frozen (used as a field
    // elsewhere) or one that would embed itself by value cannot define fields.
    // Checking up front keeps a rejected `_fields_` from leaving half-installed
    // descriptors and a final flag behind.
    if stginfo::stginfo_of(cls()).is_some_and(stginfo::stginfo_is_final)
        || entries.iter().any(|entry| entry.ty == cls())
    {
        return Err(pyre_interpreter::PyError::attribute_error(
            "Structure or union cannot contain itself",
        ));
    }

    let (base_size, base_length, mut max_align) = match first_base_stginfo(cls()) {
        Some(bi) => {
            fields_roots.set(bi_slot, bi);
            (
                stginfo::stginfo_size(fields_roots.get(bi_slot)),
                stginfo::stginfo_length(fields_roots.get(bi_slot)),
                stginfo::stginfo_align(fields_roots.get(bi_slot)).max(forced),
            )
        }
        None => (0usize, 0usize, forced),
    };
    // A union inherits its base's footprint as a floor, so a derived union with
    // only smaller fields cannot shrink below the base; a struct starts laying
    // its own fields out after the inherited prefix.
    let mut offset = base_size;
    let mut union_max = base_size;
    let mut has_pointer = false;

    // Pass 1 — compute the full layout without touching the class.
    // Active allocation unit for consecutive bitfields.  MS layout (selected
    // implicitly by `_pack_`) only coalesces identical declared types; this is
    // the behavior exercised by CPython's `test_pack_layout_switch`.
    let mut active_bits: Option<(PyObjectRef, usize, usize, usize)> = None;
    let little_bits = is_swapped ^ cfg!(target_endian = "little");
    let mut pending: Vec<(usize, usize, usize, Option<(usize, usize)>)> =
        Vec::with_capacity(entries.len());
    for entry in &entries {
        let name = &entry.name;
        fields_roots.set(ftype_slot, entry.ty);
        let ftype = || fields_roots.get(ftype_slot);
        if is_swapped
            && (proto_kind(ftype()) == ParamFunc::Pointer
                || cdata::type_code_of(ftype())
                    .is_some_and(|code| matches!(code.as_str(), "u" | "P" | "z" | "Z" | "O")))
        {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "This type does not support other endian: {name}",
            )));
        }
        let size = stginfo::field_size_of(ftype()).ok_or_else(|| {
            pyre_interpreter::PyError::type_error(format!("field '{name}' has no size"))
        })?;
        let align = stginfo::field_align_of(ftype()).unwrap_or(1).max(1);
        let eff = if pack > 0 { pack.min(align) } else { align };
        max_align = max_align.max(eff);

        if let Some(fi) = stginfo::stginfo_of(ftype())
            && stginfo::stginfo_flags(fi)
                & (stginfo::TYPEFLAG_ISPOINTER | stginfo::TYPEFLAG_HASPOINTER)
                != 0
        {
            has_pointer = true;
        }

        if let Some(bits) = entry.bits {
            if is_union {
                let bit_offset = if little_bits { 0 } else { size * 8 - bits };
                pending.push((0, size, align, Some((bits, bit_offset))));
                union_max = union_max.max(size);
                continue;
            }
            let reuse = active_bits.filter(|(active_ty, active_size, _, used)| {
                *active_ty == ftype() && *active_size == size && *used + bits <= size * 8
            });
            let (field_offset, used) = if let Some((_, _, at, used)) = reuse {
                (at, used)
            } else {
                if eff > 0 && offset % eff != 0 {
                    offset += eff - (offset % eff);
                }
                let at = offset;
                offset += size;
                (at, 0)
            };
            let bit_offset = if little_bits {
                used
            } else {
                size * 8 - used - bits
            };
            active_bits = Some((ftype(), size, field_offset, used + bits));
            pending.push((field_offset, size, align, Some((bits, bit_offset))));
        } else {
            active_bits = None;
            if !is_union && eff > 0 && offset % eff != 0 {
                offset += eff - (offset % eff);
            }
            let field_offset = if is_union { 0 } else { offset };
            pending.push((field_offset, size, align, None));
            if is_union {
                union_max = union_max.max(size);
            } else {
                offset += size;
            }
        }
    }

    let total_align = max_align.max(forced);
    let raw = if is_union { union_max } else { offset };
    let aligned = if total_align > 0 {
        raw.div_ceil(total_align) * total_align
    } else {
        raw
    };

    // Pass 2 — commit: freeze field types, install `CField` descriptors, then
    // the class `StgInfo` and `_fields_`.
    for (index, entry) in entries.iter().enumerate() {
        let (field_offset, size, align, bitfield) = pending[index];
        mark_type_final(entry.ty, size, align);
        let cf = cfield_new(&entry.name, entry.ty, field_offset, size, index);
        // The descriptor's `dict` moves and each insertion allocates both the
        // value and the key string, so the word is read back out of a root
        // slot with the value already built.  `cf` is unreferenced until
        // `set_type_attr` below, so it takes one pin.
        let roots = pyre_object::gc_roots::push_roots();
        let cf_slot = roots.base();
        let cf = roots.pin_root(cf);
        let d_slot = roots.base() + 1;
        let _ = roots.pin_root(pyre_interpreter::baseobjspace::getdict_native(cf));
        let put = |key: &str, value: PyObjectRef| {
            // SAFETY: the slot holds the descriptor dict pinned above.
            unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), key, value) };
        };
        if let Some((bits, bit_offset)) = bitfield {
            put(
                "size",
                pyre_object::w_int_new(((bits << 16) | bit_offset) as i64),
            );
            put("bit_size", pyre_object::w_int_new(bits as i64));
            put("bit_offset", pyre_object::w_int_new(bit_offset as i64));
            put("is_bitfield", pyre_object::w_bool_from(true));
        }
        if anonymous
            .iter()
            .any(|anonymous_name| anonymous_name == &entry.name)
        {
            put("is_anonymous", pyre_object::w_bool_from(true));
        }
        set_type_attr(cls(), &entry.name, roots.get(cf_slot));
    }
    for name in &anonymous {
        let (index, entry) = entries
            .iter()
            .enumerate()
            .find(|(_, entry)| &entry.name == name)
            .expect("anonymous fields validated above");
        promote_anonymous_fields(cls(), entry.ty, pending[index].0)?;
    }

    let mut flags = stginfo::DICTFLAG_FINAL;
    if has_pointer {
        flags |= stginfo::TYPEFLAG_HASPOINTER;
    }
    if is_union {
        flags |= stginfo::TYPEFLAG_HASUNION;
    }
    let mut data = StgInfoData::new(
        aligned,
        total_align,
        if is_union {
            ParamFunc::Union
        } else {
            ParamFunc::Struct
        },
    );
    // Field count includes the inherited prefix, not just the own fields.
    data.length = base_length + entries.len();
    data.flags = flags;
    data.big_endian = is_swapped ^ cfg!(target_endian = "big");
    fields_roots.set(info_slot, stginfo::stginfo_new(data));
    let w = fields_roots.get(pointer_type_slot);
    let cached_pointer_type = if w.is_null() { None } else { Some(w) };
    if let Some(pointer_type) = cached_pointer_type {
        stginfo::stginfo_set_pointer_type(fields_roots.get(info_slot), pointer_type);
    }
    stginfo::stginfo_set(cls(), fields_roots.get(info_slot));

    // Store the raw `_fields_` so the metaclass getset can return it.
    set_type_attr(cls(), "_fields_", fields_roots.get(fields_slot));
    Ok(pyre_object::w_none())
}

// ── metaclass namespace methods ────────────────────────────────────────

/// `ctype * n` — build (and cache) the `n`-element array type of `ctype`.
fn meta_mul(args: &[PyObjectRef]) -> PyResult {
    let cls = args[0];
    let count = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if count.is_null() || !unsafe { pyre_object::is_int(count) || pyre_object::is_long(count) } {
        return Err(pyre_interpreter::PyError::not_implemented(
            "array length must be an int",
        ));
    }
    if unsafe { pyre_object::is_long(count) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(count) };
        if big.get_sign() < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "array length must not be negative",
            ));
        }
        return Err(pyre_interpreter::PyError::overflow_error("array too large"));
    }
    let n = unsafe { pyre_object::w_int_get_value(count) };
    if n < 0 {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "Array length must be >= 0, not {n}"
        )));
    }
    array_type_from_ctype(cls, n as usize)
}

fn meta_from_param(args: &[PyObjectRef]) -> PyResult {
    if args
        .first()
        .copied()
        .is_some_and(cdata::is_simplecdata_type)
    {
        return cdata::simple_from_param(args);
    }
    Ok(args.get(1).copied().unwrap_or_else(pyre_object::w_none))
}

/// `%R must have storage info`: `ctype_get_pointer_type` and
/// `ctype_set_pointer_type` open with this refusal, and the setter is what
/// serves a delete.
fn no_storage_info(cls: PyObjectRef) -> pyre_interpreter::PyError {
    let rendered = match unsafe { pyre_interpreter::display::py_repr_wtf8(cls) } {
        Ok(rendered) => rendered,
        Err(error) => return error,
    };
    pyre_interpreter::PyError::type_error(pyre_interpreter::display::wtf8_format!(
        rendered,
        " must have storage info"
    ))
}

fn pointer_type_get(args: &[PyObjectRef]) -> PyResult {
    let mut cls = args[1];
    // `ctype_get_pointer_type` splits the two absences: a class with no
    // storage info at all is the `TypeError` above, and one whose
    // `pointer_type` is unset names itself with `%R`.
    let Some(info) = stginfo::stginfo_of(cls) else {
        return Err(no_storage_info(cls));
    };
    if let Some(pt) = pyre_object::with_roots!(cls => stginfo::stginfo_pointer_type(info)) {
        return Ok(pt);
    }
    let rendered = unsafe { pyre_interpreter::display::py_repr_wtf8(cls) }?;
    Err(pyre_interpreter::PyError::attribute_error(
        pyre_interpreter::display::wtf8_format!(rendered, " has no attribute '__pointer_type__'"),
    ))
}

/// The setter's operand reaches `Py_XSETREF` unchecked, so deleting clears
/// the cached pointer type and the next read reports it absent again.
/// `stginfo_pointer_type` spells absence as `None`, which is what the clear
/// stores; a raw null would instead read back as a getter that answered
/// nothing, and the attribute machinery would fall through to the descriptor.
fn pointer_type_del(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    match stginfo::stginfo_of(cls) {
        Some(info) => {
            stginfo::stginfo_set_pointer_type(info, pyre_object::w_none());
            Ok(pyre_object::w_none())
        }
        None => Err(no_storage_info(cls)),
    }
}

fn pointer_type_set(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    let value = args[2];
    match stginfo::stginfo_of(cls) {
        Some(info) => {
            stginfo::stginfo_set_pointer_type(info, value);
            Ok(pyre_object::w_none())
        }
        None => Err(no_storage_info(cls)),
    }
}

fn fields_get(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_fields_") }
        .filter(|&f| !f.is_null())
        .ok_or_else(|| pyre_interpreter::PyError::attribute_error("_fields_"))
}

fn fields_set(args: &[PyObjectRef]) -> PyResult {
    let mut cls = args[1];
    let mut value = args[2];
    let Some(mut info) = stginfo::stginfo_of(cls) else {
        return Err(pyre_interpreter::PyError::type_error(
            "ctypes state is not initialized",
        ));
    };
    if pyre_object::with_roots!(cls, info, value => stginfo::stginfo_is_final(info)) {
        return Err(pyre_interpreter::PyError::attribute_error(
            "_fields_ is final",
        ));
    }
    let is_union = pyre_object::with_roots!(cls, value => stginfo::stginfo_paramfunc(info))
        == ParamFunc::Union;
    process_fields(cls, value, is_union)
}

// ── CField descriptor ──────────────────────────────────────────────────

fn cfield_new(
    name: &str,
    proto: PyObjectRef,
    offset: usize,
    size: usize,
    index: usize,
) -> PyObjectRef {
    let inst = pyre_object::w_instance_new(cfield_type());
    // The instance `dict` moves, and every insertion below allocates: the
    // value, the key string `w_dict_setitem_str` builds, and the dict's own
    // storage when it grows.  A minor collection anywhere in that run
    // relocates the dict, so the word is read back out of a root slot for
    // each insertion.
    let roots = pyre_object::gc_roots::push_roots();
    let inst_slot = roots.base();
    let inst = roots.pin_root(inst);
    let proto_slot = inst_slot + 1;
    let _ = roots.pin_root(proto);
    // The dict lands in the slot after `proto`.
    let d_slot = inst_slot + 2;
    let _ = roots.pin_root(pyre_interpreter::baseobjspace::getdict_native(inst));
    let put = |key: &str, value: PyObjectRef| {
        // SAFETY: the slot holds the instance dict pinned above.
        unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), key, value) };
    };
    put("name", pyre_object::w_str_new(name));
    put("proto", roots.get(proto_slot));
    put("type", roots.get(proto_slot));
    put("offset", pyre_object::w_int_new(offset as i64));
    put("byte_offset", pyre_object::w_int_new(offset as i64));
    put("size", pyre_object::w_int_new(size as i64));
    put("byte_size", pyre_object::w_int_new(size as i64));
    let bit_size = BigInt::from(size).int_mul(8);
    put("bit_size", pyre_object::longobject::w_long_new(bit_size));
    put("bit_offset", pyre_object::w_int_new(0));
    put("is_bitfield", pyre_object::w_bool_from(false));
    put("is_anonymous", pyre_object::w_bool_from(false));
    put("index", pyre_object::w_int_new(index as i64));
    roots.get(inst_slot)
}

fn cf_obj(cfield: PyObjectRef, key: &str) -> PyObjectRef {
    unsafe {
        pyre_object::w_dict_getitem_str(pyre_interpreter::baseobjspace::getdict_native(cfield), key)
    }
    .unwrap_or(pyre_object::PY_NULL)
}

fn cf_usize(cfield: PyObjectRef, key: &str) -> usize {
    let o = cf_obj(cfield, key);
    if !o.is_null() && unsafe { pyre_object::is_int(o) } {
        (unsafe { pyre_object::w_int_get_value(o) }).max(0) as usize
    } else {
        0
    }
}

fn cf_bool(cfield: PyObjectRef, key: &str) -> bool {
    let value = cf_obj(cfield, key);
    !value.is_null() && pyre_interpreter::baseobjspace::is_true(value).unwrap_or(false)
}

fn bytes_to_native_uint(bytes: &[u8]) -> u64 {
    if cfg!(target_endian = "little") {
        bytes
            .iter()
            .take(8)
            .enumerate()
            .fold(0u64, |value, (i, byte)| value | ((*byte as u64) << (i * 8)))
    } else {
        bytes
            .iter()
            .take(8)
            .fold(0u64, |value, byte| (value << 8) | *byte as u64)
    }
}

fn native_uint_to_bytes(value: u64, size: usize) -> Vec<u8> {
    let bytes = if cfg!(target_endian = "little") {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    if cfg!(target_endian = "little") {
        bytes[..size.min(8)].to_vec()
    } else {
        bytes[8 - size.min(8)..].to_vec()
    }
}

/// The storage kind of a field's `proto`.  A type with no `StgInfo` of its own
/// is simple when it names a `_type_` code, and nothing otherwise.
fn proto_kind(mut proto: PyObjectRef) -> ParamFunc {
    if let Some(info) = stginfo::stginfo_of(proto) {
        let pf = pyre_object::with_roots!(proto => stginfo::stginfo_paramfunc(info));
        if pf != ParamFunc::Other {
            return pf;
        }
    }
    if cdata::type_code_of(proto).is_some() {
        return ParamFunc::Simple;
    }
    ParamFunc::Other
}

fn field_needs_swap(obj: PyObjectRef, proto: PyObjectRef, size: usize) -> bool {
    if size <= 1 {
        return false;
    }
    let oc = unsafe { pyre_object::w_instance_get_type(obj) };
    unsafe { pyre_interpreter::baseobjspace::lookup_in_type(oc, "_swappedbytes_") }.is_some()
        || unsafe { pyre_interpreter::baseobjspace::lookup_in_type(proto, "_swappedbytes_") }
            .is_some()
}

fn cfield_get(args: &[PyObjectRef]) -> PyResult {
    let mut cfield = args[0];
    let mut obj = args.get(1).copied().unwrap_or_else(pyre_object::w_none);
    // Accessed on the class (`Point.x`) → return the descriptor itself.
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        return Ok(cfield);
    }
    if !cdata::is_cdata_instance(obj) {
        return Err(pyre_interpreter::PyError::type_error(
            "not a ctypes instance",
        ));
    }
    let mut proto = pyre_object::with_roots!(cfield, obj => cf_obj(cfield, "proto"));
    let offset = pyre_object::with_roots!(cfield, obj, proto => cf_usize(cfield, "offset"));
    let size = pyre_object::with_roots!(cfield, obj, proto => cf_usize(cfield, "byte_size"));
    let index = pyre_object::with_roots!(cfield, obj, proto => cf_usize(cfield, "index"));

    match pyre_object::with_roots!(cfield, obj, proto => proto_kind(proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("field has no '_type_'"))?;
            let all = pyre_object::with_roots!(cfield, obj, proto => cdata::cdata_bytes(obj))
                .ok_or_else(|| pyre_interpreter::PyError::type_error("instance has no buffer"))?;
            let start = offset.min(all.len());
            let end = (offset + size).min(all.len());
            let mut field_bytes = all[start..end].to_vec();
            if field_needs_swap(obj, proto, size) {
                field_bytes.reverse();
            }
            if pyre_object::with_roots!(cfield => cf_bool(cfield, "is_bitfield")) {
                let bits = pyre_object::with_roots!(cfield => cf_usize(cfield, "bit_size"));
                let shift = cf_usize(cfield, "bit_offset");
                let mask = if bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                let raw = (bytes_to_native_uint(&field_bytes) >> shift) & mask;
                if matches!(tc.as_str(), "b" | "h" | "i" | "l" | "q") {
                    let signed = if bits < 64 && raw & (1u64 << (bits - 1)) != 0 {
                        (raw | !mask) as i64
                    } else {
                        raw as i64
                    };
                    return Ok(pyre_object::w_int_new(signed));
                }
                return Ok(pyre_object::longobject::w_long_new(BigInt::from(raw)));
            }
            Ok(cdata::decode_slot(&tc, &field_bytes))
        }
        ParamFunc::Array => {
            let element = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            let has_element = element.is_some();
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[obj, proto, element.unwrap_or(pyre_object::PY_NULL)]);
            let all = cdata::cdata_bytes(roots.get(base))
                .ok_or_else(|| pyre_interpreter::PyError::type_error("instance has no buffer"))?;
            let start = offset.min(all.len());
            let end = (offset + size).min(all.len());
            let field = &all[start..end];
            let element = has_element.then(|| roots.get(base + 2));
            match element.and_then(cdata::type_code_of).as_deref() {
                Some("c") => {
                    let n = field.iter().position(|&b| b == 0).unwrap_or(field.len());
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&field[..n]))
                }
                Some("u") => Ok(pyre_object::w_str_new_managed(
                    &host_ctypes::wstring_from_bytes(field),
                )),
                _ => Ok(cdata::make_indexed_subview(
                    roots.get(base + 1),
                    roots.get(base),
                    offset,
                    size,
                    index,
                )),
            }
        }
        ParamFunc::Struct | ParamFunc::Union | ParamFunc::Pointer => {
            Ok(cdata::make_indexed_subview(proto, obj, offset, size, index))
        }
        _ => Err(pyre_interpreter::PyError::type_error(
            "field type has no storage info",
        )),
    }
}

fn cfield_set(args: &[PyObjectRef]) -> PyResult {
    let mut cfield = args[0];
    let mut obj = args[1];
    let mut value = args[2];
    if !cdata::is_cdata_instance(obj) {
        return Err(pyre_interpreter::PyError::type_error(
            "not a ctypes instance",
        ));
    }
    let mut proto = pyre_object::with_roots!(cfield, obj, value => cf_obj(cfield, "proto"));
    let offset = pyre_object::with_roots!(cfield, obj, proto, value => cf_usize(cfield, "offset"));
    let size = pyre_object::with_roots!(cfield, obj, proto, value => cf_usize(cfield, "byte_size"));
    let index = pyre_object::with_roots!(cfield, obj, proto, value => cf_usize(cfield, "index"));

    match pyre_object::with_roots!(cfield, obj, proto, value => proto_kind(proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("field has no '_type_'"))?;
            if pyre_object::with_roots!(cfield, obj, proto, value => cf_bool(cfield, "is_bitfield"))
            {
                let bits = pyre_object::with_roots!(cfield, obj, proto, value => cf_usize(cfield, "bit_size"));
                let shift =
                    pyre_object::with_roots!(obj, proto, value => cf_usize(cfield, "bit_offset"));
                let mask = if bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                let integer = pyre_object::with_roots!(obj, proto => pyre_interpreter::baseobjspace::int_w(value))?
                    as u64;
                let mut storage = pyre_object::with_roots!(obj, proto => cdata::cdata_bytes(obj))
                    .unwrap_or(&[])
                    .get(offset..offset.saturating_add(size))
                    .unwrap_or(&[])
                    .to_vec();
                storage.resize(size, 0);
                if field_needs_swap(obj, proto, size) {
                    storage.reverse();
                }
                let old = bytes_to_native_uint(&storage);
                let combined = (old & !(mask << shift)) | ((integer & mask) << shift);
                let mut bytes = native_uint_to_bytes(combined, size);
                if field_needs_swap(obj, proto, size) {
                    bytes.reverse();
                }
                cdata::cdata_write(obj, offset, &bytes);
                return Ok(pyre_object::w_none());
            }
            // `value` is an arbitrary object under `"O"`, and the encode both
            // allocates and can run Python. `obj` and `proto` stay pinned
            // across that call; later reads use the slots.
            let _roots = pyre_object::gc_roots::push_roots();
            let base = _roots.pin_roots(&[value, obj, proto]);
            let mut bytes = cdata::encode_instance_or_value(
                &tc,
                _roots.get(base),
                _roots.get(base + 1),
                &index.to_string(),
            )?;
            if field_needs_swap(_roots.get(base + 1), _roots.get(base + 2), size) {
                bytes.reverse();
            }
            cdata::release_bstr_slot(
                &tc,
                cdata::cdata_addr(_roots.get(base + 1)).unwrap_or(0) + offset,
            );
            cdata::cdata_write(_roots.get(base + 1), offset, &bytes);
            let value = _roots.get(base);
            if let Some(keep) = cdata::value_for_keep(&tc, value, &bytes) {
                cdata::keep_ref(_roots.get(base + 1), &index.to_string(), keep);
            }
            Ok(pyre_object::w_none())
        }
        ParamFunc::Pointer => {
            if unsafe { pyre_object::is_none(value) } {
                cdata::cdata_write(obj, offset, &vec![0; size]);
                return Ok(pyre_object::w_none());
            }
            let direct = unsafe { pyre_interpreter::baseobjspace::isinstance_w(value, proto) };
            let expected = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            let array_decay = if cdata::is_cdata_instance(value) {
                let value_cls = unsafe { pyre_object::w_instance_get_type(value) };
                stginfo::stginfo_of(value_cls)
                    .filter(|&i| stginfo::stginfo_paramfunc(i) == ParamFunc::Array)
                    .and_then(stginfo::stginfo_proto)
                    == expected
            } else {
                false
            };
            let bytes = if direct {
                pyre_object::with_roots!(obj, value => cdata::cdata_bytes(value))
                    .unwrap_or(&[])
                    .to_vec()
            } else if array_decay {
                let addr = pyre_object::with_roots!(obj, value => cdata::cdata_addr(value))
                    .ok_or_else(|| pyre_interpreter::PyError::type_error("incompatible types"))?;
                host_ctypes::simple_storage_value_to_bytes_endian(
                    "P",
                    host_ctypes::SimpleStorageValue::Pointer(addr),
                    false,
                )
            } else {
                return Err(pyre_interpreter::PyError::type_error("incompatible types"));
            };
            pyre_object::with_roots!(obj, value => cdata::cdata_write(obj, offset, &bytes[..size.min(bytes.len())]));
            let keep = if array_decay {
                value
            } else {
                pyre_object::with_roots!(obj => cdata::objects_for_keep(value))
            };
            cdata::keep_ref(obj, &index.to_string(), keep);
            Ok(pyre_object::w_none())
        }
        ParamFunc::Array => {
            let element = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            match element.and_then(cdata::type_code_of).as_deref() {
                Some("c") => {
                    let source = pyre_interpreter::typedef::buffer_as_bytes_like(value)?
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::type_error("bytes-like object expected")
                        })?;
                    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(source) };
                    if bytes.len() > size {
                        return Err(pyre_interpreter::PyError::value_error("bytes too long"));
                    }
                    // PyCField_set: a character-array field is assigned with
                    // string semantics, so bytes beyond the first NUL do not
                    // overwrite the existing tail of the field.
                    let written = bytes
                        .iter()
                        .position(|&byte| byte == 0)
                        .map_or(bytes.len(), |nul| nul + 1);
                    cdata::cdata_write(obj, offset, &bytes[..written]);
                    Ok(pyre_object::w_none())
                }
                Some("u") => {
                    if !unsafe { pyre_object::is_str(value) } {
                        return Err(pyre_interpreter::PyError::type_error(
                            "unicode string expected",
                        ));
                    }
                    let mut field = pyre_object::with_roots!(obj, value => cdata::cdata_bytes(obj))
                        .unwrap_or(&[])
                        .get(offset..offset.saturating_add(size))
                        .unwrap_or(&[])
                        .to_vec();
                    field.resize(size, 0);
                    host_ctypes::write_wchar_array_value(&mut field, unsafe {
                        pyre_object::w_str_get_wtf8(value)
                    })
                    .map_err(|_| pyre_interpreter::PyError::value_error("string too long"))?;
                    cdata::cdata_write(obj, offset, &field);
                    Ok(pyre_object::w_none())
                }
                _ => {
                    let mut array_value = if unsafe {
                        pyre_interpreter::baseobjspace::isinstance_w(value, proto)
                    } {
                        value
                    } else if unsafe { pyre_object::is_tuple(value) } {
                        let values = seq_items(value).unwrap_or_default();
                        pyre_object::with_roots!(obj => pyre_interpreter::call::type_call_instantiate(proto, &values)).map_err(
                            |error| pyre_interpreter::PyError::runtime_error(error.message_wtf8()),
                        )?
                    } else {
                        return Err(pyre_interpreter::PyError::type_error("incompatible types"));
                    };
                    let source = pyre_object::with_roots!(array_value, obj => cdata::cdata_bytes(array_value))
                        .unwrap_or(&[])
                        .to_vec();
                    pyre_object::with_roots!(array_value, obj => cdata::cdata_write(obj, offset, &source[..source.len().min(size)]));
                    let keep =
                        pyre_object::with_roots!(obj => cdata::objects_for_keep(array_value));
                    cdata::keep_ref(obj, &index.to_string(), keep);
                    Ok(pyre_object::w_none())
                }
            }
        }
        // Structures and unions are copied by value.
        ParamFunc::Struct | ParamFunc::Union => {
            if !unsafe { pyre_interpreter::baseobjspace::isinstance_w(value, proto) } {
                return Err(pyre_interpreter::PyError::type_error("incompatible types"));
            }
            // Snapshot the source: `s.f = s.f` aliases the destination buffer,
            // and `cdata_write`'s `copy_from_slice` assumes non-overlap.
            let src = pyre_object::with_roots!(obj, value => cdata::cdata_bytes(value))
                .unwrap_or(&[])
                .to_vec();
            let n = size.min(src.len());
            pyre_object::with_roots!(obj, value => cdata::cdata_write(obj, offset, &src[..n]));
            let keep = pyre_object::with_roots!(obj => cdata::objects_for_keep(value));
            cdata::keep_ref(obj, &index.to_string(), keep);
            Ok(pyre_object::w_none())
        }
        _ => Err(pyre_interpreter::PyError::type_error(
            "assignment to this field type is not supported in this slice",
        )),
    }
}

fn cfield_repr(args: &[PyObjectRef]) -> PyResult {
    let mut cfield = args[0];
    let proto = pyre_object::with_roots!(cfield => cf_obj(cfield, "proto"));
    let tyname = if !proto.is_null() && unsafe { pyre_object::is_type(proto) } {
        unsafe { pyre_object::typeobject::w_type_get_name(proto) }.to_string()
    } else {
        "?".to_string()
    };
    let offset = pyre_object::with_roots!(cfield => cf_usize(cfield, "offset"));
    let s = format!(
        "<Field type={}, ofs={}, size={}>",
        tyname,
        offset,
        cf_usize(cfield, "size"),
    );
    Ok(pyre_object::w_str_new_managed(&s))
}

fn cfield_new_internal(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    let get = |name: &str, position: usize| {
        pyre_interpreter::builtins::resolve_pos_or_kw(
            pos.get(position).copied(),
            kwargs,
            name,
            "CField",
            position,
        )
    };
    let internal = get("_internal_use", 6)?
        .is_some_and(|value| pyre_interpreter::baseobjspace::is_true(value).unwrap_or(false));
    if !internal {
        return Err(pyre_interpreter::PyError::type_error(
            "CField is not intended to be used directly",
        ));
    }
    let mut name_obj =
        get("name", 1)?.ok_or_else(|| pyre_interpreter::PyError::type_error("missing name"))?;
    let mut proto =
        get("type", 2)?.ok_or_else(|| pyre_interpreter::PyError::type_error("missing type"))?;
    let byte_size = get("byte_size", 3)?
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .ok_or_else(|| pyre_interpreter::PyError::type_error("missing byte_size"))?;
    let byte_offset = get("byte_offset", 4)?
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    let index = get("index", 5)?
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    if !unsafe { pyre_object::is_str(name_obj) } || !unsafe { pyre_object::is_type(proto) } {
        return Err(pyre_interpreter::PyError::type_error(
            "invalid CField arguments",
        ));
    }
    let expected = pyre_object::with_roots!(name_obj, proto => stginfo::field_size_of(proto))
        .ok_or_else(|| pyre_interpreter::PyError::type_error("type has no size"))?;
    if byte_size < 0 || byte_size as usize != expected {
        return Err(pyre_interpreter::PyError::value_error(
            "byte_size does not match type size",
        ));
    }
    let field_name =
        pyre_object::with_roots!(proto => pyre_interpreter::baseobjspace::str_utf8_w(name_obj))?;
    let field = cfield_new(
        field_name,
        proto,
        byte_offset.max(0) as usize,
        byte_size as usize,
        index.max(0) as usize,
    );
    // `field` does not move, but nothing else references it before the return
    // and the argument decoding below allocates, so it takes one pin.
    let field_roots = pyre_object::gc_roots::push_roots();
    let field = field_roots.pin_root(field);
    let bit_size = get("bit_size", 7)?
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?;
    let bit_offset = get("bit_offset", 8)?
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    if let Some(bits) = bit_size {
        if bits <= 0 || bit_offset < 0 || bit_offset + bits > byte_size * 8 {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "bit field '{}' overflows its type ({} + {} > {})",
                field_name,
                bit_offset,
                bits,
                byte_size * 8,
            )));
        }
        // The field's `dict` moves and each insertion allocates both the value
        // and the key string, so the word is read back out of a root slot with
        // the value already built.
        let roots = pyre_object::gc_roots::push_roots();
        let d_slot = roots.base();
        let _ = roots.pin_root(pyre_interpreter::baseobjspace::getdict_native(field));
        let put = |key: &str, value: PyObjectRef| {
            // SAFETY: the slot holds the field dict pinned above.
            unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), key, value) };
        };
        put("bit_size", pyre_object::w_int_new(bits));
        put("bit_offset", pyre_object::w_int_new(bit_offset));
        put("is_bitfield", pyre_object::w_bool_from(true));
    }
    Ok(field)
}

// ── Structure / Union instances ────────────────────────────────────────

fn structure_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "Structure.__new__ requires a type",
        ));
    }
    let mut cls = args[0];
    if unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_abstract_") }.is_some() {
        return Err(pyre_interpreter::PyError::type_error("abstract class"));
    }
    let mut info = stginfo::stginfo_of(cls)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("abstract class"))?;
    let size = pyre_object::with_roots!(cls, info => stginfo::stginfo_size(info));
    pyre_object::with_roots!(cls => stginfo::stginfo_mark_final(info));
    let mut obj = pyre_object::w_instance_new(cls);
    let d = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "ctypes instance has no dict",
        ));
    }
    // The instance `dict` moves and the buffer allocates, so the buffer is
    // built first and the dict word is read back out of a root slot for the
    // store; `obj` does not move and takes one pin for liveness.
    let roots = pyre_object::gc_roots::push_roots();
    let obj = roots.pin_root(obj);
    let d_slot = roots.base() + 1;
    let _ = roots.pin_root(d);
    let buffer = pyre_object::w_bytearray_new(size);
    unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), "_b_", buffer) };
    Ok(obj)
}

/// Field names in base-first order (`init_pos_args`).
fn field_names_base_first(cls: PyObjectRef) -> Vec<String> {
    let mut names = Vec::new();
    for &t in mro_items(cls).iter().rev() {
        if let Some(f) = pyre_interpreter::type_dict_lookup(t, "_fields_")
            && let Ok(entries) = field_entries(f)
        {
            names.extend(entries.into_iter().map(|entry| entry.name));
        }
    }
    names
}

fn structure_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "__init__ requires self",
        ));
    }
    // The argument slice is the caller's own native array, which no root
    // walker updates, so an initialiser that moves is named by its pre-move
    // address once an earlier `setattr_str` has allocated.  Pin every argument
    // and take the movable operands back out of their slots.
    let roots = pyre_object::gc_roots::push_roots();
    let args_slot = roots.base();
    for &arg in args {
        let _ = roots.pin_root(arg);
    }
    let obj = || roots.get(args_slot);
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(&args[1..]);
    let pos_len = pos.len();
    // The keyword carrier, when present, is the trailing argument.
    let kw_slot = kwargs.map(|_| args_slot + args.len() - 1);
    let cls = unsafe { pyre_object::w_instance_get_type(obj()) };
    let names = field_names_base_first(cls);

    if pos_len > names.len() {
        return Err(pyre_interpreter::PyError::type_error(
            "too many initializers",
        ));
    }
    for i in 0..pos_len {
        pyre_interpreter::baseobjspace::setattr_str(
            obj(),
            &names[i],
            roots.get(args_slot + 1 + i),
        )?;
    }

    if let Some(kw_slot) = kw_slot {
        // `w_dict_items` returns a native `Vec` snapshot the collector does not
        // update either, and a string-keyed strategy mints its key objects
        // there, so each pair takes a pin and the values are read back.
        let items = unsafe { pyre_object::w_dict_items(roots.get(kw_slot)) };
        let items_slot = pyre_object::gc_roots::shadow_stack_len();
        for &(key_obj, val) in &items {
            let _ = roots.pin_root(key_obj);
            let _ = roots.pin_root(val);
        }
        for i in 0..items.len() {
            let key_obj = roots.get(items_slot + i * 2);
            if !unsafe { pyre_object::is_str(key_obj) } {
                continue;
            }
            let key = pyre_interpreter::baseobjspace::str_utf8_w(key_obj)?.to_string();
            if key == "__pyre_kw__" {
                continue;
            }
            // Duplicate positional + keyword assignment for the same field.
            if let Some(pos_idx) = names.iter().position(|n| *n == key)
                && pos_idx < pos_len
            {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "duplicate values for field '{key}'"
                )));
            }
            pyre_interpreter::baseobjspace::setattr_str(
                obj(),
                &key,
                roots.get(items_slot + i * 2 + 1),
            )?;
        }
    }
    Ok(pyre_object::w_none())
}

// ── shared metaclass helpers (arrays / pointers) ───────────────────────

/// Read a key from `cls`'s **own** dict (never the MRO).
fn own_dict_get(cls: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    pyre_interpreter::type_dict_lookup(cls, key)
}

/// Store a class attribute directly and invalidate the type cache for it.
fn set_type_attr(cls: PyObjectRef, key: &str, value: PyObjectRef) {
    if pyre_interpreter::type_dict_store(cls, key, value) {
        pyre_object::gc_hook::try_gc_write_barrier(cls as pyre_object::gc_hook::GCREF);
        unsafe { pyre_interpreter::baseobjspace::mutated(cls, key) };
    }
}

/// `cls.__name__`, read the way `type.__name__` reads it.
///
/// An MRO lookup never finds it: the name is a `type` getset over the type
/// object's own field, not an entry in any class dict along the chain.  Every
/// caller here read `?` instead, so `c_int * 3` was named `?_Array_3`.
fn type_name(cls: PyObjectRef) -> String {
    if !unsafe { pyre_object::is_type(cls) } {
        return "?".to_string();
    }
    pyre_interpreter::baseobjspace::str_utf8_w(unsafe { pyre_object::w_type_get_name_obj(cls) })
        .ok()
        .map(str::to_string)
        .unwrap_or_else(|| "?".to_string())
}

// ── PyCArrayType + Array ───────────────────────────────────────────────

/// Reserved key under which an element type caches the array types built over
/// it (`_ctypes._array_type_cache`), an `int`-keyed (by length) dict.  Holding
/// the cache on the element type's own dict keeps the built array types
/// GC-traced through the rooted element type rather than an untraced side table.
const ARRAY_TYPE_CACHE_KEY: &str = "_array_type_cache";

fn carraytype_new(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::builtins::type_descr_new(args)
}

fn carraytype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, array_init_stginfo)
}

/// `PyCArrayType` layout: resolve `_length_` + `_type_` and build the array
/// `StgInfo` (`size = element_size * length`, align = element align).
fn array_init_stginfo(mut cls: PyObjectRef) -> PyResult {
    let length = match own_dict_get(cls, "_length_") {
        Some(v) => {
            if !unsafe { pyre_object::is_int(v) || pyre_object::is_long(v) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "The '_length_' attribute must be an integer",
                ));
            }
            let n = if unsafe { pyre_object::is_long(v) } {
                let big = unsafe { pyre_object::longobject::w_long_get_value(v) };
                if big.get_sign() < 0 {
                    return Err(pyre_interpreter::PyError::value_error(
                        "The '_length_' attribute must not be negative",
                    ));
                }
                return Err(pyre_interpreter::PyError::overflow_error("array too large"));
            } else {
                unsafe { pyre_object::w_int_get_value(v) }
            };
            if n < 0 {
                return Err(pyre_interpreter::PyError::value_error(
                    "The '_length_' attribute must not be negative",
                ));
            }
            n as usize
        }
        None => match first_base_stginfo(cls) {
            Some(bi) => pyre_object::with_roots!(cls => stginfo::stginfo_length(bi)),
            None => {
                return Err(pyre_interpreter::PyError::attribute_error(
                    "class must define a '_length_' attribute",
                ));
            }
        },
    };

    let mut elem = match own_dict_get(cls, "_type_") {
        Some(t) => {
            if !unsafe { pyre_object::is_type(t) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "_type_ must be a type",
                ));
            }
            t
        }
        None => match first_base_stginfo(cls).and_then(stginfo::stginfo_proto) {
            Some(p) => p,
            None => {
                return Err(pyre_interpreter::PyError::attribute_error(
                    "class must define a '_type_' attribute",
                ));
            }
        },
    };

    let elem_size = pyre_object::with_roots!(cls, elem => stginfo::field_size_of(elem))
        .ok_or_else(|| pyre_interpreter::PyError::type_error("_type_ must have storage info"))?;
    let elem_align = pyre_object::with_roots!(cls, elem => stginfo::field_align_of(elem))
        .unwrap_or(1)
        .max(1);
    if elem_size != 0 && (length > usize::MAX / elem_size || elem_size * length > i64::MAX as usize)
    {
        return Err(pyre_interpreter::PyError::overflow_error("array too large"));
    }

    let mut data = StgInfoData::new(elem_size * length, elem_align, ParamFunc::Array);
    data.length = length;
    data.element_size = elem_size;
    data.proto = Some(elem);
    if let Some(ei) = stginfo::stginfo_of(elem)
        && pyre_object::with_roots!(cls, elem => stginfo::stginfo_flags(ei))
            & (stginfo::TYPEFLAG_ISPOINTER | stginfo::TYPEFLAG_HASPOINTER)
            != 0
    {
        data.flags |= stginfo::TYPEFLAG_HASPOINTER;
    }
    let info = pyre_object::with_roots!(cls, elem => stginfo::stginfo_new(data));
    stginfo::stginfo_set(cls, info);

    set_type_attr(cls, "_type_", elem);
    set_type_attr(cls, "_length_", pyre_object::w_int_new(length as i64));

    // Character arrays expose the CPython/PyPy convenience descriptors.
    match cdata::type_code_of(elem).as_deref() {
        Some("c") => install_char_array_getsets(cls),
        Some("u") => install_wchar_array_getsets(cls),
        _ => {}
    }
    Ok(pyre_object::w_none())
}

/// `ctype * n` — cache lookup on `(ctype, n)`, else create `ctype_Array_n`.
fn array_type_from_ctype(elem: PyObjectRef, n: usize) -> PyResult {
    // Both dicts below move.  Storing the cache on `elem`'s dict keeps it
    // reachable but leaves a plain local naming the pre-move address, and the
    // class construction that sits between the cache read and the cache write
    // allocates throughout — so both words live in root slots and are read
    // back at every use.
    let roots = pyre_object::gc_roots::push_roots();
    let elem_slot = roots.base();
    let elem = roots.pin_root(elem);
    let cache_slot = elem_slot + 1;
    let cached = pyre_interpreter::type_dict_lookup(elem, ARRAY_TYPE_CACHE_KEY);
    let _ = roots.pin_root(cached.unwrap_or_else(pyre_object::w_dict_new));
    if cached.is_none()
        && pyre_interpreter::type_dict_store(
            roots.get(elem_slot),
            ARRAY_TYPE_CACHE_KEY,
            roots.get(cache_slot),
        )
    {
        pyre_object::gc_hook::try_gc_write_barrier(
            roots.get(elem_slot) as pyre_object::gc_hook::GCREF
        );
    }
    if let Some(found) = unsafe { pyre_object::w_dict_getitem(roots.get(cache_slot), n as i64) } {
        return Ok(found);
    }
    let name = format!("{}_Array_{}", type_name(roots.get(elem_slot)), n);
    let ns_roots = pyre_object::gc_roots::push_roots();
    let ns_slot = ns_roots.base();
    let _ = ns_roots.pin_root(pyre_object::w_dict_new());
    let w_length = pyre_object::w_int_new(n as i64);
    unsafe {
        pyre_object::w_dict_setitem_str(ns_roots.get(ns_slot), "_type_", roots.get(elem_slot));
        pyre_object::w_dict_setitem_str(ns_roots.get(ns_slot), "_length_", w_length);
    }
    let bases = ns_roots.pin_root(pyre_object::w_tuple_new(vec![array_type()]));
    let w_name = ns_roots.pin_root(pyre_object::w_str_new_managed(&name));
    let new_cls = pyre_interpreter::call::type_call_instantiate(
        pycarraytype_type(),
        &[w_name, bases, ns_roots.get(ns_slot)],
    )?;
    unsafe { pyre_object::w_dict_setitem(roots.get(cache_slot), n as i64, new_cls) };
    Ok(new_cls)
}

/// Resolved array-instance metadata.
struct ArrayMeta {
    length: usize,
    element_size: usize,
    proto: PyObjectRef,
}

fn array_meta(obj: PyObjectRef) -> Result<ArrayMeta, pyre_interpreter::PyError> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let mut info = stginfo::stginfo_of(cls)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("abstract class"))?;
    let mut proto = pyre_object::with_roots!(info => stginfo::stginfo_proto(info))
        .ok_or_else(|| pyre_interpreter::PyError::type_error("abstract class"))?;
    let length = pyre_object::with_roots!(info, proto => stginfo::stginfo_length(info));
    let element_size = pyre_object::with_roots!(proto => stginfo::stginfo_element_size(info));
    Ok(ArrayMeta {
        length,
        element_size,
        proto,
    })
}

fn array_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "Array.__new__ requires a type",
        ));
    }
    let mut cls = args[0];
    let info = stginfo::stginfo_of(cls)
        .filter(|&i| stginfo::stginfo_proto(i).is_some())
        .ok_or_else(|| pyre_interpreter::PyError::type_error("abstract class"))?;
    let size = pyre_object::with_roots!(cls => stginfo::stginfo_size(info));
    let mut obj = pyre_object::w_instance_new(cls);
    let d = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "ctypes instance has no dict",
        ));
    }
    // The instance `dict` moves under the buffer allocation; `obj` does not
    // move and takes one pin for liveness.
    let roots = pyre_object::gc_roots::push_roots();
    let obj = roots.pin_root(obj);
    let d_slot = roots.base() + 1;
    let _ = roots.pin_root(d);
    let buffer = pyre_object::w_bytearray_new(size);
    unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), "_b_", buffer) };
    Ok(obj)
}

fn array_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "__init__ requires self",
        ));
    }
    let obj = args[0];
    let (pos, _kw) = pyre_interpreter::builtins::split_builtin_kwargs(&args[1..]);
    let npos = pos.len();
    let roots = pyre_object::gc_roots::push_roots();
    let pos_base = roots.publish(pos);
    let obj_slot = roots.publish(&[obj]);
    roots.normalize(pos_base, npos + 1);
    let meta = array_meta(roots.get(obj_slot))?;
    if npos > meta.length {
        return Err(pyre_interpreter::PyError::index_error(
            "too many initializers",
        ));
    }
    for i in 0..npos {
        array_set_index(roots.get(obj_slot), &meta, i, roots.get(pos_base + i))?;
    }
    Ok(pyre_object::w_none())
}

fn array_len(args: &[PyObjectRef]) -> PyResult {
    let meta = array_meta(args[0])?;
    Ok(pyre_object::w_int_new(meta.length as i64))
}

fn normalize_index(mut i: i64, length: usize) -> Result<usize, pyre_interpreter::PyError> {
    if i < 0 {
        i += length as i64;
    }
    if i < 0 || i >= length as i64 {
        return Err(pyre_interpreter::PyError::index_error("invalid index"));
    }
    Ok(i as usize)
}

fn array_getitem(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[0];
    let mut key = args[1];
    let meta = pyre_object::with_roots!(key, obj => array_meta(obj))?;
    if unsafe { pyre_object::is_int(key) } {
        let idx = normalize_index(unsafe { pyre_object::w_int_get_value(key) }, meta.length)?;
        return array_get_index(obj, &meta, idx);
    }
    if unsafe { pyre_object::is_slice(key) } {
        return array_get_slice(obj, &meta, key);
    }
    Err(pyre_interpreter::PyError::type_error(
        "indices must be integers",
    ))
}

fn array_get_index(mut obj: PyObjectRef, meta: &ArrayMeta, idx: usize) -> PyResult {
    let offset = idx * meta.element_size;
    match pyre_object::with_roots!(obj => proto_kind(meta.proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(meta.proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("element has no '_type_'"))?;
            let all = cdata::cdata_bytes(obj)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("instance has no buffer"))?;
            let end = (offset + meta.element_size).min(all.len());
            let start = offset.min(end);
            Ok(cdata::decode_slot(&tc, &all[start..end]))
        }
        ParamFunc::Struct | ParamFunc::Union | ParamFunc::Array | ParamFunc::Pointer => Ok(
            cdata::make_indexed_subview(meta.proto, obj, offset, meta.element_size, idx),
        ),
        _ => Err(pyre_interpreter::PyError::type_error(
            "element type has no storage info",
        )),
    }
}

fn array_setitem(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[0];
    let mut key = args[1];
    let mut value = args[2];
    let meta = pyre_object::with_roots!(key, obj, value => array_meta(obj))?;
    if unsafe { pyre_object::is_int(key) } {
        let idx = normalize_index(unsafe { pyre_object::w_int_get_value(key) }, meta.length)?;
        return array_set_index(obj, &meta, idx, value);
    }
    if unsafe { pyre_object::is_slice(key) } {
        return array_set_slice(obj, &meta, key, value);
    }
    Err(pyre_interpreter::PyError::type_error(
        "indices must be integers",
    ))
}

fn array_set_index(
    mut obj: PyObjectRef,
    meta: &ArrayMeta,
    idx: usize,
    mut value: PyObjectRef,
) -> PyResult {
    let offset = idx * meta.element_size;
    match pyre_object::with_roots!(obj, value => proto_kind(meta.proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(meta.proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("element has no '_type_'"))?;
            // `value` is an arbitrary object under `"O"`, and the encode both
            // allocates and can run Python. `obj` stays pinned across that
            // call; later reads use the slot.
            let _roots = pyre_object::gc_roots::push_roots();
            let base = _roots.pin_roots(&[value, obj]);
            let bytes = cdata::encode_instance_or_value(
                &tc,
                _roots.get(base),
                _roots.get(base + 1),
                &idx.to_string(),
            )?;
            cdata::release_bstr_slot(
                &tc,
                cdata::cdata_addr(_roots.get(base + 1)).unwrap_or(0) + offset,
            );
            cdata::cdata_write(_roots.get(base + 1), offset, &bytes);
            let value = _roots.get(base);
            if let Some(keep) = cdata::value_for_keep(&tc, value, &bytes) {
                cdata::keep_ref(_roots.get(base + 1), &idx.to_string(), keep);
            }
            Ok(pyre_object::w_none())
        }
        ParamFunc::Struct | ParamFunc::Union | ParamFunc::Array | ParamFunc::Pointer => {
            if !unsafe { pyre_interpreter::baseobjspace::isinstance_w(value, meta.proto) } {
                return Err(pyre_interpreter::PyError::type_error("incompatible types"));
            }
            // Snapshot the source: `a[i] = a[i]` aliases the destination
            // buffer, and `cdata_write`'s `copy_from_slice` assumes non-overlap.
            let src = pyre_object::with_roots!(obj, value => cdata::cdata_bytes(value))
                .unwrap_or(&[])
                .to_vec();
            let n = meta.element_size.min(src.len());
            pyre_object::with_roots!(obj, value => cdata::cdata_write(obj, offset, &src[..n]));
            let keep = pyre_object::with_roots!(obj => cdata::objects_for_keep(value));
            cdata::keep_ref(obj, &idx.to_string(), keep);
            Ok(pyre_object::w_none())
        }
        _ => Err(pyre_interpreter::PyError::type_error(
            "assignment to this element type is not supported in this slice",
        )),
    }
}

/// The concrete indices a slice selects over `length`, PySlice-adjusted.
fn slice_index_list(
    slice: PyObjectRef,
    length: usize,
) -> Result<Vec<usize>, pyre_interpreter::PyError> {
    let (start, stop, step) = pyre_interpreter::sliceobject::slice_unpack(
        unsafe { pyre_object::w_slice_get_start(slice) },
        unsafe { pyre_object::w_slice_get_stop(slice) },
        unsafe { pyre_object::w_slice_get_step(slice) },
    )?;
    let (start, _, step, slice_length) =
        pyre_interpreter::sliceobject::slice_adjust_indices(start, stop, step, length as i64);
    let mut out = Vec::with_capacity(slice_length as usize);
    let mut i = start;
    for _ in 0..slice_length {
        out.push(i as usize);
        i = i.saturating_add(step);
    }
    Ok(out)
}

fn array_get_slice(mut obj: PyObjectRef, meta: &ArrayMeta, slice: PyObjectRef) -> PyResult {
    let idxs = pyre_object::with_roots!(obj => slice_index_list(slice, meta.length))?;
    // Character arrays slice to bytes/str; other elements slice to a list.
    let tc = cdata::type_code_of(meta.proto);
    if tc.as_deref() == Some("c") {
        let all = cdata::cdata_bytes(obj).unwrap_or(&[]);
        let bytes: Vec<u8> = idxs
            .iter()
            .map(|&i| all.get(i * meta.element_size).copied().unwrap_or(0))
            .collect();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes));
    }
    if tc.as_deref() == Some("u") {
        let mut value = String::new();
        for i in idxs {
            let offset = i * meta.element_size;
            let all = pyre_object::with_roots!(obj => cdata::cdata_bytes(obj)).unwrap_or(&[]);
            let end = (offset + meta.element_size).min(all.len());
            if let host_ctypes::DecodedValue::String(s) =
                host_ctypes::decode_type_code("u", &all[offset.min(end)..end])
            {
                value.push_str(&s);
            }
        }
        return Ok(pyre_object::w_str_new_managed(&value));
    }
    // Each element is freshly boxed and the next index allocates again, so they
    // are pinned as they arrive.
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for i in idxs {
        items.push(pyre_object::with_roots!(obj => array_get_index(obj, meta, i))?);
    }
    Ok(pyre_object::w_list_new(items.take()))
}

fn array_set_slice(
    mut obj: PyObjectRef,
    meta: &ArrayMeta,
    slice: PyObjectRef,
    mut value: PyObjectRef,
) -> PyResult {
    let idxs = pyre_object::with_roots!(obj, value => slice_index_list(slice, meta.length))?;
    // `Array_ass_subscript` reads the right-hand side through
    // `PySequence_Length` and then `PySequence_GetItem` per index, so any
    // sequence will do — an `array.array`, a `range`, a `memoryview`, a class
    // with `__len__` and `__getitem__` — and each element then reaches the
    // same conversion an indexed assignment uses.  A character array needs no
    // case of its own: `b"abc"` indexes to ints, which the `c` converter takes
    // as `chr` values, and a `str` indexes to one-character strings.
    //
    // A length that cannot be read is not reported.  Upstream leaves it at -1,
    // which the size check refuses in place of whatever the failed read raised;
    // pyre reads the length and the item through `__len__` / `__getitem__`
    // where upstream uses the sequence-only pair, so a mapping of the matching
    // size reaches the item read rather than that refusal.
    let other_len =
        pyre_object::with_roots!(obj, value => pyre_interpreter::baseobjspace::len_w(value))
            .unwrap_or(-1);
    if other_len != idxs.len() as i64 {
        return Err(pyre_interpreter::PyError::value_error(
            "Can only assign sequence of same size",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_roots(&[obj, value]);
    let idx_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_none());
    for (i, index) in idxs.into_iter().enumerate() {
        pyre_object::gc_roots::shadow_stack_set(idx_slot, pyre_object::w_int_new(i as i64));
        let item = pyre_interpreter::baseobjspace::getitem(
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(idx_slot),
        )?;
        let item = pyre_object::gc_roots::pin_root(item);
        array_set_index(
            pyre_object::gc_roots::shadow_stack_get(base),
            meta,
            index,
            item,
        )?;
    }
    Ok(pyre_object::w_none())
}

// ── c_char array `.value` / `.raw` ─────────────────────────────────────

fn install_char_array_getsets(cls: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let cls = pyre_object::gc_roots::pin_root(cls);
    let value_get =
        pyre_interpreter::make_builtin_function_with_arity("value", char_array_get_value, 2);
    let value_get = pyre_object::gc_roots::pin_root(value_get);
    let value_set =
        pyre_interpreter::make_builtin_function_with_arity("value", char_array_set_value, 3);
    let value_set = pyre_object::gc_roots::pin_root(value_set);
    let value_del = pyre_interpreter::make_builtin_function_with_arity(
        "value",
        |_args| {
            Err(pyre_interpreter::PyError::type_error(
                "can't delete attribute",
            ))
        },
        2,
    );
    let value_del = pyre_object::gc_roots::pin_root(value_del);
    set_type_attr(
        cls,
        "value",
        pyre_interpreter::typedef::make_getset_property_named(
            value_get, value_set, value_del, "value",
        ),
    );
    let raw_get = pyre_interpreter::make_builtin_function_with_arity("raw", char_array_get_raw, 2);
    let raw_get = pyre_object::gc_roots::pin_root(raw_get);
    let raw_set = pyre_interpreter::make_builtin_function_with_arity("raw", char_array_set_raw, 3);
    let raw_set = pyre_object::gc_roots::pin_root(raw_set);
    let raw_del = pyre_interpreter::make_builtin_function_with_arity(
        "raw",
        |_args| {
            Err(pyre_interpreter::PyError::attribute_error(
                "cannot delete attribute",
            ))
        },
        2,
    );
    let raw_del = pyre_object::gc_roots::pin_root(raw_del);
    set_type_attr(
        cls,
        "raw",
        pyre_interpreter::typedef::make_getset_property_named(raw_get, raw_set, raw_del, "raw"),
    );
}

fn char_array_get_value(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let buf = cdata::cdata_bytes(obj).unwrap_or(&[]);
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        host_ctypes::char_array_field_value(buf),
    ))
}

fn char_array_set_value(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[1];
    let value = args[2];
    if !unsafe { pyre_object::is_bytes(value) } {
        return Err(pyre_interpreter::PyError::type_error("bytes expected"));
    }
    let src = unsafe { pyre_object::bytesobject::w_bytes_data(value) };
    let size = pyre_object::with_roots!(obj => cdata::cdata_len(obj)).unwrap_or(0);
    if src.len() > size {
        return Err(pyre_interpreter::PyError::value_error(
            "byte string too long",
        ));
    }
    // Copy `src`, then a NUL terminator when the buffer has room (no tail zero).
    let mut buf = src.to_vec();
    if src.len() < size {
        buf.push(0);
    }
    cdata::cdata_write(obj, 0, &buf);
    Ok(pyre_object::w_none())
}

fn char_array_get_raw(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let buf = cdata::cdata_bytes(obj).unwrap_or(&[]);
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(buf))
}

fn char_array_set_raw(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[1];
    let value = args[2];
    let source = pyre_interpreter::typedef::buffer_as_bytes_like(value)?
        .ok_or_else(|| pyre_interpreter::PyError::type_error("bytes-like object expected"))?;
    let src = unsafe { pyre_object::bytesobject::bytes_like_data(source) };
    let size = pyre_object::with_roots!(obj => cdata::cdata_len(obj)).unwrap_or(0);
    if src.len() > size {
        return Err(pyre_interpreter::PyError::value_error(
            "byte string too long",
        ));
    }
    cdata::cdata_write(obj, 0, src);
    Ok(pyre_object::w_none())
}

fn install_wchar_array_getsets(cls: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let cls = pyre_object::gc_roots::pin_root(cls);
    let value_get =
        pyre_interpreter::make_builtin_function_with_arity("value", wchar_array_get_value, 2);
    let value_get = pyre_object::gc_roots::pin_root(value_get);
    let value_set =
        pyre_interpreter::make_builtin_function_with_arity("value", wchar_array_set_value, 3);
    let value_set = pyre_object::gc_roots::pin_root(value_set);
    let value_del = pyre_interpreter::make_builtin_function_with_arity(
        "value",
        |_args| {
            Err(pyre_interpreter::PyError::type_error(
                "can't delete attribute",
            ))
        },
        2,
    );
    let value_del = pyre_object::gc_roots::pin_root(value_del);
    set_type_attr(
        cls,
        "value",
        pyre_interpreter::typedef::make_getset_property_named(
            value_get, value_set, value_del, "value",
        ),
    );
}

fn wchar_array_get_value(args: &[PyObjectRef]) -> PyResult {
    Ok(pyre_object::w_str_new_managed(
        &host_ctypes::wstring_from_bytes(cdata::cdata_bytes(args[1]).unwrap_or(&[])),
    ))
}

fn wchar_array_set_value(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[1];
    let value = args[2];
    if !unsafe { pyre_object::is_str(value) } {
        return Err(pyre_interpreter::PyError::type_error(
            "unicode string expected",
        ));
    }
    let text = unsafe { pyre_object::w_str_get_wtf8(value) };
    let size = pyre_object::with_roots!(obj => cdata::cdata_len(obj)).unwrap_or(0);
    // PyCArray_set_value overwrites the characters and one terminator only;
    // bytes after that terminator retain their previous values.
    let mut bytes = pyre_object::with_roots!(obj => cdata::cdata_bytes(obj))
        .unwrap_or(&[])
        .to_vec();
    bytes.resize(size, 0);
    host_ctypes::write_wchar_array_value(&mut bytes, text)
        .map_err(|_| pyre_interpreter::PyError::value_error("string too long"))?;
    cdata::cdata_write(obj, 0, &bytes);
    Ok(pyre_object::w_none())
}

// ── PyCPointerType + _Pointer ──────────────────────────────────────────

fn cpointertype_new(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::builtins::type_descr_new(args)
}

fn cpointertype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, pointer_init_stginfo)
}

/// `PyCPointerType` layout: pointer-sized `StgInfo` with `ISPOINTER`, and
/// memoise the pointer type on the pointed-to type (`POINTER` identity).
fn pointer_init_stginfo(mut cls: PyObjectRef) -> PyResult {
    let proto = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_type_") }
        .filter(|&t| !t.is_null() && unsafe { pyre_object::is_type(t) });
    let has_proto = proto.is_some();
    // `proto` is read again after every layout query below.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[proto.unwrap_or(pyre_object::PY_NULL), cls]);
    let proto_slot = base;
    let cls_slot = base + 1;
    cls = roots.get(cls_slot);
    if has_proto
        && pyre_object::with_roots!(cls => stginfo::field_size_of(roots.get(proto_slot))).is_none()
    {
        return Err(pyre_interpreter::PyError::type_error(
            "_type_ must have storage info",
        ));
    }
    let psize = host_ctypes::pointer_size();
    let mut data = StgInfoData::new(psize, psize, ParamFunc::Pointer);
    data.length = 1;
    data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    data.format = Some(
        if has_proto && {
            cls = roots.get(cls_slot);
            pyre_object::with_roots!(cls => stginfo::field_size_of(roots.get(proto_slot)))
                .unwrap_or(0)
                > 0
        } {
            let shape = {
                let mut dims = Vec::new();
                let mut p = roots.get(proto_slot);
                let mut current = p;
                while let Some(mut info) = stginfo::stginfo_of(current) {
                    cls = roots.get(cls_slot);
                    p = roots.get(proto_slot);
                    if pyre_object::with_roots!(cls, info, p => stginfo::stginfo_paramfunc(info))
                        != ParamFunc::Array
                    {
                        break;
                    }
                    cls = roots.get(cls_slot);
                    p = roots.get(proto_slot);
                    dims.push(
                        pyre_object::with_roots!(cls, info, p => stginfo::stginfo_length(info)),
                    );
                    cls = roots.get(cls_slot);
                    p = roots.get(proto_slot);
                    let Some(next) =
                        pyre_object::with_roots!(cls, p => stginfo::stginfo_proto(info))
                    else {
                        break;
                    };
                    current = next;
                }
                dims
            };
            let prefix = if shape.is_empty() {
                String::new()
            } else {
                format!(
                    "({})",
                    shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            cls = roots.get(cls_slot);
            format!(
                "&{prefix}{}",
                pyre_object::with_roots!(cls => {
                    cdata::ctype_pep3118_format(roots.get(proto_slot), None)
                })
            )
        } else {
            "&B".to_string()
        },
    );
    cls = roots.get(cls_slot);
    let info = pyre_object::with_roots!(cls => {
        data.proto = has_proto.then(|| roots.get(proto_slot));
        stginfo::stginfo_new(data)
    });
    stginfo::stginfo_set(cls, info);

    if has_proto {
        let mut p = roots.get(proto_slot);
        let pinfo = match stginfo::stginfo_of(p) {
            Some(i) => i,
            None => {
                cls = roots.get(cls_slot);
                p = roots.get(proto_slot);
                let size =
                    pyre_object::with_roots!(cls, p => stginfo::field_size_of(p)).unwrap_or(0);
                cls = roots.get(cls_slot);
                p = roots.get(proto_slot);
                let align =
                    pyre_object::with_roots!(cls, p => stginfo::field_align_of(p)).unwrap_or(1);
                cls = roots.get(cls_slot);
                p = roots.get(proto_slot);
                let info = pyre_object::with_roots!(cls, p => stginfo::stginfo_new(StgInfoData::new(
                    size,
                    align,
                    ParamFunc::Simple
                )));
                stginfo::stginfo_set(p, info);
                info
            }
        };
        cls = roots.get(cls_slot);
        stginfo::stginfo_set_pointer_type(pinfo, cls);
    }
    Ok(pyre_object::w_none())
}

fn pointer_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "_Pointer.__new__ requires a type",
        ));
    }
    let cls = args[0];
    if stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .is_none()
    {
        return Err(pyre_interpreter::PyError::type_error(
            "Cannot create instance: has no _type_",
        ));
    }
    let mut obj = pyre_object::w_instance_new(cls);
    let d = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "ctypes instance has no dict",
        ));
    }
    // The instance `dict` moves under the buffer allocation; `obj` does not
    // move and takes one pin for liveness.
    let roots = pyre_object::gc_roots::push_roots();
    let obj = roots.pin_root(obj);
    let d_slot = roots.base() + 1;
    let _ = roots.pin_root(d);
    let psize = host_ctypes::pointer_size();
    let buffer = pyre_object::w_bytearray_new(psize);
    unsafe { pyre_object::w_dict_setitem_str(roots.get(d_slot), "_b_", buffer) };
    Ok(obj)
}

fn pointer_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "__init__ requires self",
        ));
    }
    let obj = args[0];
    let (pos, _kw) = pyre_interpreter::builtins::split_builtin_kwargs(&args[1..]);
    if let Some(&val) = pos.first()
        && !unsafe { pyre_object::is_none(val) }
    {
        pointer_set_contents(obj, val)?;
    }
    Ok(pyre_object::w_none())
}

/// Store `value`'s buffer address in the pointer and keep `value` alive.
fn pointer_set_contents(mut obj: PyObjectRef, mut value: PyObjectRef) -> PyResult {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| {
            pyre_interpreter::PyError::type_error("Cannot create instance: has no _type_")
        })?;
    if !cdata::is_cdata_instance(value)
        || !unsafe { pyre_interpreter::baseobjspace::isinstance_w(value, proto) }
    {
        let proto_name = pyre_object::with_roots!(value => type_name(proto));
        return Err(pyre_interpreter::PyError::type_error(format!(
            "expected {} instead of {}",
            proto_name,
            type_name(unsafe { pyre_object::w_instance_get_type(value) })
        )));
    }
    let addr = pyre_object::with_roots!(obj, value => cdata::cdata_addr(value))
        .ok_or_else(|| pyre_interpreter::PyError::type_error("target has no buffer"))?;
    let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
        "P",
        host_ctypes::SimpleStorageValue::Pointer(addr),
        false,
    );
    pyre_object::with_roots!(obj, value => cdata::cdata_write(obj, 0, &bytes));
    cdata::keep_ref(obj, "1", value);
    Ok(pyre_object::w_none())
}

fn contents_get(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[1];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let mut proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("has no _type_"))?;
    let ptr = host_ctypes::read_pointer_from_buffer(
        pyre_object::with_roots!(obj, proto => cdata::cdata_bytes(obj)).unwrap_or(&[]),
    );
    if ptr == 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "NULL pointer access",
        ));
    }
    let size = pyre_object::with_roots!(obj, proto => stginfo::field_size_of(proto))
        .unwrap_or_else(host_ctypes::pointer_size);
    Ok(cdata::make_at_address(proto, ptr, size, obj))
}

fn contents_set(args: &[PyObjectRef]) -> PyResult {
    pointer_set_contents(args[1], args[2])
}

/// `(proto, element_size, ptr_value)` for a pointer instance.
fn pointer_meta(
    mut obj: PyObjectRef,
) -> Result<(PyObjectRef, usize, usize), pyre_interpreter::PyError> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let mut proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("has no _type_"))?;
    let element_size = pyre_object::with_roots!(obj, proto => stginfo::field_size_of(proto))
        .unwrap_or_else(host_ctypes::pointer_size);
    let ptr = host_ctypes::read_pointer_from_buffer(
        pyre_object::with_roots!(proto => cdata::cdata_bytes(obj)).unwrap_or(&[]),
    );
    Ok((proto, element_size, ptr))
}

fn pointer_getitem(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[0];
    let key = args[1];
    if unsafe { pyre_object::is_slice(key) } {
        let mut start_obj = unsafe { pyre_object::sliceobject::w_slice_get_start(key) };
        let mut stop_obj = unsafe { pyre_object::sliceobject::w_slice_get_stop(key) };
        let step_obj = unsafe { pyre_object::sliceobject::w_slice_get_step(key) };
        if unsafe { pyre_object::is_none(stop_obj) } {
            return Err(pyre_interpreter::PyError::value_error(
                "slice stop is required",
            ));
        }
        let step = if unsafe { pyre_object::is_none(step_obj) } {
            1
        } else {
            pyre_object::with_roots!(obj, start_obj, stop_obj => pyre_interpreter::sliceobject::eval_slice_index(step_obj))?
        };
        if step == 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "slice step cannot be zero",
            ));
        }
        let start = if unsafe { pyre_object::is_none(start_obj) } {
            if step < 0 { -1 } else { 0 }
        } else {
            pyre_object::with_roots!(obj, stop_obj => pyre_interpreter::sliceobject::eval_slice_index(start_obj))?
        };
        let stop = pyre_object::with_roots!(obj => pyre_interpreter::sliceobject::eval_slice_index(stop_obj))?;
        // Each element is freshly allocated and the next index allocates again,
        // so they are pinned as they arrive.
        let mut values = pyre_object::gc_roots::RootedItems::new();
        let mut index = start;
        while if step > 0 { index < stop } else { index > stop } {
            values.push(pyre_object::with_roots!(obj => pointer_get_index(obj, index as isize))?);
            index = index.saturating_add(step);
        }
        return Ok(pyre_object::w_list_new(values.take()));
    }
    if !unsafe { pyre_object::is_int(key) } {
        return Err(pyre_interpreter::PyError::type_error(
            "Pointer indices must be integer",
        ));
    }
    pointer_get_index(obj, unsafe { pyre_object::w_int_get_value(key) } as isize)
}

fn pointer_get_index(mut obj: PyObjectRef, index: isize) -> PyResult {
    let (mut proto, element_size, ptr) = pyre_object::with_roots!(obj => pointer_meta(obj))?;
    if ptr == 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "NULL pointer access",
        ));
    }
    let addr = host_ctypes::pointer_item_address(ptr, index, element_size);
    match pyre_object::with_roots!(obj, proto => proto_kind(proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("element has no '_type_'"))?;
            let bytes = unsafe { host_ctypes::borrow_memory(addr as *const u8, element_size) };
            Ok(cdata::decode_slot(&tc, bytes))
        }
        _ => Ok(cdata::make_at_address(proto, addr, element_size, obj)),
    }
}

fn pointer_setitem(args: &[PyObjectRef]) -> PyResult {
    let mut obj = args[0];
    let mut key = args[1];
    let mut value = args[2];
    if !unsafe { pyre_object::is_int(key) } {
        return Err(pyre_interpreter::PyError::type_error(
            "Pointer indices must be integer",
        ));
    }
    let (mut proto, element_size, ptr) =
        pyre_object::with_roots!(key, obj, value => pointer_meta(obj))?;
    if ptr == 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "NULL pointer access",
        ));
    }
    let index = unsafe { pyre_object::w_int_get_value(key) } as isize;
    let addr = host_ctypes::pointer_item_address(ptr, index, element_size);
    match pyre_object::with_roots!(obj, proto, value => proto_kind(proto)) {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("element has no '_type_'"))?;
            // `value` is an arbitrary object under `"O"`, and the encode both
            // allocates and can run Python. `obj` stays pinned across that
            // call; later reads use the slot.
            let _roots = pyre_object::gc_roots::push_roots();
            let base = _roots.pin_roots(&[value, obj]);
            let bytes = cdata::encode_instance_or_value(
                &tc,
                _roots.get(base),
                _roots.get(base + 1),
                &index.to_string(),
            )?;
            cdata::release_bstr_slot(&tc, addr);
            unsafe { host_ctypes::copy_bytes_to_address(addr, &bytes, element_size) };
            let value = _roots.get(base);
            if let Some(keep) = cdata::value_for_keep(&tc, value, &bytes) {
                cdata::keep_ref(_roots.get(base + 1), &index.to_string(), keep);
            }
            Ok(pyre_object::w_none())
        }
        _ => {
            if !unsafe { pyre_interpreter::baseobjspace::isinstance_w(value, proto) } {
                return Err(pyre_interpreter::PyError::type_error("incompatible types"));
            }
            let src = cdata::cdata_bytes(value).unwrap_or(&[]);
            unsafe { host_ctypes::copy_bytes_to_address(addr, src, element_size) };
            Ok(pyre_object::w_none())
        }
    }
}

fn pointer_bool(args: &[PyObjectRef]) -> PyResult {
    let ptr = host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(args[0]).unwrap_or(&[]));
    Ok(pyre_object::w_bool_from(ptr != 0))
}
