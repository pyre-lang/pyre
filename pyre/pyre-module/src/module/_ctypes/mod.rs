//! _ctypes module — PyPy: `pypy/module/_rawffi/` plus `lib_pypy/_ctypes/`.
//!
//! `interp_ctypes` holds the module-level surface: library handles (dlopen /
//! dlsym / dlclose), size/align queries, and the raw-memory helpers.  The type
//! machinery is split across the submodules — `stginfo` carries a ctypes type's
//! layout, `metaclass` builds the types and their fields, `cdata` holds the
//! scalar instance buffer, and `funcptr` marshals and performs the foreign
//! call.  The submodules need `host_env`; without it the module is limited to
//! the placeholder surface at the foot of `interp_ctypes`.

pyre_interpreter::pyre_module_init!(interp_ctypes);

/// CPython 3.14 constructs the native `_ctypes` type family from immutable
/// `PyType_Spec`s (`Modules/_ctypes/_ctypes.c:6270-6318` and
/// `callproc.c:carg_spec`).  PyPy implements the same family as ordinary
/// classes rooted at `_CDataMeta`; pyre preserves that common-metaclass owner
/// while projecting CPython's observable heap owner, immutability, and module.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn finish_cpython_type(
    tp: pyre_object::PyObjectRef,
    module: &str,
    immutable: bool,
) -> pyre_object::PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let _ = roots.pin_root(tp);
    let slot = roots.base();
    let w_module = pyre_object::w_str_new(module);
    let tp = roots.get(slot);
    pyre_interpreter::type_dict_store(tp, "__module__", w_module);
    pyre_interpreter::typedef::mark_cpython_heap_type(tp, immutable);
    tp
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod callbacks;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod cdata;
#[cfg(all(windows, feature = "host_env"))]
mod com;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod funcptr;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod metaclass;
#[cfg(all(any(unix, windows), feature = "host_env"))]
mod seh;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod stginfo;

#[cfg(all(test, any(unix, windows), feature = "host_env"))]
mod tests {
    use super::*;

    #[test]
    fn native_type_family_has_common_meta_and_cpython_314_owner() {
        pyre_interpreter::typedef::init_typeobjects();
        const IMMUTABLETYPE: i64 =
            pyre_object::typeobject::TpFlags::PY_TPFLAGS_IMMUTABLETYPE.as_int();
        const HEAPTYPE: i64 = pyre_object::typeobject::TpFlags::PY_TPFLAGS_HEAPTYPE.as_int();
        const MASK: i64 = IMMUTABLETYPE | HEAPTYPE;

        let ctype = metaclass::ctype_type();
        let metas = [
            metaclass::pycsimpletype_type(),
            metaclass::pycstructtype_type(),
            metaclass::pycuniontype_type(),
            metaclass::pycarraytype_type(),
            metaclass::pycpointertype_type(),
            metaclass::pycfuncptrtype_type(),
        ];
        for meta in metas {
            assert_eq!(
                pyre_interpreter::baseobjspace::getattr_str(meta, "__base__").unwrap(),
                ctype
            );
            assert!(pyre_interpreter::baseobjspace::getattr_str(meta, "from_address").is_ok());
        }

        let simple = cdata::simplecdata_type();
        let funcptr = funcptr::cfuncptr_type();
        assert_eq!(
            pyre_interpreter::typedef::r#type(simple).unwrap().as_ptr(),
            metas[0]
        );
        assert_eq!(
            pyre_interpreter::typedef::r#type(funcptr).unwrap().as_ptr(),
            metas[5]
        );

        for (name, ty, module) in [
            ("CType_Type", ctype, "_ctypes"),
            ("PyCSimpleType", metas[0], "_ctypes"),
            ("PyCStructType", metas[1], "_ctypes"),
            ("UnionType", metas[2], "_ctypes"),
            ("PyCArrayType", metas[3], "_ctypes"),
            ("PyCPointerType", metas[4], "_ctypes"),
            ("PyCFuncPtrType", metas[5], "_ctypes"),
            ("_CData", cdata::cdata_type(), "_ctypes"),
            ("_SimpleCData", simple, "_ctypes"),
            ("CFuncPtr", funcptr, "_ctypes"),
            ("Structure", metaclass::structure_type(), "_ctypes"),
            ("Union", metaclass::union_type(), "_ctypes"),
            ("Array", metaclass::array_type(), "_ctypes"),
            ("_Pointer", metaclass::pointer_base_type(), "_ctypes"),
            ("CField", metaclass::cfield_type(), "ctypes"),
            ("CArgObject", interp_ctypes::carg_type(), "_ctypes"),
        ] {
            assert_eq!(
                unsafe { pyre_object::w_type_get_flags(ty) } & MASK,
                MASK,
                "{name}.__flags__"
            );
            let w_module = pyre_interpreter::baseobjspace::getattr_str(ty, "__module__").unwrap();
            assert_eq!(
                unsafe { pyre_object::w_str_get_wtf8(w_module) },
                module,
                "{name}.__module__"
            );
            assert!(
                !unsafe { pyre_object::w_type_is_heaptype(ty) },
                "{name} must keep the PyPy builtin storage owner"
            );
        }
    }
}
