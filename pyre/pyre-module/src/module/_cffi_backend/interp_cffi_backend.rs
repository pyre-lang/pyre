//! `_cffi_backend`'s module surface — PyPy:
//! `pypy/module/_cffi_backend/moduledef.py`.

use super::parse_c_type;

/// `pypy/module/_cffi_backend/__init__.py:VERSION`.  cffi's Python half
/// compares this against its own version and refuses to run on a mismatch, so
/// it tracks the vendored `lib_pypy/cffi` rather than anything of pyre's.
pub const VERSION: &str = "1.18.0.dev0";

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(ns_slot, "__version__", pyre_object::w_str_new(VERSION));

    // `clibffi.FFI_DEFAULT_ABI`.  `FFI_CDECL` is the win32 spelling of the
    // same value and is defined on every platform.
    let abi = default_abi() as i64;
    pyre_interpreter::__pyre_put_new!(ns_slot, "FFI_DEFAULT_ABI", pyre_object::w_int_new(abi));
    pyre_interpreter::__pyre_put_new!(ns_slot, "FFI_CDECL", pyre_object::w_int_new(abi));
    if let Some(stdcall) = super::ctypefunc::stdcall_abi() {
        pyre_interpreter::__pyre_put_new!(ns_slot, "FFI_STDCALL", pyre_object::w_int_new(stdcall));
    }

    register_rtld_constants(pyre_object::gc_roots::shadow_stack_get(ns_slot));

    // `moduledef.py interpleveldefs` — the fixed-arity entry points first.
    // `interp2app` reads each parameter's name off the interp-level function
    // and drops the `w_` prefix, so every one of these binds by keyword too.
    for (name, f, argnames) in [
        (
            "new_primitive_type",
            super::func::new_primitive_type as pyre_interpreter::gateway::BuiltinCodeFn,
            &["name"] as &[&'static str],
        ),
        (
            "new_pointer_type",
            super::func::new_pointer_type,
            &["ctype"],
        ),
        (
            "new_array_type",
            super::func::new_array_type,
            &["ctptr", "length"],
        ),
        ("new_void_type", super::func::new_void_type, &[]),
        ("cast", super::func::cast, &["ctype", "ob"]),
        ("typeof", super::func::typeof_, &["cdata"]),
        ("sizeof", super::func::sizeof, &["obj"]),
        ("alignof", super::func::alignof, &["ctype"]),
        (
            "getcname",
            super::func::getcname,
            &["ctype", "replace_with"],
        ),
        ("unpack", super::func::unpack, &["cdata", "length"]),
        ("release", super::func::release, &["cdata"]),
        ("newp_handle", super::handle::newp_handle, &["ctype", "x"]),
        ("from_handle", super::handle::from_handle, &["cdata"]),
        ("get_errno", super::cerrno::get_errno, &[]),
        ("set_errno", super::cerrno::set_errno, &["errno"]),
        ("_get_types", super::func::get_types, &[]),
        ("_get_common_types", get_common_types, &["dict"]),
        ("new_struct_type", super::func::new_struct_type, &["name"]),
        ("new_union_type", super::func::new_union_type, &["name"]),
        (
            "new_enum_type",
            super::func::new_enum_type,
            &["name", "enumerators", "enumvalues", "basectype"],
        ),
    ] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::gateway::with_module(
                MODULE,
                pyre_interpreter::gateway::make_module_builtin_function_with_arity_and_sig(
                    name,
                    f,
                    argnames.len() as u16,
                    pyre_interpreter::gateway::Signature::new(argnames.to_vec(), None, None, 0, 0),
                ),
            )
        );
    }
    // `newp(ctype, init=None)`, `string(cdata, maxlen=-1)`,
    // `typeoffsetof(ctype, field_or_index, following=0)` carry a default, and
    // the rest of this group binds its own arguments for the same reason.
    for (name, f) in [
        (
            "newp",
            super::func::newp as pyre_interpreter::gateway::BuiltinCodeFn,
        ),
        ("string", super::func::string),
        ("typeoffsetof", super::func::typeoffsetof),
        ("rawaddressof", super::func::rawaddressof),
        (
            "complete_struct_or_union",
            super::func::complete_struct_or_union,
        ),
        ("new_function_type", super::func::new_function_type),
        ("load_library", super::func::load_library),
        ("from_buffer", super::func::from_buffer),
        ("gcp", super::func::gcp),
        ("_offset_in_bytes", super::func::offset_in_bytes),
        ("callback", super::func::callback),
    ] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::gateway::with_module(
                MODULE,
                pyre_interpreter::make_module_builtin_function(name, f),
            )
        );
    }
    // `func.py memmove(dest, src, n)` — a fixed arity, but its parameters are
    // positional-or-keyword upstream, so the registration carries the names.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "memmove",
        pyre_interpreter::gateway::with_module(
            MODULE,
            pyre_interpreter::gateway::make_module_builtin_function_with_arity_and_sig(
                "memmove",
                super::func::memmove as pyre_interpreter::gateway::BuiltinCodeFn,
                3,
                pyre_interpreter::gateway::Signature::new(
                    vec!["dest", "src", "n"],
                    None,
                    None,
                    0,
                    0,
                ),
            ),
        )
    );
    // The types `moduledef.py` publishes.
    pyre_interpreter::__pyre_put_new!(ns_slot, "CType", super::ctypeobj::ctype_type());
    pyre_interpreter::__pyre_put_new!(ns_slot, "_CDataBase", super::cdataobj::cdata_type());
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__CData_iterator",
        super::ctypearray::cdata_iter_type()
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "CField", super::ctypestruct::cfield_type());
    pyre_interpreter::__pyre_put_new!(ns_slot, "CLibrary", super::libraryobj::clibrary_type());
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__FFIAllocator",
        super::allocator::allocator_type()
    );
    pyre_interpreter::__pyre_put_new!(ns_slot, "buffer", super::cbuffer::buffer_type());
    pyre_interpreter::__pyre_put_new!(ns_slot, "FFI", super::ffi_obj::ffi_type_object());
    pyre_interpreter::__pyre_put_new!(ns_slot, "Lib", super::lib_obj::lib_type());
    pyre_interpreter::__pyre_put_new!(ns_slot, "__FFIGlobSupport", super::cglob::glob_type());
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__FFIFunctionWrapper",
        super::wrapper::function_wrapper_type()
    );

    #[cfg(windows)]
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "getwinerror",
        pyre_interpreter::gateway::with_module(
            MODULE,
            pyre_interpreter::make_module_builtin_function(
                "getwinerror",
                super::cerrno::getwinerror,
            ),
        )
    );
    Ok(())
}

const MODULE: &str = "_cffi_backend";

/// `clibffi.FFI_DEFAULT_ABI`.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
))]
pub fn default_abi() -> u32 {
    libffi::raw::ffi_abi_FFI_DEFAULT_ABI
}

#[cfg(not(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
)))]
pub fn default_abi() -> u32 {
    // No libffi on this target, so no foreign call can be made at all; the
    // constant still has to exist because cffi's Python half reads it at
    // import time.
    0
}

/// `moduledef.get_dict_rtld_constants` — the names `rdynload` found, with
/// the four cffi always needs defaulted to 0 where the platform lacks them.
pub(super) fn register_rtld_constants(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ns = pyre_object::gc_roots::pin_root(ns);
    #[cfg(unix)]
    let found: &[(&str, i64)] = &[
        ("RTLD_LAZY", libc::RTLD_LAZY as i64),
        ("RTLD_NOW", libc::RTLD_NOW as i64),
        ("RTLD_GLOBAL", libc::RTLD_GLOBAL as i64),
        ("RTLD_LOCAL", libc::RTLD_LOCAL as i64),
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
        ("RTLD_NODELETE", libc::RTLD_NODELETE as i64),
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
        ("RTLD_NOLOAD", libc::RTLD_NOLOAD as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("RTLD_DEEPBIND", libc::RTLD_DEEPBIND as i64),
    ];
    #[cfg(not(unix))]
    let found: &[(&str, i64)] = &[
        ("RTLD_LAZY", 0),
        ("RTLD_NOW", 0),
        ("RTLD_GLOBAL", 0),
        ("RTLD_LOCAL", 0),
    ];
    for (name, value) in found {
        pyre_interpreter::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(*value));
    }
}

/// `func._get_common_types(dict)` — fill the mapping cffi's parser consults
/// for the type names every platform spells the same way.
fn get_common_types(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let w_dict = args.first().copied().ok_or_else(|| {
        pyre_interpreter::PyError::type_error("_get_common_types() missing dict argument")
    })?;
    let roots = pyre_object::gc_roots::push_roots();
    let _ = roots.pin_root(w_dict);
    let slot = roots.base();
    let mut index = 0;
    while let Some((key, value)) = parse_c_type::enum_common_types(index) {
        let w_value = pyre_object::w_str_new(value);
        // `space.setitem_str` takes the dict fast path only for a real dict and
        // otherwise dispatches `__setitem__`, so a mapping of any other kind
        // reaches its own method instead of the raw accessor.
        if unsafe { pyre_object::pyobject::is_dict(roots.get(slot)) } {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(roots.get(slot), key, w_value);
            }
        } else {
            let w_key = pyre_object::w_str_new(key);
            pyre_interpreter::baseobjspace::setitem(roots.get(slot), w_key, w_value)?;
        }
        index += 1;
    }
    Ok(pyre_object::w_none())
}
