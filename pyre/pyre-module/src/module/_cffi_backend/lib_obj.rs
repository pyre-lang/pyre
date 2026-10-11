//! CFFI generated-library objects — PyPy: `pypy/module/_cffi_backend/lib_obj.py`.

use pyre_interpreter::{PyError, PyErrorKind};
use pyre_object::PyObjectRef;
use std::ffi::CStr;

use super::ffi_obj::W_FFIObject;
use super::{
    cdataobj, cglob, ctypeobj, ffi_obj, libraryobj, parse_c_type, realize_c_type, wrapper,
};

pub const FLAVOR_STATIC: i64 = 0;
pub const FLAVOR_DLOPEN: i64 = 1;

/// `W_LibObject` and its `W_DlOpenLibObject` flavor.
#[pyre_interpreter::pyre_class("_cffi_backend.Lib")]
#[derive(Default)]
pub struct W_LibObject {
    pub w_ffi: PyObjectRef,
    pub dict_w: PyObjectRef,
    pub w_libname: PyObjectRef,
    pub flavor: i64,
    pub libhandle: i64,
    pub autoclose: i64,
}

fn lib_arg(w_lib: PyObjectRef) -> Result<&'static mut W_LibObject, PyError> {
    W_LibObject::from_obj(w_lib).ok_or_else(|| {
        PyError::type_error(format!(
            "expected a Lib object, got '{}'",
            pyre_interpreter::type_methods::arg_type_name(w_lib)
        ))
    })
}

fn ffi_of(lib: &W_LibObject) -> Result<&'static mut W_FFIObject, PyError> {
    W_FFIObject::from_obj(lib.w_ffi).ok_or_else(|| PyError::system_error("Lib object lost its FFI"))
}

fn libname(lib: &W_LibObject) -> Result<&'static str, PyError> {
    pyre_interpreter::baseobjspace::text_w(lib.w_libname)
}

pub fn new_lib(
    w_ffi: PyObjectRef,
    libname: &str,
    flavor: i64,
    libhandle: usize,
    autoclose: bool,
) -> Result<PyObjectRef, PyError> {
    let _ = lib_type();
    let _ = W_FFIObject::from_obj(w_ffi)
        .ok_or_else(|| PyError::type_error("expected an FFI object"))?;
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let dict_slot = ffi_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_module_dict_new());
    let name_slot = dict_slot + 1;
    let _ = roots.pin_root(pyre_object::w_str_new(libname));
    let obj = W_LibObject::allocate_stable(W_LibObject {
        flavor,
        libhandle: libhandle as i64,
        autoclose: i64::from(autoclose),
        ..Default::default()
    });
    let lib = W_LibObject::from_obj(obj).expect("allocate_stable returns this layout");
    lib.w_ffi = roots.get(ffi_slot);
    lib.dict_w = roots.get(dict_slot);
    lib.w_libname = roots.get(name_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    Ok(obj)
}

/// `W_LibObject.make_includes_from`.
pub fn make_includes_from(
    w_lib: PyObjectRef,
    c_includes: *const *const core::ffi::c_char,
) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let includes_slot = lib_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let mut num = 0usize;
    while !unsafe { *c_includes.add(num) }.is_null() {
        let include_name = unsafe { CStr::from_ptr(*c_includes.add(num)) }
            .to_string_lossy()
            .into_owned();
        let part_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new_managed(&include_name));
        let _ = roots.pin_root(pyre_object::w_str_new("ffi"));
        let _ = roots.pin_root(pyre_object::w_str_new("lib"));
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
            roots.get(part_slot + 1),
            roots.get(part_slot + 2),
        ]));
        let module = pyre_interpreter::importing::dunder_import_name_obj(
            roots.get(part_slot),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            roots.get(part_slot + 3),
            0,
        )
        .map_err(|_| {
            let name = lib_arg(roots.get(lib_slot))
                .and_then(|lib| libname(lib))
                .unwrap_or("");
            PyError::new(
                pyre_interpreter::PyErrorKind::ImportError,
                format!("while loading {name}: failed to import ffi, lib from {include_name}"),
            )
        })?;
        let _ = roots.pin_root(module);
        let w_lib1 = pyre_interpreter::baseobjspace::getattr_str(roots.get(part_slot + 4), "lib")?;
        let _ = roots.pin_root(w_lib1);
        let lib1 = lib_arg(roots.get(part_slot + 5))?;
        let _ = roots.pin_root(lib1.w_ffi);
        // `w_list_append` can grow the items block, so the pair is rooted
        // rather than held only in a Rust local across that allocation.
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
            roots.get(part_slot + 6),
            roots.get(part_slot + 5),
        ]));
        unsafe {
            pyre_object::listobject::w_list_append(
                roots.get(includes_slot),
                roots.get(part_slot + 7),
            )
        };
        num += 1;
    }
    ffi_of(lib_arg(roots.get(lib_slot))?)?.included_ffis_libs = roots.get(includes_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        lib_arg(roots.get(lib_slot))?.w_ffi as pyre_object::gc_hook::GCREF,
    );
    Ok(())
}

fn build_cpython_func(
    w_lib: PyObjectRef,
    g: parse_c_type::GlobalS,
    fnname: &str,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let lib = lib_arg(roots.get(lib_slot))?;
    let ffi = ffi_of(lib)?;
    let raw_slot = lib_slot + 1;
    let _ = roots.pin_root(realize_c_type::realize_c_type_or_func(
        lib.w_ffi,
        unsafe { (*ffi.ctxobj).ctx.types },
        parse_c_type::getarg(g.type_op),
    )?);
    let _ = realize_c_type::W_RawFuncType::from_obj(roots.get(raw_slot))
        .ok_or_else(|| PyError::system_error("builtin global is not a raw function type"))?;
    realize_c_type::W_RawFuncType::prepare_nostruct_fnptr(
        roots.get(raw_slot),
        lib_arg(roots.get(lib_slot))?.w_ffi,
    )?;
    assert!(!g.address.is_null());
    let modulename = libname(lib_arg(roots.get(lib_slot))?)?.to_string();
    wrapper::new_function_wrapper(
        lib_arg(roots.get(lib_slot))?.w_ffi,
        g.address as usize,
        g.size_or_direct_fn as usize,
        roots.get(raw_slot),
        fnname,
        &modulename,
    )
}

/// `W_LibObject._get_attr_elidable` — `self.dict_w[attr]`, `KeyError` if
/// not found.
///
/// The miss raises rather than answering `None`: `_build_attr` fills
/// `dict_w` right after it, and an elidable call that raised is never folded
/// to its result. `w_attr` is the attribute's string object, the one word
/// `elidable_promote` guards on.
#[majit_macros::elidable_promote]
fn get_attr_elidable(lib: &W_LibObject, w_attr: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let mut w_attr = w_attr;
    let attr = pyre_object::with_roots!(w_attr => pyre_interpreter::baseobjspace::text_w(w_attr))?;
    match unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(lib.dict_w, attr) } {
        Some(w_value) => Ok(w_value),
        None => Err(PyError::key_error_with_key(w_attr)),
    }
}

/// `W_LibObject._build_attr`.
#[majit_macros::dont_look_inside]
fn build_attr(w_lib: PyObjectRef, w_attr: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let _ = roots.pin_root(w_attr);
    let w_attr = || roots.get(lib_slot + 1);
    let attr = pyre_interpreter::baseobjspace::text_w(w_attr())?.to_string();
    let attr = attr.as_str();
    let ffi_slot = lib_slot + 2;
    let _ = roots.pin_root(lib_arg(roots.get(lib_slot))?.w_ffi);
    let ffi = W_FFIObject::from_obj(roots.get(ffi_slot))
        .ok_or_else(|| PyError::system_error("Lib object lost its FFI"))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    let index = parse_c_type::search_in_globals(ctx, attr);
    let w_result = if index < 0 {
        let included_slot = ffi_slot + 1;
        let _ = roots.pin_root(ffi.included_ffis_libs);
        let included = pyre_interpreter::baseobjspace::fixedview(roots.get(included_slot), -1)?;
        let pair_roots = pyre_object::gc_roots::shadow_stack_len();
        for &pair in &included {
            let _ = roots.pin_root(pair);
        }
        let mut result = None;
        for i in 0..included.len() {
            let pair = pyre_interpreter::baseobjspace::fixedview(roots.get(pair_roots + i), 2)?;
            let base = pyre_object::gc_roots::shadow_stack_len();
            for &item in &pair {
                let _ = roots.pin_root(item);
            }
            if !unsafe { pyre_object::pyobject::is_none(roots.get(base + 1)) } {
                result = match get_attr_elidable(lib_arg(roots.get(base + 1))?, w_attr()) {
                    Ok(w_value) => Some(w_value),
                    Err(e) if e.kind == PyErrorKind::KeyError => {
                        build_attr(roots.get(base + 1), w_attr())?
                    }
                    Err(e) => return Err(e),
                };
            } else {
                result = ffi_obj::fetch_int_constant(roots.get(base), attr)?;
            }
            if result.is_some() {
                break;
            }
        }
        let Some(result) = result else {
            return Ok(None);
        };
        result
    } else {
        let g = unsafe { *ctx.globals.offset(index) };
        let op = parse_c_type::getop(g.type_op);
        match op {
            parse_c_type::OP_CPYTHON_BLTN_V
            | parse_c_type::OP_CPYTHON_BLTN_N
            | parse_c_type::OP_CPYTHON_BLTN_O => build_cpython_func(roots.get(lib_slot), g, attr)?,
            parse_c_type::OP_GLOBAL_VAR => {
                let w_ct = realize_c_type::realize_c_type(
                    roots.get(ffi_slot),
                    ctx.types,
                    parse_c_type::getarg(g.type_op),
                )?;
                let ct = ctypeobj::ctype_arg(w_ct)?;
                let g_size = g.size_or_direct_fn as isize as i64;
                if g_size != ct.size && g_size != 0 && ct.size > 0 {
                    return Err(ffi_obj::ffi_error(format!(
                        "global variable '{attr}' should be {} bytes according to the cdef, but is actually {g_size}",
                        ct.size
                    )));
                }
                let ct_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = roots.pin_root(w_ct);
                let ptr = if g.address.is_null() {
                    cdlopen_fetch(roots.get(lib_slot), attr)?
                } else {
                    g.address.cast()
                };
                cglob::new_glob(attr, roots.get(ct_slot), ptr, std::ptr::null_mut())
            }
            parse_c_type::OP_GLOBAL_VAR_F => {
                let w_ct = realize_c_type::realize_c_type(
                    roots.get(ffi_slot),
                    ctx.types,
                    parse_c_type::getarg(g.type_op),
                )?;
                cglob::new_glob(attr, w_ct, std::ptr::null_mut(), g.address.cast())
            }
            parse_c_type::OP_CONSTANT_INT | parse_c_type::OP_ENUM => {
                realize_c_type::realize_global_int(roots.get(ffi_slot), g, index)?
            }
            parse_c_type::OP_CONSTANT | parse_c_type::OP_DLOPEN_CONST => {
                let w_ct = realize_c_type::realize_c_type(
                    roots.get(ffi_slot),
                    ctx.types,
                    parse_c_type::getarg(g.type_op),
                )?;
                let ct = ctypeobj::ctype_arg(w_ct)?;
                if ct.size <= 0 {
                    return Err(ffi_obj::ffi_error(format!(
                        "constant '{attr}' is of type '{}', whose size is not known",
                        ct.name()
                    )));
                }
                let ptr = if g.address.is_null() {
                    assert_eq!(op, parse_c_type::OP_DLOPEN_CONST);
                    cdlopen_fetch(roots.get(lib_slot), attr)?
                } else {
                    assert_eq!(op, parse_c_type::OP_CONSTANT);
                    let ptr = ffi_obj::allocate_free_mem(roots.get(ffi_slot), ct.size as usize)?;
                    type Fetch = unsafe extern "C" fn(*mut u8);
                    let fetch: Fetch = unsafe { core::mem::transmute(g.address) };
                    unsafe { fetch(ptr) };
                    ptr
                };
                unsafe { ctypeobj::convert_to_object(ct, ptr as usize)? }
            }
            parse_c_type::OP_DLOPEN_FUNC => {
                let ptr = cdlopen_fetch(roots.get(lib_slot), attr)?;
                let w_raw = realize_c_type::realize_c_type_or_func(
                    roots.get(ffi_slot),
                    ctx.types,
                    parse_c_type::getarg(g.type_op),
                )?;
                let raw = realize_c_type::W_RawFuncType::from_obj(w_raw).ok_or_else(|| {
                    PyError::system_error("dlopen function is not a raw function type")
                })?;
                let w_ctfnptr =
                    realize_c_type::W_RawFuncType::unwrap_as_fnptr(w_raw, roots.get(ffi_slot))?;
                let _ = raw;
                cdataobj::new_cdata(ptr as usize, w_ctfnptr)
            }
            _ => {
                return Err(PyError::not_implemented(format!(
                    "in lib_build_attr: op={op}"
                )));
            }
        }
    };
    let result_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_result);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            lib_arg(roots.get(lib_slot))?.dict_w,
            attr,
            roots.get(result_slot),
        )
    };
    Ok(Some(roots.get(result_slot)))
}

fn get_attr(
    w_lib: PyObjectRef,
    w_attr: PyObjectRef,
    is_getattr: bool,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let attr_slot = lib_slot + 1;
    let _ = roots.pin_root(w_attr);
    let attr = pyre_interpreter::baseobjspace::text_w(roots.get(attr_slot))?.to_string();
    // `_get_attr`: `_get_attr_elidable` first, `_build_attr` on KeyError.
    let value = match get_attr_elidable(lib_arg(roots.get(lib_slot))?, roots.get(attr_slot)) {
        Ok(value) => Some(value),
        Err(e) if e.kind == PyErrorKind::KeyError => {
            build_attr(roots.get(lib_slot), roots.get(attr_slot))?
        }
        Err(e) => return Err(e),
    };
    if let Some(value) = value {
        return Ok(value);
    }
    if is_getattr {
        match attr.as_str() {
            "__all__" => return dir1(roots.get(lib_slot), true),
            "__dict__" => return full_dict_copy(roots.get(lib_slot)),
            "__class__" => {
                return pyre_interpreter::typedef::gettypefor(
                    &pyre_object::MODULE_TYPE as *const _,
                )
                .map(|tp| tp.as_ptr())
                .ok_or_else(|| PyError::system_error("module type is not initialized"));
            }
            "__name__" => {
                return Ok(pyre_object::w_str_new_managed(&format!(
                    "{}.lib",
                    libname(lib_arg(roots.get(lib_slot))?)?
                )));
            }
            "__loader__" | "__spec__" => return Ok(pyre_object::w_none()),
            _ => {}
        }
    }
    Err(PyError::attribute_error(format!(
        "cffi library '{}' has no function, constant or global variable named '{attr}'",
        libname(lib_arg(roots.get(lib_slot))?)?
    )))
}

fn lib_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.pin_roots(&[args[0], args[1]]);
    let value = get_attr(roots.get(lib_slot), roots.get(lib_slot + 1), true)?;
    if cglob::W_GlobSupport::from_obj(value).is_some() {
        cglob::read_global_var(value)
    } else {
        Ok(value)
    }
}

fn lib_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &arg in args {
        let _ = roots.pin_root(arg);
    }
    let value = get_attr(roots.get(base), roots.get(base + 1), false)?;
    if cglob::W_GlobSupport::from_obj(value).is_some() {
        cglob::write_global_var(value, roots.get(base + 2))?;
        Ok(pyre_object::w_none())
    } else {
        Err(PyError::attribute_error(format!(
            "cannot write to function or constant '{}'",
            pyre_interpreter::baseobjspace::text_w(roots.get(base + 1))?
        )))
    }
}

fn lib_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let _ = get_attr(args[0], args[1], false)?;
    Err(PyError::attribute_error("C attribute cannot be deleted"))
}

fn lib_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    dir1(args[0], false)
}

fn dir1(w_lib: PyObjectRef, ignore_global_vars: bool) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let result_slot = lib_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let lib = lib_arg(roots.get(lib_slot))?;
    let ffi = ffi_of(lib)?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    for i in 0..ctx.num_globals as isize {
        let g = unsafe { *ctx.globals.offset(i) };
        if ignore_global_vars
            && matches!(
                parse_c_type::getop(g.type_op),
                parse_c_type::OP_GLOBAL_VAR | parse_c_type::OP_GLOBAL_VAR_F
            )
        {
            continue;
        }
        let name = unsafe { CStr::from_ptr(g.name) }.to_string_lossy();
        // The name pin lives across this one append. The result list stays
        // on `roots`.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new_managed(&name));
        unsafe {
            pyre_object::listobject::w_list_append(roots.get(result_slot), roots.get(name_slot))
        };
    }
    Ok(roots.get(result_slot))
}

fn full_dict_copy(w_lib: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let result_slot = lib_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let ffi = ffi_of(lib_arg(roots.get(lib_slot))?)?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    for i in 0..ctx.num_globals as isize {
        let g = unsafe { *ctx.globals.offset(i) };
        let name = unsafe { CStr::from_ptr(g.name) }
            .to_string_lossy()
            .into_owned();
        // The value pin lives across this one setitem. The result dict stays
        // on `roots`.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        let value = get_attr(
            roots.get(lib_slot),
            pyre_object::w_str_new_managed(&name),
            false,
        )?;
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(value);
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(
                roots.get(result_slot),
                &name,
                roots.get(value_slot),
            )
        };
    }
    Ok(roots.get(result_slot))
}

/// `W_LibObject.address_of_func_or_global_var`.
pub fn address_of_func_or_global_var(
    w_lib: PyObjectRef,
    varname: &str,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let lib_slot = roots.base();
    let _ = roots.pin_root(w_lib);
    let name_slot = lib_slot + 1;
    let _ = roots.pin_root(pyre_object::w_str_new_managed(varname));
    let value_slot = lib_slot + 2;
    let _ = roots.pin_root(get_attr(roots.get(lib_slot), roots.get(name_slot), false)?);
    let value = roots.get(value_slot);
    if cglob::W_GlobSupport::from_obj(value).is_some() {
        return cglob::address(value);
    }
    if wrapper::W_FunctionWrapper::from_obj(value).is_some() {
        return wrapper::try_extract_direct_fnptr_as_cdata(value);
    }
    if let Some(cdata) = cdataobj::W_CData::from_obj(value)
        && ctypeobj::ctype_at(cdata.ctype).is_some_and(|ct| ct.kind == ctypeobj::KIND_FUNC)
    {
        return Ok(value);
    }
    Err(PyError::attribute_error(format!(
        "cannot take the address of the constant '{varname}'"
    )))
}

fn cdlopen_fetch(w_lib: PyObjectRef, name: &str) -> Result<*mut u8, PyError> {
    let lib = lib_arg(w_lib)?;
    if lib.flavor != FLAVOR_DLOPEN {
        return Err(ffi_obj::ffi_error(format!(
            "library '{}' has no function or global variable named '{name}'",
            libname(lib)?
        )));
    }
    if lib.libhandle == 0 {
        return Err(ffi_obj::ffi_error(format!(
            "library '{}' has been closed",
            libname(lib)?
        )));
    }
    libraryobj::dlsym(lib.libhandle as usize, name, true)
        .or_else(|| libraryobj::dlsym(lib.libhandle as usize, name, false))
        .map(|address| address as *mut u8)
        .ok_or_else(|| {
            ffi_obj::ffi_error(format!(
                "symbol '{name}' not found in library '{}'",
                libname(lib).unwrap_or("")
            ))
        })
}

/// `W_DlOpenLibObject.cdlopen_close` and the base `W_LibObject` arm.
pub fn cdlopen_close(w_lib: PyObjectRef) -> Result<(), PyError> {
    let lib = lib_arg(w_lib)?;
    if lib.flavor != FLAVOR_DLOPEN {
        return Err(ffi_obj::ffi_error(format!(
            "library '{}' was not created with ffi.dlopen()",
            libname(lib)?
        )));
    }
    let handle = lib.libhandle;
    lib.libhandle = 0;
    if handle == 0 {
        return Ok(());
    }
    unsafe { pyre_object::dictmultiobject::w_dict_clear(lib.dict_w) };
    if lib.autoclose != 0 {
        libraryobj::drop_library(handle as usize);
    }
    Ok(())
}

/// `W_DlOpenLibObject._finalize_`.
pub fn close_lib(lib: &mut W_LibObject) {
    if lib.flavor == FLAVOR_DLOPEN && lib.libhandle != 0 {
        let handle = lib.libhandle;
        lib.libhandle = 0;
        if lib.autoclose != 0 {
            libraryobj::drop_library(handle as usize);
        }
    }
}

/// Sweep-time half of `W_DlOpenLibObject._finalize_`.
///
/// # Safety
/// `obj` must be a GC-dead `W_LibObject`.
pub unsafe fn w_lib_dealloc(obj: PyObjectRef) {
    close_lib(unsafe { &mut *(obj as *mut W_LibObject) });
}

fn lib_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_str_new_managed(&format!(
        "<Lib object for '{}'>",
        libname(lib_arg(args[0])?)?
    )))
}

static LIB_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.Lib`.
pub fn lib_type() -> PyObjectRef {
    LIB_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.Lib",
            init_lib_type,
            pyre_interpreter::typedef::w_object(),
            <W_LibObject as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_LibObject as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            pyre_object::w_type_set_dispatch_own_getattribute(tp);
        }
        tp
    })
}

fn init_lib_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    for (name, function, arity) in [
        (
            "__repr__",
            lib_repr as pyre_interpreter::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("__getattribute__", lib_getattribute, 2),
        ("__setattr__", lib_setattr, 3),
        ("__delattr__", lib_delattr, 2),
        ("__dir__", lib_dir, 1),
    ] {
        store(
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, function, arity),
        );
    }
}
