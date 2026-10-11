//! `_ctypes` — the native surface the CPython `ctypes` package sits on.
//!
//! With the `host_env` feature this provides a working end-to-end slice: the
//! dynamic-linker primitives (`dlopen`/`dlsym`/`dlclose` on posix,
//! `LoadLibrary`/`FreeLibrary` on Windows), the scalar data type
//! (`_SimpleCData`, see [`super::cdata`]), the foreign
//! function object (`CFuncPtr`, see [`super::funcptr`]), `sizeof`/`addressof`/
//! `byref`/`alignment`/`resize`, and the import-time constants the package
//! requires.  `Structure`/`Union`/`Array`/`_Pointer`/`CField` are real
//! (see [`super::metaclass`]); the metaclasses compute each type's layout and
//! the buffer-view machinery aliases nested/pointed-to memory.
//!
//! All host/FFI work is delegated to `rustpython_host_env::ctypes`; the module
//! contains no direct `libc::` FFI.  Windows `LoadLibrary` is
//! `host_ctypes::load_library_ex_w`, so the flags reach `LoadLibraryExW`.

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    #[cfg(all(any(unix, windows), feature = "host_env"))]
    register_host_ctypes(ns);
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    register_stub_ctypes(ns);
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Functional surface (host_env)
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn register_host_ctypes(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    use rustpython_host_env::ctypes as host_ctypes;

    // ── dlopen flags ──
    //
    // `ctypes/__init__.py` imports `RTLD_LOCAL`/`RTLD_GLOBAL` before it
    // branches on `os.name`, so both names have to exist wherever the module
    // is real; where there is no `dlfcn.h` they are 0, which is the value
    // `host_env`'s own pair carries.  `RTLD_LAZY`/`RTLD_NOW` have no such
    // caller and stay with the platform that defines them.
    #[cfg(unix)]
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_LOCAL",
            pyre_object::w_int_new(libc::RTLD_LOCAL as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_GLOBAL",
            pyre_object::w_int_new(libc::RTLD_GLOBAL as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_LAZY",
            pyre_object::w_int_new(libc::RTLD_LAZY as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_NOW",
            pyre_object::w_int_new(libc::RTLD_NOW as i64)
        );
    }
    #[cfg(not(unix))]
    {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_LOCAL",
            pyre_object::w_int_new(host_ctypes::RTLD_LOCAL as i64)
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "RTLD_GLOBAL",
            pyre_object::w_int_new(host_ctypes::RTLD_GLOBAL as i64)
        );
    }
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "DEFAULT_MODE",
        pyre_object::w_int_new(host_ctypes::dlopen_mode(None) as i64)
    );

    #[cfg(windows)]
    register_windows_loader(pyre_object::gc_roots::shadow_stack_get(ns_slot));

    let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);

    // ── dlopen(name, mode=DEFAULT_MODE) → integer handle into host libcache ──
    #[cfg(unix)]
    pyre_interpreter::__pyre_store!(
        ns,
        "dlopen",
        pyre_interpreter::make_builtin_function("dlopen", |args| {
            if args.is_empty() {
                return Err(pyre_interpreter::PyError::type_error(
                    "dlopen() missing library name",
                ));
            }
            let w_name = args[0];
            let mut w_mode = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let name = unsafe {
                if pyre_object::is_none(w_name) {
                    // dlopen(None) → process handle
                    let load_flags = if args.len() >= 2 {
                        Some(pyre_interpreter::baseobjspace::int_w(w_mode)? as libc::c_int)
                    } else {
                        None
                    };
                    let mode = host_ctypes::dlopen_mode(load_flags);
                    let ptr = rustpython_host_env::ctypes::dlopen_self(mode).map_err(|e| {
                        pyre_interpreter::PyError::os_error(format!("dlopen(None): {e}"))
                    })?;
                    let h = rustpython_host_env::ctypes::insert_raw_library_handle(ptr);
                    return Ok(pyre_object::w_int_new(h as i64));
                }
                // A library name is a path, so it reaches `dlopen` in the
                // filesystem's own units: a byte with no UTF-8 spelling names
                // a real file and must not be replaced with U+FFFD.
                if pyre_object::is_bytes(w_name) {
                    pyre_interpreter::gateway::os_string_from_fs_bytes(
                        pyre_object::bytesobject::w_bytes_data(w_name),
                    )
                } else if pyre_object::is_str(w_name) {
                    pyre_interpreter::gateway::os_string_from_fs_bytes(&pyre_object::with_roots!(
                        w_mode => pyre_interpreter::gateway::fsencode(w_name)
                    )?)
                } else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "dlopen: name must be a string, bytes or None",
                    ));
                }
            };
            let load_flags = if args.len() >= 2 {
                Some(pyre_interpreter::baseobjspace::int_w(w_mode)? as i32)
            } else {
                None
            };
            let mode = host_ctypes::dlopen_mode(load_flags);
            let h =
                rustpython_host_env::ctypes::open_library_with_mode(&name, mode).map_err(|e| {
                    let mut msg = rustpython_wtf8::Wtf8Buf::from_string("dlopen(".to_string());
                    msg.push_wtf8(&pyre_interpreter::gateway::fsdecode_os_str_wtf8(&name));
                    msg.push_str(&format!("): {}", pyre_interpreter::with_causes(&e)));
                    pyre_interpreter::PyError::os_error(msg)
                })?;
            Ok(pyre_object::w_int_new(h as i64))
        })
    );

    // ── dlsym(handle, name) → address (int) ──
    #[cfg(unix)]
    pyre_interpreter::__pyre_store!(
        ns,
        "dlsym",
        pyre_interpreter::make_builtin_function_with_arity(
            "dlsym",
            |args| {
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "dlsym() needs 2 arguments",
                    ));
                }
                let w_handle = args[0];
                let mut w_name = args[1];
                let h = pyre_object::with_roots!(w_name =>
                    pyre_interpreter::baseobjspace::int_w(w_handle)
                )? as usize;
                let name = unsafe {
                    if !pyre_object::is_str(w_name) {
                        return Err(pyre_interpreter::PyError::type_error(
                            "dlsym: name must be a string",
                        ));
                    }
                    pyre_interpreter::baseobjspace::str_utf8_w(w_name)?.to_string()
                };
                let addr = lookup_symbol(h, name.as_bytes()).map_err(|e| {
                    use rustpython_host_env::ctypes::LookupSymbolError as L;
                    let msg = match e {
                        L::LibraryNotFound => "library not found".to_string(),
                        L::LibraryClosed => "library closed".to_string(),
                        L::Load(s) => s,
                    };
                    pyre_interpreter::PyError::os_error(format!("dlsym({name}): {msg}"))
                })?;
                Ok(pyre_object::w_int_new(addr as i64))
            },
            2,
        )
    );

    // ── dlclose(handle) → None ──
    #[cfg(unix)]
    pyre_interpreter::__pyre_store!(
        ns,
        "dlclose",
        pyre_interpreter::make_builtin_function_with_arity(
            "dlclose",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "dlclose() needs handle",
                    ));
                }
                let h = pyre_interpreter::baseobjspace::int_w(args[0])? as usize;
                rustpython_host_env::ctypes::drop_library(h);
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // ── get_errno / set_errno — routed through host_env's ctypes-local errno ──
    pyre_interpreter::__pyre_store!(
        ns,
        "get_errno",
        pyre_interpreter::make_builtin_function_with_arity(
            "get_errno",
            |_| {
                Ok(pyre_object::w_int_new(
                    rustpython_host_env::ctypes::get_errno() as i64,
                ))
            },
            0,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "set_errno",
        pyre_interpreter::make_builtin_function_with_arity(
            "set_errno",
            |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "set_errno() needs value",
                    ));
                }
                let v = pyre_interpreter::baseobjspace::int_w(args[0])? as i32;
                let prev = rustpython_host_env::ctypes::set_errno(v);
                Ok(pyre_object::w_int_new(prev as i64))
            },
            1,
        )
    );

    // ── sizeof / alignment of raw type codes ('i', 'l', 'd', …) ──
    pyre_interpreter::__pyre_store!(
        ns,
        "_sizeof_typecode",
        pyre_interpreter::make_builtin_function_with_arity(
            "_sizeof_typecode",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "_sizeof_typecode() needs typecode string",
                    ));
                }
                let code = pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string();
                match rustpython_host_env::ctypes::simple_type_size(&code) {
                    Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                    None => Err(pyre_interpreter::PyError::value_error(format!(
                        "unknown type code: {code}"
                    ))),
                }
            },
            1,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "_alignof_typecode",
        pyre_interpreter::make_builtin_function_with_arity(
            "_alignof_typecode",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(pyre_interpreter::PyError::type_error(
                        "_alignof_typecode() needs typecode string",
                    ));
                }
                let code = pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string();
                match rustpython_host_env::ctypes::simple_type_align(&code) {
                    Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                    None => Err(pyre_interpreter::PyError::value_error(format!(
                        "unknown type code: {code}"
                    ))),
                }
            },
            1,
        )
    );

    // ── string_at(ptr, size=-1) → bytes ──
    pyre_interpreter::__pyre_store!(
        ns,
        "string_at",
        pyre_interpreter::make_builtin_function("string_at", |args| {
            if args.is_empty() {
                return Err(pyre_interpreter::PyError::type_error(
                    "string_at() needs ptr",
                ));
            }
            let w_ptr = args[0];
            let mut w_size = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let ptr = pyre_object::with_roots!(w_size =>
                pyre_interpreter::baseobjspace::int_w(w_ptr)
            )? as usize;
            let size = if args.len() >= 2 {
                pyre_interpreter::baseobjspace::int_w(w_size)?
            } else {
                -1
            };
            // The address comes from the caller, so the read faults as
            // readily as any foreign call's does and needs the same fence.
            let read =
                super::seh::guard(|| rustpython_host_env::ctypes::string_at(ptr, size as isize))?;
            let bytes = read.map_err(|e| {
                use rustpython_host_env::ctypes::StringAtError as S;
                let msg = match e {
                    S::NullPointer => "NULL pointer access",
                    S::TooLong => "size too large",
                };
                pyre_interpreter::PyError::os_error(format!("string_at: {msg}"))
            })?;
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
        })
    );

    pyre_interpreter::__pyre_store!(
        ns,
        "SIZEOF_TIME_T",
        pyre_object::w_int_new(rustpython_host_env::ctypes::SIZEOF_TIME_T as i64)
    );

    // The callable `_CData.__reduce__` names.  It has to be reachable by name
    // for a pickle of a ctypes value to load at all, since that is what the
    // stream carries.
    pyre_interpreter::__pyre_store!(ns, "_unpickle", super::cdata::unpickle_function());

    // ── import-time constants ──
    pyre_interpreter::__pyre_store!(ns, "__version__", pyre_object::w_str_new("1.1.0"));
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_CDECL",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_CDECL))
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_PYTHONAPI",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_PYTHONAPI))
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_USE_ERRNO",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_USE_ERRNO))
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_USE_LASTERROR",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_USE_LASTERROR))
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "CTYPES_MAX_ARGCOUNT",
        pyre_object::w_int_new(host_ctypes::CTYPES_MAX_ARGCOUNT as i64)
    );

    #[cfg(target_os = "macos")]
    pyre_interpreter::__pyre_store!(
        ns,
        "_dyld_shared_cache_contains_path",
        pyre_interpreter::make_builtin_function("_dyld_shared_cache_contains_path", |args| {
            let Some(&path) = args.first() else {
                return Ok(pyre_object::w_bool_from(false));
            };
            if unsafe { pyre_object::is_none(path) } {
                return Ok(pyre_object::w_bool_from(false));
            }
            if !unsafe { pyre_object::is_str(path) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "path must be a string",
                ));
            }
            let path = pyre_interpreter::baseobjspace::str_utf8_w(path)?;
            let found = host_ctypes::dyld_shared_cache_contains_path(path)
                .map_err(|_| pyre_interpreter::PyError::value_error("path contains null byte"))?;
            Ok(pyre_object::w_bool_from(found))
        })
    );

    // Real addresses so `memmove`/`memset = CFUNCTYPE(...)(_memmove_addr)`
    // build callable foreign functions; the other three are import-only
    // sentinels (cast/string_at/memoryview PYFUNCTYPE targets are not
    // exercised by this slice).
    pyre_interpreter::__pyre_store!(
        ns,
        "_memmove_addr",
        pyre_object::w_int_new(host_ctypes::memmove_addr() as i64)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "_memset_addr",
        pyre_object::w_int_new(host_ctypes::memset_addr() as i64)
    );
    // RustPython's internal non-FFI targets.  CFuncPtr.__call__ recognizes
    // these values before entering libffi, exactly like function.rs.
    pyre_interpreter::__pyre_store!(ns, "_cast_addr", pyre_object::w_int_new(1));
    pyre_interpreter::__pyre_store!(ns, "_string_at_addr", pyre_object::w_int_new(2));
    pyre_interpreter::__pyre_store!(ns, "_wstring_at_addr", pyre_object::w_int_new(3));
    pyre_interpreter::__pyre_store!(ns, "_memoryview_at_addr", pyre_object::w_int_new(4));

    // ── ArgumentError — a real Exception subclass ──
    let mut w_exception = pyre_interpreter::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _ctypes init");
    let mut argument_error = pyre_object::with_roots!(ns, w_exception => pyre_interpreter::builtins::new_exception_class(
        "ArgumentError",
        pyre_interpreter::builtins::exc_exception_new,
        w_exception,
    ));
    // Both CPython's module exception and PyPy's app-level ArgumentError are
    // mutable heap classes, unlike the immutable native `_ctypes` types.
    let mut argument_error = pyre_object::with_roots!(ns, argument_error => super::finish_cpython_type(
        argument_error,
        "ctypes",
        false,
    ));
    pyre_interpreter::__pyre_store!(ns, "ArgumentError", argument_error);

    // ── aggregate + array/pointer types: real `Structure`/`Union`/`Array`/
    //    `_Pointer`/`CField` ──
    use super::metaclass;
    let mut structure_tp = pyre_object::with_roots!(ns => metaclass::structure_type());
    pyre_interpreter::__pyre_store!(ns, "Structure", structure_tp);
    let mut union_tp = pyre_object::with_roots!(ns => metaclass::union_type());
    pyre_interpreter::__pyre_store!(ns, "Union", union_tp);
    let mut array_tp = pyre_object::with_roots!(ns => metaclass::array_type());
    pyre_interpreter::__pyre_store!(ns, "Array", array_tp);
    let mut pointer_tp = pyre_object::with_roots!(ns => metaclass::pointer_base_type());
    pyre_interpreter::__pyre_store!(ns, "_Pointer", pointer_tp);
    pyre_interpreter::__pyre_store!(ns, "CField", metaclass::cfield_type());

    // ── the functional scalar + foreign-function types ──
    let mut simplecdata_tp = pyre_object::with_roots!(ns => super::cdata::simplecdata_type());
    pyre_interpreter::__pyre_store!(ns, "_SimpleCData", simplecdata_tp);
    let mut cfuncptr_tp = pyre_object::with_roots!(ns => super::funcptr::cfuncptr_type());
    pyre_interpreter::__pyre_store!(ns, "CFuncPtr", cfuncptr_tp);

    // ── sizeof / addressof / byref / alignment / resize ──
    pyre_interpreter::__pyre_store!(
        ns,
        "sizeof",
        pyre_interpreter::make_builtin_function("sizeof", ctypes_sizeof)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "alignment",
        pyre_interpreter::make_builtin_function("alignment", ctypes_alignment)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "addressof",
        pyre_interpreter::make_builtin_function("addressof", ctypes_addressof)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "byref",
        pyre_interpreter::make_builtin_function("byref", ctypes_byref)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "resize",
        pyre_interpreter::make_builtin_function("resize", ctypes_resize)
    );
    // `call_function` / `call_cdeclfunction` are the module's own
    // "so that we can call CFunction instances" pair, described as debugging
    // only and private; the readline extension `_ctypes` names in that comment
    // is why they are still here.  The two differ by the calling convention
    // they pass, which x86-64 has one of.
    for name in ["call_function", "call_cdeclfunction"] {
        pyre_interpreter::__pyre_store!(
            ns,
            name,
            pyre_interpreter::make_builtin_function_with_arity(
                name,
                super::funcptr::call_function,
                2,
            )
        );
    }
}

/// Resolve `symbol` in the library a `_handle` names, for `CFuncPtr((name,
/// dll))` and `in_dll`.
///
/// The two platforms disagree on what a `_handle` is. On posix it is a key
/// into `host_env`'s library cache, which owns the `dlopen` handle and does
/// the `dlsym`. On Windows it is the `HMODULE` `LoadLibrary` returned — that
/// cache cannot take one, and `GetProcAddress` on the module is what
/// `_ctypes.c` does with it anyway.
#[cfg(all(unix, feature = "host_env"))]
pub(super) fn lookup_symbol(
    handle: usize,
    symbol: &[u8],
) -> Result<usize, rustpython_host_env::ctypes::LookupSymbolError> {
    use rustpython_host_env::ctypes::LookupSymbolError as Error;
    let address = rustpython_host_env::ctypes::lookup_function_symbol_addr(handle, symbol)?;
    // `dlsym` reports a miss by returning NULL, and a resolver that itself
    // returns NULL — an IFUNC — leaves `dlerror` unset, so the load succeeds
    // with address 0. `rdynload.dlsym` rejects that: address 0 is not a symbol.
    if address == 0 {
        return Err(Error::Load(format!(
            "symbol '{}' not found",
            String::from_utf8_lossy(symbol)
        )));
    }
    Ok(address)
}

#[cfg(all(windows, feature = "host_env"))]
pub(super) fn lookup_symbol(
    handle: usize,
    symbol: &[u8],
) -> Result<usize, rustpython_host_env::ctypes::LookupSymbolError> {
    use rustpython_host_env::ctypes::LookupSymbolError as Error;
    if handle == 0 {
        return Err(Error::LibraryNotFound);
    }
    // `GetProcAddress` names the export in the module's own narrow spelling,
    // and an embedded NUL would silently truncate the name it looks for.
    let Ok(name) = std::ffi::CString::new(symbol) else {
        return Err(Error::Load("symbol name contains a null byte".to_string()));
    };
    rustpython_host_env::ctypes::get_proc_address(handle as _, &name).ok_or_else(|| {
        Error::Load(format!(
            "symbol '{}' not found",
            String::from_utf8_lossy(symbol)
        ))
    })
}

// ──────────────────────────────────────────────────────────────────────
// Windows surface — `_ctypes.c`'s `#ifdef MS_WIN32` module methods
// ──────────────────────────────────────────────────────────────────────

/// `PyErr_SetFromWindowsErr(GetLastError())` — the code lands in `.winerror`
/// and the errmap picks the `.errno` its subclass comes from.
#[cfg(all(windows, feature = "host_env"))]
fn last_win32_error() -> pyre_interpreter::PyError {
    let code = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    pyre_interpreter::PyError::os_error_win32_syscall2(
        code,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// The names `ctypes/__init__.py` reaches for once `os.name == "nt"`.
///
/// The dynamic loader is the only part of the module that is written twice:
/// `libloading` already spells `LoadLibrary`/`FreeLibrary` for the same
/// `host_env` library cache the posix `dlopen` uses, so what differs is the
/// module surface, not the machinery below it.
#[cfg(all(windows, feature = "host_env"))]
fn register_windows_loader(mut ns: pyre_object::PyObjectRef) {
    use rustpython_host_env::ctypes as host_ctypes;

    // `_ctypes.h`: `STDCALL` is the absence of the `CDECL` bit, and `HRESULT`
    // marks a return value `_check_HRESULT` inspects.  Both sit inside the
    // same `#ifdef MS_WIN32` as the functions below, so neither is defined on
    // the posix side.
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_STDCALL",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_STDCALL))
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "FUNCFLAG_HRESULT",
        pyre_object::w_int_new(i64::from(host_ctypes::FUNCFLAG_HRESULT))
    );

    // ── LoadLibrary(name, load_flags=0) → HMODULE ──
    //
    // `LoadLibraryExW(name, NULL, load_flags)`.  The flags are not optional
    // decoration: `CDLL._load_library` (`ctypes/__init__.py`) defaults
    // `winmode` to `nt._LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` and adds
    // `_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` for a name carrying a separator, so
    // *every* `CDLL` on this platform arrives with a search policy to apply.
    // Dropping it would silently widen the search back to `LoadLibraryW`'s
    // default order, which is what those flags exist to narrow.
    //
    // The handle is the `HMODULE` itself rather than a key into `host_env`'s
    // library cache: that cache's Windows door is `Library::new`, which has no
    // flags parameter, and its raw-handle door is unix-only.  Keeping the
    // module handle is also what `_ctypes.c` stores, so `FreeLibrary` and the
    // symbol lookup in [`lookup_symbol`] are the plain Win32 calls on it.
    pyre_interpreter::__pyre_store!(
        ns,
        "LoadLibrary",
        pyre_interpreter::make_builtin_function("LoadLibrary", |args| {
            let Some(&name) = args.first() else {
                return Err(pyre_interpreter::PyError::type_error(
                    "LoadLibrary() missing library name",
                ));
            };
            // `PyArg_ParseTuple(args, "U|i:LoadLibrary")` — a str, and the
            // path reaches the loader in the filesystem's own units so a
            // surrogate-bearing name round-trips instead of folding to U+FFFD.
            if !unsafe { pyre_object::is_str(name) } {
                return Err(pyre_interpreter::PyError::type_error(
                    "LoadLibrary() argument 1 must be str",
                ));
            }
            let name = pyre_interpreter::gateway::os_string_from_fs_bytes(
                &pyre_interpreter::gateway::fsencode(name)?,
            );
            let load_flags = match args.get(1) {
                Some(&flags) => pyre_interpreter::baseobjspace::int_w(flags)? as u32,
                None => 0,
            };
            let wide = widestring::WideCString::from_os_str(&name)
                .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?;
            let module = match host_ctypes::load_library_ex_w(&wide, load_flags) {
                Ok(module) => module,
                Err(error) => {
                    // ERROR_MOD_NOT_FOUND is answered with a plain
                    // FileNotFoundError naming the module rather than the winerror
                    // OSError every other failure gets, because the DLL that is
                    // missing is as often a dependency as the name asked for.
                    const ERROR_MOD_NOT_FOUND: i32 = 126;
                    if error.raw_os_error() != Some(ERROR_MOD_NOT_FOUND) {
                        return Err(last_win32_error());
                    }
                    let mut msg = rustpython_wtf8::Wtf8Buf::from_string(
                        "Could not find module '".to_string(),
                    );
                    msg.push_wtf8(&pyre_interpreter::gateway::fsdecode_os_str_wtf8(&name));
                    msg.push_str(
                        "' (or one of its dependencies). Try using the full path with \
                         constructor syntax.",
                    );
                    return Err(pyre_interpreter::PyError::new(
                        pyre_interpreter::error::PyErrorKind::FileNotFoundError,
                        msg,
                    ));
                }
            };
            Ok(pyre_object::w_int_new(module as isize as i64))
        })
    );

    // ── FreeLibrary(handle) → None ──
    pyre_interpreter::__pyre_store!(
        ns,
        "FreeLibrary",
        pyre_interpreter::make_builtin_function_with_arity(
            "FreeLibrary",
            |args| {
                let Some(&handle) = args.first() else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "FreeLibrary() needs handle",
                    ));
                };
                let module = pyre_interpreter::baseobjspace::int_w(handle)? as isize;
                host_ctypes::free_library(module as _).map_err(|_| last_win32_error())?;
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // ── FormatError(code=GetLastError()) → str ──
    pyre_interpreter::__pyre_store!(
        ns,
        "FormatError",
        pyre_interpreter::make_builtin_function("FormatError", |args| {
            // `if (code == 0) code = GetLastError();` — an explicit zero is
            // the same request as no argument, not a request to describe
            // ERROR_SUCCESS.
            let code = match args.first() {
                Some(&code) => match pyre_interpreter::baseobjspace::int_w(code)? as u32 {
                    0 => None,
                    code => Some(code),
                },
                None => None,
            };
            // The code the system cannot describe still has to produce a
            // string: `FormatMessageW` failing is not an error here.
            let message = host_ctypes::format_error_message(code)
                .unwrap_or_else(|| "<no description>".to_string());
            Ok(pyre_object::w_str_new_managed(&message))
        })
    );

    // ── get_last_error / set_last_error — the ctypes-local copy, which is
    //    separate from the thread's own Win32 last error.  The setter answers
    //    with the value it replaced, the same contract `set_errno` above
    //    carries and the one the documented signature promises. ──
    pyre_interpreter::__pyre_store!(
        ns,
        "get_last_error",
        pyre_interpreter::make_builtin_function_with_arity(
            "get_last_error",
            |_| Ok(pyre_object::w_int_new(host_ctypes::get_last_error() as i64)),
            0,
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "set_last_error",
        pyre_interpreter::make_builtin_function_with_arity(
            "set_last_error",
            |args| {
                let Some(&value) = args.first() else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "set_last_error() needs value",
                    ));
                };
                let previous = host_ctypes::set_last_error(pyre_interpreter::baseobjspace::int_w(
                    value,
                )? as u32);
                Ok(pyre_object::w_int_new(previous as i64))
            },
            1,
        )
    );

    // ── _check_HRESULT(hr) → hr, or the Win32 error it names ──
    //
    // `HRESULT`'s `_check_retval_`.  `FAILED(hr)` is the sign bit, and the
    // raise is `PyErr_SetFromWindowsErr(hr)` — the code lands in `.winerror`
    // and the errmap picks `.errno`.
    pyre_interpreter::__pyre_store!(
        ns,
        "_check_HRESULT",
        pyre_interpreter::make_builtin_function_with_arity(
            "_check_HRESULT",
            |args| {
                let Some(&hr) = args.first() else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "_check_HRESULT() needs hresult",
                    ));
                };
                let hr = pyre_interpreter::baseobjspace::int_w(hr)? as i32;
                if hr < 0 {
                    return Err(pyre_interpreter::PyError::os_error_win32_syscall2(
                        hr,
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    ));
                }
                Ok(pyre_object::w_int_new(hr as i64))
            },
            1,
        )
    );

    // ── CopyComPointer(src, dst) → HRESULT ──
    //
    // `dst` is a `byref()` carrier, so the destination is the address that
    // carrier already resolved; `src` is a COM interface pointer held in a
    // cdata buffer.  The `AddRef` before the store is what makes this a copy
    // rather than a move.
    pyre_interpreter::__pyre_store!(
        ns,
        "CopyComPointer",
        pyre_interpreter::make_builtin_function_with_arity(
            "CopyComPointer",
            |args| {
                use super::cdata;
                if args.len() < 2 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "CopyComPointer() needs (src, dst)",
                    ));
                }
                let (src, dst) = (args[0], args[1]);
                let destination = if is_carg(dst) { carg_ptr(dst) } else { 0 };
                if destination == 0 {
                    return Ok(pyre_object::w_int_new(
                        host_ctypes::HRESULT_E_POINTER as i64,
                    ));
                }
                let source = if unsafe { pyre_object::is_none(src) } {
                    0
                } else if cdata::is_cdata_instance(src) {
                    let (Some(addr), Some(len)) = (cdata::cdata_addr(src), cdata::cdata_len(src))
                    else {
                        return Ok(pyre_object::w_int_new(
                            host_ctypes::HRESULT_E_POINTER as i64,
                        ));
                    };
                    let buffer = unsafe { host_ctypes::borrow_memory(addr as *const u8, len) };
                    host_ctypes::read_pointer_from_buffer(buffer)
                } else {
                    return Ok(pyre_object::w_int_new(
                        host_ctypes::HRESULT_E_POINTER as i64,
                    ));
                };
                Ok(pyre_object::w_int_new(
                    host_ctypes::copy_com_pointer(source, destination) as i64,
                ))
            },
            2,
        )
    );

    // ── COMError — `args` is the (text, details) tail, not the whole tuple ──
    let mut w_exception = pyre_interpreter::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _ctypes init");
    let mut comerror = pyre_object::with_roots!(ns, w_exception => pyre_interpreter::builtins::make_exc_type_with_init(
        "COMError",
        Some("Raised when a COM method call failed."),
        pyre_interpreter::builtins::exc_exception_new,
        Some(comerror_init),
        w_exception,
    ));
    COMERROR_TYPE_OBJ.set(comerror);
    pyre_interpreter::__pyre_store!(ns, "COMError", comerror);
}

#[cfg(all(windows, feature = "host_env"))]
static COMERROR_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// The `COMError` class, once `_ctypes` has been set up.
#[cfg(all(windows, feature = "host_env"))]
pub(super) fn comerror_type() -> Option<pyre_object::PyObjectRef> {
    COMERROR_TYPE_OBJ.get()
}

/// `comerror_init` — stamps the three slots and re-points `args` at the tail.
#[cfg(all(windows, feature = "host_env"))]
fn comerror_init(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let Some(&w_self) = args.first() else {
        return Err(pyre_interpreter::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(&args[1..]);
    if kwargs.is_some_and(|dict| {
        unsafe { pyre_object::w_dict_str_entries(dict) }
            .iter()
            .any(|(key, _)| key != "__pyre_kw__")
    }) {
        return Err(pyre_interpreter::PyError::type_error(
            "COMError() takes no keyword arguments",
        ));
    }
    let [hresult, text, details] = positional else {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "COMError expected 3 arguments, got {}",
            positional.len()
        )));
    };
    // Every `setattr_str` allocates its key string and may run a Python
    // `__setattr__`, and the argument slice is a plain array no root walker
    // updates, so a `list` or `dict` argument in it goes stale after the first
    // store.  Pin the three before that store and read them back at each use.
    let roots = pyre_object::gc_roots::push_roots();
    let args_slot = roots.base();
    let _ = roots.pin_root(*hresult);
    let _ = roots.pin_root(*text);
    let _ = roots.pin_root(*details);
    pyre_interpreter::baseobjspace::setattr_str(w_self, "hresult", roots.get(args_slot))?;
    pyre_interpreter::baseobjspace::setattr_str(w_self, "text", roots.get(args_slot + 1))?;
    pyre_interpreter::baseobjspace::setattr_str(w_self, "details", roots.get(args_slot + 2))?;
    // `args` carries all three, hresult included, and `str()`/`repr()` show
    // all three with it.
    let w_args = pyre_object::tupleobject::w_tuple_new(vec![
        roots.get(args_slot),
        roots.get(args_slot + 1),
        roots.get(args_slot + 2),
    ]);
    pyre_interpreter::baseobjspace::setattr_str(w_self, "args", w_args)?;
    Ok(pyre_object::w_none())
}

// ── sizeof / alignment (types and instances) ──────────────────────────

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn ctypes_sizeof(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use super::cdata;
    let obj = *args
        .first()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("sizeof() missing argument"))?;
    if unsafe { pyre_object::is_type(obj) } {
        if let Some(size) = cdata::ctype_size_of(obj) {
            return Ok(pyre_object::w_int_new(size as i64));
        }
        return if cdata::type_code_of(obj).is_some() {
            Err(cdata::invalid_type_code_error())
        } else {
            Err(pyre_interpreter::PyError::type_error(
                "this type has no size",
            ))
        };
    }
    if cdata::is_cdata_instance(obj) {
        return Ok(pyre_object::w_int_new(
            cdata::cdata_len(obj).unwrap_or(0) as i64
        ));
    }
    Err(pyre_interpreter::PyError::type_error(
        "this type has no size",
    ))
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn ctypes_alignment(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use super::{cdata, stginfo};
    use rustpython_host_env::ctypes as host_ctypes;
    let obj = *args
        .first()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("alignment() missing argument"))?;
    let target = if unsafe { pyre_object::is_type(obj) } {
        obj
    } else if cdata::is_cdata_instance(obj) {
        unsafe { pyre_object::w_instance_get_type(obj) }
    } else {
        return Err(pyre_interpreter::PyError::type_error("no alignment info"));
    };
    if let Some(info) = stginfo::stginfo_of(target) {
        return Ok(pyre_object::w_int_new(stginfo::stginfo_align(info) as i64));
    }
    match cdata::type_code_of(target) {
        Some(tc) => match host_ctypes::simple_type_align(&tc) {
            Some(a) => Ok(pyre_object::w_int_new(a as i64)),
            None => Err(cdata::invalid_type_code_error()),
        },
        None => Err(pyre_interpreter::PyError::type_error("no alignment info")),
    }
}

// ── addressof / byref / resize (instances) ────────────────────────────

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn ctypes_addressof(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use super::cdata;
    let obj = *args
        .first()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("addressof() missing argument"))?;
    if !cdata::is_cdata_instance(obj) {
        return Err(pyre_interpreter::PyError::type_error("invalid type"));
    }
    let addr = cdata::cdata_addr(obj)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("instance has no buffer"))?;
    Ok(pyre_object::w_int_new(addr as i64))
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn ctypes_byref(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use super::cdata;
    use rustpython_host_env::ctypes as host_ctypes;
    let mut obj = *args
        .first()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("byref() missing argument"))?;
    if !cdata::is_cdata_instance(obj) {
        return Err(pyre_interpreter::PyError::type_error(
            "byref() argument must be a ctypes instance",
        ));
    }
    let offset = if args.len() >= 2 {
        pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::int_w(args[1]))? as isize
    } else {
        0
    };
    let base = pyre_object::with_roots!(obj => cdata::cdata_addr(obj))
        .ok_or_else(|| pyre_interpreter::PyError::type_error("instance has no buffer"))?;
    let addr = host_ctypes::offset_address(base, offset);
    Ok(make_carg(addr, obj))
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn ctypes_resize(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    use super::{cdata, stginfo};
    use rustpython_host_env::ctypes as host_ctypes;
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "resize() needs (obj, size)",
        ));
    }
    if !cdata::is_cdata_instance(args[0]) {
        return Err(pyre_interpreter::PyError::type_error(
            "excepted ctypes instance",
        ));
    }
    // The instance and the requested size stay pinned across the ownership
    // check and the integer conversion.  Both reads come back from those slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    if !cdata::owns_buffer(pyre_object::gc_roots::shadow_stack_get(args_base)) {
        return Err(pyre_interpreter::PyError::value_error(
            "Memory cannot be resized because this object doesn't own it",
        ));
    }
    // The floor is the type's natural size, not the current buffer length, so a
    // previously enlarged object can be shrunk back down to it.  A negative
    // request would wrap to a huge `usize`, so reject the signed value first.
    let requested = pyre_interpreter::baseobjspace::int_w(
        pyre_object::gc_roots::shadow_stack_get(args_base + 1),
    )?;
    let cls = unsafe {
        pyre_object::w_instance_get_type(pyre_object::gc_roots::shadow_stack_get(args_base))
    };
    let min = stginfo::stginfo_of(cls)
        .map(stginfo::stginfo_size)
        .or_else(|| cdata::type_code_of(cls).and_then(|tc| host_ctypes::simple_type_size(&tc)))
        .unwrap_or(0);
    if requested < min as i64 {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "minimum size is {min}"
        )));
    }
    let size = requested as usize;
    if let Some(ba) = cdata::cdata_buffer(pyre_object::gc_roots::shadow_stack_get(args_base)) {
        unsafe {
            let old_size = pyre_object::w_bytearray_len(ba);
            pyre_object::w_bytearray_vec_mut(ba).resize(size, 0);
            pyre_object::bytearrayobject::w_bytearray_sync_alloc(ba, old_size);
        };
    }
    Ok(pyre_object::w_none())
}

// ── byref carrier ──────────────────────────────────────────────────────

#[cfg(all(any(unix, windows), feature = "host_env"))]
static CARG_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// The minimal `byref` carrier type — holds `_ptr` (address) and `_obj`
/// (the referenced instance, kept alive).  Foreign-call consumption of the
/// carrier (the CArgObject P-tag path) is a later slice.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub(super) fn carg_type() -> pyre_object::PyObjectRef {
    CARG_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("CArgObject", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let mut ns = pyre_object::gc_roots::pin_root(ns);
            pyre_interpreter::__pyre_store!(
                ns,
                "__repr__",
                pyre_interpreter::make_builtin_function("__repr__", |args| {
                    let d = pyre_interpreter::baseobjspace::getdict_native(args[0]);
                    let value = unsafe { pyre_object::w_dict_getitem_str(d, "_obj") }
                        .unwrap_or_else(pyre_object::w_none);
                    let rendered = unsafe { pyre_interpreter::display::py_repr_wtf8(value) }?;
                    Ok(pyre_object::w_str_from_wtf8_managed(
                        pyre_interpreter::display::wtf8_format!("<cparam ", rendered, ">"),
                    ))
                })
            );
        });
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        super::finish_cpython_type(tp, "_ctypes", true)
    })
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
pub(super) fn make_carg(
    addr: usize,
    mut obj: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let mut carg = pyre_object::w_instance_new(carg_type());
    let d =
        pyre_object::with_roots!(carg, obj => pyre_interpreter::baseobjspace::getdict_native(carg));
    if !d.is_null() {
        // The instance dictionary moves under a minor collection and both
        // stores allocate — the address value, the key string each store
        // builds, and the dictionary's own storage when it grows — so it is
        // pinned on acquisition and read back for every store.  The address is
        // hoisted out of the call: arguments evaluate left to right, so an
        // inline slot read would precede that allocation and go stale.
        let roots = pyre_object::gc_roots::push_roots();
        let dict_slot = roots.base();
        let _ = roots.pin_root(d);
        let ptr = pyre_object::w_int_new(addr as i64);
        unsafe {
            pyre_object::w_dict_setitem_str(roots.get(dict_slot), "_ptr", ptr);
            pyre_object::w_dict_setitem_str(roots.get(dict_slot), "_obj", obj);
        }
    }
    carg
}

/// Whether `obj` is a `byref()` carrier (consumed by [`super::funcptr`]).
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub(super) fn is_carg(obj: pyre_object::PyObjectRef) -> bool {
    !obj.is_null() && unsafe { pyre_object::w_instance_get_type(obj) } == carg_type()
}

/// The address a `byref()` carrier points at.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub(super) fn carg_ptr(carg: pyre_object::PyObjectRef) -> usize {
    let d = pyre_interpreter::baseobjspace::getdict_native(carg);
    if d.is_null() {
        return 0;
    }
    match unsafe { pyre_object::w_dict_getitem_str(d, "_ptr") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            (unsafe { pyre_object::w_int_get_value(o) }) as usize
        }
        _ => 0,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Stub surface (non-unix or no host_env) — keeps names importable.
// ──────────────────────────────────────────────────────────────────────

#[cfg(not(all(any(unix, windows), feature = "host_env")))]
fn register_stub_ctypes(ns: pyre_object::PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(ns, "ArgumentError", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "_Pointer", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "Structure", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "Union", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "Array", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "CField", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "CFuncPtr", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "_SimpleCData", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "sizeof", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "alignment", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "addressof", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "byref", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "resize", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(ns, "call_function", pyre_interpreter::typedef::w_object());
    pyre_interpreter::__pyre_store!(
        ns,
        "call_cdeclfunction",
        pyre_interpreter::typedef::w_object()
    );
}
