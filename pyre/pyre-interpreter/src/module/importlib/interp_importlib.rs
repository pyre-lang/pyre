//! importlib submodule stubs — PyPy: Lib/importlib/* (app-level).
//!
//! Verbatim move of the four init_importlib_* blocks previously in
//! importing.rs.  Each one is renamed to register_<suffix> to match
//! the moduledef entry points.

/// importlib stub — PyPy uses `pypy/module/_frozen_importlib/` plus
/// `lib-python/3/importlib/`.
/// Avoid loading the real importlib.__init__ since it drags in
/// _bootstrap and _bootstrap_external.
pub fn register_pkg(ns: pyre_object::PyObjectRef) {
    // importlib.import_module(name, package=None) — return an imported
    // module by name. PyPy: Lib/importlib/__init__.py import_module →
    // _bootstrap._gcd_import. We defer to the interpreter's importhook
    // since it handles both builtins and source modules.
    crate::module_ns_store(
        ns,
        "import_module",
        crate::make_builtin_function("import_module", |args| {
            let name = args.first().copied().unwrap_or(pyre_object::w_none());
            unsafe {
                if !pyre_object::is_str(name) {
                    return Err(crate::PyError::type_error(
                        "import_module: name must be str",
                    ));
                }
                // The name stays WTF-8. A lone surrogate has no `&str`
                // spelling; `str_utf8_w` would raise UnicodeEncodeError
                // before the import could miss.
                let name_wtf8 = pyre_object::w_str_get_wtf8(name);
                // `load_source_module` seeds a fresh module namespace from
                // `space.builtin`, so it must be the live EC, not null —
                // pass the thread's current context.
                crate::importing::importhook(
                    name_wtf8,
                    pyre_object::w_none(),
                    pyre_object::w_list_new(vec![pyre_object::w_str_new("*")]),
                    0,
                    crate::call::getexecutioncontext(),
                )
            }
        }),
    );
    crate::module_ns_store(
        ns,
        "invalidate_caches",
        crate::make_builtin_function_with_arity(
            "invalidate_caches",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "reload",
        crate::make_builtin_function_with_arity(
            "reload",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // Mark as a package so dotted imports treat it as such. Point __path__
    // at the on-disk importlib directory so unregistered submodules
    // (importlib._bootstrap / importlib._bootstrap_external) load their real
    // source from there; the builtin machinery/util/abc stubs still win
    // because the full-name builtin check precedes the __path__ disk search.
    #[cfg(feature = "host_env")]
    let path_items = match crate::importing::detect_stdlib_path() {
        Some(dir) => vec![crate::gateway::fsdecode_os_str(
            dir.join("importlib").as_os_str(),
        )],
        None => vec![],
    };
    #[cfg(not(feature = "host_env"))]
    let path_items: Vec<pyre_object::PyObjectRef> = vec![];
    crate::module_ns_store(ns, "__path__", pyre_object::w_list_new(path_items));
}

/// importlib.abc stub — abstract base classes.
pub fn register_abc(ns: pyre_object::PyObjectRef) {
    for name in [
        "Loader",
        "Finder",
        "MetaPathFinder",
        "PathEntryFinder",
        "ResourceLoader",
        "InspectLoader",
        "ExecutionLoader",
        "FileLoader",
        "SourceLoader",
    ] {
        crate::module_ns_store(ns, name, crate::typedef::w_object());
    }
}

/// importlib.machinery stub — provides the names inspect.py references.
/// PyPy ships the real importlib; we shortcut it with a stub so pyre does
/// not have to execute _bootstrap_external.
pub fn register_machinery(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
        ns,
        "SOURCE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".py")]),
    );
    crate::module_ns_store(
        ns,
        "BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::module_ns_store(
        ns,
        "EXTENSION_SUFFIXES",
        pyre_object::w_list_new({
            #[cfg(all(
                feature = "cpyext",
                not(feature = "sandbox"),
                any(target_os = "macos", target_os = "linux")
            ))]
            {
                vec![pyre_object::w_str_new(crate::cpyext::extension_suffix())]
            }
            #[cfg(not(all(
                feature = "cpyext",
                not(feature = "sandbox"),
                any(target_os = "macos", target_os = "linux")
            )))]
            {
                vec![]
            }
        }),
    );
    crate::module_ns_store(
        ns,
        "DEBUG_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::module_ns_store(
        ns,
        "OPTIMIZED_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::module_ns_store(
        ns,
        "all_suffixes",
        crate::make_builtin_function_with_arity(
            "all_suffixes",
            |_| {
                let mut suffixes = vec![
                    pyre_object::w_str_new(".py"),
                    pyre_object::w_str_new(".pyc"),
                ];
                #[cfg(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                ))]
                suffixes.push(pyre_object::w_str_new(crate::cpyext::extension_suffix()));
                Ok(pyre_object::w_list_new(suffixes))
            },
            0,
        ),
    );
    crate::module_ns_store(ns, "ModuleSpec", crate::typedef::w_object());
    crate::module_ns_store(ns, "BuiltinImporter", crate::typedef::w_object());
    crate::module_ns_store(ns, "FrozenImporter", crate::typedef::w_object());
    crate::module_ns_store(ns, "PathFinder", crate::typedef::w_object());
    crate::module_ns_store(ns, "FileFinder", crate::typedef::w_object());
    crate::module_ns_store(ns, "SourceFileLoader", crate::typedef::w_object());
    crate::module_ns_store(ns, "SourcelessFileLoader", crate::typedef::w_object());
    crate::module_ns_store(ns, "ExtensionFileLoader", crate::typedef::w_object());
    crate::module_ns_store(ns, "AppleFrameworkLoader", crate::typedef::w_object());
    crate::module_ns_store(ns, "NamespaceLoader", crate::typedef::w_object());
    crate::module_ns_store(ns, "WindowsRegistryFinder", crate::typedef::w_object());
}
