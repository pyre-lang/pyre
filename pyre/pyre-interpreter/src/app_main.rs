//! `pypy/interpreter/app_main.py` — the startup steps every launcher runs
//! around the program: creating and registering `__main__` with its identity
//! attributes, the `__main__.__loader__` seed, `-X faulthandler`,
//! `import site`, the startup `sys.path[0]` and the warnings bootstrap.  `pyrex` and the `pyre-wasm`
//! guest both call them, so the two entry points start the same way.

use crate::importing;

/// `app_main.py` `run_command_line` — `mainmodule = type(sys)('__main__')`
/// and `sys.modules['__main__'] = mainmodule`.  The module aliases the fresh
/// globals dict, so `__main__.__dict__` and the program's `globals()` are one
/// object.  Returns `(globals, module)`.
#[majit_macros::not_rpython]
pub fn prepare_main_module(
    execution_context: &crate::PyExecutionContext,
) -> (pyre_object::PyObjectRef, pyre_object::PyObjectRef) {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let globals_slot = shadow_stack_len();
    let _ = pin_root(execution_context.fresh_module_globals());
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new("__main__"));
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            shadow_stack_get(globals_slot),
            "__name__",
            shadow_stack_get(name_slot),
        )
    };
    // PyPy `module.py Module` is an ordinary GC object. The managed twin
    // matches that, so detaching `sys.modules` can make the module/dict
    // cycle unreachable. The allocation can collect: the dict is already
    // on `_roots`, and the result is pinned before `set_sys_module`.
    let module_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::module::w_module_new_aliasing_dict_managed(
        shadow_stack_get(name_slot),
        shadow_stack_get(globals_slot),
    ));
    importing::set_sys_module("__main__", shadow_stack_get(module_slot));
    (
        shadow_stack_get(globals_slot),
        shadow_stack_get(module_slot),
    )
}

/// Seed the module-identity attributes every `__main__` namespace carries
/// (`type(sys)('__main__')` leaves them `None`; `runpy` does the `-m` case).
/// Without them a bare `__spec__` / `__package__` / `__doc__` reference falls
/// through to the `builtins` module and returns its values, so
/// `if __spec__ is None` main-detection (multiprocessing spawn, runpy, pytest)
/// inverts.
///
/// A script run by path also gets `__file__` / `__cached__`; the
/// `-c "<string>"` command path does not.  `main_file` is the absolutized
/// path, while `sys.argv[0]` keeps the literal command-line spelling.
#[majit_macros::not_rpython]
pub fn seed_main_module_attrs(
    canonical: pyre_object::PyObjectRef,
    main_module: pyre_object::PyObjectRef,
    main_file: Option<&str>,
) {
    let _seed_roots = pyre_object::gc_roots::push_roots();
    let canonical_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(canonical);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(main_module);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(canonical_slot),
            "__spec__",
            pyre_object::w_none(),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(canonical_slot),
            "__package__",
            pyre_object::w_none(),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(canonical_slot),
            "__doc__",
            pyre_object::w_none(),
        );
    }
    if let Some(file) = main_file {
        let _roots = pyre_object::gc_roots::push_roots();
        let file_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(file));
        let _ = crate::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            "__file__",
            pyre_object::gc_roots::shadow_stack_get(file_slot),
        );
        let _ = crate::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            "__cached__",
            pyre_object::w_none(),
        );
    }
}

/// `add_main_module` / `set_main_loader` — seed `__main__.__loader__`.
///
/// BuiltinImporter is the initial setting for every `__main__`; a source script
/// replaces it with `SourceFileLoader`, and a `.pyc` script with
/// `SourcelessFileLoader`, bound to that file. `-m` is not covered here:
/// `runpy._run_module_as_main` installs the module's own loader. Must run after
/// the importlib bootstrap, which is what supplies all three classes.
#[majit_macros::not_rpython]
pub fn seed_main_loader(
    w_main_globals: pyre_object::PyObjectRef,
    script_file: Option<&str>,
    sourceless: bool,
    ec_ptr: *const crate::PyExecutionContext,
) {
    use crate::baseobjspace::getattr_str;

    // The imports below run module bodies; the globals are read from their
    // slot at each use.
    let roots = pyre_object::gc_roots::push_roots();
    let globals_slot = roots.pin_roots(&[w_main_globals]);
    let load = |module: &str, attr: &str| -> Option<pyre_object::PyObjectRef> {
        importing::importhook(
            rustpython_wtf8::Wtf8::new(module),
            roots.get(globals_slot),
            pyre_object::PY_NULL,
            0,
            ec_ptr,
        )
        .ok()?;
        let w_mod = importing::get_sys_module(module)?;
        getattr_str(w_mod, attr).ok()
    };

    let loader = match script_file {
        Some(path) => load(
            "_frozen_importlib_external",
            if sourceless {
                "SourcelessFileLoader"
            } else {
                "SourceFileLoader"
            },
        )
        .and_then(|ty| {
            let _roots = pyre_object::gc_roots::push_roots();
            let ty_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ty);
            let name_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new("__main__"));
            let path_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(path));
            crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(ty_slot),
                &[
                    pyre_object::gc_roots::shadow_stack_get(name_slot),
                    pyre_object::gc_roots::shadow_stack_get(path_slot),
                ],
            )
            .ok()
        }),
        None => load("_frozen_importlib", "BuiltinImporter"),
    };
    let Some(loader) = loader else {
        return;
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            roots.get(globals_slot),
            "__loader__",
            loader,
        );
    }
}

/// `app_main.py` `run_command_line` — `launch_env` has already assembled the
/// dev-mode,
/// `PYTHONWARNINGS`, `-W` and BytesWarning entries onto `sys.warnoptions`; what
/// remains is driving the warnings machinery through them before user code runs.
/// A module already in `sys.modules` (a `sitecustomize` may have pulled it in)
/// has run its body once, so it is re-driven through `_processoptions`;
/// otherwise the plain import runs `_processoptions(sys.warnoptions)` from the
/// module body (`warnings.py`, its module-level `_processoptions` call).  This
/// is the step that emits
/// "Invalid -W option ignored: ..." and that leaves `sys.modules['warnings']`
/// populated on a `-W` run.
///
/// `run_command_line` writes the two branches under one `try` and catches
/// `ImportError` alone, so a `-W ignore::mod.W` whose `mod` raises anything
/// else leaves the block and ends the process before user code runs.  The
/// spec runs that program: the filters apply as far as they got and the
/// failure is reported on stderr.  So the structure below is that block's,
/// and the non-`ImportError` arm reports rather than either dropping the
/// failure or carrying it out of here.
fn init_warnoptions(
    w_main_globals: pyre_object::PyObjectRef,
    ec_ptr: *const crate::PyExecutionContext,
) {
    if importing::warnoptions().is_empty() {
        return;
    }
    let attempt = (|| -> Result<(), crate::PyError> {
        let Some(w_warnings) = importing::get_sys_module("warnings") else {
            importing::importhook(
                rustpython_wtf8::Wtf8::new("warnings"),
                w_main_globals,
                pyre_object::PY_NULL,
                0,
                ec_ptr,
            )?;
            return Ok(());
        };
        let Some(mut w_sys) = importing::get_interpreter_sys_module() else {
            return Ok(());
        };
        // `from warnings import _processoptions` raises `ImportError` when the
        // name is absent, which the `except` covers; the attribute read that
        // spells it here raises `AttributeError`, which it would not, so that
        // lookup keeps its own swallowing arm.
        let Ok(mut process) = pyre_object::with_roots!(w_sys =>
            crate::baseobjspace::getattr_str(w_warnings, "_processoptions"))
        else {
            return Ok(());
        };
        let options = pyre_object::with_roots!(process =>
            crate::baseobjspace::getattr_str(w_sys, "warnoptions"))?;
        crate::call::call_function_impl_result(process, &[options])?;
        Ok(())
    })();
    if let Err(e) = attempt
        && !is_import_error(&e)
    {
        // Through the same seam the traceback takes, so the two stay ordered
        // on a mediated stderr.
        crate::host_seam::emit_stderr(b"'import warnings' failed; traceback:\n");
        crate::eprint_exception(&e, true);
    }
}

/// The failures that leave the install silent rather than ending startup.
/// `app_main.py run_command_line` swallows the ValueError a descriptor that is
/// no longer open raises -- and only that one -- while the whole block sits
/// behind an `'faulthandler' in sys.builtin_module_names` guard, which is what
/// the other two stand in for: a build with no host seam carries the module but
/// can only answer NotImplementedError, and one with no reachable stdlib cannot
/// import it.  Neither is a failed install; both mean the guard would have kept
/// the block from running at all.
fn faulthandler_init_is_silent(error: &crate::PyError) -> bool {
    matches!(
        error.kind,
        crate::PyErrorKind::ValueError | crate::PyErrorKind::NotImplementedError
    ) || is_import_error(error)
}

/// `app_main.py run_command_line` — `-X faulthandler`, `-X dev` or
/// PYTHONFAULTHANDLER installs the fatal-signal handlers ahead of
/// `import site`, so a crash in anything that runs from here on dumps the
/// Python traceback it was in.
///
/// The descriptor is named rather than left for `enable` to resolve, which is
/// what `faulthandler.enable(2)` does upstream: 2 is the file a fatal dump has
/// to reach whatever the program later does to `sys.stderr`, and passing it
/// also keeps the install from running a Python-level `fileno()` and `flush()`
/// during startup.
///
/// Returns false when the failed install ends startup with status 1.
fn init_faulthandler(
    w_main_globals: pyre_object::PyObjectRef,
    ec_ptr: *const crate::PyExecutionContext,
) -> bool {
    if !importing::faulthandler_flag() {
        return true;
    }
    // `_PyFaulthandler_Init` runs once per process, from `pyinit_core`, ahead
    // of the program `run_command_line` goes on to dispatch, so whatever that
    // program does to the handlers is what a following `-i` inherits.  pyre
    // installs from `import site` instead, which each entry point reaches --
    // one of them per process -- so this holds the installer idempotent at the
    // point that depends on it: a second install would undo a
    // `faulthandler.disable()` the program had just made.
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    let mut first = false;
    INSTALLED.call_once(|| first = true);
    if !first {
        return true;
    }
    let attempt = (|| -> Result<(), crate::PyError> {
        use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

        let module = importing::importhook(
            rustpython_wtf8::Wtf8::new("faulthandler"),
            w_main_globals,
            pyre_object::PY_NULL,
            0,
            ec_ptr,
        )?;
        // Both the module and the descriptor have to survive an allocation the
        // other one drives, so neither is held in a Rust local across it.
        let _roots = push_roots();
        let module_slot = shadow_stack_len();
        let _ = pin_root(module);
        let fd_slot = shadow_stack_len();
        let _ = pin_root(pyre_object::w_int_new(2));
        let enable_slot = shadow_stack_len();
        let _ = pin_root(crate::baseobjspace::getattr_str(
            shadow_stack_get(module_slot),
            "enable",
        )?);
        crate::call::call_function_impl_result(
            shadow_stack_get(enable_slot),
            &[shadow_stack_get(fd_slot)],
        )?;
        Ok(())
    })();
    if let Err(e) = attempt
        && !faulthandler_init_is_silent(&e)
    {
        // `run_command_line` catches the one error it means to ignore and
        // nothing else, so a failed install leaves startup the way an
        // unhandled exception does -- the traceback, and no program run.  The
        // handlers were asked for by name; continuing without them would run
        // the program in exactly the configuration the option refused.
        crate::eprint_exception(&e, true);
        return false;
    }
    true
}

/// app_main.py — unless `-S` (`no_site`) was given, `import site` once
/// `__main__` is registered so the standard `site` initialization runs before
/// user code (sys.path finalization, the `quit`/`exit`/`help` builtins). The
/// import failing is non-fatal (the bare `except`): print "'import site'
/// failed" to stderr and continue.
///
/// `pymain_run_python` then prepends the startup `sys.path[0]`, so that step
/// happens here too — after `site`, whose `removeduppaths()` would otherwise
/// rewrite the `-c` / REPL empty entry into the absolute cwd.
///
/// Returns false when startup ends here with status 1: a failed
/// `-X faulthandler` install, which `run_command_line` does not catch.
#[majit_macros::not_rpython]
pub fn import_site(
    no_site: bool,
    mut w_main_globals: pyre_object::PyObjectRef,
    ec_ptr: *const crate::PyExecutionContext,
) -> bool {
    // `run_command_line` installs it ahead of its own `import site`, so a
    // crash inside `site` itself is already covered.
    // `init_faulthandler` and `import site` run module bodies; the globals
    // are read from their slot at each use.
    let roots = pyre_object::gc_roots::push_roots();
    let globals_slot = roots.pin_roots(&[w_main_globals]);
    if !init_faulthandler(roots.get(globals_slot), ec_ptr) {
        return false;
    }
    // Through the `builtins.__import__` binding, the way the `import site`
    // statement this stands in for reaches the importer -- a replacement of
    // that binding sees this import.  Behind it, `dunder_import` hands the
    // name to the installed `_bootstrap.__import__`.  The native `importhook`
    // this called before resolves a name by its own filesystem search, so
    // reaching `site` through it consults no `sys.path_hooks` entry and parses
    // the source instead of reading the module's bytecode cache, leaving
    // `__cached__` as `None`.
    if !no_site
        && importing::call_dunder_import(
            "site",
            roots.get(globals_slot),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            0,
            ec_ptr,
        )
        .is_err()
    {
        crate::host_seam::emit_stderr(b"'import site' failed\n");
    }
    let mut w_main_globals = roots.get(globals_slot);
    pyre_object::with_roots!(w_main_globals => importing::add_sys_path_0());
    // The warnings bootstrap sits outside the `no_site` guard in `app_main.py`,
    // so `-S -Wxxx` still reports a bad filter.
    init_warnoptions(roots.get(globals_slot), ec_ptr);
    // `pylifecycle.c`: apply `PyConfig.lazy_imports == 1` after site.
    crate::lazyimport::apply_lazy_imports_mode_after_site();
    true
}

/// `app_main.py:1057` treats only `ImportError` as "this hook does not handle
/// the path" and lets anything else propagate.  `zipimport.ZipImportError` is
/// a subclass, so a hook that rejects an ordinary file lands here too.  The
/// kind test comes first because a natively-raised error carries its class in
/// `kind` and may have no exception object yet.
pub fn is_import_error(error: &crate::PyError) -> bool {
    matches!(
        error.kind,
        crate::PyErrorKind::ImportError | crate::PyErrorKind::ModuleNotFoundError
    ) || raised_is_instance_of(error, "ImportError")
}

/// Whether the raised exception is an instance of the named builtin exception
/// class or of a subclass of it.
pub fn raised_is_instance_of(error: &crate::PyError, class_name: &str) -> bool {
    let exc = error.exc_object;
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return false;
    }
    let Some(raised_type) = crate::typedef::r#type(exc) else {
        return false;
    };
    crate::builtins::lookup_exc_class(class_name).is_some_and(|target| unsafe {
        crate::baseobjspace::exception_issubclass_w(raised_type.as_ptr(), target)
    })
}
