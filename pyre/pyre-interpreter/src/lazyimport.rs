//! PEP 810 lazy imports (`lazyimportobject.c`, `ceval.c` `_PyEval_LazyImportName`,
//! `import.c` `_PyImport_LazyImportModuleLevelObject`).

use pyre_object::*;
use rustpython_compiler::core::{PositionEncoding, SourceFileBuilder};

use crate::compile::codegen::error::{CodegenError, CodegenErrorType};
use crate::compile::{CompileError, Mode};
use crate::pyframe::PyFrame;
use crate::typedef::make_new_descr;

const LAZY_NORMAL: i32 = 0;
const LAZY_ALL: i32 = 1;

/// `executioncontext.py ExecutionContext.gettopframe`.
fn current_frame() -> *mut PyFrame {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec).gettopframe() }
    }
}

fn str_value(obj: PyObjectRef) -> Option<&'static str> {
    if obj.is_null() || unsafe { is_none(obj) } {
        None
    } else {
        unsafe { w_str_get_value_opt(obj) }
    }
}

pub fn lazy_import_new(
    w_builtins: PyObjectRef,
    w_from: PyObjectRef,
    w_attr: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    lazy_import_new_ex(w_builtins, w_from, w_attr, true)
}

/// `_PyLazyImport_New` with `frame = NULL` (`lz_code` null, `lz_instr_offset` -1).
fn lazy_import_new_no_frame(
    w_builtins: PyObjectRef,
    w_from: PyObjectRef,
    w_attr: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    lazy_import_new_ex(w_builtins, w_from, w_attr, false)
}

fn lazy_import_new_ex(
    w_builtins: PyObjectRef,
    w_from: PyObjectRef,
    w_attr: PyObjectRef,
    capture_frame: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if w_from.is_null() || unsafe { !(is_str(w_from) || is_lazy_import(w_from)) } {
        return Err(crate::PyError::type_error(
            "expected str or lazy_import for name",
        ));
    }
    let w_attr = if w_attr.is_null() || unsafe { is_none(w_attr) } {
        PY_NULL
    } else if unsafe { is_str(w_attr) || is_tuple(w_attr) } {
        w_attr
    } else {
        return Err(crate::PyError::type_error(
            "lazy_import: fromlist must be None, a string, or a tuple",
        ));
    };
    // `lazyimportobject.c _PyLazyImport_New`: store the current frame's code
    // and LASTI. `PyFrame.last_instr` is an instruction-unit index (same unit
    // as LASTI); `w_code_addr2line` takes a byte offset, so callers multiply
    // by 2. `LoadChild` passes `frame = NULL`.
    let (w_code, instr_offset) = if !capture_frame {
        (PY_NULL, -1)
    } else {
        let frame = current_frame();
        if frame.is_null() {
            (PY_NULL, -1)
        } else {
            (
                unsafe { (*frame).getcode_w() },
                unsafe { (*frame).last_instr } as i64,
            )
        }
    };
    Ok(w_lazy_import_new(
        w_builtins,
        w_from,
        w_attr,
        w_code,
        instr_offset,
    ))
}

/// `_PyEval_LazyImportFrom`
pub fn lazy_import_from(mut v: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let v_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(v);
    v = pyre_object::gc_roots::shadow_stack_get(v_slot);
    let w_attr = unsafe { (*w_lazy_import_cast(v)).w_attr };
    if !w_attr.is_null()
        && unsafe { is_tuple(w_attr) && w_tuple_len(w_attr) > 0 }
        && let Some(attr) = {
            let from = unsafe { (*w_lazy_import_cast(v)).w_from };
            lazy_import_get_loaded_attr(from, name)?
        }
    {
        return Ok(attr);
    }
    v = pyre_object::gc_roots::shadow_stack_get(v_slot);
    let w_name = w_str_new(name);
    v = pyre_object::gc_roots::shadow_stack_get(v_slot);
    lazy_import_new(PY_NULL, v, w_name)
}

fn lazy_import_get_loaded_attr(
    name: PyObjectRef,
    attr_name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let Some(name_s) = str_value(name) else {
        return Ok(None);
    };
    let Some(module) = crate::importing::sys_modules_entry(name_s) else {
        return Ok(None);
    };
    if unsafe { !is_module(module) } {
        return Ok(None);
    }
    let dict = unsafe { w_module_get_w_dict(module) };
    if dict.is_null() {
        return Ok(None);
    }
    match crate::baseobjspace::finditem_str(dict, attr_name) {
        Ok(Some(attr)) if unsafe { is_lazy_import(attr) } => Ok(None),
        Ok(other) => Ok(other),
        Err(_) => Ok(None),
    }
}

fn lazy_import_path(mut m: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let m_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(m);
    m = pyre_object::gc_roots::shadow_stack_get(m_slot);
    let from = unsafe { (*w_lazy_import_cast(m)).w_from };
    if unsafe { is_lazy_import(from) } {
        let base = lazy_import_path(from)?;
        m = pyre_object::gc_roots::shadow_stack_get(m_slot);
        let attr = unsafe { (*w_lazy_import_cast(m)).w_attr };
        let attr_s = str_value(attr).unwrap_or("").to_owned();
        let base_s = str_value(base).unwrap_or("").to_owned();
        return Ok(w_str_new(&format!("{base_s}.{attr_s}")));
    }
    let attr = unsafe { (*w_lazy_import_cast(m)).w_attr };
    if !attr.is_null() && unsafe { !is_tuple(attr) || w_tuple_len(attr) > 0 } {
        return Ok(unsafe { (*w_lazy_import_cast(m)).w_from });
    }
    let from = unsafe { (*w_lazy_import_cast(m)).w_from };
    let from_s = str_value(from).unwrap_or("").to_owned();
    match from_s.find('.') {
        Some(dot) => Ok(w_str_new(&from_s[..dot])),
        None => Ok(from),
    }
}

fn lazy_import_name(m: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let lz = unsafe { &*w_lazy_import_cast(m) };
    if unsafe { is_lazy_import(lz.w_from) } {
        return lazy_import_path(m);
    }
    if !lz.w_attr.is_null() && unsafe { !is_tuple(lz.w_attr) || w_tuple_len(lz.w_attr) > 0 } {
        let from = str_value(lz.w_from).unwrap_or("");
        return Ok(w_str_new(&format!("{from}...")));
    }
    Ok(lz.w_from)
}

fn lazy_import_repr_impl(op: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let name = lazy_import_name(op)?;
    let name_s = str_value(name).unwrap_or("");
    Ok(w_str_new(&format!("<lazy_import '{name_s}'>")))
}

fn is_resolving(op: PyObjectRef) -> bool {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return false;
    }
    unsafe {
        (*ec)
            .lazy_imports_resolving()
            .borrow()
            .iter()
            .any(|&p| std::ptr::eq(p, op))
    }
}

fn resolving_push(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if !ec.is_null() {
        unsafe { (*ec).lazy_imports_push(op) };
    }
}

fn resolving_pop(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return;
    }
    unsafe { (*ec).lazy_imports_pop(op) };
}

fn import_cycle_error(name: PyObjectRef, root_from: PyObjectRef) -> crate::PyError {
    let shown = str_value(name).unwrap_or("");
    let msg = format!("cannot import name '{shown}' (most likely due to a circular import)");
    let w_type = crate::builtins::lookup_exc_class("ImportCycleError")
        .unwrap_or_else(|| crate::typedef::gettypeobject(&pyre_object::EXC_IMPORT_ERROR_TYPE));
    let _ = root_from;
    crate::PyError::from_type_and_value(w_type, w_str_new(&msg))
}

fn root_and_first(lz: PyObjectRef) -> (PyObjectRef, PyObjectRef) {
    let mut root = lz;
    let mut first = PY_NULL;
    loop {
        let from = unsafe { (*w_lazy_import_cast(root)).w_from };
        if unsafe { !is_lazy_import(from) } {
            break;
        }
        first = root;
        root = from;
    }
    (root, first)
}

fn replay_from(mod_: PyObjectRef, mut lz: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let from = unsafe { (*w_lazy_import_cast(lz)).w_from };
    if unsafe { !is_lazy_import(from) } {
        return Ok(mod_);
    }
    let parent = pyre_object::with_roots!(lz => replay_from(mod_, from))?;
    let attr = unsafe { (*w_lazy_import_cast(lz)).w_attr };
    let name = str_value(attr).unwrap_or("").to_owned();
    let obj = crate::importing::import_from(parent, &name)?;
    resolve_result(obj)
}

fn resolve_result(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || unsafe { !is_lazy_import(obj) } {
        Ok(obj)
    } else {
        load_lazy_import(obj)
    }
}

/// `_PyImport_LoadLazyImportTstate` / `lazy_import_resolve_impl`
#[majit_macros::dont_look_inside]
pub fn load_lazy_import(lazy_import: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    load_lazy_import_impl(lazy_import).map(|(value, _imported)| value)
}

/// `lazyimportobject.c lazy_import_add_exception_cause`
fn lazy_import_add_exception_cause(lz: PyObjectRef, mut err: crate::PyError) -> crate::PyError {
    let roots = pyre_object::gc_roots::push_roots();
    let lz_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lz);
    let err_slot = err.pin(&roots);
    let lz = pyre_object::gc_roots::shadow_stack_get(lz_slot);
    let w_code = unsafe { (*w_lazy_import_cast(lz)).w_code };
    let instr_offset = unsafe { (*w_lazy_import_cast(lz)).instr_offset };
    if w_code.is_null() || instr_offset < 0 {
        err.reload(&roots, err_slot);
        return err;
    }
    let code_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_code);
    err.reload(&roots, err_slot);
    let exc = err.to_exc_object();
    err.reload(&roots, err_slot);
    if exc.is_null() || unsafe { !is_exception(exc) } {
        return err;
    }
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let lz = pyre_object::gc_roots::shadow_stack_get(lz_slot);
    let name = match lazy_import_name(lz) {
        Ok(n) => n,
        Err(_) => {
            err.reload(&roots, err_slot);
            return err;
        }
    };
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    err.reload(&roots, err_slot);
    let w_code = pyre_object::gc_roots::shadow_stack_get(code_slot);
    // `PyCode_Addr2Line(lz_code, lz_instr_offset * 2)`: LASTI / `last_instr`
    // is an instruction index; `w_code_addr2line` wants a byte offset.
    let lineno = unsafe { crate::pycode::w_code_addr2line(w_code, instr_offset.saturating_mul(2)) };
    let filename_obj = unsafe { crate::pycode::w_code_filename_obj(w_code) };
    if filename_obj.is_null() {
        err.reload(&roots, err_slot);
        return err;
    }
    let filename_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(filename_obj);
    let w_code = pyre_object::gc_roots::shadow_stack_get(code_slot);
    let funcname_obj = unsafe { crate::pycode::w_code_name_obj(w_code) };
    if funcname_obj.is_null() {
        err.reload(&roots, err_slot);
        return err;
    }
    let funcname_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(funcname_obj);
    err.reload(&roots, err_slot);
    let Some(filename) = str_value(pyre_object::gc_roots::shadow_stack_get(filename_slot)) else {
        err.reload(&roots, err_slot);
        return err;
    };
    let filename = filename.to_owned();
    let Some(funcname) = str_value(pyre_object::gc_roots::shadow_stack_get(funcname_slot)) else {
        err.reload(&roots, err_slot);
        return err;
    };
    let funcname = funcname.to_owned();
    let Some(name_s) = str_value(pyre_object::gc_roots::shadow_stack_get(name_slot)) else {
        err.reload(&roots, err_slot);
        return err;
    };
    let name_s = name_s.to_owned();
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let cause = unsafe { w_exception_get_cause(exc) };
    let context = unsafe { w_exception_get_context(exc) };
    let suppress = unsafe { w_exception_get_suppress_context(exc) };
    if !cause.is_null() || !context.is_null() || suppress {
        let note = w_str_new_managed(&format!(
            "lazy import of '{name_s}' declared in {funcname} at {filename}:{lineno}"
        ));
        let note_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(note);
        let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        let notes = crate::baseobjspace::getattr_str_impl(exc, "__notes__", true, true);
        let add = match notes {
            Ok(notes) if notes.is_null() => true,
            Ok(notes) => {
                let notes_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(notes);
                match crate::baseobjspace::contains(
                    pyre_object::gc_roots::shadow_stack_get(notes_slot),
                    pyre_object::gc_roots::shadow_stack_get(note_slot),
                ) {
                    Ok(false) => true,
                    _ => false,
                }
            }
            Err(_) => false,
        };
        if add {
            let _ = crate::baseobjspace::base_exception_add_note(
                pyre_object::gc_roots::shadow_stack_get(exc_slot),
                pyre_object::gc_roots::shadow_stack_get(note_slot),
            );
        }
        err.reload(&roots, err_slot);
        return err;
    }
    let mut cause_err = crate::PyError::new(
        crate::PyErrorKind::ImportError,
        format!("lazy import of '{name_s}' raised an exception during resolution"),
    );
    err.reload(&roots, err_slot);
    let cause = cause_err.to_exc_object();
    err.reload(&roots, err_slot);
    if cause.is_null() || unsafe { !is_exception(cause) } {
        return err;
    }
    let cause_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cause);
    crate::pytraceback::traceback_add(
        pyre_object::gc_roots::shadow_stack_get(cause_slot),
        &funcname,
        &filename,
        lineno as i64,
    );
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let cause = pyre_object::gc_roots::shadow_stack_get(cause_slot);
    unsafe {
        w_exception_set_cause(exc, cause);
        w_exception_set_suppress_context(exc, true);
    }
    err.reload(&roots, err_slot);
    err
}

fn load_lazy_import_impl(
    mut lazy_import: PyObjectRef,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let (mut root, first) = root_and_first(lazy_import);
    let _roots = pyre_object::gc_roots::push_roots();
    let lazy_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lazy_import);
    let root_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(root);
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(first);
    lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    if is_resolving(lazy_import) {
        let name = lazy_import_name(lazy_import)?;
        root = pyre_object::gc_roots::shadow_stack_get(root_slot);
        let root_from = unsafe { (*w_lazy_import_cast(root)).w_from };
        return Err(import_cycle_error(name, root_from));
    }
    resolving_push(pyre_object::gc_roots::shadow_stack_get(lazy_slot));
    let result = load_lazy_import_resolve(lazy_slot, root_slot, first_slot);
    resolving_pop(pyre_object::gc_roots::shadow_stack_get(lazy_slot));
    match result {
        Ok(v) => Ok(v),
        Err(e) => Err(lazy_import_add_exception_cause(
            pyre_object::gc_roots::shadow_stack_get(lazy_slot),
            e,
        )),
    }
}

fn load_lazy_import_resolve(
    lazy_slot: usize,
    root_slot: usize,
    first_slot: usize,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let mut root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let mut first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    let root_attr = unsafe { (*w_lazy_import_cast(root)).w_attr };
    let mut fromlist = if !first.is_null()
        && !root_attr.is_null()
        && unsafe { is_tuple(root_attr) && w_tuple_len(root_attr) > 0 }
    {
        let mut attr = unsafe { (*w_lazy_import_cast(first)).w_attr };
        pyre_object::with_roots!(attr => w_tuple_new(vec![attr]))
    } else if root_attr.is_null() {
        w_none()
    } else {
        root_attr
    };
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let fromlist_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(fromlist);
    let frame = current_frame();
    let mut globals = if frame.is_null() {
        w_none()
    } else {
        unsafe { (*frame).get_w_globals() }
    };
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(globals);
    let builtins = unsafe { (*w_lazy_import_cast(root)).w_builtins };
    let import_func = if builtins.is_null() {
        None
    } else {
        crate::baseobjspace::finditem_str(builtins, "__import__")?
    };
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    let mut import_func = import_func.ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found")
    })?;
    let import_func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(import_func);
    import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
    let mut name = unsafe { (*w_lazy_import_cast(root)).w_from };
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let fromlist_empty =
        unsafe { is_none(fromlist) || (is_tuple(fromlist) && w_tuple_len(fromlist) == 0) };
    let builtins = unsafe { (*w_lazy_import_cast(root)).w_builtins };
    if first.is_null()
        && fromlist_empty
        && is_default_import_func(import_func)
        && is_interp_builtins(builtins)
    {
        name = pyre_object::gc_roots::shadow_stack_get(name_slot);
        let truncated = maybe_truncate_import_name(name)?;
        pyre_object::gc_roots::shadow_stack_set(name_slot, truncated);
    }
    let mut level = w_int_new(0);
    let mut lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
    first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let level_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(level);
    level = pyre_object::gc_roots::shadow_stack_get(level_slot);
    let mut obj = crate::call::call_function_impl_result(
        import_func,
        &[name, globals, globals, fromlist, level],
    )?;
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let root_from = unsafe { (*w_lazy_import_cast(root)).w_from };
    if !std::ptr::eq(name, root_from) {
        unsafe { (*w_lazy_import_cast(root)).active = 1 };
        obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        if !module_dict_contains_path(obj)? {
            import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
            globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            level = pyre_object::gc_roots::shadow_stack_get(level_slot);
            root = pyre_object::gc_roots::shadow_stack_get(root_slot);
            let mut root_from = unsafe { (*w_lazy_import_cast(root)).w_from };
            let root_from_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(root_from);
            root_from = pyre_object::gc_roots::shadow_stack_get(root_from_slot);
            import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
            globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            level = pyre_object::gc_roots::shadow_stack_get(level_slot);
            obj = crate::call::call_function_impl_result(
                import_func,
                &[root_from, globals, globals, fromlist, level],
            )?;
            pyre_object::gc_roots::shadow_stack_set(obj_slot, obj);
        }
    }
    obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    import_func = pyre_object::gc_roots::shadow_stack_get(import_func_slot);
    let root_from = unsafe { (*w_lazy_import_cast(root)).w_from };
    if std::ptr::eq(name, root_from) && is_default_import_func(import_func) {
        name = pyre_object::gc_roots::shadow_stack_get(name_slot);
        clear_lazy_submodule(name, false)?;
        obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    }
    lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
    first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    obj = resolve_result(obj)?;
    lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
    first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    pyre_object::gc_roots::shadow_stack_set(obj_slot, obj);
    obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    if !first.is_null() {
        obj = replay_from(obj, lazy_import)?;
        lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
        pyre_object::gc_roots::shadow_stack_set(obj_slot, obj);
        obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    }
    if let Ok(path) = lazy_import_path(lazy_import) {
        discard_lazy_module(path);
    }
    obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let imported = if unsafe { is_module(obj) } {
        obj
    } else {
        PY_NULL
    };
    Ok((obj, imported))
}

fn discard_lazy_module(name: PyObjectRef) {
    let set = crate::module::sys::state::lazy_modules();
    if !set.is_null() && unsafe { is_set_or_frozenset(set) } {
        unsafe { w_set_discard(set, name) };
    }
}

/// `_imp__set_lazy_attributes_impl`: drop `name` from `sys.lazy_modules`
/// and clear any pending lazy child with `bind=0`.
pub fn set_lazy_attributes(_modobj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    discard_lazy_module(pyre_object::gc_roots::shadow_stack_get(name_slot));
    clear_lazy_submodule(pyre_object::gc_roots::shadow_stack_get(name_slot), false)
}

fn module_is_initializing(module: PyObjectRef) -> Result<bool, crate::PyError> {
    if module.is_null() || unsafe { is_none(module) } {
        return Ok(false);
    }
    match crate::baseobjspace::getattr_str(module, "__spec__") {
        Ok(spec) => crate::importing::is_spec_initializing(spec),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(false),
        Err(e) => Err(e),
    }
}

/// `PyDict_Pop(dict, key, NULL)`: a missing key is success.
fn dict_pop_discard(dict: PyObjectRef, key: PyObjectRef) -> Result<(), crate::PyError> {
    crate::type_methods::dict_method_pop(&[dict, key, w_none()]).map(|_| ())
}

/// `_PyImport_ClearLazySubmodule`. `bind` is `_imp._set_lazy_attributes`'s 0
/// or the completed-import 1: a true bind may copy the loaded child onto
/// an already-finished parent.
///
/// Residual from look-inside `__import__` (`PyImport_ImportModuleLevelObject`'s
/// cached-initialized arm); `getattr_str` on `__spec__` must stay off that graph.
#[majit_macros::dont_look_inside]
pub fn clear_lazy_submodule(name: PyObjectRef, bind: bool) -> Result<(), crate::PyError> {
    let pending = crate::module::sys::state::lazy_pending_submodules();
    if pending.is_null() {
        return Ok(());
    }
    if unsafe { pyre_object::dictmultiobject::w_dict_len(pending) } == 0 {
        return Ok(());
    }
    let Some(name_s) = str_value(name) else {
        return Ok(());
    };
    let Some(dot) = name_s.rfind('.') else {
        return Ok(());
    };
    let parent_s = name_s[..dot].to_owned();
    let child_s = name_s[dot + 1..].to_owned();
    let full_s = name_s.to_owned();
    let _roots = pyre_object::gc_roots::push_roots();
    let pending_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending);
    let Some(pending_set) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(pending_slot),
        &parent_s,
    )?
    else {
        return Ok(());
    };
    let pending_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending_set);
    if crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
        &child_s,
    )?
    .is_none()
    {
        return Ok(());
    }
    if let Some(loaded) = crate::importing::sys_modules_entry(&full_s) {
        let loaded_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(loaded);
        match module_is_initializing(pyre_object::gc_roots::shadow_stack_get(loaded_slot)) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(_) => return Ok(()),
        }
    }
    let parent_slot = pyre_object::gc_roots::shadow_stack_len();
    let has_parent = match crate::importing::sys_modules_entry(&parent_s) {
        Some(module) => {
            let _ = pyre_object::gc_roots::pin_root(module);
            true
        }
        None => false,
    };
    let initializing = if has_parent {
        module_is_initializing(pyre_object::gc_roots::shadow_stack_get(parent_slot))?
    } else {
        false
    };
    if bind && !initializing && has_parent {
        if unsafe { is_module(pyre_object::gc_roots::shadow_stack_get(parent_slot)) }
            && let Some(value) = crate::importing::sys_modules_entry(&full_s)
            && unsafe { !is_none(value) }
        {
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(value);
            let dict = unsafe {
                w_module_get_w_dict(pyre_object::gc_roots::shadow_stack_get(parent_slot))
            };
            if !dict.is_null() {
                let dict_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(dict);
                if crate::baseobjspace::finditem_str(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    &child_s,
                )?
                .is_none()
                {
                    let key = w_str_new(&child_s);
                    let key_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(key);
                    crate::baseobjspace::setitem(
                        pyre_object::gc_roots::shadow_stack_get(dict_slot),
                        pyre_object::gc_roots::shadow_stack_get(key_slot),
                        pyre_object::gc_roots::shadow_stack_get(value_slot),
                    )?;
                }
            }
        }
    }
    if initializing {
        let key = w_str_new(&child_s);
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(key);
        let none = w_none();
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            none,
        )?;
    } else {
        let key = w_str_new(&child_s);
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(key);
        dict_pop_discard(
            pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
        )?;
    }
    Ok(())
}

fn replace_item_if(
    mut namespace: PyObjectRef,
    mut name: PyObjectRef,
    mut placeholder: PyObjectRef,
    mut value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if namespace.is_null() {
        return Ok(());
    }
    match pyre_object::with_roots!(name, namespace, placeholder, value => {
        crate::baseobjspace::finditem(namespace, name)
    }) {
        Ok(Some(current)) if std::ptr::eq(current, placeholder) => {
            crate::baseobjspace::setitem(namespace, name, value)?;
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(()),
        Err(e) => Err(e),
    }
}

/// `_PyLazyImport_Reify`
#[majit_macros::dont_look_inside]
pub fn reify(
    mut placeholder: PyObjectRef,
    mut name: PyObjectRef,
    mut namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let placeholder_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(placeholder);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let namespace_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(namespace);
    let (value, _imported) =
        load_lazy_import_impl(pyre_object::gc_roots::shadow_stack_get(placeholder_slot))?;
    let mut value = value;
    placeholder = pyre_object::gc_roots::shadow_stack_get(placeholder_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    namespace = pyre_object::gc_roots::shadow_stack_get(namespace_slot);
    pyre_object::with_roots!(value, name, namespace, placeholder => {
        replace_item_if(namespace, name, placeholder, value)
    })?;
    Ok(value)
}

/// Promote `ob_type` so `is_lazy_import` (`py_type_check` vs the
/// `lazy_import` vtable) is a constant compare on the hot LOAD_GLOBAL /
/// LOAD_NAME path. `reify` is `dont_look_inside`; a residual type test
/// would keep that call in the trace.
#[inline]
fn promote_value_type(value: PyObjectRef) {
    if !value.is_null()
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(value))
    {
        let ob_type = unsafe { (*value).ob_type };
        let _ = majit_metainterp::jit::promote(ob_type);
    }
}

/// Interpreter-visible lazy check used by LOAD_GLOBAL / LOAD_NAME.
#[inline]
pub fn maybe_reify(
    value: PyObjectRef,
    name: PyObjectRef,
    namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    promote_value_type(value);
    if unsafe { is_lazy_import(value) } {
        reify(value, name, namespace)
    } else {
        Ok(value)
    }
}

#[inline]
pub fn maybe_reify_str(
    value: PyObjectRef,
    name: &str,
    namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    promote_value_type(value);
    if unsafe { is_lazy_import(value) } {
        reify(value, w_str_new(name), namespace)
    } else {
        Ok(value)
    }
}

/// gh-144957: give module `__getattr__` a chance before resolving.
pub fn module_getattro_lazy(
    module: PyObjectRef,
    name: &str,
    mut placeholder: PyObjectRef,
    mut dict: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if name != "__getattr__"
        && let Some(getattr) = pyre_object::with_roots!(dict, placeholder => {
            crate::baseobjspace::finditem_str(dict, "__getattr__")
        })?
    {
        let getattr = pyre_object::with_roots!(dict, placeholder => {
            maybe_reify_str(getattr, "__getattr__", dict)
        })?;
        match pyre_object::with_roots!(dict, placeholder => {
            crate::call::call_function_impl_result(getattr, &[w_str_new(name)])
        }) {
            Ok(result) => return Ok(result),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
            Err(e) => return Err(e),
        }
    }
    if is_resolving(placeholder) {
        return Err(crate::PyError::attribute_error(format!(
            "module has no attribute '{name}'"
        )));
    }
    reify(placeholder, w_str_new(name), dict)
}

/// `lazyimportobject.c _PyLazyImport_LoadChild`.
pub fn load_lazy_import_child(
    mut source: PyObjectRef,
    mut name: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let source_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(source);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let builtins = unsafe { (*w_lazy_import_cast(source)).w_builtins };
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let mut lz = lazy_import_new_no_frame(builtins, name, PY_NULL)?;
    let lz_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(lz);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let name_s = str_value(name).unwrap_or("").to_owned();
    if let Some(first_dot) = name_s.find('.') {
        let mut start = first_dot + 1;
        loop {
            let rel = name_s[start..].find('.');
            let attr_end = rel.map(|d| start + d).unwrap_or(name_s.len());
            let mut attr = w_str_new(&name_s[start..attr_end]);
            let attr_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(attr);
            lz = pyre_object::gc_roots::shadow_stack_get(lz_slot);
            attr = pyre_object::gc_roots::shadow_stack_get(attr_slot);
            let next = lazy_import_new_no_frame(PY_NULL, lz, attr)?;
            pyre_object::gc_roots::shadow_stack_set(lz_slot, next);
            match rel {
                Some(_) => start = attr_end + 1,
                None => break,
            }
        }
    }
    source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    lz = pyre_object::gc_roots::shadow_stack_get(lz_slot);
    unsafe {
        let w_code = (*w_lazy_import_cast(source)).w_code;
        (*w_lazy_import_cast(lz)).w_code = w_code;
        (*w_lazy_import_cast(lz)).instr_offset = (*w_lazy_import_cast(source)).instr_offset;
        if !w_code.is_null() {
            pyre_object::gc_hook::try_gc_write_barrier(lz as pyre_object::gc_hook::GCREF);
        }
    }
    lz = pyre_object::gc_roots::shadow_stack_get(lz_slot);
    let mut result = load_lazy_import(lz)?;
    let result_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(result);
    source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let mut from = unsafe { (*w_lazy_import_cast(source)).w_from };
    let from_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(from);
    from = pyre_object::gc_roots::shadow_stack_get(from_slot);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let names_differ =
        if !from.is_null() && !name.is_null() && unsafe { is_str(from) && is_str(name) } {
            !unsafe { w_str_eq_w(from, name) }
        } else {
            !std::ptr::eq(from, name)
        };
    result = pyre_object::gc_roots::shadow_stack_get(result_slot);
    if unsafe { is_module(result) } && names_differ && !module_dict_contains_path(result)? {
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        from = unsafe { (*w_lazy_import_cast(source)).w_from };
        pyre_object::gc_roots::shadow_stack_set(from_slot, from);
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        from = pyre_object::gc_roots::shadow_stack_get(from_slot);
        return load_lazy_import_child(source, from);
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(result_slot))
}

/// `_PyImport_TryLoadLazySubmodule`.
///
/// Residual (`@jit.dont_look_inside`, as `importing.py load_compiled_module`):
/// the body loads a submodule through the import machinery.
#[majit_macros::dont_look_inside]
pub fn try_load_lazy_submodule(
    mut module: PyObjectRef,
    attr_name: &str,
    suppress: bool,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { !is_module(module) } {
        return Ok(None);
    }
    let attr_s = attr_name.to_owned();
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(module);
    module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    let mut dict = unsafe { w_module_get_w_dict(module) };
    if dict.is_null() {
        return Ok(None);
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
    let Some(mod_name) = crate::baseobjspace::finditem_str(dict, "__name__")? else {
        return Ok(None);
    };
    let Some(mod_name_s) = str_value(mod_name) else {
        return Ok(None);
    };
    let mod_name_s = mod_name_s.to_owned();
    let pending = crate::module::sys::state::lazy_pending_submodules();
    if pending.is_null() {
        return Ok(None);
    }
    let pending_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending);
    let Some(pending_set) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(pending_slot),
        &mod_name_s,
    )?
    else {
        return Ok(None);
    };
    let pending_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending_set);
    let Some(source) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
        &attr_s,
    )?
    else {
        return Ok(None);
    };
    let source_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(source);
    let full_s = format!("{mod_name_s}.{attr_s}");
    let mut full_name = w_str_new(&full_s);
    let full_name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(full_name);
    dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
    let no_getattr = crate::baseobjspace::finditem_str(dict, "__getattr__")?.is_none();
    let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let load_child =
        !unsafe { is_none(source) } && !suppress && (is_lazy_import_active(source) || no_getattr);
    full_name = pyre_object::gc_roots::shadow_stack_get(full_name_slot);
    let load_result = if load_child {
        let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        load_lazy_import_child(source, full_name)
    } else {
        match crate::importing::sys_modules_entry(&full_s) {
            Some(loaded) if unsafe { !is_none(loaded) } => Ok(loaded),
            _ => {
                full_name = pyre_object::gc_roots::shadow_stack_get(full_name_slot);
                find_and_load_lazy_submodule(full_name)
            }
        }
    };
    match load_result {
        Err(mut e) => {
            let e_slot = e.pin(&roots);
            if let Some(loaded) = crate::importing::sys_modules_entry(&full_s)
                && unsafe { !is_none(loaded) }
            {
                let loaded_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(loaded);
                let key = w_str_new(&attr_s);
                let key_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(key);
                dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
                let _ = crate::baseobjspace::dict_delitem_if_value_is(
                    dict,
                    pyre_object::gc_roots::shadow_stack_get(key_slot),
                    pyre_object::gc_roots::shadow_stack_get(loaded_slot),
                );
                let pending_set = pyre_object::gc_roots::shadow_stack_get(pending_set_slot);
                if crate::baseobjspace::finditem_str(pending_set, &attr_s)
                    .ok()
                    .flatten()
                    .is_none()
                {
                    let key = w_str_new(&attr_s);
                    let key_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(key);
                    let _ = crate::baseobjspace::setitem(
                        pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
                        pyre_object::gc_roots::shadow_stack_get(key_slot),
                        pyre_object::gc_roots::shadow_stack_get(source_slot),
                    );
                }
            }
            e.reload(&roots, e_slot);
            Err(e)
        }
        Ok(mod_) if mod_.is_null() || unsafe { is_none(mod_) } => Ok(None),
        Ok(mod_) => {
            let mod_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(mod_);
            let initializing =
                lazy_submodule_is_initializing(pyre_object::gc_roots::shadow_stack_get(mod_slot))?;
            if initializing {
                return Ok(Some(pyre_object::gc_roots::shadow_stack_get(mod_slot)));
            }
            let key = w_str_new(&attr_s);
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(key);
            crate::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(mod_slot),
            )?;
            let key = w_str_new(&attr_s);
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(key);
            dict_pop_discard(
                pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
            )?;
            Ok(Some(pyre_object::gc_roots::shadow_stack_get(mod_slot)))
        }
    }
}

/// `globals is locals`: module exec, and `FunctionType` of the same code.
/// `CO_NEWLOCALS` frames (functions, class bodies) bind a distinct mapping
/// (`pyframe.py initialize_frame_scopes`), so the test is not a PyCode flag.
fn is_module_level_frame_of(frame: &PyFrame) -> bool {
    let globals = frame.get_w_globals();
    let locals = frame.get_w_locals();
    !globals.is_null() && !locals.is_null() && std::ptr::eq(globals, locals)
}

fn is_module_level_frame() -> bool {
    let frame = current_frame();
    if frame.is_null() {
        return false;
    }
    is_module_level_frame_of(unsafe { &*frame })
}

fn check_lazy_import_compatibility(
    mut globals: PyObjectRef,
    mut name: PyObjectRef,
    level: i32,
) -> Result<bool, crate::PyError> {
    if globals.is_null() || unsafe { is_none(globals) } {
        return Ok(false);
    }
    let Some(lazy_modules) = pyre_object::with_roots!(globals, name => {
        crate::baseobjspace::finditem_str(globals, "__lazy_modules__")
    })?
    else {
        return Ok(false);
    };
    let mut lazy_modules = lazy_modules;
    let abs = pyre_object::with_roots!(lazy_modules => import_abs_name(name, globals, level))?;
    crate::baseobjspace::contains(lazy_modules, abs)
}

fn import_abs_name(
    mut name: PyObjectRef,
    mut globals: PyObjectRef,
    level: i32,
) -> Result<PyObjectRef, crate::PyError> {
    if level <= 0 {
        return Ok(name);
    }
    if globals.is_null() || unsafe { is_none(globals) } {
        return Err(crate::PyError::key_error("'__name__' not in globals"));
    }
    let package = match pyre_object::with_roots!(globals, name => {
        crate::baseobjspace::finditem_str(globals, "__package__")
    })? {
        Some(p) if !unsafe { is_none(p) } => p,
        _ => match pyre_object::with_roots!(name => {
            crate::baseobjspace::finditem_str(globals, "__name__")
        })? {
            Some(n) => n,
            None => return Err(crate::PyError::key_error("'__name__' not in globals")),
        },
    };
    let Some(pkg) = str_value(package) else {
        return Err(crate::PyError::type_error("package must be a string"));
    };
    let mut end = pkg.len();
    for _ in 1..level {
        match pkg[..end].rfind('.') {
            Some(dot) => end = dot,
            None => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    "attempted relative import beyond top-level package",
                ));
            }
        }
    }
    let base = &pkg[..end];
    let name_s = str_value(name).unwrap_or("");
    if name_s.is_empty() {
        Ok(w_str_new(base))
    } else {
        Ok(w_str_new(&format!("{base}.{name_s}")))
    }
}

/// `import.c lazy_modules_add`. Skip `sys.lazy_modules` when *name* is already
/// imported and fully initialized. `__spec__` is read from the module dict;
/// `_initializing` from mapdict instance storage only (no descriptors, no
/// `__bool__`). Returns the `sys.modules` entry (`existing_mod_p`).
fn lazy_modules_add(mut name: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let name_s = str_value(name).unwrap_or("").to_owned();
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let existing = crate::importing::sys_modules_entry(&name_s);
    let existing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(existing.unwrap_or(PY_NULL));
    let mut loaded = {
        let m = pyre_object::gc_roots::shadow_stack_get(existing_slot);
        !m.is_null() && unsafe { !is_none(m) }
    };
    if loaded {
        let existing_mod = pyre_object::gc_roots::shadow_stack_get(existing_slot);
        if unsafe { is_module(existing_mod) } {
            let mut dict = unsafe { w_module_get_w_dict(existing_mod) };
            let spec = if dict.is_null() {
                None
            } else {
                pyre_object::with_roots!(dict => {
                    crate::baseobjspace::finditem_str(dict, "__spec__")
                })?
            };
            let spec_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(spec.unwrap_or(PY_NULL));
            let spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
            loaded = spec.is_null() || unsafe { is_none(spec) };
            if !loaded {
                let mut spec = spec;
                if unsafe { crate::objspace::std::mapdict::has_mapdict_storage(spec) } {
                    loaded = match pyre_object::with_roots!(spec => {
                        crate::baseobjspace::getdictvalue(spec, "_initializing")
                    })? {
                        None => true,
                        Some(init) => is_w(init, w_bool_from(false)),
                    };
                }
            }
        }
    }
    if !loaded {
        name = pyre_object::gc_roots::shadow_stack_get(name_slot);
        let set = crate::module::sys::state::lazy_modules();
        if !set.is_null() {
            unsafe { w_set_add(set, name) };
        }
    }
    let existing = pyre_object::gc_roots::shadow_stack_get(existing_slot);
    Ok(if existing.is_null() {
        None
    } else {
        Some(existing)
    })
}

/// `import.c register_from_lazy_on_parent`.
///
/// `IMPORT_FROM` returns a stored attribute directly, so skip the alias when
/// `from` is already on the existing module.
fn register_from_lazy_on_parent(
    existing_module: Option<PyObjectRef>,
    mut abs_name: PyObjectRef,
    mut from: PyObjectRef,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut existing_module = existing_module.unwrap_or(PY_NULL);
    let existing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(existing_module);
    let abs_name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(abs_name);
    let from_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(from);
    existing_module = pyre_object::gc_roots::shadow_stack_get(existing_slot);
    from = pyre_object::gc_roots::shadow_stack_get(from_slot);
    let from_s = str_value(from).unwrap_or("").to_owned();
    existing_module = pyre_object::gc_roots::shadow_stack_get(existing_slot);
    if !existing_module.is_null() && unsafe { is_module(existing_module) } {
        let mut dict = unsafe { w_module_get_w_dict(existing_module) };
        if !dict.is_null()
            && pyre_object::with_roots!(dict => {
                crate::baseobjspace::finditem_str(dict, &from_s)
            })?
            .is_some()
        {
            return Ok(());
        }
    }
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    let abs = str_value(abs_name).unwrap_or("").to_owned();
    let mut fromname = w_str_new(&format!("{abs}.{from_s}"));
    let fromname_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(fromname);
    fromname = pyre_object::gc_roots::shadow_stack_get(fromname_slot);
    lazy_modules_add(fromname)?;
    fromname = pyre_object::gc_roots::shadow_stack_get(fromname_slot);
    register_lazy_on_parent(fromname, w_none())
}

/// `import.c register_lazy_on_parent`. A cached parent (`sys.modules` hit
/// that is not `None`) stores `None` instead of `source`. `source is None`
/// is `PyDict_SetDefaultRef` (write only if `child` is absent); otherwise
/// `PyDict_SetItem`.
fn register_lazy_on_parent(
    name: PyObjectRef,
    mut source: PyObjectRef,
) -> Result<(), crate::PyError> {
    let Some(name_s) = str_value(name) else {
        return Ok(());
    };
    let name_s = name_s.to_owned();
    let _roots = pyre_object::gc_roots::push_roots();
    let source_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(source);
    let mut pending = crate::module::sys::state::ensure_lazy_pending_submodules();
    if pending.is_null() {
        return Ok(());
    }
    let pending_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending);
    let mut end = name_s.len();
    while let Some(dot) = name_s[..end].rfind('.') {
        let parent = &name_s[..dot];
        let child = &name_s[dot + 1..end];
        let fullname = &name_s[..end];
        pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        let parent_key = pyre_object::with_roots!(pending, source => w_str_new(parent));
        let parent_key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(parent_key);
        pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        let mut parent_key = pyre_object::gc_roots::shadow_stack_get(parent_key_slot);
        let child_key = pyre_object::with_roots!(pending, source, parent_key => w_str_new(child));
        let child_key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(child_key);
        pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
        parent_key = pyre_object::gc_roots::shadow_stack_get(parent_key_slot);
        let mut submodules = match crate::baseobjspace::finditem(pending, parent_key)? {
            Some(d) => d,
            None => {
                let mut d = w_dict_new();
                pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
                parent_key = pyre_object::gc_roots::shadow_stack_get(parent_key_slot);
                source = pyre_object::gc_roots::shadow_stack_get(source_slot);
                pyre_object::with_roots!(d, pending, source, parent_key => {
                    crate::baseobjspace::setitem(pending, parent_key, d)
                })?;
                d
            }
        };
        let submodules_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(submodules);
        let cached = crate::importing::sys_modules_entry(fullname);
        let cached_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cached.unwrap_or(PY_NULL));
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        let cached = pyre_object::gc_roots::shadow_stack_get(cached_slot);
        let value = if !cached.is_null() && unsafe { !is_none(cached) } {
            w_none()
        } else {
            source
        };
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        submodules = pyre_object::gc_roots::shadow_stack_get(submodules_slot);
        let mut child_key = pyre_object::gc_roots::shadow_stack_get(child_key_slot);
        let mut value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        if unsafe { is_none(source) } {
            if crate::baseobjspace::finditem(submodules, child_key)?.is_none() {
                submodules = pyre_object::gc_roots::shadow_stack_get(submodules_slot);
                child_key = pyre_object::gc_roots::shadow_stack_get(child_key_slot);
                value = pyre_object::gc_roots::shadow_stack_get(value_slot);
                crate::baseobjspace::setitem(submodules, child_key, value)?;
            }
        } else {
            crate::baseobjspace::setitem(submodules, child_key, value)?;
        }
        end = dot;
    }
    Ok(())
}

/// `_PyImport_LazyImportModuleLevelObject`
pub fn lazy_import_module_level(
    mut name: PyObjectRef,
    mut builtins: PyObjectRef,
    mut globals: PyObjectRef,
    mut _locals: PyObjectRef,
    mut fromlist: PyObjectRef,
    level: i32,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { !is_str(name) } {
        return Err(crate::PyError::type_error(format!(
            "module name must be a string, got {}",
            crate::type_methods::arg_type_name(name)
        )));
    }
    if level < 0 {
        return Err(crate::PyError::value_error("level must be >= 0"));
    }
    if !is_module_level_frame() {
        return Err(crate::PyError::syntax_error(
            "'lazy import' is only allowed at module level",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let builtins_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(builtins);
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(globals);
    let locals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(_locals);
    let fromlist_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(fromlist);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
    globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    let mut abs_name = import_abs_name(name, globals, level)?;
    builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    let abs_name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(abs_name);
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    fromlist = if fromlist.is_null() || unsafe { is_none(fromlist) } {
        w_none()
    } else if unsafe { is_str(fromlist) } {
        w_tuple_new(vec![fromlist])
    } else {
        fromlist
    };
    pyre_object::gc_roots::shadow_stack_set(fromlist_slot, fromlist);
    builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    if let Some(filter) = crate::module::sys::state::lazy_imports_filter() {
        let mut filter = filter;
        let filter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(filter);
        filter = pyre_object::gc_roots::shadow_stack_get(filter_slot);
        globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
        let mut modname =
            crate::baseobjspace::finditem_str(globals, "__name__")?.unwrap_or_else(w_none);
        builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
        fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
        abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
        filter = pyre_object::gc_roots::shadow_stack_get(filter_slot);
        let modname_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(modname);
        modname = pyre_object::gc_roots::shadow_stack_get(modname_slot);
        let decision =
            crate::call::call_function_impl_result(filter, &[modname, abs_name, fromlist])?;
        builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
        fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
        abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
        if !crate::baseobjspace::is_true(decision)? {
            name = pyre_object::gc_roots::shadow_stack_get(name_slot);
            globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
            _locals = pyre_object::gc_roots::shadow_stack_get(locals_slot);
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            return crate::importing::dunder_import_name_obj(
                name,
                globals,
                _locals,
                fromlist,
                level as i64,
            );
        }
        builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);
        fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
        abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    }
    if unsafe { is_tuple(fromlist) } {
        let n = unsafe { w_tuple_len(fromlist) };
        for i in 0..n {
            let item = unsafe { w_tuple_getitem(fromlist, i as i64) }.unwrap_or(PY_NULL);
            if unsafe { !is_str(item) } {
                return Err(crate::PyError::type_error(format!(
                    "Item in ``from list'' must be str, not {}",
                    crate::type_methods::arg_type_name(item)
                )));
            }
        }
    }
    let mut res = lazy_import_new(builtins, abs_name, fromlist)?;
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    let res_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(res);
    res = pyre_object::gc_roots::shadow_stack_get(res_slot);
    let existing_module = lazy_modules_add(abs_name)?;
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    let existing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(existing_module.unwrap_or(PY_NULL));
    if unsafe { is_tuple(fromlist) && w_tuple_len(fromlist) > 0 } {
        let n = unsafe { w_tuple_len(fromlist) };
        for i in 0..n {
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
            let existing = pyre_object::gc_roots::shadow_stack_get(existing_slot);
            let existing_module = if existing.is_null() {
                None
            } else {
                Some(existing)
            };
            let item = unsafe { w_tuple_getitem(fromlist, i as i64) }.unwrap_or(PY_NULL);
            register_from_lazy_on_parent(existing_module, abs_name, item)?;
        }
        res = pyre_object::gc_roots::shadow_stack_get(res_slot);
    } else {
        register_lazy_on_parent(abs_name, res)?;
        res = pyre_object::gc_roots::shadow_stack_get(res_slot);
    }
    Ok(res)
}

fn level_from_obj(level: PyObjectRef) -> Result<i32, crate::PyError> {
    if level.is_null() || unsafe { is_none(level) } {
        return Ok(0);
    }
    crate::builtins::space_index_w(level).map(|v| v as i32)
}

fn builtins_mapping_from_frame(frame: &PyFrame) -> PyObjectRef {
    let w_builtin = frame.get_builtin();
    if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        let dict = unsafe { w_module_get_w_dict(w_builtin) };
        if !dict.is_null() {
            return dict;
        }
    }
    w_builtin
}

/// `_PyEval_LazyImportName`
#[majit_macros::dont_look_inside]
pub fn eval_lazy_import_name(
    frame: &mut PyFrame,
    mut name: PyObjectRef,
    mut fromlist: PyObjectRef,
    mut level_obj: PyObjectRef,
    explicit_lazy: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let mut lazy = explicit_lazy;
    let mode = crate::module::sys::state::lazy_imports_mode();
    if mode == LAZY_ALL && !lazy {
        lazy = is_module_level_frame_of(frame);
    } else if mode == LAZY_NORMAL {
        // keep the IMPORT_NAME explicit-lazy bit
    }
    if !lazy && is_module_level_frame_of(frame) {
        let mut globals = frame.get_w_globals();
        let level = pyre_object::with_roots!(fromlist, globals, level_obj, name => {
            level_from_obj(level_obj)
        })?;
        lazy = pyre_object::with_roots!(fromlist, level_obj, name => {
            check_lazy_import_compatibility(globals, name, level)
        })?;
    }
    if !lazy {
        return crate::importing::import_name(frame, name, fromlist, level_obj);
    }
    let mut builtins = pyre_object::with_roots!(fromlist, level_obj, name => {
        builtins_mapping_from_frame(frame)
    });
    let lazy_import_func = if builtins.is_null() {
        None
    } else {
        pyre_object::with_roots!(builtins, fromlist, level_obj, name => {
            crate::baseobjspace::finditem_str(builtins, "__lazy_import__")
        })?
    };
    let lazy_import_func = lazy_import_func.ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "__lazy_import__ not found")
    })?;
    if is_default_lazy_import(lazy_import_func) {
        let mut globals = frame.get_w_globals();
        let mut locals = crate::importing::import_locals(frame);
        let level = pyre_object::with_roots!(builtins, fromlist, globals, locals, name => {
            level_from_obj(level_obj)
        })?;
        return lazy_import_module_level(name, builtins, globals, locals, fromlist, level);
    }
    let globals = frame.get_w_globals();
    let locals = crate::importing::import_locals(frame);
    crate::call::call_function_impl_result(
        lazy_import_func,
        &[name, globals, locals, fromlist, level_obj],
    )
}

/// One-word residual-call ABI for [`eval_lazy_import_name`].
///
/// IMPORT_NAME with the eager bit clear on a `!CO_NEWLOCALS` code object
/// can still take the lazy route (`sys.set_lazy_imports("all")`,
/// `__lazy_modules__`, the explicit-lazy bit). The bytecode codewriter
/// residualises that decision here (`MayForce` / can-raise, the same
/// flavor as `import_from`). `explicit_lazy` is the IMPORT_NAME flag bit 0.
pub extern "C" fn eval_lazy_import_name_jit_abi(
    frame: *mut PyFrame,
    name: PyObjectRef,
    fromlist: PyObjectRef,
    level: PyObjectRef,
    explicit_lazy: i64,
) -> PyObjectRef {
    if frame.is_null() {
        return crate::runtime_ops::jit_publish_residual_error_ref(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "IMPORT_NAME received a null frame",
        ));
    }
    // `is_module_level_frame` and `lazy_import_module_level` read the
    // executing frame off the TLS slot; install the red frame so an
    // inlined or residual call matches `pyopcode.py IMPORT_NAME`.
    let _guard = crate::eval::install_current_frame_tls_only(unsafe { &mut *frame });
    match eval_lazy_import_name(
        unsafe { &mut *frame },
        name,
        fromlist,
        level,
        explicit_lazy != 0,
    ) {
        Ok(result) => result,
        Err(error) => crate::runtime_ops::jit_publish_residual_error_ref(error),
    }
}

/// Capture the interp-level `__lazy_import__`, the sibling of
/// `set_default_importlib_import` / `space.w_default_importlib_import`.
pub fn set_default_lazy_import(w_func: PyObjectRef) {
    crate::baseobjspace::object_space().set_w_default_lazy_import(w_func);
}

fn is_default_lazy_import(func: PyObjectRef) -> bool {
    let stored = crate::baseobjspace::object_space().w_default_lazy_import();
    !stored.is_null() && std::ptr::eq(stored, func)
}

/// `_PyImport_IsDefaultImportFunc`: pointer identity with
/// `space.w_default_importlib_import`.
fn is_default_import_func(func: PyObjectRef) -> bool {
    let stored = crate::importing::default_importlib_import().unwrap_or(PY_NULL);
    !stored.is_null() && std::ptr::eq(stored, func)
}

/// `root->lz_builtins == tstate->interp->builtins`.
fn is_interp_builtins(w_builtins: PyObjectRef) -> bool {
    if w_builtins.is_null() {
        return false;
    }
    let interp = crate::baseobjspace::space_builtin_dict();
    !interp.is_null() && std::ptr::eq(w_builtins, interp)
}

/// `_PyLazyImport_IsActive`.
fn is_lazy_import_active(declaration: PyObjectRef) -> bool {
    if declaration.is_null() || unsafe { !is_lazy_import(declaration) } {
        return false;
    }
    unsafe { (*w_lazy_import_cast(declaration)).active != 0 }
}

fn module_dict_contains_path(obj: PyObjectRef) -> Result<bool, crate::PyError> {
    if obj.is_null() || unsafe { !is_module(obj) } {
        return Ok(false);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let dict = unsafe { w_module_get_w_dict(pyre_object::gc_roots::shadow_stack_get(obj_slot)) };
    if dict.is_null() {
        return Ok(false);
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    Ok(crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        "__path__",
    )?
    .is_some())
}

fn err_matches_exception(err: &mut crate::PyError) -> bool {
    let roots = pyre_object::gc_roots::push_roots();
    let err_slot = err.pin(&roots);
    let w_exc = crate::builtins::lookup_exc_class("Exception");
    err.reload(&roots, err_slot);
    let Some(w_exc) = w_exc else {
        return false;
    };
    err.match_(PY_NULL, w_exc)
}

/// `_PyModuleSpec_IsInitializing` after optional `__spec__` getattr.
/// An `Exception`-subclass error counts as not initializing.
fn lazy_submodule_is_initializing(module: PyObjectRef) -> Result<bool, crate::PyError> {
    if module.is_null() || unsafe { is_none(module) } {
        return Ok(false);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(module);
    let spec = match crate::baseobjspace::findattr(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "__spec__",
    ) {
        Ok(None) => return Ok(false),
        Ok(Some(spec)) => spec,
        Err(mut e) => {
            if err_matches_exception(&mut e) {
                return Ok(false);
            }
            return Err(e);
        }
    };
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(spec);
    match crate::importing::is_spec_initializing(pyre_object::gc_roots::shadow_stack_get(spec_slot))
    {
        Ok(v) => Ok(v),
        Err(mut e) => {
            if err_matches_exception(&mut e) {
                Ok(false)
            } else {
                Err(e)
            }
        }
    }
}

/// `lazyimportobject.c lazy_import_resolve_impl` dotted-name walk: keep eager
/// traversal through aliased or non-package parents, otherwise bind the root
/// package and load pending children on attribute access.
fn maybe_truncate_import_name(mut name: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let Some(name_s) = str_value(name) else {
        return Ok(name);
    };
    let name_s = name_s.to_owned();
    let Some(dot) = name_s.find('.') else {
        return Ok(pyre_object::gc_roots::shadow_stack_get(name_slot));
    };
    let mut regular = true;
    let mut complete = true;
    for (i, _) in name_s.match_indices('.') {
        if !regular {
            break;
        }
        let mut prefix = w_str_new(&name_s[..i]);
        let prefix_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(prefix);
        prefix = pyre_object::gc_roots::shadow_stack_get(prefix_slot);
        let cached = lazy_import_get_loaded_attr(prefix, "__name__")?;
        if cached.is_none() {
            complete = false;
            break;
        }
        let mut cached = cached.unwrap();
        let cached_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cached);
        prefix = pyre_object::gc_roots::shadow_stack_get(prefix_slot);
        let path = lazy_import_get_loaded_attr(prefix, "__path__")?;
        cached = pyre_object::gc_roots::shadow_stack_get(cached_slot);
        prefix = pyre_object::gc_roots::shadow_stack_get(prefix_slot);
        regular =
            path.is_some() && unsafe { is_str(cached) } && unsafe { w_str_eq_w(cached, prefix) };
    }
    name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let loaded = lazy_import_get_loaded_attr(name, "__name__")?;
    if regular && (!complete || loaded.is_none()) {
        Ok(w_str_new(&name_s[..dot]))
    } else {
        Ok(pyre_object::gc_roots::shadow_stack_get(name_slot))
    }
}

/// `import.c import_find_and_load_with_name` for `_find_and_load_lazy_submodule`.
fn find_and_load_lazy_submodule(full_name: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(full_name);
    let Some(w_bootstrap) = crate::importing::importlib_bootstrap_module() else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "importlib bootstrap is not available",
        ));
    };
    let bootstrap_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_bootstrap);
    let Some(w_find) = crate::baseobjspace::findattr_result(
        pyre_object::gc_roots::shadow_stack_get(bootstrap_slot),
        "_find_and_load_lazy_submodule",
    )?
    else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "_find_and_load_lazy_submodule not found",
        ));
    };
    let find_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_find);
    let w_import = crate::importing::default_importlib_import().unwrap_or_else(w_none);
    let import_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_import);
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(find_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            pyre_object::gc_roots::shadow_stack_get(import_slot),
        ],
    )
}

/// `builtin___lazy_import___impl`
pub fn builtin_lazy_import(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let scope = crate::builtins::bind_builtin_kwargs(
        args,
        &["name", "globals", "locals", "fromlist", "level"],
        &[true, false, false, false, false],
        "__lazy_import__",
    )?;
    let mut name = scope[0];
    let mut globals = scope[1];
    let mut locals = scope[2];
    let mut fromlist = if scope[3].is_null() {
        w_tuple_new(Vec::new())
    } else {
        scope[3]
    };
    let level = if scope[4].is_null() {
        0
    } else {
        pyre_object::with_roots!(fromlist, globals, locals, name => level_from_obj(scope[4]))?
    };
    if globals.is_null() {
        let frame = current_frame();
        if frame.is_null() {
            return Err(crate::PyError::type_error(
                "__lazy_import__() missing globals when called without a frame",
            ));
        }
        globals = unsafe { (*frame).get_w_globals() };
    }
    if locals.is_null() {
        locals = globals;
    }
    if unsafe { !is_dict(globals) } {
        return Err(crate::PyError::type_error(format!(
            "expect dict for globals, got {}",
            crate::type_methods::arg_type_name(globals)
        )));
    }
    let mut builtins = pyre_object::with_roots!(fromlist, globals, locals, name => {
        crate::baseobjspace::finditem_str(globals, "__builtins__")
    })?
    .ok_or_else(|| crate::PyError::value_error("unable to get builtins for lazy import"))?;
    if unsafe { is_module(builtins) } {
        let dict = unsafe { w_module_get_w_dict(builtins) };
        if dict.is_null() {
            return Err(crate::PyError::attribute_error(
                "builtins module has no dict",
            ));
        }
        builtins = dict;
    }
    lazy_import_module_level(name, builtins, globals, locals, fromlist, level)
}

fn lazy_import_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor '__repr__' requires a 'lazy_import' object",
        ));
    }
    lazy_import_repr_impl(self_)
}

fn lazy_import_descr_resolve(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor 'resolve' requires a 'lazy_import' object",
        ));
    }
    load_lazy_import(self_)
}

fn lazy_import_descr_getattro(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut self_ = args.first().copied().unwrap_or(PY_NULL);
    let mut name = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor '__getattribute__' requires a 'lazy_import' object",
        ));
    }
    let name_s = str_value(name).unwrap_or("");
    match pyre_object::with_roots!(name, self_ => {
        crate::baseobjspace::object_getattribute(self_, name_s)
    }) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let mut w_name = lazy_import_name(self_)?;
            let name_r =
                pyre_object::with_roots!(w_name => crate::builtins::builtin_repr(&[name]))?;
            let mut name_r = name_r;
            let w_r = pyre_object::with_roots!(name_r => crate::builtins::builtin_repr(&[w_name]))?;
            let name_s = str_value(name_r).unwrap_or("?");
            let w_s = str_value(w_r).unwrap_or("?");
            Err(crate::PyError::attribute_error(format!(
                "cannot access attribute {name_s} on unresolved lazy import {w_s}"
            )))
        }
        Err(e) => Err(e),
    }
}

pub(crate) fn init_lazy_import_type(ns: PyObjectRef) {
    let _root_scope = gc_roots::push_roots();
    let ns_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            make_new_descr(|_| {
                Err(crate::PyError::type_error(
                    "cannot create 'lazy_import' instances",
                ))
            })
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::gateway::make_slot_wrapper_with_arity_and_doc(
                "__repr__",
                lazy_import_descr_repr,
                1,
                "Return repr(self).",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "resolve",
            crate::gateway::make_method_descriptor_with_arity_and_doc(
                "resolve",
                lazy_import_descr_resolve,
                1,
                "Resolve the lazy import and return the imported object.",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getattribute__",
            crate::gateway::make_slot_wrapper_with_arity(
                "__getattribute__",
                lazy_import_descr_getattro,
                2,
            )
        )
    };
}

pub fn apply_lazy_imports_mode_after_site() {
    if crate::importing::lazy_imports_config() == 1 {
        crate::module::sys::state::set_lazy_imports_mode(LAZY_ALL);
    }
}

// ── compile-wrapper: `from ... lazy import` check ──────────────────

fn lazy_syntax_error(
    source: &str,
    filename: &str,
    range: ruff_text_size::TextRange,
    message: String,
) -> CompileError {
    let sf = SourceFileBuilder::new(filename, source).finish();
    let loc = sf
        .to_source_code()
        .source_location(range.start(), PositionEncoding::Utf8);
    let end = sf
        .to_source_code()
        .source_location(range.end(), PositionEncoding::Utf8);
    CompileError::Codegen(CodegenError {
        location: Some(loc),
        end_location: Some(end),
        error: CodegenErrorType::SyntaxError(message),
        source_path: filename.to_owned(),
    })
}

fn misplaced_from_lazy_import(source: &str, filename: &str) -> Result<(), CompileError> {
    // Parser/parser.c: `from ... lazy import` is rejected; ruff has no equivalent.
    let bytes = source.as_bytes();
    let mut i = 0usize;
    let mut at_stmt = true;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            at_stmt = true;
            continue;
        }
        if c == b'\'' || c == b'"' {
            i = skip_string(bytes, i);
            at_stmt = false;
            continue;
        }
        if c == b'\n' || c == b';' {
            at_stmt = true;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if at_stmt && rest_is_ident(bytes, i, b"from") {
            i += 4;
            i = skip_ws_and_comments(bytes, i);
            // Leading dots, then an optional dotted name: a name ends at the
            // first whitespace that is not followed by `.`.
            let mut after_dot = true;
            while i < bytes.len() {
                let ch = bytes[i];
                if ch == b'.' {
                    after_dot = true;
                    i = skip_ws_and_comments(bytes, i + 1);
                    continue;
                }
                if after_dot && (ch.is_ascii_alphanumeric() || ch == b'_') {
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    {
                        i += 1;
                    }
                    after_dot = false;
                    let next = skip_ws_and_comments(bytes, i);
                    if next < bytes.len() && bytes[next] == b'.' {
                        i = next;
                        continue;
                    }
                    break;
                }
                break;
            }
            i = skip_ws_and_comments(bytes, i);
            if rest_is_ident(bytes, i, b"lazy") {
                let after = skip_ws_and_comments(bytes, i + 4);
                if rest_is_ident(bytes, after, b"import") {
                    let range = ruff_text_size::TextRange::new(
                        ruff_text_size::TextSize::from(i as u32),
                        ruff_text_size::TextSize::from((i + 4) as u32),
                    );
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        range,
                        "use 'lazy from ... ' instead of 'from ... lazy import'".to_string(),
                    ));
                }
            }
            at_stmt = false;
            continue;
        }
        at_stmt = false;
        i += 1;
    }
    Ok(())
}

fn rest_is_ident(bytes: &[u8], i: usize, word: &[u8]) -> bool {
    if i + word.len() > bytes.len() {
        return false;
    }
    if !bytes[i..].starts_with(word) {
        return false;
    }
    let next = i + word.len();
    next == bytes.len() || !(bytes[next].is_ascii_alphanumeric() || bytes[next] == b'_')
}

fn skip_ws_and_comments(bytes: &[u8], mut i: usize) -> usize {
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'\n' {
            i += 2;
            continue;
        }
        break;
    }
    i
}

fn skip_string(bytes: &[u8], i: usize) -> usize {
    let quote = bytes[i];
    let triple = i + 2 < bytes.len() && bytes[i + 1] == quote && bytes[i + 2] == quote;
    let mut j = if triple { i + 3 } else { i + 1 };
    while j < bytes.len() {
        if bytes[j] == b'\\' {
            j += 2;
            continue;
        }
        if bytes[j] == quote {
            if triple {
                if j + 2 < bytes.len() && bytes[j + 1] == quote && bytes[j + 2] == quote {
                    return j + 3;
                }
            } else {
                return j + 1;
            }
        }
        j += 1;
    }
    bytes.len()
}

pub fn check_misplaced_lazy_import(
    source: &str,
    filename: &str,
    mode: Mode,
) -> Result<(), CompileError> {
    if !matches!(mode, Mode::Exec | Mode::Single | Mode::BlockExpr) {
        return Ok(());
    }
    misplaced_from_lazy_import(source, filename)
}
