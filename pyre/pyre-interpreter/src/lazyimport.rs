//! PEP 810 lazy imports (`lazyimportobject.c`, `ceval.c` `_PyEval_LazyImportName`,
//! `import.c` `_PyImport_LazyImportModuleLevelObject`).

use pyre_object::*;
use rustpython_compiler::core::{PositionEncoding, SourceFileBuilder};
use rustpython_compiler_core::bytecode::{CodeObject, CodeUnit, Instruction};

use crate::compile::codegen::error::{CodegenError, CodegenErrorType};
use crate::compile::{CompileError, Mode, ast, parser};
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
    Ok(w_lazy_import_new(w_builtins, w_from, w_attr))
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
            .lazy_imports_resolving
            .borrow()
            .iter()
            .any(|&p| std::ptr::eq(p, op))
    }
}

fn resolving_push(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if !ec.is_null() {
        unsafe { (*ec).lazy_imports_resolving.borrow_mut().push(op) };
    }
}

fn resolving_pop(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return;
    }
    let mut stack = unsafe { (*ec).lazy_imports_resolving.borrow_mut() };
    if let Some(i) = stack.iter().rposition(|&p| std::ptr::eq(p, op)) {
        stack.remove(i);
    }
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
    result
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
    lazy_import = pyre_object::gc_roots::shadow_stack_get(lazy_slot);
    first = pyre_object::gc_roots::shadow_stack_get(first_slot);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
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

/// `_PyImport_ClearLazySubmodule`. `bind` is `_imp._set_lazy_attributes`'s 0
/// or the completed-import 1: a true bind may copy the loaded child onto
/// an already-finished parent.
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
                    crate::baseobjspace::setitem(
                        pyre_object::gc_roots::shadow_stack_get(dict_slot),
                        w_str_new(&child_s),
                        pyre_object::gc_roots::shadow_stack_get(value_slot),
                    )?;
                }
            }
        }
    }
    if initializing {
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
            w_str_new(&child_s),
            w_none(),
        )?;
    } else {
        let _ = crate::baseobjspace::delitem(
            pyre_object::gc_roots::shadow_stack_get(pending_set_slot),
            w_str_new(&child_s),
        );
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

/// `_PyImport_TryLoadLazySubmodule` (simplified).
pub fn try_load_lazy_submodule(
    module: PyObjectRef,
    attr_name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { !is_module(module) } {
        return Ok(None);
    }
    let mut dict = unsafe { w_module_get_w_dict(module) };
    if dict.is_null() {
        return Ok(None);
    }
    let Some(mod_name) = pyre_object::with_roots!(dict => {
        crate::baseobjspace::finditem_str(dict, "__name__")
    })?
    else {
        return Ok(None);
    };
    let Some(mod_name_s) = str_value(mod_name) else {
        return Ok(None);
    };
    let pending = crate::module::sys::state::lazy_pending_submodules();
    if pending.is_null() {
        return Ok(None);
    }
    let Some(pending_set) = pyre_object::with_roots!(dict => {
        crate::baseobjspace::finditem_str(pending, mod_name_s)
    })?
    else {
        return Ok(None);
    };
    let mut pending_set = pending_set;
    let Some(_source) = pyre_object::with_roots!(dict, pending_set => {
        crate::baseobjspace::finditem_str(pending_set, attr_name)
    })?
    else {
        return Ok(None);
    };
    let full_name = format!("{mod_name_s}.{attr_name}");
    if let Some(loaded) = pyre_object::with_roots!(dict, pending_set => {
        crate::importing::sys_modules_entry(&full_name)
    }) && unsafe { !is_none(loaded) }
    {
        let mut loaded = loaded;
        let _ = pyre_object::with_roots!(loaded, pending_set => {
            crate::baseobjspace::setitem(dict, w_str_new(attr_name), loaded)
        });
        let _ = pyre_object::with_roots!(loaded => {
            crate::baseobjspace::delitem(pending_set, w_str_new(attr_name))
        });
        return Ok(Some(loaded));
    }
    Ok(None)
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

fn lazy_modules_add(mut name: PyObjectRef) -> Result<(), crate::PyError> {
    if let Some(name_s) = str_value(name)
        && let Some(existing) = pyre_object::with_roots!(name => {
            crate::importing::sys_modules_entry(name_s)
        })
        && unsafe { is_module(existing) }
    {
        return Ok(());
    }
    let set = crate::module::sys::state::lazy_modules();
    if !set.is_null() {
        unsafe { w_set_add(set, name) };
    }
    Ok(())
}

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
    source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let pending_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pending);
    pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
    let mut end = name_s.len();
    while let Some(dot) = name_s[..end].rfind('.') {
        let parent = &name_s[..dot];
        let child = &name_s[dot + 1..end];
        pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
        let mut submodules = match crate::baseobjspace::finditem_str(pending, parent)? {
            Some(d) => d,
            None => {
                let mut d = w_dict_new();
                pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
                source = pyre_object::gc_roots::shadow_stack_get(source_slot);
                let parent_key = pyre_object::with_roots!(d, pending, source => w_str_new(parent));
                pyre_object::with_roots!(d, pending, source => {
                    crate::baseobjspace::setitem(pending, parent_key, d)
                })?;
                d
            }
        };
        pending = pyre_object::gc_roots::shadow_stack_get(pending_slot);
        source = pyre_object::gc_roots::shadow_stack_get(source_slot);
        let child_key = pyre_object::with_roots!(pending, source, submodules => w_str_new(child));
        crate::baseobjspace::setitem(submodules, child_key, source)?;
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
    lazy_modules_add(abs_name)?;
    fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
    abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
    if unsafe { is_tuple(fromlist) && w_tuple_len(fromlist) > 0 } {
        let n = unsafe { w_tuple_len(fromlist) };
        for i in 0..n {
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
            let item = unsafe { w_tuple_getitem(fromlist, i as i64) }.unwrap_or(PY_NULL);
            let abs = str_value(abs_name).unwrap_or("").to_owned();
            let from = str_value(item).unwrap_or("").to_owned();
            let mut fromname = w_str_new(&format!("{abs}.{from}"));
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
            let fromname_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(fromname);
            fromname = pyre_object::gc_roots::shadow_stack_get(fromname_slot);
            lazy_modules_add(fromname)?;
            fromname = pyre_object::gc_roots::shadow_stack_get(fromname_slot);
            fromlist = pyre_object::gc_roots::shadow_stack_get(fromlist_slot);
            abs_name = pyre_object::gc_roots::shadow_stack_get(abs_name_slot);
            register_lazy_on_parent(fromname, w_none())?;
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

// ── compile-wrapper: syntax checks + IMPORT_NAME bit0 ──────────────

#[derive(Clone, Copy)]
struct LazyCheckCtx {
    in_function: bool,
    in_class: bool,
    in_try: bool,
}

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

fn validate_lazy_stmts(
    stmts: &[ast::Stmt],
    ctx: LazyCheckCtx,
    source: &str,
    filename: &str,
) -> Result<(), CompileError> {
    for stmt in stmts {
        match stmt {
            ast::Stmt::Import(node) if node.is_lazy => {
                check_lazy_context(ctx, source, filename, node.range, "import")?;
            }
            ast::Stmt::ImportFrom(node) if node.is_lazy => {
                let module = node.module.as_ref().map(|s| s.as_str()).unwrap_or("");
                if node.level == 0 && module == "__future__" {
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        node.range,
                        "lazy from __future__ import is not allowed".to_string(),
                    ));
                }
                check_lazy_context(ctx, source, filename, node.range, "from ... import")?;
                if node.names.first().is_some_and(|n| n.name.as_str() == "*") {
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        node.range,
                        "lazy from ... import * is not allowed".to_string(),
                    ));
                }
            }
            ast::Stmt::FunctionDef(node) => {
                validate_lazy_stmts(
                    &node.body,
                    LazyCheckCtx {
                        in_function: true,
                        in_class: false,
                        in_try: false,
                    },
                    source,
                    filename,
                )?;
            }
            ast::Stmt::ClassDef(node) => {
                validate_lazy_stmts(
                    &node.body,
                    LazyCheckCtx {
                        in_function: false,
                        in_class: true,
                        in_try: false,
                    },
                    source,
                    filename,
                )?;
            }
            ast::Stmt::Try(node) => {
                let try_ctx = LazyCheckCtx {
                    in_try: true,
                    ..ctx
                };
                validate_lazy_stmts(&node.body, try_ctx, source, filename)?;
                for handler in &node.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    validate_lazy_stmts(&h.body, try_ctx, source, filename)?;
                }
                validate_lazy_stmts(&node.orelse, try_ctx, source, filename)?;
                validate_lazy_stmts(&node.finalbody, try_ctx, source, filename)?;
            }
            ast::Stmt::If(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                for clause in &node.elif_else_clauses {
                    validate_lazy_stmts(&clause.body, ctx, source, filename)?;
                }
            }
            ast::Stmt::For(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                validate_lazy_stmts(&node.orelse, ctx, source, filename)?;
            }
            ast::Stmt::While(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                validate_lazy_stmts(&node.orelse, ctx, source, filename)?;
            }
            ast::Stmt::With(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
            }
            ast::Stmt::Match(node) => {
                for case in &node.cases {
                    validate_lazy_stmts(&case.body, ctx, source, filename)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_lazy_context(
    ctx: LazyCheckCtx,
    source: &str,
    filename: &str,
    range: ruff_text_size::TextRange,
    import_type: &str,
) -> Result<(), CompileError> {
    let message = if ctx.in_try {
        format!("lazy {import_type} not allowed inside try/except blocks")
    } else if ctx.in_function {
        format!("lazy {import_type} not allowed inside functions")
    } else if ctx.in_class {
        format!("lazy {import_type} not allowed inside classes")
    } else {
        return Ok(());
    };
    Err(lazy_syntax_error(source, filename, range, message))
}

fn collect_import_name_lazy(stmts: &[ast::Stmt], out: &mut Vec<bool>) {
    for stmt in stmts {
        match stmt {
            ast::Stmt::Import(node) => {
                for _ in &node.names {
                    out.push(node.is_lazy);
                }
            }
            ast::Stmt::ImportFrom(node) => out.push(node.is_lazy),
            ast::Stmt::FunctionDef(_) | ast::Stmt::ClassDef(_) => {}
            ast::Stmt::Try(node) => {
                collect_import_name_lazy(&node.body, out);
                for handler in &node.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_import_name_lazy(&h.body, out);
                }
                collect_import_name_lazy(&node.orelse, out);
                collect_import_name_lazy(&node.finalbody, out);
            }
            ast::Stmt::If(node) => {
                collect_import_name_lazy(&node.body, out);
                for clause in &node.elif_else_clauses {
                    collect_import_name_lazy(&clause.body, out);
                }
            }
            ast::Stmt::For(node) => {
                collect_import_name_lazy(&node.body, out);
                collect_import_name_lazy(&node.orelse, out);
            }
            ast::Stmt::While(node) => {
                collect_import_name_lazy(&node.body, out);
                collect_import_name_lazy(&node.orelse, out);
            }
            ast::Stmt::With(node) => collect_import_name_lazy(&node.body, out),
            ast::Stmt::Match(node) => {
                for case in &node.cases {
                    collect_import_name_lazy(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

pub fn patch_import_name_bit0(code: &mut CodeObject, flags: &[bool]) {
    let mut flag_i = 0usize;
    let mut i = 0usize;
    let len = code.instructions.len();
    let ptr = code.instructions.units_ptr() as *mut CodeUnit;
    while i < len {
        let unit = code.instructions[i];
        if matches!(unit.op, Instruction::ImportName { .. }) {
            if flag_i < flags.len() && flags[flag_i] {
                let arg = u8::from(unit.arg);
                if arg & 2 == 0 {
                    unsafe {
                        (*ptr.add(i)).arg = (arg | 1).into();
                    }
                }
            }
            flag_i += 1;
        }
        i += 1 + unit.op.cache_entries();
    }
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
            let start = i;
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
                        ruff_text_size::TextSize::from(start as u32),
                        ruff_text_size::TextSize::from((after + 6).min(bytes.len()) as u32),
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

pub fn collect_lazy_import_flags(
    source: &str,
    filename: &str,
    mode: Mode,
) -> Result<Vec<bool>, CompileError> {
    if !matches!(mode, Mode::Exec | Mode::Single | Mode::BlockExpr) {
        return Ok(Vec::new());
    }
    misplaced_from_lazy_import(source, filename)?;
    let parsed = parser::parse_unchecked(source, parser::Mode::Module.into());
    flags_from_mod(parsed.syntax(), source, filename)
}

pub fn flags_from_mod(
    module: &ast::Mod,
    source: &str,
    filename: &str,
) -> Result<Vec<bool>, CompileError> {
    let body: &[ast::Stmt] = match module {
        ast::Mod::Module(m) => &m.body,
        ast::Mod::Expression(_) => return Ok(Vec::new()),
    };
    validate_lazy_stmts(
        body,
        LazyCheckCtx {
            in_function: false,
            in_class: false,
            in_try: false,
        },
        source,
        filename,
    )?;
    let mut flags = Vec::new();
    collect_import_name_lazy(body, &mut flags);
    Ok(flags)
}
