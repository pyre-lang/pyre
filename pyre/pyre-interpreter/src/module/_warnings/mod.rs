//! `_warnings` — structural port of `pypy/module/_warnings/interp_warnings.py`.

use crate::PyError;
use pyre_object::*;

const VERSION_ATTR: &str = "_filters_version_state";

pub(crate) fn pin_root_slot(value: PyObjectRef) -> usize {
    let _ = pyre_object::gc_roots::pin_root(value);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

fn module_attr(module: &str, name: &str) -> Option<PyObjectRef> {
    let w_module = crate::importing::get_sys_module(module)?;
    crate::baseobjspace::getattr_str(w_module, name).ok()
}

fn warnings_attr(name: &str) -> Option<PyObjectRef> {
    module_attr("warnings", name)
}

fn import_module(name: &str) -> Result<PyObjectRef, PyError> {
    if let Some(module) = crate::importing::get_sys_module(name) {
        return Ok(module);
    }
    // The fresh `"*"` has to be rooted across the list's own allocation.
    let w_fromlist = {
        let mut items = pyre_object::gc_roots::RootedItems::new();
        items.push(w_str_new("*"));
        w_list_new(items.take())
    };
    crate::importing::importhook(
        rustpython_wtf8::Wtf8::new(name),
        w_none(),
        w_fromlist,
        0,
        crate::call::getexecutioncontext(),
    )?;
    crate::importing::get_sys_module(name).ok_or_else(|| {
        PyError::new(
            crate::PyErrorKind::ImportError,
            format!("No module named '{name}'"),
        )
    })
}

/// The namespace holding the `State` fields, captured when the module is
/// built.  Upstream reaches them as plain attributes of a translated-away
/// `State` instance that `space.fromcache` hands out and that no Python name
/// can reach.  pyre keeps them in the module's own namespace, so holding the
/// dict itself is what gives the same reachability: resolving
/// `sys.modules['_warnings']` per access would put a lookup on the path of
/// every warning and would lose the state the moment app code rebinds the
/// name.  Module namespaces are Box-immortal, so the captured pointer stays
/// valid and unmoved.
static STATE_NS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The captured `State` namespace, or `PY_NULL` before the module is built —
/// the single read seam every traced reader of `STATE_NS` goes through.
///
/// `setup_after_space_initialization` stamps the pointer at runtime, so it is
/// not a build-time constant and the JIT residualizes the read rather than
/// folding whatever the build process happened to see
/// (`@dont_look_inside`, the `importing::sys_modules_dict` shape).  The
/// `-> PyObjectRef` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub(crate) fn state_ns() -> PyObjectRef {
    STATE_NS.load(std::sync::atomic::Ordering::Acquire) as PyObjectRef
}

/// One `State` field.  Errs only before the module is built, which
/// `state_is_readable` screens for.
fn native_attr(name: &str) -> Result<PyObjectRef, PyError> {
    let ns = state_ns();
    if !ns.is_null()
        && let Some(value) = unsafe { w_dict_getitem_str(ns, name) }
    {
        return Ok(value);
    }
    Err(PyError::runtime_error(format!(
        "_warnings.{name} is not initialized"
    )))
}

fn native_store(name: &str, value: PyObjectRef) -> Result<(), PyError> {
    let ns = state_ns();
    if ns.is_null() {
        return Err(PyError::runtime_error("_warnings is not initialized"));
    }
    crate::module_ns_store(ns, name, value);
    Ok(())
}

fn get_default_action() -> Result<PyObjectRef, PyError> {
    if let Some(action) = warnings_attr("defaultaction") {
        native_store("_defaultaction", action)?;
        Ok(action)
    } else {
        native_attr("_defaultaction")
    }
}

fn get_once_registry() -> Result<PyObjectRef, PyError> {
    if let Some(registry) = warnings_attr("onceregistry") {
        native_store("_onceregistry", registry)?;
        Ok(registry)
    } else {
        native_attr("_onceregistry")
    }
}

fn create_filter(category: PyObjectRef, action: &str, module: Option<&str>) -> PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_str_new_managed(action));
    fields.push(w_none());
    fields.push(category);
    fields.push(module.map(w_str_new_managed).unwrap_or_else(w_none));
    fields.push(w_int_new(0));
    w_tuple_new(fields.take())
}

fn warning_class(name: &str) -> PyObjectRef {
    crate::builtins::lookup_exc_class(name)
        .unwrap_or_else(|| panic!("{name} must be installed before _warnings"))
}

fn current_version() -> Result<PyObjectRef, PyError> {
    native_attr(VERSION_ATTR)
}

fn new_version() -> PyObjectRef {
    // PyPy `State.filters_mutated`: `space.call_function(space.w_object)`.
    // The registry compares this opaque sentinel by identity, never value.
    w_instance_new(crate::typedef::w_object())
}

fn filters_mutated_impl() -> Result<(), PyError> {
    native_store(VERSION_ATTR, new_version())
}

fn get_category(message: PyObjectRef, category: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let warning = warning_class("Warning");
    // `space.isinstance_w` — the type-only test, not the `__instancecheck__`
    // protocol the `isinstance()` builtin runs.
    if unsafe { crate::baseobjspace::isinstance_w(message, warning) } {
        return crate::typedef::r#type(message)
            .map(|p| p.as_ptr())
            .ok_or_else(|| PyError::type_error("warning instance has no type"));
    }
    let mut category = if category.is_null() || unsafe { is_none(category) } {
        warning_class("UserWarning")
    } else {
        category
    };
    // `interp_warnings.py` `get_category` raises "category is not a subclass of Warning"
    // from inside its own `try`, so the enclosing `except OperationError` catches
    // that raise and re-reports it in the `'%T'` form.  A plain False and a
    // subclass check that itself fails therefore reach the caller identically.
    match pyre_object::with_roots!(category => crate::baseobjspace::issubclass(category, warning)) {
        Ok(true) => Ok(category),
        Ok(false) | Err(_) => Err(PyError::type_error(format!(
            "category must be a Warning subclass, not '{}'",
            crate::baseobjspace::object_functionstr_type_name(category)
        ))),
    }
}

fn is_internal_frame(frame: *mut crate::PyFrame) -> bool {
    if frame.is_null() {
        return false;
    }
    let filename = unsafe { &*crate::pyframe::pyframe_get_pycode(&*frame) }
        .source_path
        .as_str();
    filename.contains("importlib") && filename.contains("_bootstrap")
}

fn filename_has_prefix(frame: *mut crate::PyFrame, skip_file_prefixes: &[String]) -> bool {
    if frame.is_null() {
        return false;
    }
    let filename = unsafe { &*crate::pyframe::pyframe_get_pycode(&*frame) }
        .source_path
        .as_str();
    skip_file_prefixes
        .iter()
        .any(|prefix| filename.starts_with(prefix))
}

fn get_frame(mut stacklevel: i64, skip_file_prefixes: &[String]) -> *mut crate::PyFrame {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return std::ptr::null_mut();
    }
    let mut frame = unsafe { (*ec).gettopframe_nohidden() };
    if stacklevel <= 0 || is_internal_frame(frame) {
        while stacklevel > 1 && !frame.is_null() {
            frame = crate::executioncontext::ExecutionContext::getnextframe_nohidden(frame);
            stacklevel -= 1;
        }
    } else {
        while stacklevel > 1 && !frame.is_null() {
            loop {
                frame = crate::executioncontext::ExecutionContext::getnextframe_nohidden(frame);
                if frame.is_null()
                    || (!is_internal_frame(frame)
                        && !filename_has_prefix(frame, skip_file_prefixes))
                {
                    break;
                }
            }
            stacklevel -= 1;
        }
    }
    frame
}

fn setup_context(
    stacklevel: i64,
    skip_file_prefixes: &[String],
) -> (PyObjectRef, i64, PyObjectRef, PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let frame = get_frame(stacklevel, skip_file_prefixes);
    let (filename, lineno, globals) = if frame.is_null() {
        let globals = crate::importing::get_interpreter_sys_module()
            .map(|module| unsafe { w_module_get_w_dict(module) })
            .unwrap_or_else(w_dict_new);
        let globals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(globals);
        (
            w_str_new_managed("<sys>"),
            0,
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
        )
    } else {
        let frame = unsafe { &*frame };
        (
            unsafe { crate::pycode::w_code_filename_obj(frame.fget_f_code()) },
            frame.get_last_lineno() as i64,
            frame.get_w_globals(),
        )
    };
    let pair = pyre_object::gc_roots::pin_roots(&[filename, globals]);
    let filename_slot = pair;
    let globals_slot = pair + 1;
    let registry = unsafe {
        w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            "__warningregistry__",
        )
    }
    .unwrap_or_else(|| {
        let registry = w_dict_new();
        unsafe {
            w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(globals_slot),
                "__warningregistry__",
                registry,
            )
        };
        registry
    });
    let registry_slot = pin_root_slot(registry);
    let module = unsafe {
        w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            "__name__",
        )
    }
    .unwrap_or_else(|| w_str_new_managed("<string>"));
    let module_slot = pin_root_slot(module);
    (
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
        lineno,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
    )
}

fn check_matched(filter: PyObjectRef, value: PyObjectRef) -> Result<bool, PyError> {
    if filter.is_null() || unsafe { is_none(filter) } {
        return Ok(true);
    }
    if unsafe { pyre_object::pyobject::is_exact_type(filter, &pyre_object::STR_TYPE) } {
        if unsafe { !is_str(value) } {
            return Ok(false);
        }
        return Ok(unsafe { w_str_get_wtf8(filter) == w_str_get_wtf8(value) });
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let filter_slot = pin_root_slot(filter);
    let value_slot = pin_root_slot(value);
    let matcher = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(filter_slot),
        "match",
    )?;
    let matcher_slot = pin_root_slot(matcher);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(matcher_slot),
        &[pyre_object::gc_roots::shadow_stack_get(value_slot)],
    )?;
    let result_slot = pin_root_slot(result);
    crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(result_slot))
}

fn get_filter(
    category: PyObjectRef,
    text: PyObjectRef,
    lineno: i64,
    module: PyObjectRef,
) -> Result<(String, PyObjectRef), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pin_root_slot(category);
    let text_slot = pin_root_slot(text);
    let module_slot = pin_root_slot(module);
    let filters = if let Some(filters) = warnings_attr("filters") {
        native_store("filters", filters)?;
        filters
    } else {
        native_attr("filters")?
    };
    let filters_slot = pin_root_slot(filters);
    // PyPy `space.fixedview(w_filters)` snapshots any iterable before the
    // loop because user code may mutate `warnings.filters` while matching.
    let items =
        crate::baseobjspace::fixedview(pyre_object::gc_roots::shadow_stack_get(filters_slot), -1)?;
    let item_slots: Vec<_> = items.into_iter().map(pin_root_slot).collect();
    for item_slot in item_slots {
        let item = pyre_object::gc_roots::shadow_stack_get(item_slot);
        let fields = crate::baseobjspace::fixedview(item, 5)?;
        let field_slots: Vec<_> = fields.into_iter().map(pin_root_slot).collect();
        let filter_lineno =
            crate::baseobjspace::int_w(pyre_object::gc_roots::shadow_stack_get(field_slots[4]))?;
        if check_matched(
            pyre_object::gc_roots::shadow_stack_get(field_slots[1]),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
        )? && check_matched(
            pyre_object::gc_roots::shadow_stack_get(field_slots[3]),
            pyre_object::gc_roots::shadow_stack_get(module_slot),
        )? && crate::baseobjspace::issubclass(
            pyre_object::gc_roots::shadow_stack_get(category_slot),
            pyre_object::gc_roots::shadow_stack_get(field_slots[2]),
        )? && (filter_lineno == 0 || filter_lineno == lineno)
        {
            return Ok((
                crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(
                    field_slots[0],
                ))?
                .to_string(),
                pyre_object::gc_roots::shadow_stack_get(item_slot),
            ));
        }
    }
    let action = get_default_action()?;
    Ok((crate::baseobjspace::text_w(action)?.to_string(), PY_NULL))
}

fn already_warned(
    registry: PyObjectRef,
    key: PyObjectRef,
    should_set: bool,
) -> Result<bool, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let registry_slot = pin_root_slot(registry);
    let key_slot = pin_root_slot(key);
    let version = current_version()?;
    let version_slot = pin_root_slot(version);
    let registry_version = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        "version",
    )?;
    if registry_version != Some(pyre_object::gc_roots::shadow_stack_get(version_slot)) {
        let clear = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            "clear",
        )?;
        let clear_slot = pin_root_slot(clear);
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(clear_slot),
            &[],
        )?;
        let version_key_slot =
            pin_root_slot(pyre_object::unicodeobject::intern_str_value("version"));
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(version_key_slot),
            pyre_object::gc_roots::shadow_stack_get(version_slot),
        )?;
    } else if let Some(value) = crate::baseobjspace::finditem(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )? && crate::baseobjspace::is_true(value)?
    {
        return Ok(true);
    }
    if should_set {
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            w_bool_from(true),
        )?;
    }
    Ok(false)
}

fn update_registry(
    registry: PyObjectRef,
    text: PyObjectRef,
    category: PyObjectRef,
) -> Result<bool, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let registry_slot = pin_root_slot(registry);
    let text_slot = pin_root_slot(text);
    let category_slot = pin_root_slot(category);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::gc_roots::shadow_stack_get(text_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(category_slot));
    let key = w_tuple_new(fields.take());
    let key_slot = pin_root_slot(key);
    already_warned(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
        true,
    )
}

/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the
/// line is assembled in a `Wtf8Buf`, which has no counterpart in the
/// immutable lifted string model, and the whole body is a write to
/// `sys.stderr` — an opaque host action with nothing for a trace to reuse.
#[majit_macros::dont_look_inside]
pub(crate) fn show_warning(
    message: PyObjectRef,
    text: PyObjectRef,
    category: PyObjectRef,
    filename: PyObjectRef,
    lineno: i64,
    source_line: PyObjectRef,
    source: PyObjectRef,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let message_slot = pin_root_slot(message);
    let text_slot = pin_root_slot(text);
    let category_slot = pin_root_slot(category);
    let filename_slot = pin_root_slot(filename);
    let source_line_slot = pin_root_slot(source_line);
    let source_slot = pin_root_slot(source);
    if let Some(show) = warnings_attr("_showwarnmsg") {
        let show_slot = pin_root_slot(show);
        if !crate::baseobjspace::callable_w(pyre_object::gc_roots::shadow_stack_get(show_slot)) {
            return Err(PyError::type_error(
                "warnings._showwarnmsg() must be set to a callable",
            ));
        }
        let cls = warnings_attr("WarningMessage")
            .ok_or_else(|| PyError::runtime_error("unable to get warnings.WarningMessage"))?;
        let cls_slot = pin_root_slot(cls);
        let lineno_slot = pin_root_slot(w_int_new(lineno));
        let warning_message = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
            &[
                pyre_object::gc_roots::shadow_stack_get(message_slot),
                pyre_object::gc_roots::shadow_stack_get(category_slot),
                pyre_object::gc_roots::shadow_stack_get(filename_slot),
                pyre_object::gc_roots::shadow_stack_get(lineno_slot),
                w_none(),
                w_none(),
                if pyre_object::gc_roots::shadow_stack_get(source_slot).is_null() {
                    w_none()
                } else {
                    pyre_object::gc_roots::shadow_stack_get(source_slot)
                },
            ],
        )?;
        let warning_message_slot = pin_root_slot(warning_message);
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(show_slot),
            &[pyre_object::gc_roots::shadow_stack_get(
                warning_message_slot,
            )],
        )?;
        return Ok(());
    }
    let name = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        "__name__",
    )?;
    let name_slot = pin_root_slot(name);
    // Assembled in WTF-8: the message is whatever `warn()` was given, so a
    // filename or text carrying a lone surrogate is written out as-is and
    // stderr's error handler decides how it appears.
    let mut line = rustpython_wtf8::Wtf8Buf::new();
    line.push_wtf8(crate::baseobjspace::text_wtf8_w(
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
    )?);
    line.push_str(&format!(":{lineno}: "));
    line.push_wtf8(crate::baseobjspace::text_wtf8_w(
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    )?);
    line.push_str(": ");
    line.push_wtf8(crate::baseobjspace::text_wtf8_w(
        pyre_object::gc_roots::shadow_stack_get(text_slot),
    )?);
    line.push_str("\n");
    // The interpreter-owned `sys`, not `sys.modules["sys"]`: upstream reads
    // `space.sys` and a program that rebinds or deletes the mapping entry
    // still gets its warnings written.
    if let Some(sys) = crate::importing::get_interpreter_sys_module()
        && let Ok(stderr) = crate::baseobjspace::getattr_str(sys, "stderr")
        && !unsafe { is_none(stderr) }
        && let Ok(write) = crate::baseobjspace::getattr_str(stderr, "write")
    {
        let write_slot = pin_root_slot(write);
        let line_slot = pin_root_slot(pyre_object::w_str_from_wtf8_managed(line));
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(write_slot),
            &[pyre_object::gc_roots::shadow_stack_get(line_slot)],
        )?;
        let source_line = pyre_object::gc_roots::shadow_stack_get(source_line_slot);
        let source_line = if source_line.is_null() || unsafe { is_none(source_line) } {
            if let Ok(linecache) = import_module("linecache") {
                crate::baseobjspace::getattr_str(linecache, "getline")
                    .and_then(|getline| {
                        crate::call::call_function_impl_result(
                            getline,
                            &[
                                pyre_object::gc_roots::shadow_stack_get(filename_slot),
                                pyre_object::gc_roots::shadow_stack_get(pin_root_slot(w_int_new(
                                    lineno,
                                ))),
                            ],
                        )
                    })
                    .ok()
            } else {
                None
            }
        } else {
            Some(source_line)
        };
        if let Some(source_line) = source_line {
            let source_line_slot = pin_root_slot(source_line);
            let stripped = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(source_line_slot),
                "strip",
            )
            .and_then(|strip| crate::call::call_function_impl_result(strip, &[]))?;
            let stripped = crate::baseobjspace::text_w(stripped)?;
            let visible = stripped.trim_start_matches([' ', '\t', '\u{000c}']);
            if !visible.is_empty() {
                crate::call::call_function_impl_result(
                    pyre_object::gc_roots::shadow_stack_get(write_slot),
                    &[w_str_new_managed(&format!("  {visible}\n"))],
                )?;
            }
        }
    }
    Ok(())
}

/// One-word residual-call ABI for [`show_warning`].
pub extern "C" fn show_warning_jit_abi(
    message: PyObjectRef,
    text: PyObjectRef,
    category: PyObjectRef,
    filename: PyObjectRef,
    lineno: i64,
    source_line: PyObjectRef,
    source: PyObjectRef,
) -> i64 {
    match show_warning(
        message,
        text,
        category,
        filename,
        lineno,
        source_line,
        source,
    ) {
        Ok(()) => 0,
        Err(error) => crate::runtime_ops::jit_publish_residual_error(error),
    }
}

fn get_source_line(module_globals: PyObjectRef, lineno: i64) -> Result<PyObjectRef, PyError> {
    if module_globals.is_null() || unsafe { is_none(module_globals) } {
        return Ok(PY_NULL);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let globals_slot = pin_root_slot(module_globals);
    // `importlib/_bootstrap_external.py` `_bless_my_loader` reconciles
    // `__loader__` against `__spec__.loader` before either is used: it raises
    // when the globals name no loader at all, and warns `DeprecationWarning`
    // when only the deprecated `__loader__` names one.  `interp_warnings.py`
    // `get_source_line`
    // reads `__loader__` straight out of the globals; that module predates the
    // reconciliation and carries no counterpart for it, so this answers to the
    // 3.14 stdlib shipped here.
    let Some(external) = crate::importing::importlib_bootstrap_external_module() else {
        // A warning raised before the external bootstrap is wired has no loader
        // to ask for a source line.
        return Ok(PY_NULL);
    };
    let external_slot = pin_root_slot(external);
    let bless = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(external_slot),
        "_bless_my_loader",
    )?;
    let bless_slot = pin_root_slot(bless);
    // `None` when the globals name no loader; the `get_source` lookup below then
    // takes the `AttributeError` arm and yields no source line.
    let loader = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(bless_slot),
        &[pyre_object::gc_roots::shadow_stack_get(globals_slot)],
    )?;
    let loader_slot = pin_root_slot(loader);
    let Some(module_name) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(globals_slot),
        "__name__",
    )?
    else {
        return Ok(PY_NULL);
    };
    let module_name_slot = pin_root_slot(module_name);
    let get_source = match crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(loader_slot),
        "get_source",
    ) {
        Ok(get_source) => get_source,
        Err(err) if matches!(err.kind, crate::PyErrorKind::AttributeError) => return Ok(PY_NULL),
        Err(err) => return Err(err),
    };
    let get_source_slot = pin_root_slot(get_source);
    let source = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(get_source_slot),
        &[pyre_object::gc_roots::shadow_stack_get(module_name_slot)],
    )?;
    if unsafe { is_none(source) } {
        return Ok(PY_NULL);
    }
    let source_slot = pin_root_slot(source);
    // `interp_warnings.py` `get_source_line` splits through the `str` type
    // itself, so a `str`
    // subclass that overrides `splitlines` does not get to decide what the line
    // list is, and the descriptor's own receiver check rejects a `get_source`
    // result that is not a `str` — the same type error `PyUnicode_Splitlines`
    // reports for that input.
    let splitlines = crate::baseobjspace::getattr_str(
        crate::typedef::gettypeobject(&pyre_object::STR_TYPE),
        "splitlines",
    )?;
    let splitlines_slot = pin_root_slot(splitlines);
    let lines = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(splitlines_slot),
        &[pyre_object::gc_roots::shadow_stack_get(source_slot)],
    )?;
    let lines_slot = pin_root_slot(lines);
    match crate::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(lines_slot),
        pyre_object::gc_roots::shadow_stack_get(pin_root_slot(w_int_new(lineno - 1))),
    ) {
        Ok(line) => Ok(line),
        Err(err)
            if matches!(
                err.kind,
                crate::PyErrorKind::TypeError | crate::PyErrorKind::IndexError
            ) =>
        {
            Ok(PY_NULL)
        }
        Err(err) => Err(err),
    }
}

pub(crate) fn do_warn_explicit(
    category: PyObjectRef,
    message: PyObjectRef,
    filename: PyObjectRef,
    lineno: i64,
    module: PyObjectRef,
    registry: PyObjectRef,
    source_line: PyObjectRef,
    source: PyObjectRef,
) -> Result<(), PyError> {
    // RPython keeps these values in livevars across every allocating helper.
    // Native Rust locals are invisible to the moving collector, so mirror
    // that lifetime with the established temporary shadow-stack bracket.
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pin_root_slot(category);
    let input_message_slot = pin_root_slot(message);
    let filename_slot = pin_root_slot(filename);
    let input_module_slot = pin_root_slot(module);
    let registry_slot = pin_root_slot(registry);
    let source_line_slot = pin_root_slot(source_line);
    let source_slot = pin_root_slot(source);

    let input_module = pyre_object::gc_roots::shadow_stack_get(input_module_slot);
    // A warning issued this late during shutdown has lost the `warnings`
    // module, so no filter can be matched and there is nothing safe to do
    // with it.
    if !input_module.is_null() && unsafe { is_none(input_module) } {
        return Ok(());
    }
    let registry = pyre_object::gc_roots::shadow_stack_get(registry_slot);
    if !registry.is_null()
        && unsafe { !crate::baseobjspace::isinstance_dict_w(registry) && !is_none(registry) }
    {
        return Err(PyError::type_error(
            "'registry' must be a dict or None".to_owned(),
        ));
    }
    let module = if input_module.is_null() {
        let filename = pyre_object::gc_roots::shadow_stack_get(filename_slot);
        if unsafe { !is_str(filename) } {
            // Preserve `space.fsencode_w`'s TypeError for non-text filenames.
            let _ = crate::baseobjspace::text_w(filename)?;
            unreachable!();
        }
        let filename_text = unsafe { w_str_get_wtf8(filename) };
        let filename_bytes = filename_text.as_bytes();
        if filename_bytes.is_empty() {
            w_str_new_managed("<unknown>")
        } else if filename_bytes.ends_with(b".py") {
            let stem = rustpython_wtf8::Wtf8Buf::from_bytes(
                filename_bytes[..filename_bytes.len() - 3].to_vec(),
            )
            .expect("removing an ASCII suffix preserves WTF-8");
            w_str_from_wtf8_managed(stem)
        } else {
            filename
        }
    } else {
        input_module
    };
    let module_slot = pin_root_slot(module);
    let warning = warning_class("Warning");
    let input_message = pyre_object::gc_roots::shadow_stack_get(input_message_slot);
    let category = pyre_object::gc_roots::shadow_stack_get(category_slot);
    let (text, message, category) =
        if unsafe { crate::baseobjspace::isinstance_w(input_message, warning) } {
            (
                crate::builtins::builtin_str(&[pyre_object::gc_roots::shadow_stack_get(
                    input_message_slot,
                )])?,
                pyre_object::gc_roots::shadow_stack_get(input_message_slot),
                crate::typedef::r#type(pyre_object::gc_roots::shadow_stack_get(input_message_slot))
                    .map(|p| p.as_ptr())
                    .unwrap_or(category),
            )
        } else {
            let input_message = pyre_object::gc_roots::shadow_stack_get(input_message_slot);
            let text = if unsafe { is_str(input_message) || is_bytes(input_message) } {
                input_message
            } else {
                crate::builtins::builtin_str(&[input_message])?
            };
            let text_slot = pin_root_slot(text);
            let instance = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(category_slot),
                &[pyre_object::gc_roots::shadow_stack_get(input_message_slot)],
            )?;
            let text = pyre_object::gc_roots::shadow_stack_get(text_slot);
            (text, instance, category)
        };
    let text_slot = pin_root_slot(text);
    let message_slot = pin_root_slot(message);
    let category_slot = pin_root_slot(category);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::gc_roots::shadow_stack_get(text_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(category_slot));
    fields.push(w_int_new(lineno));
    let key = w_tuple_new(fields.take());
    let key_slot = pin_root_slot(key);
    let registry = pyre_object::gc_roots::shadow_stack_get(registry_slot);
    let has_registry = !registry.is_null() && unsafe { !is_none(registry) };
    if has_registry
        && already_warned(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            false,
        )?
    {
        return Ok(());
    }
    let (action, item) = get_filter(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(text_slot),
        lineno,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    let item_slot = pin_root_slot(item);
    if action == "error" {
        return Err(unsafe {
            PyError::from_exc_object(pyre_object::gc_roots::shadow_stack_get(message_slot))
        });
    }
    if action == "ignore" {
        return Ok(());
    }
    let mut warned = false;
    if action != "always" && action != "all" {
        if has_registry {
            crate::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(registry_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                w_bool_from(true),
            )?;
        }
        if action == "once" {
            let once = if has_registry {
                pyre_object::gc_roots::shadow_stack_get(registry_slot)
            } else {
                get_once_registry()?
            };
            warned = update_registry(
                once,
                pyre_object::gc_roots::shadow_stack_get(text_slot),
                pyre_object::gc_roots::shadow_stack_get(category_slot),
            )?;
        } else if action == "module" {
            if has_registry {
                warned = update_registry(
                    pyre_object::gc_roots::shadow_stack_get(registry_slot),
                    pyre_object::gc_roots::shadow_stack_get(text_slot),
                    pyre_object::gc_roots::shadow_stack_get(category_slot),
                )?;
            }
        } else if action != "default" {
            return Err(PyError::runtime_error(format!(
                "Unrecognized action ({action}) in warnings.filters: {}",
                if pyre_object::gc_roots::shadow_stack_get(item_slot).is_null() {
                    "???"
                } else {
                    "filter item"
                }
            )));
        }
    }
    if !warned {
        show_warning(
            pyre_object::gc_roots::shadow_stack_get(message_slot),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
            pyre_object::gc_roots::shadow_stack_get(category_slot),
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
            lineno,
            pyre_object::gc_roots::shadow_stack_get(source_line_slot),
            pyre_object::gc_roots::shadow_stack_get(source_slot),
        )?;
    }
    Ok(())
}

/// Whether the `State` has been installed yet.
///
/// Upstream builds `space.fromcache(State)` during space construction, so
/// `space.warn` can always match a warning.  pyre builds it with the module,
/// and a warning issued before that point still has to reach stderr rather
/// than raise out of the operator that issued it.
pub fn state_is_readable() -> bool {
    !state_ns().is_null()
}

/// Publish the `State` by creating the module that owns it, for a startup that
/// never imports `_warnings`.
///
/// `moduledef.py setup_after_space_initialization` runs for every space, so
/// upstream's `space.fromcache(State)` is populated before any code runs.
/// pyre publishes it from this module's own init, which makes it depend on
/// something importing `_warnings`.  The native startup does, through the
/// importlib bootstrap; a build that runs no bootstrap does not, and reaches
/// user code with `sys.modules` holding `__main__` and `sys` alone — so every
/// warning it raises takes the unfiltered stderr fallback, for the whole
/// process rather than for the window the fallback was written for.
///
/// Goes through `getbuiltinmodule`, so the module lands in `sys.modules`
/// and a later `import _warnings` binds this one.  A private second namespace
/// would carry a second `filters` list, and the interpreter would keep
/// matching against the one the application never touched.
///
/// Attempted once.  A failure leaves the State absent, which the caller
/// already handles; retrying would pay a whole module init per warning.  The
/// `extra_init` that publishes the State runs inside this call, so a warning
/// raised from within it finds the flag already set and falls back instead of
/// re-entering.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn install_state() -> i64 {
    static TRIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if TRIED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return 0;
    }
    match crate::importing::getbuiltinmodule("_warnings", false, true, std::ptr::null()) {
        Ok(_) => 0,
        Err(_) => 0,
    }
}

/// `interp_warnings.do_warn` — the interpreter-level entry point.
///
/// `category` must already be a `Warning` subclass; `_warnings.warn` runs
/// `get_category` first, and `space.warn` names the class outright.
/// `skip_file_prefixes` reaches `setup_context`'s frame walk; a caller with
/// no prefix to skip passes an empty slice.
///
/// Reachable without `warnings` being imported: the filters, the default
/// action and the once-registry all live on the native module, so an
/// interpreter warning is matched and deduplicated whether or not the Python
/// wrapper has been loaded.
pub fn do_warn(
    message: PyObjectRef,
    category: PyObjectRef,
    stacklevel: i64,
    source: PyObjectRef,
    skip_file_prefixes: &[String],
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let message_slot = pin_root_slot(message);
    let category_slot = pin_root_slot(category);
    let source_slot = pin_root_slot(source);
    let (filename, lineno, module, registry) = setup_context(stacklevel, skip_file_prefixes);
    do_warn_explicit(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(message_slot),
        filename,
        lineno,
        module,
        registry,
        PY_NULL,
        pyre_object::gc_roots::shadow_stack_get(source_slot),
    )
}

fn filters_mutated(_: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    filters_mutated_impl()?;
    Ok(w_none())
}

// Python 3.14's `warnings.py` expects the accelerator to provide the lock
// hooks. PyPy's State has no separate lock, and pyre currently executes
// Python on one interpreter thread (`module/thread/mod.rs`), so there is no
// concurrent access to serialize. Keep these process-wide no-ops rather than
// inventing thread-local ownership for shared warning state.
fn lock_noop(_: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(w_none())
}

crate::py_module! {
    "_warnings",
    interpleveldefs: {
        "_warnings_context" => w_none(),
    },
    inline_functions: {
        fn _warn_unawaited_coroutine(coro: PyObjectRef) -> Result<PyObjectRef, PyError> {
            // module/_warnings/moduledef.py exposes the app-level
            // `app_warnings._warn_unawaited_coroutine`.  The bundled 3.14
            // `_py_warnings` implementation is that same app-level body,
            // including coroutine-origin formatting and `source=coro`.
            let _roots = pyre_object::gc_roots::push_roots();
            let coro_slot = pin_root_slot(coro);
            let w_mod = import_module("_py_warnings")?;
            let module_slot = pin_root_slot(w_mod);
            let w_f = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(module_slot),
                "_warn_unawaited_coroutine",
            )?;
            let function_slot = pin_root_slot(w_f);
            crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(function_slot),
                &[pyre_object::gc_roots::shadow_stack_get(coro_slot)],
            )
        }

        fn warn(
            mut message: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut category: PyObjectRef,
            #[default(1i64)] stacklevel: i64,
            // `source` is positional-or-keyword; only `skip_file_prefixes`
            // sits behind the clinic's `*`.
            #[default(pyre_object::PY_NULL)] mut source: PyObjectRef,
            #[kwonly] #[default(pyre_object::PY_NULL)] skip_file_prefixes: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // CPython 3.14 `_warnings.warn`: `skip_file_prefixes` is a
            // tuple-only fast-path argument.  Its frame walk is the current
            // PyPy `get_frame` walk with matching filenames skipped alongside
            // importlib-internal frames.
            let skip_file_prefixes = if skip_file_prefixes.is_null() {
                Vec::new()
            } else {
                if unsafe { !pyre_object::is_tuple(skip_file_prefixes) } {
                    return Err(PyError::type_error(format!(
                        "warn() argument 'skip_file_prefixes' must be tuple, not {}",
                        crate::type_methods::clinic_arg_type_name(skip_file_prefixes),
                    )));
                }
                let mut prefixes = Vec::new();
                for prefix in pyre_object::with_roots!(category, message, source => crate::baseobjspace::fixedview(skip_file_prefixes, -1))? {
                    if unsafe { !pyre_object::is_str(prefix) } {
                        return Err(PyError::type_error(format!(
                            "Found non-str '{}' in skip_file_prefixes.",
                            crate::baseobjspace::object_functionstr_type_name(prefix),
                        )));
                    }
                    prefixes.push(pyre_object::with_roots!(category, message, source => crate::baseobjspace::text_w(prefix))?.to_string());
                }
                prefixes
            };
            let stacklevel = if skip_file_prefixes.is_empty() {
                stacklevel
            } else {
                stacklevel.max(2)
            };
            let _roots = pyre_object::gc_roots::push_roots();
            let message_slot = pin_root_slot(message);
            let source_slot = pin_root_slot(source);
            let category = get_category(
                pyre_object::gc_roots::shadow_stack_get(message_slot),
                category,
            )?;
            do_warn(
                pyre_object::gc_roots::shadow_stack_get(message_slot),
                category,
                stacklevel,
                pyre_object::gc_roots::shadow_stack_get(source_slot),
                &skip_file_prefixes,
            )?;
            Ok(w_none())
        }

        fn warn_explicit(
            mut message: PyObjectRef,
            mut category: PyObjectRef,
            mut filename: PyObjectRef,
            lineno: i64,
            #[default(pyre_object::PY_NULL)] mut module: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut registry: PyObjectRef,
            #[default(pyre_object::PY_NULL)] module_globals: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut source: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let source_line = pyre_object::with_roots!(category, filename, message, module, registry, source => get_source_line(module_globals, lineno))?;
            let _roots = pyre_object::gc_roots::push_roots();
            // `get_category` allocates.  The source line and every argument
            // the warning still needs are pinned together and read back after.
            let base = pyre_object::gc_roots::pin_roots(&[
                source_line, message, filename, module, registry, source,
            ]);
            let category = get_category(pyre_object::gc_roots::shadow_stack_get(base + 1), category)?;
            do_warn_explicit(
                category,
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                pyre_object::gc_roots::shadow_stack_get(base + 2),
                lineno,
                pyre_object::gc_roots::shadow_stack_get(base + 3),
                pyre_object::gc_roots::shadow_stack_get(base + 4),
                pyre_object::gc_roots::shadow_stack_get(base),
                pyre_object::gc_roots::shadow_stack_get(base + 5),
            )?;
            Ok(w_none())
        }
    },
    functions: {
        "_filters_mutated" / 0 = filters_mutated,
        "_filters_mutated_lock_held" / 0 = filters_mutated,
        "_acquire_lock" / 0 = lock_noop,
        "_release_lock" / 0 = lock_noop,
    },
    extra_init: |ns| {
        // `create_filter` and the allocations below can collect; the
        // namespace is read back from its slot.
        let _filter_roots = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        let ns = || pyre_object::gc_roots::shadow_stack_get(ns_slot);
        let mut filter_slots = Vec::new();
        let mut put_filter = |filter: PyObjectRef| {
            filter_slots.push(pyre_object::gc_roots::shadow_stack_len());
            let _ = pyre_object::gc_roots::pin_root(filter);
        };
        put_filter(create_filter(
            warning_class("DeprecationWarning"),
            "default",
            Some("__main__"),
        ));
        put_filter(create_filter(
            warning_class("DeprecationWarning"),
            "ignore",
            None,
        ));
        put_filter(create_filter(
            warning_class("PendingDeprecationWarning"),
            "ignore",
            None,
        ));
        put_filter(create_filter(warning_class("ImportWarning"), "ignore", None));
        put_filter(create_filter(
            warning_class("ResourceWarning"),
            "ignore",
            None,
        ));
        drop(put_filter);
        let filters = w_list_new(
            filter_slots
                .into_iter()
                .map(pyre_object::gc_roots::shadow_stack_get)
                .collect(),
        );
        // `moduledef.py:18-22 setup_after_space_initialization` publishes the
        // State fields into the module dict.  Capture the namespace only once
        // every field is in place, so `state_is_readable` never reports a
        // half-filled State to a warning raised in between.
        crate::__pyre_put_new!(ns_slot, "filters", filters);
        crate::__pyre_put_new!(ns_slot, "_onceregistry", w_dict_new());
        crate::__pyre_put_new!(ns_slot, "_defaultaction", w_str_new("default"));
        crate::__pyre_put_new!(ns_slot, VERSION_ATTR, new_version());
        STATE_NS.store(ns() as usize, std::sync::atomic::Ordering::Release);
    },
}
