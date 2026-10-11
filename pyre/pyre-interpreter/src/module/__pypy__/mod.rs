//! `__pypy__` module — PyPy: pypy/module/__pypy__/
//!
//! Pyre exposes the small slice of the `__pypy__` surface that the
//! PyPy-flavored stdlib needs.  `pickle.py` imports `identity_dict`
//! (an identity-keyed memo dict) and `builders.BytesBuilder` in one
//! shared `try` block; both must resolve for the optimized path to
//! activate, so both are provided here as app-level classes.
//!
//! `collections.OrderedDict` (`_collections/app_odict.py`) imports
//! `reversed_dict`, `move_to_end`, and `objects_in_repr` — the dict-order
//! primitives PyPy keeps interp-level in `interp_dict.py` / `interp_magic.py`.
//! Pyre matches that placement (the `functions:` block below), so they are
//! non-binding `BuiltinFunction`s; only `identity_dict` stays an app-level
//! class.
//!
//! `PickleBuffer` (`interp_buffer.py W_PickleBuffer`) is exposed here as
//! an interp-level class; `pickle.py` re-exports it and the `_pickle`
//! accelerator serializes it in-band or out-of-band under protocol 5.

pub mod interp_buffer;

pub use interp_buffer::W_PickleBuffer;

/// `interp_magic.get_contextvar_context` — read the value from the live
/// PyPy-style ExecutionContext, returning None before the first Context is
/// installed.
fn get_contextvar_context(_: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return Ok(pyre_object::w_none());
    }
    let context = unsafe { (*ec).contextvar_context };
    if context.is_null() {
        Ok(pyre_object::w_none())
    } else {
        Ok(context)
    }
}

/// `interp_magic.set_contextvar_context` — replace the current
/// ExecutionContext's PEP 567 Context slot.
fn set_contextvar_context(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let Some(&context) = args.first() else {
        return Err(crate::PyError::type_error(
            "set_contextvar_context() missing required argument",
        ));
    };
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return Err(crate::PyError::runtime_error(
            "no current execution context",
        ));
    }
    unsafe { (*ec).contextvar_context = context };
    Ok(pyre_object::w_none())
}

/// `interp_magic.py newlist_hint`: an empty list to be filled with about
/// `sizehint` items.
///
/// The hint names a storage length, not a length, so what comes back is `[]`.
/// SizeListStrategy retains it without allocating; the first append chooses
/// the concrete strategy and asks its `get_empty_storage(sizehint)` for the
/// exact backing capacity.
fn newlist_hint(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let sizehint = crate::baseobjspace::int_w(args[0])?;
    isize::try_from(sizehint)
        .map_err(|_| crate::PyError::overflow_error("integer does not fit in signed word"))?;
    Ok(pyre_object::listobject::w_list_new_with_sizehint(sizehint))
}

/// `interp_magic.py resizelist_hint`: forward to the live list strategy.
fn resizelist_hint(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let mut w_list = args[0];
    if !unsafe { pyre_object::is_list(w_list) } {
        return Err(crate::PyError::type_error("arg 1 must be a 'list'"));
    }
    let sizehint = pyre_object::with_roots!(w_list => crate::baseobjspace::int_w(args[1]))?;
    isize::try_from(sizehint)
        .map_err(|_| crate::PyError::overflow_error("integer does not fit in signed word"))?;
    if !unsafe { pyre_object::listobject::w_list_resize_hint(w_list, sizehint) } {
        return Err(crate::PyError::memory_error(""));
    }
    Ok(pyre_object::w_none())
}

/// `interp_magic.py list_get_physical_size`.
fn list_get_physical_size(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let w_list = args[0];
    if !unsafe { pyre_object::is_list(w_list) } {
        return Err(crate::PyError::type_error("expected list"));
    }
    let Some(size) = (unsafe { pyre_object::listobject::w_list_physical_size(w_list) }) else {
        return Err(crate::PyError::value_error(
            "can't get physical size of list",
        ));
    };
    Ok(pyre_object::w_int_new(size as i64))
}

/// `interp_magic.py add_memory_pressure`: report a raw allocation to
/// incminimark so the next major collection is scheduled sooner. The public
/// builtin has no owner object; translated internal callers use the same GC
/// surface with an object when their type carries `special_memory_pressure`.
fn add_memory_pressure(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let estimate = crate::baseobjspace::int_w(args[0])?;
    let estimate = isize::try_from(estimate)
        .map_err(|_| crate::PyError::overflow_error("integer does not fit in signed word"))?;
    majit_gc::add_memory_pressure(estimate, majit_ir::GcRef::NULL);
    Ok(pyre_object::w_none())
}

/// `interp_magic.py hidden_applevel` — `func.getcode().hidden_applevel = True`,
/// returning the function so it can be spelled as a decorator.
///
/// One function at a time, rather than a whole compilation unit: a module that
/// is otherwise the program's own reaches for this on the single frame that is
/// not, the way `lib_pypy/_contextvars.py` marks `Context.run`, which stands
/// between a callable and whoever asked the context to run it.
fn hidden_applevel(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let w_func = args[0];
    // `space.interp_w(Function, w_func)`, whose mismatch raises the same
    // `'%s' object expected, got '%T' instead` as `descr_call_mismatch`.
    if w_func.is_null() || !unsafe { crate::is_function(w_func) } {
        return Err(crate::baseobjspace::descr_call_mismatch(
            w_func,
            "hidden_applevel",
            crate::typedef::gettypeobject(&crate::FUNCTION_TYPE),
        ));
    }
    let w_code = unsafe { crate::function_get_code(w_func) } as pyre_object::PyObjectRef;
    unsafe { crate::pycode::w_code_set_hidden_applevel(w_code, true) };
    Ok(w_func)
}

/// `interp_magic.py strategy` — expose the live implementation strategy of a
/// dict, list, set, or mapdict-backed instance.
///
/// This is intentionally a diagnostic of the representation pyre actually
/// uses. In particular, ascii lists and integer sets currently report their
/// Object strategies; that makes those remaining PyPy strategy ports
/// visible instead of hiding them behind a missing `__pypy__` function.
fn strategy(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let obj = args[0];
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let dict = crate::type_methods::resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(
        obj_slot,
    ));
    if !dict.is_null() && unsafe { pyre_object::is_dict(dict) } {
        return Ok(pyre_object::w_str_new_managed(unsafe {
            pyre_object::dictmultiobject::w_dict_strategy_name(dict)
        }));
    }
    if unsafe { pyre_object::is_list(pyre_object::gc_roots::shadow_stack_get(obj_slot)) } {
        return Ok(pyre_object::w_str_new_managed(unsafe {
            pyre_object::listobject::w_list_strategy_name(pyre_object::gc_roots::shadow_stack_get(
                obj_slot,
            ))
        }));
    }
    if unsafe {
        pyre_object::setobject::is_set_or_frozenset(pyre_object::gc_roots::shadow_stack_get(
            obj_slot,
        ))
    } {
        // W_SetObject currently has one ObjectKey-backed representation.  The
        // helper reports that real shape; EmptySetStrategy and
        // IntegerSetStrategy remain explicit builtin-type porting work.
        return Ok(pyre_object::w_str_new_managed("ObjectSetStrategy"));
    }
    if let Some(name) = unsafe {
        crate::objspace::std::mapdict::mapdict_strategy_repr(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        )
    } {
        return Ok(pyre_object::w_str_from_wtf8_managed(name));
    }
    Err(crate::PyError::type_error(
        "expecting dict or list or set object, or instance of some kind",
    ))
}

/// `interp_dict.py:43-45 / 79-81` `isinstance(w_obj, W_DictMultiObject)`:
/// resolve the backing of an exact dict, module dict, or dict subclass,
/// rejecting a read-only `mappingproxy` (a `W_Root`, not a `W_DictMultiObject`)
/// and any non-dict.  `what` names the primitive for the `TypeError`.
fn dict_backing_or_type_error(
    obj: pyre_object::PyObjectRef,
    what: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let is_proxy = unsafe { pyre_object::is_dict_proxy(obj) };
    if !is_proxy {
        let backing = crate::type_methods::resolve_dict_backing(obj);
        if !backing.is_null() {
            return Ok(backing);
        }
    }
    Err(crate::PyError::type_error(format!(
        "{what}() argument must be a dict"
    )))
}

/// `interp_dict.py reversed_dict` — `W_DictMultiObject.descr_reversed`
/// (`dictmultiobject.py`): a reverse iterator over the dict keys, resolved
/// through the backing dict so a subclass `__reversed__` is bypassed.
fn reversed_dict(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let backing = dict_backing_or_type_error(args[0], "reversed_dict")?;
    Ok(
        pyre_object::dictmultiobject::w_dict_view_reverse_iterator_new(
            backing,
            pyre_object::dictmultiobject::DictViewKind::Keys,
        ),
    )
}

/// `interp_dict.py move_to_end` — `W_DictMultiObject.nondescr_move_to_end`
/// (`dictmultiobject.py`): move an existing key to the back (`last`, the
/// default) or front of the insertion order.  `@unwrap_spec(last=bool)`: `last`
/// may be supplied positionally or by keyword and is coerced by truthiness.
fn move_to_end(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["last"], "move_to_end")?;
    let (mut d, mut key) = match positional {
        [d, key] | [d, key, _] => (*d, *key),
        _ => {
            return Err(crate::PyError::type_error(
                "move_to_end() takes 2 or 3 positional arguments",
            ));
        }
    };
    let last =
        match crate::builtins::bind_pos_or_kw(positional, kwargs, 2, "last", "move_to_end", 3)? {
            Some(w) => pyre_object::with_roots!(d, key => crate::baseobjspace::is_true(w))?,
            None => true,
        };
    let _roots = pyre_object::gc_roots::push_roots();
    // Both words already exist. Sequential `pin_root` would normalize after
    // `d` and leave `key` unpublished (`RootScope::pin_roots`).
    let base = pyre_object::gc_roots::pin_roots(&[d, key]);
    let backing =
        dict_backing_or_type_error(pyre_object::gc_roots::shadow_stack_get(base), "move_to_end")?;
    if crate::baseobjspace::dict_move_to_end(
        backing,
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        last,
    )? {
        Ok(pyre_object::w_none())
    } else {
        Err(crate::PyError::key_error_with_key(
            pyre_object::gc_roots::shadow_stack_get(base + 1),
        ))
    }
}

/// `interp_magic.py objects_in_repr` — `space.get_objects_in_repr()`
/// (`objspace.py:134`): the execution-context-owned identity dict of objects
/// currently being `repr()`'d, lazily built and cached on the EC.  `identity_dict`
/// stays an app-level class, so the instance is constructed by calling that type.
///
/// `get_objects_in_repr` builds `W_IdentityDict(self)` directly (`objspace.py`),
/// so the recursion guard cannot be swapped through module monkeypatching.  pyre
/// mirrors that by constructing from the canonical type captured at module init
/// (`CANONICAL_IDENTITY_DICT_KEY`), not the publicly reassignable
/// `__pypy__.identity_dict` attribute.
fn objects_in_repr(_: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return Err(crate::PyError::runtime_error(
            "no current execution context",
        ));
    }
    let cached = unsafe { (*ec).py_repr };
    if !cached.is_null() {
        return Ok(cached);
    }
    let module = crate::importing::check_sys_modules("__pypy__")
        .ok_or_else(|| crate::PyError::runtime_error("__pypy__ module not loaded"))?;
    let ty =
        crate::baseobjspace::getattr_str(module, CANONICAL_IDENTITY_DICT_KEY).map_err(|_| {
            crate::PyError::runtime_error("__pypy__: canonical identity_dict unavailable")
        })?;
    let inst = crate::call::call_function_impl_result(ty, &[])?;
    // Stored immediately (no intervening allocation); thereafter forwarded by
    // `ExecutionContext::walk_builtin_roots`.
    unsafe { (*ec).py_repr = inst };
    Ok(inst)
}

/// Module-dict key holding the `identity_dict` type captured at module init.
/// Not a valid Python identifier, so it cannot be reached — or rewritten —
/// through `__pypy__.identity_dict` attribute access; the module dict is
/// Box-immortal, so the entry roots the type for the lifetime of the process.
const CANONICAL_IDENTITY_DICT_KEY: &str = "@objects_in_repr_identity_dict";

/// `interp_magic.py write_unraisable` — turn the supplied exception
/// value back into an OperationError and report it through `sys.unraisablehook`.
fn write_unraisable(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let where_desc = crate::baseobjspace::text_wtf8_w(args[0])?.to_wtf8_buf();
    // `OperationError(space.type(w_exc), w_exc)` accepts any object, so the
    // exception tag cannot be read unconditionally: it lives past the header
    // of a `W_BaseException`, and a plain instance is smaller than that.
    // Classify first, and fall back to the kind a bare `BaseException` maps to.
    let mut error = match unsafe {
        pyre_object::interp_exceptions::w_exception_kind_checked(args[1])
    } {
        Some(_) => unsafe { crate::PyError::from_exc_object(args[1]) },
        None => {
            let mut error = crate::PyError::new(crate::PyErrorKind::RuntimeError, String::new());
            error.set_exc_object(args[1]);
            error
        }
    };
    // `interp_magic.write_unraisable` passes `with_traceback=True`: unlike
    // the default finalizer form, `where` already contains its preposition
    // (for example `"in: testplace"`).
    error.write_unraisable_with_traceback(pyre_object::w_none(), &where_desc, args[2], true);
    Ok(pyre_object::w_none())
}

crate::py_module! {
    "__pypy__",
    // `PickleBuffer` wraps a bytes-like object for proto-5 out-of-band
    // buffers; `identity_dict` keys a memo by object identity (id(key))
    // so the Pickler can memoize unhashable containers.
    interpleveldefs: {
        "PickleBuffer" => interp_buffer::picklebuffer_type_object(),
    },
    appleveldefs: {
        "identity_dict_app.py" => ["identity_dict"],
    },
    functions: {
        "get_contextvar_context" / 0 = get_contextvar_context,
        "set_contextvar_context" / 1 = set_contextvar_context,
        "add_memory_pressure" / 1 = add_memory_pressure,
        "newlist_hint" / 1 = newlist_hint,
        "resizelist_hint" / 2 = resizelist_hint,
        "list_get_physical_size" / 1 = list_get_physical_size,
        "reversed_dict" / 1 = reversed_dict,
        "move_to_end" / * = move_to_end,
        "objects_in_repr" / 0 = objects_in_repr,
        "write_unraisable" / 3 = write_unraisable,
        "hidden_applevel" / 1 = hidden_applevel,
        "strategy" / 1 = strategy,
        "newmemoryview" / * = interp_buffer::newmemoryview,
    },
    extra_init: |ns| {
        // Mark as a package so `from __pypy__.builders import ...`
        // treats `__pypy__` as a package with submodules.
        let mut ns = ns;
        let mut w_path = pyre_object::with_roots!(ns => pyre_object::w_list_new(vec![]));
        crate::__pyre_store!(ns, "__path__", w_path);
        // Snapshot the canonical `identity_dict` type before any app code can
        // reassign `__pypy__.identity_dict`, keyed so attribute access cannot
        // reach it.  `objects_in_repr` builds its recursion guard from this
        // snapshot, matching PyPy's direct `W_IdentityDict` construction.
        if let Some(mut ty) = crate::module_ns_get(ns, "identity_dict") {
            crate::__pyre_store!(ns, CANONICAL_IDENTITY_DICT_KEY, ty);
        }
    }
}

/// `__pypy__.builders` submodule — exposes the string/bytes builders.
pub mod builders {
    crate::py_module! {
        "__pypy__.builders",
        // BytesBuilder is the append-only byte buffer pickle.py writes
        // frames into; StringBuilder is its text analogue.
        appleveldefs: {
            "builders_app.py" => ["BytesBuilder", "StringBuilder"],
        }
    }
}

/// `__pypy__.bufferable` — PyPy keeps `bufferable` in a submodule rather
/// than exporting the class directly from `__pypy__`.
pub mod bufferable {
    crate::py_module! {
        "__pypy__.bufferable",
        interpleveldefs: {
            "bufferable" => crate::module::__pypy__::interp_buffer::bufferable_impl::type_object(),
        }
    }
}
