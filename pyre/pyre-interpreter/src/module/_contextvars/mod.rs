//! `_contextvars` module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! `Context` is the app-level line-by-line port because its persistent Map
//! operations and `run()`'s try/finally are already expressed exactly there.
//! ContextVar and Token remain interpreter-level while their state operations
//! are ported incrementally.  `copy_context` is interp-level too — a
//! non-binding builtin over `current_context`; PyPy keeps it in
//! `lib_pypy/_contextvars.py`, but `Python/context.c` ships it as a C function.

use pyre_object::*;
use std::sync::OnceLock;

pub(crate) fn context_var_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("_contextvars.ContextVar", |ns| {
            let _roots = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            let signature = pyre_interpreter::gateway::Signature::new(
                vec!["cls", "name", "default"],
                None,
                None,
                1,
                2,
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__new__",
                pyre_interpreter::typedef::make_new_descr_with_signature(
                    context_var_new,
                    signature.clone()
                )
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__init__",
                pyre_interpreter::make_builtin_function_with_signature(
                    "__init__",
                    |_| Ok(w_none()),
                    pyre_interpreter::gateway::Signature::new(
                        vec!["self", "name", "default"],
                        None,
                        None,
                        1,
                        2,
                    )
                )
            );
            crate::__pyre_put_new!(
                ns_slot,
                "get",
                pyre_interpreter::make_builtin_function("get", context_var_get)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "set",
                pyre_interpreter::make_builtin_function_with_arity("set", context_var_set, 2)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "reset",
                pyre_interpreter::make_builtin_function_with_arity("reset", context_var_reset, 2)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__repr__",
                pyre_interpreter::make_builtin_function_with_arity("__repr__", context_var_repr, 1)
            );
            {
                let getter = pyre_interpreter::make_builtin_function_with_arity(
                    "name",
                    context_var_name_get,
                    2,
                );
                let getter_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(getter);
                crate::__pyre_put_new!(
                    ns_slot,
                    "name",
                    pyre_interpreter::typedef::make_getset_descriptor_named(
                        pyre_object::gc_roots::shadow_stack_get(getter_slot),
                        "name"
                    )
                );
            }
            {
                let func = pyre_interpreter::make_builtin_function(
                    "__class_getitem__",
                    pyre_interpreter::_pypy_generic_alias::generic_alias_class_getitem,
                );
                let func_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(func);
                crate::__pyre_put_new!(
                    ns_slot,
                    "__class_getitem__",
                    pyre_object::function::w_classmethod_new(
                        pyre_object::gc_roots::shadow_stack_get(func_slot)
                    )
                );
            }
        });
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp
    })
}

fn context_var_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // PyPy lib_pypy/_contextvars.py ContextVar.__init__(name, *,
    // default=_NO_DEFAULT): the signature-aware gateway supplies
    // [cls, name, default], with PY_NULL for an omitted default.
    if args.len() < 2 || args[1].is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "ContextVar() takes exactly 1 positional argument (0 given)",
        ));
    }
    if !unsafe { is_str(args[1]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "context variable name must be a str",
        ));
    }
    // `hash_w_strict` and `context_var_type()` can collect. Pin the
    // constructor arguments and reload them at each use.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let name = || pyre_object::gc_roots::shadow_stack_get(args_base + 1);
    pyre_interpreter::baseobjspace::hash_w_strict(name())?;
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_instance_new(context_var_type()));
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    pyre_interpreter::baseobjspace::setattr_str(obj(), "_name", name())?;
    if args.len() > 2 {
        let default = pyre_object::gc_roots::shadow_stack_get(args_base + 2);
        if !default.is_null() {
            pyre_interpreter::baseobjspace::setattr_str(obj(), "_default", default)?;
        }
    }
    Ok(obj())
}

fn context_var_name_get(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::getattr_str(args[1], "_name")
}

fn context_var_get(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "get() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }
    // The lookup runs the Context mapping's `__getitem__`, which is
    // application-level Python.  The default handed back below it is whatever
    // the caller passed -- a list or a dict included -- and the gateway's
    // argument array is a native copy a collection does not rewrite, so both
    // operands are read back from their slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let var = || pyre_object::gc_roots::shadow_stack_get(args_base);
    if let Some(context) = current_context(false)? {
        match pyre_interpreter::baseobjspace::getitem(context, var()) {
            Ok(value) => return Ok(value),
            Err(err) if err.kind == pyre_interpreter::PyErrorKind::KeyError => {}
            Err(err) => return Err(err),
        }
    }
    if args.len() > 1 {
        return Ok(pyre_object::gc_roots::shadow_stack_get(args_base + 1));
    }
    if let Some(default) = pyre_interpreter::baseobjspace::findattr_result(var(), "_default")? {
        return Ok(default);
    }
    Err(pyre_interpreter::PyError::lookup_error(
        context_var_repr_string(var())?,
    ))
}

fn call_method_result(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `args` is a native copy; the attribute lookup can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    let method = pyre_interpreter::baseobjspace::getattr_str(obj, name)?;
    let mut live_args = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        live_args.push(_roots.get(base + i));
    }
    pyre_interpreter::call::call_function_impl_result(method, &live_args)
}

fn current_context(create: bool) -> Result<Option<PyObjectRef>, pyre_interpreter::PyError> {
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return Err(pyre_interpreter::PyError::runtime_error(
            "no current execution context",
        ));
    }
    let mut context = unsafe { (*ec).contextvar_context };
    if context.is_null() || std::ptr::eq(context, w_none()) {
        if !create {
            return Ok(None);
        }
        // `_context_type` is stored on the immortal ContextVar type's dict by
        // module init, which roots the app-level Context type without a raw
        // side table or thread-local copy.
        let context_type =
            pyre_interpreter::baseobjspace::getattr_str(context_var_type(), "_context_type")?;
        context = pyre_interpreter::call::call_function_impl_result(context_type, &[])?;
        unsafe { (*ec).contextvar_context = context };
    }
    Ok(Some(context))
}

fn context_var_set(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // PyPy lib_pypy/_contextvars.py ContextVar.set, line by line: read the
    // old binding, persistently replace Context._data, and return a token
    // tied to this exact Context.
    // Every step here runs Python, so each value takes its root slot before
    // the call it has to outlive rather than after.  Pinning afterwards
    // publishes a word a minor has already moved, and the collector then walks
    // that slot as if it named an object.
    //
    // The gateway hands this function a native array it rebuilt from its own
    // roots, so `args` is not rewritten by a collection either: the variable
    // and the value are read back too.  The value is whatever the caller
    // passed, a list or a dict included, and the token at the end is the only
    // thing that will ever hold the old binding.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let var = || pyre_object::gc_roots::shadow_stack_get(args_base);
    let value = || pyre_object::gc_roots::shadow_stack_get(args_base + 1);

    let context_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(
        current_context(true)?.expect("create=true returns a Context"),
    );
    let context = || pyre_object::gc_roots::shadow_stack_get(context_slot);

    let data_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
        context(),
        "_data",
    )?);
    let data = || pyre_object::gc_roots::shadow_stack_get(data_slot);

    let old_value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(
        match pyre_interpreter::baseobjspace::getitem(data(), var()) {
            Ok(value) => value,
            Err(err) if err.kind == pyre_interpreter::PyErrorKind::KeyError => token_missing(),
            Err(err) => return Err(err),
        },
    );

    let updated_data = call_method_result(data(), "set", &[var(), value()])?;
    pyre_interpreter::baseobjspace::setattr_str(context(), "_data", updated_data)?;
    new_token(
        context(),
        var(),
        pyre_object::gc_roots::shadow_stack_get(old_value_slot),
    )
}

fn context_var_reset(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `token_type()` is a first-use constructor and can collect. Pin the
    // receiver and token, then reload the token for the isinstance test.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let var = || pyre_object::gc_roots::shadow_stack_get(args_base);
    let token = || pyre_object::gc_roots::shadow_stack_get(args_base + 1);
    let cls = token_type();
    if !unsafe { pyre_interpreter::baseobjspace::isinstance_w(token(), cls) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "expected an instance of Token, got {}",
            pyre_interpreter::type_methods::arg_type_name(token()),
        )));
    }
    // Each attribute read and each mapping call is application-level Python.
    // Both identity tests below compare addresses, so an operand that moved
    // under one of these calls does not merely crash -- it reports a token as
    // belonging to a different ContextVar or Context.  Everything that
    // outlives a call is therefore read back from its slot, `args` included:
    // the gateway rebuilt that array from its own roots, so a collection does
    // not rewrite it.

    if pyre_interpreter::baseobjspace::is_true(pyre_interpreter::baseobjspace::getattr_str(
        token(),
        "_used",
    )?)? {
        return Err(pyre_interpreter::PyError::runtime_error(
            pyre_interpreter::display::wtf8_format!(
                token_repr_string(token())?,
                " has already been used once",
            ),
        ));
    }
    let token_var_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
        token(),
        "_var",
    )?);
    let token_var = || pyre_object::gc_roots::shadow_stack_get(token_var_slot);
    if !std::ptr::eq(token_var(), var()) {
        return Err(pyre_interpreter::PyError::value_error(
            "Token was created by a different ContextVar",
        ));
    }
    let context_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(
        current_context(true)?.expect("create=true returns a Context"),
    );
    let context = || pyre_object::gc_roots::shadow_stack_get(context_slot);
    if !std::ptr::eq(
        pyre_interpreter::baseobjspace::getattr_str(token(), "_context")?,
        context(),
    ) {
        return Err(pyre_interpreter::PyError::value_error(
            "Token was created in a different Context",
        ));
    }
    let data_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
        context(),
        "_data",
    )?);
    let data = || pyre_object::gc_roots::shadow_stack_get(data_slot);
    let old_value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
        token(),
        "_old_value",
    )?);
    let old_value = || pyre_object::gc_roots::shadow_stack_get(old_value_slot);
    let updated_data = if std::ptr::eq(old_value(), token_missing()) {
        call_method_result(data(), "delete", &[token_var()])?
    } else {
        call_method_result(data(), "set", &[token_var(), old_value()])?
    };
    pyre_interpreter::baseobjspace::setattr_str(context(), "_data", updated_data)?;
    pyre_interpreter::baseobjspace::setattr_str(token(), "_used", w_bool_from(true))?;
    Ok(w_none())
}

fn context_var_repr_string(
    obj: PyObjectRef,
) -> Result<rustpython_wtf8::Wtf8Buf, pyre_interpreter::PyError> {
    let Some(_guard) = pyre_interpreter::display::ReprGuard::enter(obj) else {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string("...".to_string()));
    };
    // The name and default reprs allocate. The var's own address is
    // `getaddrstring`, taken after those calls from the rooted word.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let name = pyre_interpreter::baseobjspace::getattr_str(obj(), "_name")?;
    let name_repr = unsafe { pyre_interpreter::display::py_repr_wtf8(name)? };
    let default = match pyre_interpreter::baseobjspace::findattr_result(obj(), "_default")? {
        Some(value) => pyre_interpreter::display::wtf8_format!(" default=", unsafe {
            pyre_interpreter::display::py_repr_wtf8(value)?
        }),
        None => rustpython_wtf8::Wtf8Buf::new(),
    };
    Ok(pyre_interpreter::display::wtf8_format!(
        "<ContextVar name=",
        name_repr,
        default,
        format!(" at {}>", pyre_interpreter::display::repr_gc_addr(obj())),
    ))
}

fn context_var_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(pyre_object::w_str_from_wtf8_managed(
        context_var_repr_string(args[0])?,
    ))
}

fn token_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("_contextvars.Token", |ns| {
            let _roots = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(ns_slot, "MISSING", w_instance_new(token_missing_type()));
            {
                let getter =
                    pyre_interpreter::make_builtin_function_with_arity("var", token_var_get, 2);
                let getter_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(getter);
                crate::__pyre_put_new!(
                    ns_slot,
                    "var",
                    pyre_interpreter::typedef::make_getset_descriptor_named(
                        pyre_object::gc_roots::shadow_stack_get(getter_slot),
                        "var"
                    )
                );
            }
            {
                let getter = pyre_interpreter::make_builtin_function_with_arity(
                    "old_value",
                    token_old_value_get,
                    2,
                );
                let getter_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(getter);
                crate::__pyre_put_new!(
                    ns_slot,
                    "old_value",
                    pyre_interpreter::typedef::make_getset_descriptor_named(
                        pyre_object::gc_roots::shadow_stack_get(getter_slot),
                        "old_value"
                    )
                );
            }
            crate::__pyre_put_new!(
                ns_slot,
                "__repr__",
                pyre_interpreter::make_builtin_function_with_arity("__repr__", token_repr, 1)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__enter__",
                pyre_interpreter::make_builtin_function_with_arity("__enter__", token_enter, 1)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__exit__",
                pyre_interpreter::make_builtin_function_with_arity("__exit__", token_exit, 4)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__new__",
                pyre_interpreter::typedef::make_new_descr(|_| {
                    Err(pyre_interpreter::PyError::type_error(
                        "Tokens can only be created by ContextVars",
                    ))
                })
            );
            {
                let func = pyre_interpreter::make_builtin_function(
                    "__class_getitem__",
                    pyre_interpreter::_pypy_generic_alias::generic_alias_class_getitem,
                );
                let func_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(func);
                crate::__pyre_put_new!(
                    ns_slot,
                    "__class_getitem__",
                    pyre_object::function::w_classmethod_new(
                        pyre_object::gc_roots::shadow_stack_get(func_slot)
                    )
                );
            }
        });
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp
    })
}

fn token_missing_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("_contextvars.Token.MISSING", |ns| {
            let _roots = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(
                ns_slot,
                "__repr__",
                pyre_interpreter::make_builtin_function_with_arity(
                    "__repr__",
                    |_| Ok(w_str_new_managed("<Token.MISSING>")),
                    1
                )
            );
        });
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp
    })
}

fn token_missing() -> PyObjectRef {
    // The Token type dict is the shared owner/root, matching PyPy's
    // `Token.MISSING` class attribute.  Do not cache this movable singleton
    // in a raw-pointer OnceLock.
    pyre_interpreter::baseobjspace::getattr_str(token_type(), "MISSING")
        .expect("Token.MISSING is installed with the Token type")
}

fn new_token(
    context: PyObjectRef,
    var: PyObjectRef,
    old_value: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // The three values arrive as native words and outlive the allocation
    // below and four attribute stores, each of which can collect; `old_value`
    // is whatever the caller last set, a list or a dict included.  The token
    // takes a slot of its own for liveness: an instance never moves, but one
    // nothing refers to yet is still swept.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[context, var, old_value]);
    let token_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_instance_new(token_type()));
    let token = || pyre_object::gc_roots::shadow_stack_get(token_slot);
    let value_at = |offset: usize| pyre_object::gc_roots::shadow_stack_get(base + offset);
    pyre_interpreter::baseobjspace::setattr_str(token(), "_context", value_at(0))?;
    pyre_interpreter::baseobjspace::setattr_str(token(), "_var", value_at(1))?;
    pyre_interpreter::baseobjspace::setattr_str(token(), "_old_value", value_at(2))?;
    pyre_interpreter::baseobjspace::setattr_str(token(), "_used", w_bool_from(false))?;
    Ok(token())
}

fn token_var_get(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::getattr_str(args[1], "_var")
}

fn token_old_value_get(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::getattr_str(args[1], "_old_value")
}

fn token_repr_string(
    token: PyObjectRef,
) -> Result<rustpython_wtf8::Wtf8Buf, pyre_interpreter::PyError> {
    // The var's repr allocates, and a young token moves with that collection.
    let _roots = pyre_object::gc_roots::push_roots();
    let token_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(token);
    let token = || pyre_object::gc_roots::shadow_stack_get(token_slot);
    let var = pyre_interpreter::baseobjspace::getattr_str(token(), "_var")?;
    let var_repr = unsafe { pyre_interpreter::display::py_repr_wtf8(var)? };
    let used = pyre_interpreter::baseobjspace::is_true(
        pyre_interpreter::baseobjspace::getattr_str(token(), "_used")?,
    )?;
    Ok(pyre_interpreter::display::wtf8_format!(
        if used {
            "<Token used var="
        } else {
            "<Token var="
        },
        var_repr,
        format!(" at {}>", pyre_interpreter::display::repr_gc_addr(token())),
    ))
}

fn token_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(pyre_object::w_str_from_wtf8_managed(token_repr_string(
        args[0],
    )?))
}

fn token_enter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(args[0])
}

fn token_exit(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `args` is the gateway's native copy; the lookup can collect.
    let mut w_token = args[0];
    let var = pyre_object::with_roots!(w_token => pyre_interpreter::baseobjspace::getattr_str(w_token, "_var"))?;
    context_var_reset(&[var, w_token])?;
    Ok(w_bool_from(false))
}

pyre_interpreter::py_module! {
    "_contextvars",
    interpleveldefs: {
        "ContextVar" => context_var_type(),
        "Token" => token_type(),
    },
    functions: {
        // `copy_context()` — snapshot the current context; `Context.copy()`
        // shares the persistent `_data` Map.  `current_context(true)` is the
        // same "read-or-create the thread's Context" step `set`/`reset` use.
        "copy_context" / 0 = |_| {
            let context = current_context(true)?.expect("create=true returns a Context");
            call_method_result(context, "copy", &[])
        },
    },
    extra_init: |ns| {
        let mut ns = ns;
        let mut context_var = pyre_interpreter::module_ns_get(ns, "ContextVar")
            .expect("_contextvars.ContextVar must be installed first");
        pyre_object::with_roots!(context_var, ns => pyre_interpreter::importing::appleveldef_install_seeded(
            ns,
            include_str!("_contextvars_app.py"),
            "_contextvars_app.py",
            "_contextvars",
            &["Context"],
            &[("ContextVar", context_var)],
        ))?;
        let context = pyre_interpreter::module_ns_get(ns, "Context")
            .expect("_contextvars.Context must be installed by appleveldefs");
        // [3.14-spec] PyPy keeps Context as the ordinary app-level class in
        // lib_pypy/_contextvars.py (with Unsubclassable as its metaclass), and
        // pyre keeps that owner and control-flow shape.  CPython 3.14 exposes
        // PyContext_Type as a static immutable type instead
        // (Python/context.c:750-770).  No @jit.*, _immutable_fields_, or
        // runtime reader in PyPy's class definition depends on the public
        // owner flags, so project only CPython's observable axes here.
        unsafe {
            pyre_object::w_type_set_cpython_type_flags(context, false, true, true);
            pyre_object::w_type_suppress_cpython_basetype(context);
        }
        let context_var_dict =
            unsafe { pyre_object::w_type_get_dict_ptr(context_var) as PyObjectRef };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                context_var_dict,
                "_context_type",
                context,
            );
        }
    },
}
