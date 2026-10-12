//! `pypy/module/pypyjit/interp_jit.py` — the public `set_param` entry.
//!
//! The rest of that module (the driver, dispatch, `residual_call`) stays in
//! the JIT crates. This file owns only the function `moduledef.py` binds as
//! `interp_jit.set_param`.

/// interp_jit.py — `set_param(space, __args__)`.
///
/// Accepts the PyPy calling conventions:
///   * `set_param("name=value,name=value")` / `set_param("off")` /
///     `set_param("default")` — the positional string form.
///   * `set_param(name=value, ...)` — keyword arguments.
///
/// The positional string goes through `set_user_param`
/// (`call::set_jit_param_string`). A keyword `enable_opts` goes to
/// `set_param_enable_opts` with the whole string. Every other keyword is an
/// integer (`space.int_w`) whose name is in `unroll_parameters`, then one
/// `name=value` list through the same string parser.
pub(super) fn set_param_args(
    args: &pyre_interpreter::argument::Arguments,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let (pos, kwargs) = pyre_interpreter::builtins::arguments_pos_and_kwargs(args)?;
    set_param_from(&pos, kwargs)
}

pub(super) fn set_param(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let (pos, kwds) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    set_param_from(pos, kwds)
}

fn set_param_from(
    pos: &[pyre_object::PyObjectRef],
    mut kwds: Option<pyre_object::PyObjectRef>,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    // interp_jit.py:147-148 — at most one non-keyword argument.
    if pos.len() > 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → set_user_param(None, text).
    if let Some(&text_obj) = pos.first() {
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[kwds.unwrap_or(pyre_object::PY_NULL)]);
        let text = pyre_interpreter::baseobjspace::text_w(text_obj);
        let w = roots.get(base);
        kwds = if w.is_null() { None } else { Some(w) };
        drop(roots);
        let text = text?;
        if pyre_interpreter::call::set_jit_param_string(text).is_err() {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    // interp_jit.py:159-170 — keyword arguments.
    // `enable_opts` is `jit.set_param(None, 'enable_opts', text)` and is not
    // split on commas. Other names are `space.int_w` plus `unroll_parameters`,
    // then one `name=value` string for `set_user_param`.
    if let Some(kw_dict) = kwds {
        let mut parts: Vec<String> = Vec::new();
        let mut enable_opts: Option<&str> = None;
        for (k, v) in unsafe { pyre_object::dictmultiobject::w_dict_items(kw_dict) } {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_wtf8(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            if key == "enable_opts" {
                enable_opts = Some(pyre_interpreter::baseobjspace::text_w(v)?);
            } else {
                let value = pyre_interpreter::baseobjspace::int_w(v)?;
                let known = majit_metainterp::jit::UNROLL_PARAMETERS
                    .iter()
                    .any(|&(name, _)| key == name && name != "enable_opts");
                if !known {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "no JIT parameter '{key}'"
                    )));
                }
                parts.push(format!("{key}={value}"));
            }
        }
        if !parts.is_empty()
            && pyre_interpreter::call::set_jit_param_string(&parts.join(",")).is_err()
        {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
        if let Some(value) = enable_opts {
            pyre_interpreter::call::set_jit_param_enable_opts(value);
        }
    }

    Ok(pyre_object::w_none())
}
