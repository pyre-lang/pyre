//! `types.GenericAlias` — PEP 585 parameterized generics.
//!
//! PyPy equivalent: lib_pypy/_pypy_generic_alias.py (GenericAlias) +
//! pypy/objspace/std/util.py (generic_alias_class_getitem).
//!
//! The payload lives in `pyre_object::_pypy_generic_alias::GenericAlias`;
//! this module is the behaviour surface (class-getitem constructor,
//! parameter collection, and the typedef methods).

use crate::{make_builtin_function, make_builtin_function_with_arity};
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

/// `_ATTR_EXCEPTIONS` (`_pypy_generic_alias.py`) — attribute names that
/// resolve on the alias itself; every other name delegates to the
/// `__origin__` through `__getattribute__`.
pub(crate) const ATTR_EXCEPTIONS: &[&str] = &[
    "__args__",
    "__class__",
    "__mro_entries__",
    "__origin__",
    "__parameters__",
    "__reduce__",
    "__reduce_ex__",
    "__typing_unpacked_tuple_args__",
    "__unpacked__",
];

/// `name in _ATTR_EXCEPTIONS` — used by `getattr` to decide whether to
/// delegate to `__origin__`.
pub(crate) fn is_attr_exception(name: &str) -> bool {
    // `name in _ATTR_EXCEPTIONS` membership, spelled as direct string
    // equality so it lowers to a chain of `ll_streq` rather than a
    // scan over the `&[&str]` const backing (whose body is opaque).
    matches!(
        name,
        "__args__"
            | "__class__"
            | "__mro_entries__"
            | "__origin__"
            | "__parameters__"
            | "__reduce__"
            | "__reduce_ex__"
            | "__typing_unpacked_tuple_args__"
            | "__unpacked__"
    )
}

/// CPython 3.14 `attr_blocked` (`Objects/genericaliasobject.c`) — these
/// attributes are neither proxied to the origin nor exposed by the alias.
pub(crate) fn is_attr_blocked(name: &str) -> bool {
    matches!(name, "__bases__" | "__copy__" | "__deepcopy__")
}

/// `generic_alias_class_getitem(space, w_cls, w_item)` (util.py).
///
/// Registered as the `__class_getitem__` classmethod on builtin
/// containers, so the bound call delivers `args = [w_cls, w_item]`.  The
/// `w_item` operand is mandatory (the gateway declares it positional).
pub fn generic_alias_class_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    if args.len() != 2 {
        // The message is prefixed with the bound class's name
        // (`list.__class_getitem__() takes exactly one argument`).
        let prefix = args
            .first()
            .filter(|&&c| unsafe { is_type(c) })
            .map(|&c| format!("{}.", unsafe { pyre_object::w_type_get_name(c) }))
            .unwrap_or_default();
        return Err(crate::PyError::type_error(format!(
            "{prefix}__class_getitem__() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    make_generic_alias(args[0], args[1], pyre_object::PY_NULL)
}

/// `GenericAlias.__new__` (`_pypy_generic_alias.py`) — wrap a bare item
/// into a 1-tuple, collect the free parameters, allocate.
pub fn make_generic_alias(
    origin: PyObjectRef,
    item: PyObjectRef,
    w_subtype: PyObjectRef,
) -> crate::PyResult {
    // `collect_parameters` runs Python at every turn of its walk, and the
    // argument tuple it is handed is nursery-allocated, so the word built here
    // is the pre-move one by the time the alias is stamped with it.  Stamping a
    // stale tuple is worse than reading one: the alias is a traced object, so
    // the collector follows the dead word on the next walk rather than at the
    // next use. `w_subtype` is `PY_NULL` for an exact alias and the requested
    // class for `GenericAlias.__new__`.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[origin, item, w_subtype]);
    let origin = || pyre_object::gc_roots::shadow_stack_get(base);
    let item = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let args = if unsafe { is_tuple(item()) } {
        item()
    } else {
        jit_w_tuple1(item())
    };
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args);
    let args = || pyre_object::gc_roots::shadow_stack_get(args_slot);
    let parameters = collect_parameters(args())?;
    Ok(w_generic_alias_new(
        origin(),
        args(),
        parameters,
        pyre_object::gc_roots::shadow_stack_get(base + 2),
    ))
}

/// `_collect_parameters(args)` (`_pypy_generic_alias.py`) — gather the
/// free type variables in order of first appearance.
pub(crate) fn collect_parameters(args: PyObjectRef) -> crate::PyResult {
    // Every turn of this walk runs Python — the `__typing_subst__` and
    // `__parameters__` lookups, and the `__eq__` behind the uniqueness test —
    // and `args`, the tuples and lists nested inside it, and the parameters
    // gathered out of them are all nursery-allocated.  One scope owns the whole
    // walk: the accumulator holds slot indices, and the two helpers pin into
    // this scope instead of opening their own, because an inner scope's drop
    // truncates the stack back to its base and would take the accumulated
    // slots with it.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args);
    let args = || pyre_object::gc_roots::shadow_stack_get(args_slot);
    let mut param_slots: Vec<usize> = Vec::new();
    let n = unsafe { w_tuple_len(args()) };
    let mut i = 0usize;
    while i < n {
        if let Some(t) = unsafe { w_tuple_getitem(args(), i as i64) } {
            collect_parameters_one(t, &mut param_slots)?;
        }
        i += 1;
    }
    let mut params = Vec::with_capacity(param_slots.len());
    let slots = param_slots.as_slice();
    let nslots = slots.len();
    let mut j = 0usize;
    while j < nslots {
        params.push(pyre_object::gc_roots::shadow_stack_get(slots[j]));
        j += 1;
    }
    Ok(w_tuple_new(params))
}

/// One element of the [`collect_parameters`] walk.  Pins into the caller's
/// scope: `param_slots` names slots the caller reads once this returns.
fn collect_parameters_one(
    t: PyObjectRef,
    param_slots: &mut Vec<usize>,
) -> Result<(), crate::PyError> {
    let t_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(t);
    let t = || pyre_object::gc_roots::shadow_stack_get(t_slot);
    unsafe {
        if is_type(t()) {
            // A bare class exposes no `__parameters__` descriptor of its own.
            return Ok(());
        }
        // A collection cannot change what `t` is, so the container kind is read
        // once and the recursion below re-reads only the address.
        let container_is_tuple = is_tuple(t());
        if container_is_tuple || is_list(t()) {
            let n = if container_is_tuple {
                w_tuple_len(t())
            } else {
                w_list_len(t())
            };
            for i in 0..n {
                let x = if container_is_tuple {
                    w_tuple_getitem(t(), i as i64)
                } else {
                    w_list_getitem(t(), i as i64)
                };
                if let Some(x) = x {
                    collect_parameters_one(x, param_slots)?;
                }
            }
            return Ok(());
        }
    }
    // `hasattr(t, '__typing_subst__')` → `t` is itself a parameter.  `hasattr`
    // only swallows `AttributeError`; a misbehaving descriptor that raises
    // anything else propagates.
    match crate::baseobjspace::getattr_str(t(), "__typing_subst__") {
        Ok(_) => {
            push_unique(param_slots, t())?;
            return Ok(());
        }
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    // Otherwise pull `getattr(t, '__parameters__', ())` — the `()` default
    // applies only on `AttributeError`.
    match crate::baseobjspace::getattr_str(t(), "__parameters__") {
        Ok(sub) => {
            if unsafe { is_tuple(sub) } {
                let sub_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(sub);
                let sub = || pyre_object::gc_roots::shadow_stack_get(sub_slot);
                let n = unsafe { w_tuple_len(sub()) };
                for i in 0..n {
                    if let Some(x) = unsafe { w_tuple_getitem(sub(), i as i64) } {
                        push_unique(param_slots, x)?;
                    }
                }
            }
        }
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// `if item not in parameters: parameters.append(item)`.  The `in` test is
/// `tuple.__contains__`, so a raising `__eq__` propagates — and a returning one
/// may have moved both operands, so each comparison reads its pair back out of
/// the slots.  Pins into the caller's scope, as [`collect_parameters_one`] does.
fn push_unique(param_slots: &mut Vec<usize>, item: PyObjectRef) -> Result<(), crate::PyError> {
    let item_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(item);
    for index in 0..param_slots.len() {
        let p = pyre_object::gc_roots::shadow_stack_get(param_slots[index]);
        let item = pyre_object::gc_roots::shadow_stack_get(item_slot);
        if crate::baseobjspace::eq_w(p, item)? {
            return Ok(());
        }
    }
    param_slots.push(item_slot);
    Ok(())
}

// ── typedef methods ──────────────────────────────────────────────────

/// `__origin__` getset (`GenericAlias.__origin__`).
fn ga_get_origin(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_origin(self_) })
    } else {
        Ok(w_none())
    }
}

/// `__args__` getset (`GenericAlias.__args__`).
fn ga_get_args(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_args(self_) })
    } else {
        Ok(w_none())
    }
}

/// `__parameters__` getset (`GenericAlias.__parameters__`).
fn ga_get_parameters(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_parameters(self_) })
    } else {
        Ok(w_none())
    }
}

/// Read `args[0]` as the bound `self`, rejecting a non-GenericAlias
/// before any unsafe field access (an unbound/forged direct call).
fn self_alias(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor requires a 'types.GenericAlias' object",
        ));
    }
    Ok(self_)
}

/// `GenericAlias.__repr__` (`_pypy_generic_alias.py`).
fn ga_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
        repr(self_)?
    }))
}

/// `GenericAlias.__hash__` (`_pypy_generic_alias.py`).
fn ga_hash(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    Ok(w_int_new(crate::builtins::try_hash_value(self_)?))
}

/// `GenericAlias.__call__` (`_pypy_generic_alias.py`).
fn ga_call(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_);
    let _ = pyre_object::gc_roots::pin_root(origin);
    let result = crate::builtins::call_forwarding_args(
        unsafe { pyre_object::gc_roots::shadow_stack_get(root_base + 1) },
        &args[1..],
    )?;
    let _ = pyre_object::gc_roots::pin_root(result);
    crate::call::set_orig_class(
        unsafe { pyre_object::gc_roots::shadow_stack_get(root_base + 2) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(root_base) },
    )?;
    Ok(unsafe { pyre_object::gc_roots::shadow_stack_get(root_base + 2) })
}

/// `GenericAlias.__getattribute__` (`_pypy_generic_alias.py`).
fn ga_getattribute(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_ = self_alias(args)?;
    let name_obj = args.get(1).copied().unwrap_or_else(w_none);
    let name = pyre_object::with_roots!(self_ => crate::baseobjspace::text_w(name_obj))?;
    if !is_attr_exception(name) && !is_attr_blocked(name) {
        let origin = unsafe { w_generic_alias_get_origin(self_) };
        crate::baseobjspace::getattr_str(origin, name)
    } else {
        crate::baseobjspace::object_getattribute(self_, name)
    }
}

/// `GenericAlias.__iter__` (`_pypy_generic_alias.py`).
fn ga_iter(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let self_ = pyre_object::gc_roots::pin_root(self_);
    let starred = make_starred(self_)?;
    let starred_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(starred);
    let singleton = jit_w_tuple1(unsafe { pyre_object::gc_roots::shadow_stack_get(starred_slot) });
    let singleton_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(singleton);
    crate::baseobjspace::iter(unsafe { pyre_object::gc_roots::shadow_stack_get(singleton_slot) })
}

/// `GenericAlias.__dir__` (`_pypy_generic_alias.py`).
fn ga_dir(args: &[PyObjectRef]) -> crate::PyResult {
    dir_list(self_alias(args)?)
}

/// `GenericAlias.__eq__` (`_pypy_generic_alias.py`).
fn ga_eq(args: &[PyObjectRef]) -> crate::PyResult {
    let mut self_ = args.first().copied().unwrap_or_else(w_none);
    let mut other = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } || !unsafe { is_generic_alias(other) } {
        return Ok(w_not_implemented());
    }
    let eq = unsafe {
        pyre_object::with_roots!(self_, other => crate::baseobjspace::eq_w(
            w_generic_alias_get_origin(self_),
            w_generic_alias_get_origin(other),
        ))? && pyre_object::with_roots!(self_, other => crate::baseobjspace::eq_w(
            w_generic_alias_get_args(self_),
            w_generic_alias_get_args(other),
        ))? && w_generic_alias_get_unpacked(self_) == w_generic_alias_get_unpacked(other)
    };
    Ok(w_bool_from(eq))
}

/// CPython 3.14's `ga_richcompare`: `!=` is the inverse of the structural
/// equality result; the four ordering operations return `NotImplemented`.
fn ga_ne(args: &[PyObjectRef]) -> crate::PyResult {
    let result = ga_eq(args)?;
    if unsafe { is_not_implemented(result) } {
        Ok(result)
    } else {
        Ok(w_bool_from(!unsafe { w_bool_get_value(result) }))
    }
}

fn ga_ordering(args: &[PyObjectRef]) -> crate::PyResult {
    self_alias(args)?;
    Ok(w_not_implemented())
}

/// `GenericAlias.__mro_entries__` (`_pypy_generic_alias.py`) —
/// `(self.__origin__,)`, so `class C(list[int])` resolves to `list`.
fn ga_mro_entries(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    Ok(jit_w_tuple1(origin))
}

/// `GenericAlias.__getitem__` (`_pypy_generic_alias.py`) — substitute the
/// free parameters with `items` and build the resulting alias.
fn ga_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let items_raw = args.get(1).copied().unwrap_or_else(w_none);
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_);
    let items_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(items_raw);
    let items = if unsafe { is_tuple(pyre_object::gc_roots::shadow_stack_get(items_slot)) } {
        pyre_object::gc_roots::shadow_stack_get(items_slot)
    } else {
        jit_w_tuple1(pyre_object::gc_roots::shadow_stack_get(items_slot))
    };
    let items_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(items);
    let self_ = pyre_object::gc_roots::shadow_stack_get(self_slot);
    let params = unsafe { w_generic_alias_get_parameters(self_) };
    let ga_args = unsafe { w_generic_alias_get_args(self_) };
    let newargs = subs_parameters(
        self_,
        ga_args,
        params,
        pyre_object::gc_roots::shadow_stack_get(items_slot),
    )?;
    let self_ = pyre_object::gc_roots::shadow_stack_get(self_slot);
    let res = make_generic_alias(
        unsafe { w_generic_alias_get_origin(self_) },
        w_tuple_new(newargs),
        pyre_object::PY_NULL,
    )?;
    if unsafe { w_generic_alias_get_unpacked(pyre_object::gc_roots::shadow_stack_get(self_slot)) } {
        unsafe { w_generic_alias_set_unpacked(res, true) };
    }
    Ok(res)
}

/// `tuple.index(item)` resolved through `eq_w`; `Ok(None)` plays the Python
/// `ValueError` raised when the item is absent.  A raising `__eq__`
/// propagates (`tuple.index` does not swallow comparison errors).
fn tuple_index(t: PyObjectRef, item: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
    // `__eq__` is user code and both operands are nursery-allocated, so the
    // tuple is re-read for every element rather than indexed from the word
    // that was live before the first comparison.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[t, item]);
    let t = || pyre_object::gc_roots::shadow_stack_get(base);
    let item = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let n = unsafe { w_tuple_len(t()) };
    for i in 0..n {
        let _elem_roots = pyre_object::gc_roots::push_roots();
        if let Some(x) = unsafe { w_tuple_getitem(t(), i as i64) } {
            let x = pyre_object::gc_roots::pin_root(x);
            if crate::baseobjspace::eq_w(x, item())? {
                return Ok(Some(i));
            }
        }
    }
    Ok(None)
}

/// `_unpack_args(*items)` (`typing.py`) — flatten any element that is an
/// unpacked `tuple[...]` alias (one exposing `__typing_unpacked_tuple_args__`)
/// into its members, unless those end in `...`.  Returns a fresh items tuple.
fn unpack_args(items: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    // The loop body runs Python at every turn, so the accumulator holds slot
    // indices rather than values — the same shape `push_newarg` uses above —
    // and `items` is read back before each element fetch.
    let _roots = pyre_object::gc_roots::push_roots();
    let items_slot =
        pyre_object::gc_roots::pin_roots(&[items, pyre_object::PY_NULL, pyre_object::PY_NULL]);
    let subargs_slot = items_slot + 1;
    let arg_slot = items_slot + 2;
    let items = || pyre_object::gc_roots::shadow_stack_get(items_slot);
    let n = unsafe { w_tuple_len(items()) };
    let mut newarg_slots: Vec<usize> = Vec::new();
    let mut push_newarg = |value: PyObjectRef, slots: &mut Vec<usize>| {
        slots.push(pyre_object::gc_roots::shadow_stack_len());
        let _ = pyre_object::gc_roots::pin_root(value);
    };
    for i in 0..n {
        let Some(arg) = (unsafe { w_tuple_getitem(items(), i as i64) }) else {
            continue;
        };
        // getattr_str / is_true / getitem run Python; `arg` is read back
        // from its slot for `push_newarg`.
        pyre_object::gc_roots::shadow_stack_set(arg_slot, arg);
        let subargs = match crate::baseobjspace::getattr_str(arg, "__typing_unpacked_tuple_args__")
        {
            Ok(s) => s,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        // `if subargs is not None and not (subargs and subargs[-1] is ...)`
        // — `subargs` is any object exposing `__typing_unpacked_tuple_args__`,
        // not necessarily a tuple, so the gate uses the general truthiness /
        // subscription protocol.
        let do_unpack = if unsafe { pyre_object::is_none(subargs) } {
            false
        } else {
            pyre_object::gc_roots::shadow_stack_set(subargs_slot, subargs);
            let ends_ellipsis = crate::baseobjspace::is_true(
                pyre_object::gc_roots::shadow_stack_get(subargs_slot),
            )? && {
                let idx_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(w_int_new(-1));
                let last = crate::baseobjspace::getitem(
                    pyre_object::gc_roots::shadow_stack_get(subargs_slot),
                    pyre_object::gc_roots::shadow_stack_get(idx_slot),
                )?;
                unsafe { is_ellipsis(last) }
            };
            !ends_ellipsis
        };
        if do_unpack {
            // `newargs.extend(subargs)` — any iterable, not just a tuple.
            // Publish the collected members in one go: `collect_iterable`'s own
            // scope has popped, so its Vec is untraced from here on.
            let members = crate::builtins::collect_iterable(
                pyre_object::gc_roots::shadow_stack_get(subargs_slot),
            )?;
            let member_base = pyre_object::gc_roots::pin_roots(&members);
            for index in 0..members.len() {
                newarg_slots.push(member_base + index);
            }
        } else {
            push_newarg(
                pyre_object::gc_roots::shadow_stack_get(arg_slot),
                &mut newarg_slots,
            );
        }
    }
    let mut newargs = Vec::with_capacity(newarg_slots.len());
    for &slot in &newarg_slots {
        newargs.push(pyre_object::gc_roots::shadow_stack_get(slot));
    }
    Ok(w_tuple_new(newargs))
}

/// `_is_unpacked_typevartuple(x)` (`typing.py`) — `x` is an unpacked
/// `TypeVarTuple` (`*Ts`), identified by `__typing_is_unpacked_typevartuple__
/// is True`; a bare class is never one.
fn is_unpacked_typevartuple(x: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { is_type(x) } {
        return Ok(false);
    }
    match crate::baseobjspace::getattr_str(x, "__typing_is_unpacked_typevartuple__") {
        Ok(v) => Ok(v == w_bool_from(true)),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(false),
        Err(e) => Err(e),
    }
}

/// `isinstance(param, TypeVarTuple)` — mirrors `_is_typevar`'s bootstrapping
/// shortcut (`_pypy_generic_alias.py`): match the parameter's type by
/// `__name__` + `__module__` rather than importing `typing`.
fn is_typevartuple(param: PyObjectRef) -> bool {
    let Some(t) = crate::typedef::r#type(param) else {
        return false;
    };
    if unsafe { pyre_object::w_type_get_name(t.as_ptr()) } != "TypeVarTuple" {
        return false;
    }
    matches!(
        crate::baseobjspace::getattr_str(t.as_ptr(), "__module__")
            .ok()
            .and_then(|m| crate::baseobjspace::text_w(m).ok()),
        Some("typing")
    )
}

/// `subs_parameters(self, args, params, items)` (`_pypy_generic_alias.py`)
/// — produce the substituted `__args__` for `self[items]`.  Shared by
/// `GenericAlias.__getitem__` and `UnionType.__getitem__`.
pub(crate) fn subs_parameters(
    self_: PyObjectRef,
    args: PyObjectRef,
    params: PyObjectRef,
    items: PyObjectRef,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    // The Python hooks below are arbitrary collecting calls.  RPython's GC
    // transform keeps all four live arguments in shadow-stack slots and
    // reloads them afterwards; mirror that ownership explicitly.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let self_ = pyre_object::gc_roots::pin_root(self_);
    let args = pyre_object::gc_roots::pin_root(args);
    let params = pyre_object::gc_roots::pin_root(params);
    let items = pyre_object::gc_roots::pin_root(items);
    let current_self = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let current_args = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let current_params = || pyre_object::gc_roots::shadow_stack_get(root_base + 2);
    let nparams = unsafe { w_tuple_len(current_params()) };
    if nparams == 0 {
        let repr = unsafe { crate::display::py_repr_wtf8(current_self())? };
        return Err(crate::PyError::type_error(crate::display::wtf8_format!(
            repr,
            " is not a generic class"
        )));
    }
    // Substitution runs arbitrary Python — `__typing_prepare_subst__`,
    // `__typing_subst__`, and the recursive descent into nested lists and
    // tuples — and allocates at nearly every step. The collector moves
    // objects and does not scan Rust locals, so the owners and every produced
    // argument live on the shadow stack and are reread after anything that
    // can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(self_);
    let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(params);
    let params_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // `items = _unpack_args(items)` flattens unpacked `tuple[...]` aliases.
    // `__typing_prepare_subst__` then reshapes `items` for
    // `ParamSpec`/`TypeVarTuple` parameters — honoured per param, missing
    // attribute (the `None` default) skips it.
    let _ = pyre_object::gc_roots::pin_root(items);
    let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let unpacked = unpack_args(pyre_object::gc_roots::shadow_stack_get(items_slot))?;
    pyre_object::gc_roots::shadow_stack_set(items_slot, unpacked);
    for i in 0..nparams {
        let Some(param) = (unsafe {
            w_tuple_getitem(
                pyre_object::gc_roots::shadow_stack_get(params_slot),
                i as i64,
            )
        }) else {
            continue;
        };
        let param_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(param);
        // `prepare = getattr(param, '__typing_prepare_subst__', None)` then
        // `if prepare is not None`: a missing attribute and an attribute
        // explicitly set to `None` both skip the reshape.
        let prepare = match crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(param_slot),
            "__typing_prepare_subst__",
        ) {
            Ok(p) => p,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        if !unsafe { pyre_object::is_none(prepare) } {
            let reshaped = crate::call::call_function_impl_result(
                prepare,
                &[
                    pyre_object::gc_roots::shadow_stack_get(self_slot),
                    pyre_object::gc_roots::shadow_stack_get(items_slot),
                ],
            )?;
            pyre_object::gc_roots::shadow_stack_set(items_slot, reshaped);
        }
    }
    // Non-tuple `items` (a broken `__typing_prepare_subst__`) counts as one arg
    // per CPython `_Py_subs_parameters`; a bare `w_tuple_len` would crash on it.
    let is_tuple_items = unsafe { is_tuple(pyre_object::gc_roots::shadow_stack_get(items_slot)) };
    let nitems = if is_tuple_items {
        unsafe { w_tuple_len(pyre_object::gc_roots::shadow_stack_get(items_slot)) }
    } else {
        1
    };
    if nparams != nitems {
        let direction = if nitems > nparams { "many" } else { "few" };
        let s = unsafe {
            crate::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(self_slot))?
        };
        if nitems < nparams {
            // A parameter carrying a default need not be supplied, so the
            // shortfall is measured against the required count rather than
            // `nparams` (`typing.py:1071` spells the same "expected at least"
            // form). Only the trailing run can be defaulted away: a default
            // cannot fill a missing required parameter ahead of it.
            let mut required = nparams;
            while required > 0 {
                let Some(param) = (unsafe {
                    w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(params_slot),
                        (required - 1) as i64,
                    )
                }) else {
                    break;
                };
                let has_default = match crate::baseobjspace::getattr_str(param, "has_default") {
                    Ok(method) => {
                        let result = crate::call::call_function_impl_result(method, &[])?;
                        crate::baseobjspace::is_true(result)?
                    }
                    Err(e) if e.kind == crate::PyErrorKind::AttributeError => false,
                    Err(e) => return Err(e),
                };
                if !has_default {
                    break;
                }
                required -= 1;
            }
            if nitems < required {
                return Err(crate::PyError::type_error(format!(
                    "Too few arguments for {s}; actual {nitems}, expected at least {required}"
                )));
            }
        }
        return Err(crate::PyError::type_error(format!(
            "Too {direction} arguments for {s}; actual {nitems}, expected {nparams}"
        )));
    }
    // `argitems` is the tuple view CPython indexes: `item` itself when it is a
    // tuple, otherwise a 1-tuple wrapping the single non-tuple `item`.
    let argitems = if is_tuple_items {
        pyre_object::gc_roots::shadow_stack_get(items_slot)
    } else {
        jit_w_tuple1(pyre_object::gc_roots::shadow_stack_get(items_slot))
    };
    let _ = pyre_object::gc_roots::pin_root(argitems);
    let argitems_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Slots rather than values: each entry has to survive every later
    // iteration's allocations before the caller builds the result from them.
    let mut newarg_slots: Vec<usize> = Vec::new();
    let mut push_newarg = |value: PyObjectRef, slots: &mut Vec<usize>| {
        let _ = pyre_object::gc_roots::pin_root(value);
        slots.push(pyre_object::gc_roots::shadow_stack_len() - 1);
    };
    let args_are_tuple = unsafe { is_tuple(pyre_object::gc_roots::shadow_stack_get(args_slot)) };
    let nargs = if args_are_tuple {
        unsafe { w_tuple_len(pyre_object::gc_roots::shadow_stack_get(args_slot)) }
    } else {
        unsafe { w_list_len(pyre_object::gc_roots::shadow_stack_get(args_slot)) }
    };
    for i in 0..nargs {
        let args_now = pyre_object::gc_roots::shadow_stack_get(args_slot);
        let old_arg = if args_are_tuple {
            unsafe { w_tuple_getitem(args_now, i as i64) }
        } else {
            unsafe { w_list_getitem(args_now, i as i64) }
        };
        let Some(old_arg) = old_arg else {
            continue;
        };
        if unsafe { is_type(old_arg) } {
            push_newarg(old_arg, &mut newarg_slots);
            continue;
        }
        let old_arg = pyre_object::gc_roots::pin_root(old_arg);
        let old_arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // CPython 3.14 `_Py_subs_parameters`: lists and tuples containing
        // parameters are recursively substituted, preserving their shape.
        if unsafe { is_tuple(old_arg) || is_list(old_arg) } {
            let subargs = subs_parameters(
                pyre_object::gc_roots::shadow_stack_get(self_slot),
                pyre_object::gc_roots::shadow_stack_get(old_arg_slot),
                pyre_object::gc_roots::shadow_stack_get(params_slot),
                pyre_object::gc_roots::shadow_stack_get(items_slot),
            )?;
            // The recursion returned through its own `_roots` scope, so the
            // entries are unrooted again; build the container before anything
            // else can allocate.
            let nested =
                if unsafe { is_tuple(pyre_object::gc_roots::shadow_stack_get(old_arg_slot)) } {
                    w_tuple_new(subargs)
                } else {
                    w_list_new(subargs)
                };
            push_newarg(nested, &mut newarg_slots);
            continue;
        }
        // `unpack = _is_unpacked_typevartuple(old_arg)` decides whether the
        // produced `arg` is spliced (`newargs.extend`) or appended.
        let unpack =
            is_unpacked_typevartuple(pyre_object::gc_roots::shadow_stack_get(old_arg_slot))?;
        // `meth = getattr(old_arg, '__typing_subst__', None)` then
        // `if meth is not None`: a missing attribute and an attribute
        // explicitly set to `None` both fall through to `subs_tvars`.
        let meth = match crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(old_arg_slot),
            "__typing_subst__",
        ) {
            Ok(m) => m,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        let arg = if !unsafe { pyre_object::is_none(meth) } {
            let _ = pyre_object::gc_roots::pin_root(meth);
            let meth_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let iparam = tuple_index(
                pyre_object::gc_roots::shadow_stack_get(params_slot),
                pyre_object::gc_roots::shadow_stack_get(old_arg_slot),
            )?
            .ok_or_else(|| crate::PyError::value_error("tuple.index(x): x not in tuple"))?;
            let item = unsafe {
                w_tuple_getitem(
                    pyre_object::gc_roots::shadow_stack_get(argitems_slot),
                    iparam as i64,
                )
            }
            .unwrap_or_else(w_none);
            crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(meth_slot),
                &[item],
            )?
        } else {
            subs_tvars(
                pyre_object::gc_roots::shadow_stack_get(old_arg_slot),
                pyre_object::gc_roots::shadow_stack_get(params_slot),
                pyre_object::gc_roots::shadow_stack_get(argitems_slot),
            )?
        };
        if unpack {
            let arg = pyre_object::gc_roots::pin_root(arg);
            let arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // GH-138497: an unpacked `__typing_subst__` must return a tuple.
            // (authority = CPython 3.14)
            if !unsafe { is_tuple(pyre_object::gc_roots::shadow_stack_get(arg_slot)) } {
                return Err(crate::PyError::type_error(format!(
                    "expected __typing_subst__ of {} objects to return a tuple, not {}",
                    crate::type_methods::arg_type_name(pyre_object::gc_roots::shadow_stack_get(
                        old_arg_slot
                    )),
                    crate::type_methods::arg_type_name(pyre_object::gc_roots::shadow_stack_get(
                        arg_slot
                    )),
                )));
            }
            let n = unsafe { w_tuple_len(pyre_object::gc_roots::shadow_stack_get(arg_slot)) };
            for j in 0..n {
                if let Some(x) = unsafe {
                    w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(arg_slot), j as i64)
                } {
                    push_newarg(x, &mut newarg_slots);
                }
            }
        } else {
            push_newarg(arg, &mut newarg_slots);
        }
    }
    let mut newargs = Vec::with_capacity(newarg_slots.len());
    for slot in newarg_slots {
        newargs.push(pyre_object::gc_roots::shadow_stack_get(slot));
    }
    Ok(newargs)
}

/// `subs_tvars(obj, params, argitems)` (`_pypy_generic_alias.py`) —
/// substitute the parameters of a nested generic and re-subscript it.
fn subs_tvars(
    mut obj: PyObjectRef,
    mut params: PyObjectRef,
    mut argitems: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let subparams = match pyre_object::with_roots!(obj, params, argitems =>
        crate::baseobjspace::getattr_str(obj, "__parameters__")
    ) {
        Ok(sub) => sub,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(obj),
        Err(e) => return Err(e),
    };
    if !unsafe { is_tuple(subparams) } || unsafe { w_tuple_len(subparams) } == 0 {
        return Ok(obj);
    }
    let nsub = unsafe { w_tuple_len(subparams) };
    // `tuple_index` runs a user `__eq__` on every turn, so the operands and the
    // arguments picked so far are held as shadow-stack slots and read back when
    // the substitution tuple is built.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, params, argitems, subparams]);
    let obj = || pyre_object::gc_roots::shadow_stack_get(base);
    let params = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let argitems = || pyre_object::gc_roots::shadow_stack_get(base + 2);
    let subparams = || pyre_object::gc_roots::shadow_stack_get(base + 3);
    let mut subarg_slots: Vec<usize> = Vec::with_capacity(nsub);
    for i in 0..nsub {
        let Some(param) = (unsafe { w_tuple_getitem(subparams(), i as i64) }) else {
            continue;
        };
        // `try: argitems[params.index(param)] except ValueError: param`.
        // The search runs `__eq__` on every turn, so `param` is published
        // before it and read back from the slot for each use afterwards.
        let param_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(param);
        let param = || pyre_object::gc_roots::shadow_stack_get(param_slot);
        let arg = match tuple_index(params(), param())? {
            Some(idx) => unsafe { w_tuple_getitem(argitems(), idx as i64) }.unwrap_or(param()),
            None => param(),
        };
        // `arg` is published here rather than in the branch below, because
        // the test that picks the branch reads `__module__` off the
        // parameter's type: `is_typevartuple` is a collecting call, and the
        // word `w_tuple_getitem` just returned is a pre-move address on the
        // other side of it.
        let arg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(arg);
        // `if isinstance(param, TypeVarTuple): subargs.extend(arg)` — a
        // `TypeVarTuple` captures a sequence, so its bound `arg` is spliced.
        if is_typevartuple(param()) {
            let members = crate::builtins::collect_iterable(
                pyre_object::gc_roots::shadow_stack_get(arg_slot),
            )?;
            let member_base = pyre_object::gc_roots::pin_roots(&members);
            for index in 0..members.len() {
                subarg_slots.push(member_base + index);
            }
        } else {
            subarg_slots.push(arg_slot);
        }
    }
    let mut subargs = Vec::with_capacity(subarg_slots.len());
    for &slot in &subarg_slots {
        subargs.push(pyre_object::gc_roots::shadow_stack_get(slot));
    }
    // Build the substitution tuple before reading `obj` back: `w_tuple_new`
    // allocates, so a receiver read ahead of it would be the pre-move one.
    let subs = w_tuple_new(subargs);
    crate::baseobjspace::getitem(obj(), subs)
}

/// `_make_starred(ga)` (`_pypy_generic_alias.py`) — a copy of the alias
/// flagged unpacked, so it renders `*X[...]` and `iter()` yields it.
pub(crate) fn make_starred(ga: PyObjectRef) -> crate::PyResult {
    let origin = unsafe { w_generic_alias_get_origin(ga) };
    let args = unsafe { w_generic_alias_get_args(ga) };
    let res = make_generic_alias(origin, args, pyre_object::PY_NULL)?;
    unsafe { w_generic_alias_set_unpacked(res, true) };
    Ok(res)
}

/// `GenericAlias.__reduce__` (CPython 3.14
/// `Objects/genericaliasobject.c:ga_reduce`).
fn ga_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_);
    let self_ = || pyre_object::gc_roots::shadow_stack_get(self_slot);
    let origin = unsafe { w_generic_alias_get_origin(self_()) };
    let ga_args = unsafe { w_generic_alias_get_args(self_()) };
    let pair = pyre_object::gc_roots::pin_roots(&[origin, ga_args]);
    if unsafe { w_generic_alias_get_unpacked(self_()) } {
        // 3.14 reconstructs a starred alias as `next(iter(orig))`.  This
        // replaces PyPy's app-level `_make_starred` reduce target and keeps
        // the callable globally pickleable without a synthetic module.
        let orig = make_generic_alias(
            pyre_object::gc_roots::shadow_stack_get(pair),
            pyre_object::gc_roots::shadow_stack_get(pair + 1),
            pyre_object::PY_NULL,
        )?;
        let orig_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(orig);
        let iterator =
            crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(orig_slot))?;
        let iter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(iterator);
        let args = jit_w_tuple1(pyre_object::gc_roots::shadow_stack_get(iter_slot));
        let args_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(args);
        let next_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(crate::baseobjspace::builtin_callable("next"));
        return Ok(w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(next_slot),
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        ]));
    }
    // `(type(self), (origin, args))`.
    let ga_type = crate::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE);
    let args = w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(pair),
        pyre_object::gc_roots::shadow_stack_get(pair + 1),
    ]);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args);
    Ok(w_tuple_new(vec![
        ga_type,
        pyre_object::gc_roots::shadow_stack_get(args_slot),
    ]))
}

/// `GenericAlias.__unpacked__` getset — the unpacked flag as a bool.
fn ga_get_unpacked(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(unsafe { w_generic_alias_get_unpacked(self_) }))
}

/// `GenericAlias.__typing_unpacked_tuple_args__` getset
/// (`_pypy_generic_alias.py`) — `args` when the alias is an unpacked
/// `tuple[...]`, else `None`.
fn ga_get_typing_unpacked_tuple_args(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Ok(w_none());
    }
    let unpacked = unsafe { w_generic_alias_get_unpacked(self_) };
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    let tuple_type = crate::typedef::gettypeobject(&pyre_object::TUPLE_TYPE);
    if unpacked && std::ptr::eq(origin, tuple_type) {
        Ok(unsafe { w_generic_alias_get_args(self_) })
    } else {
        Ok(w_none())
    }
}

/// `GenericAlias.__dir__` (`_pypy_generic_alias.py`) —
/// `sorted(_ATTR_EXCEPTIONS | set(dir(origin)))`.  Invoked from
/// `builtins::builtin_dir` for a GenericAlias receiver.
pub(crate) fn dir_list(ga: PyObjectRef) -> crate::PyResult {
    let origin = unsafe { w_generic_alias_get_origin(ga) };
    let dir_origin = crate::builtins::builtin_dir(&[origin])?;
    let mut names: Vec<Wtf8Buf> = ATTR_EXCEPTIONS
        .iter()
        .map(|s| {
            let mut buf = Wtf8Buf::new();
            buf.push_str(s);
            buf
        })
        .collect();
    let n = unsafe { w_list_len(dir_origin) };
    for i in 0..n {
        if let Some(item) = unsafe { w_list_getitem(dir_origin, i as i64) }
            && unsafe { is_str(item) }
        {
            // `dir(origin)` can name a lone surrogate. Keep the WTF-8 bytes;
            // their order matches code-point order, so `sorted` stays exact.
            let mut buf = Wtf8Buf::new();
            buf.push_wtf8(unsafe { w_str_get_wtf8(item) });
            names.push(buf);
        }
    }
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    names.dedup();
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in names {
        items.push(w_str_from_wtf8_managed(name));
    }
    Ok(w_list_new(items.take()))
}

/// `add_recurse` (`_pypy_generic_alias.py`) maps a bare `None`
/// operand to `type(None)` before it lands in `__args__`, so
/// `(int | None).__args__` is `(int, NoneType)`.
fn normalize_none(x: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::is_none(x) } {
        crate::typedef::gettypeobject(&pyre_object::NONE_TYPE)
    } else {
        x
    }
}

/// `_create_union(x, y)` (`_pypy_generic_alias.py`) — both operands
/// must be unionable, else `NotImplemented`; identical operands collapse.
pub fn create_union(x: PyObjectRef, y: PyObjectRef) -> crate::PyResult {
    use crate::objspace::descroperation::unionable;
    if !unionable(x) || !unionable(y) {
        return Ok(w_not_implemented());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let input_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(x);
    let _ = pyre_object::gc_roots::pin_root(y);
    // CPython 3.14 `_Py_union_type_or`: initialize a unionbuilder, then add
    // the operands in order.  The builder, rather than an eager `x == y`,
    // decides duplicates inside the construction-time hash partition.
    let raw_args = w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(input_base),
        pyre_object::gc_roots::shadow_stack_get(input_base + 1),
    ]);
    let _ = pyre_object::gc_roots::pin_root(raw_args);
    let raw_args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let parameters = collect_parameters(pyre_object::gc_roots::shadow_stack_get(raw_args_slot))?;
    build_union(
        &[
            pyre_object::gc_roots::shadow_stack_get(input_base),
            pyre_object::gc_roots::shadow_stack_get(input_base + 1),
        ],
        parameters,
    )
}

/// `UnionType(items)` construction used by `typing.Union[items]`.
///
/// Unlike `_create_union(x, y)`, the constructor does not apply the
/// operator-level `_unionable` gate: typing's Python-level `_GenericAlias`
/// instances are valid members even though a generic arbitrary object must
/// still make the direct `obj | type` operator return `NotImplemented`.
/// The remaining body is `UnionType.__init__`: collect parameters from the
/// raw items, recursively flatten nested unions, and deduplicate by equality.
pub(crate) fn union_from_items(items: &[PyObjectRef]) -> crate::PyResult {
    if items.len() == 1 {
        return Ok(normalize_none(items[0]));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let item_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in items {
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let current_items = || {
        (0..items.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(item_base + i))
            .collect::<Vec<_>>()
    };
    let raw_args = w_tuple_new(current_items());
    let _ = pyre_object::gc_roots::pin_root(raw_args);
    let raw_args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let parameters = collect_parameters(pyre_object::gc_roots::shadow_stack_get(raw_args_slot))?;
    build_union(&current_items(), parameters)
}

/// `typing._type_convert(arg)` for the string operands accepted by
/// `typing.Union[...]`, TypeVar `|`, and an already-created Union's `|`.
/// Plain `type | "name"` still follows the ordinary unionable check and
/// rejects the string before reaching this helper.
pub(crate) fn typing_type_convert(arg: PyObjectRef) -> crate::PyResult {
    if !unsafe { pyre_object::is_str(arg) } {
        return Ok(arg);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(arg);
    let arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let typing = crate::importing::check_sys_modules("typing")
        .ok_or_else(|| crate::PyError::type_error("typing module is not initialized"))?;
    let _ = pyre_object::gc_roots::pin_root(typing);
    let typing_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Module getattr may execute Python and collect.  Reload both the module
    // and operand from their PyPy-shaped roots before the call.
    let convert = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(typing_slot),
        "_type_convert",
    )?;
    let _ = pyre_object::gc_roots::pin_root(convert);
    let convert_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(convert_slot),
        &[pyre_object::gc_roots::shadow_stack_get(arg_slot)],
    )
}

/// CPython 3.14 `unionbuilder`.  Slots, rather than a side table keyed by
/// objects, are the transient equivalent of its owned list/set references and
/// keep every member live across user `__hash__` / `__eq__` callbacks.
struct UnionBuilder {
    hashable_set_slot: usize,
    member_slots: Vec<usize>,
    unhashable_slots: Vec<usize>,
}

impl UnionBuilder {
    fn new() -> Self {
        let hashable_set = pyre_object::w_set_new();
        let hashable_set_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(hashable_set);
        Self {
            hashable_set_slot,
            member_slots: Vec::new(),
            unhashable_slots: Vec::new(),
        }
    }

    fn add(&mut self, arg: PyObjectRef) -> Result<(), crate::PyError> {
        let arg = normalize_none(arg);
        if unsafe { pyre_object::is_union(arg) } {
            let _ = pyre_object::gc_roots::pin_root(arg);
            let arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let inner = unsafe {
                pyre_object::w_union_get_args(pyre_object::gc_roots::shadow_stack_get(arg_slot))
            };
            let _ = pyre_object::gc_roots::pin_root(inner);
            let inner_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let n = unsafe {
                pyre_object::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(inner_slot))
            };
            for i in 0..n {
                let member = unsafe {
                    pyre_object::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(inner_slot),
                        i as i64,
                    )
                };
                if let Some(member) = member {
                    self.add(member)?;
                }
            }
            return Ok(());
        }

        // Keep this candidate rooted for the builder's whole lifetime.  A
        // duplicate leaves one unused slot, matching the temporary strong
        // reference held by CPython's builder call and avoiding a nested root
        // scope that would also discard newly accepted member slots.
        let arg = pyre_object::gc_roots::pin_root(arg);
        let arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let unhashable = match crate::builtins::try_hash_value(
            pyre_object::gc_roots::shadow_stack_get(arg_slot),
        ) {
            Ok(hash) => {
                let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
                let key = unsafe { pyre_object::dictmultiobject::object_key_hashed(arg, hash) };
                let set = pyre_object::gc_roots::shadow_stack_get(self.hashable_set_slot);
                let present = unsafe { pyre_object::w_set_contains_key_checked(set, key) }
                    .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
                if present {
                    return Ok(());
                }
                // The contains probe may execute user equality and collect;
                // use both forwarded operands for the following set store.
                let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
                let set = pyre_object::gc_roots::shadow_stack_get(self.hashable_set_slot);
                unsafe { pyre_object::w_set_add_hashed_checked(set, arg, hash) }
                    .map_err(crate::baseobjspace::map_set_update_error)?;
                false
            }
            Err(_) => {
                // `unionbuilder_add_single_unchecked` clears the exception
                // from PyObject_Hash: any failed hash classifies this member
                // as unhashable.  Equality failures during list containment
                // remain observable.
                for &slot in &self.unhashable_slots {
                    if crate::baseobjspace::eq_w(
                        pyre_object::gc_roots::shadow_stack_get(slot),
                        pyre_object::gc_roots::shadow_stack_get(arg_slot),
                    )? {
                        return Ok(());
                    }
                }
                true
            }
        };

        self.member_slots.push(arg_slot);
        if unhashable {
            self.unhashable_slots.push(arg_slot);
        }
        Ok(())
    }

    fn finish(self, parameters_slot: usize) -> crate::PyResult {
        match self.member_slots.as_slice() {
            [] => Err(crate::PyError::type_error(
                "Cannot take a Union of no types.",
            )),
            [slot] => Ok(pyre_object::gc_roots::shadow_stack_get(*slot)),
            _ => {
                let hashable_args = pyre_object::w_frozenset_new();
                unsafe {
                    pyre_object::w_set_copy_storage_from(
                        hashable_args,
                        pyre_object::gc_roots::shadow_stack_get(self.hashable_set_slot),
                    )
                };
                let unhashable_args = if self.unhashable_slots.is_empty() {
                    pyre_object::PY_NULL
                } else {
                    pyre_object::w_tuple_new(
                        self.unhashable_slots
                            .iter()
                            .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
                            .collect(),
                    )
                };
                // Read the members and the parameters tuple out of the shadow
                // stack only here: the frozenset copy and the tuple above are
                // collection points, and a `Vec` or a by-value argument built
                // before them holds addresses the walk cannot forward.  The
                // frozenset itself is allocated old-gen and does not move.
                let members: Vec<_> = self
                    .member_slots
                    .iter()
                    .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
                    .collect();
                Ok(pyre_object::w_union_from_parts(
                    members,
                    hashable_args,
                    unhashable_args,
                    pyre_object::gc_roots::shadow_stack_get(parameters_slot),
                ))
            }
        }
    }
}

fn build_union(items: &[PyObjectRef], parameters: PyObjectRef) -> crate::PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(parameters);
    let parameters_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let item_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in items {
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let mut builder = UnionBuilder::new();
    for i in 0..items.len() {
        builder.add(pyre_object::gc_roots::shadow_stack_get(item_base + i))?;
    }
    builder.finish(parameters_slot)
}

/// CPython 3.14 `unions_equal`: compare the construction-time hashable
/// frozensets, then compare both directions of the unhashable tuple partition.
pub(crate) fn union_set_eq(a: PyObjectRef, b: PyObjectRef) -> Result<bool, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let a = pyre_object::gc_roots::pin_root(a);
        let a_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let b = pyre_object::gc_roots::pin_root(b);
        let b_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let ah = w_union_get_hashable_args(a);
        let ah_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ah);
        let bh = w_union_get_hashable_args(b);
        let bh_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(bh);
        if !crate::baseobjspace::eq_w(
            pyre_object::gc_roots::shadow_stack_get(ah_slot),
            pyre_object::gc_roots::shadow_stack_get(bh_slot),
        )? {
            return Ok(false);
        }
        let aa = w_union_get_unhashable_args(pyre_object::gc_roots::shadow_stack_get(a_slot));
        let bb = w_union_get_unhashable_args(pyre_object::gc_roots::shadow_stack_get(b_slot));
        if aa.is_null() || bb.is_null() {
            return Ok(aa.is_null() && bb.is_null());
        }
        let _ = pyre_object::gc_roots::pin_root(aa);
        let aa_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(bb);
        let bb_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let na = w_tuple_len(pyre_object::gc_roots::shadow_stack_get(aa_slot));
        let nb = w_tuple_len(pyre_object::gc_roots::shadow_stack_get(bb_slot));
        if na != nb {
            return Ok(false);
        }
        for i in 0..na {
            let Some(x) =
                w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(aa_slot), i as i64)
            else {
                return Ok(false);
            };
            let x_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(x);
            let mut found = false;
            for j in 0..nb {
                if let Some(y) =
                    w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(bb_slot), j as i64)
                {
                    let y_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(y);
                    if crate::baseobjspace::eq_w(
                        pyre_object::gc_roots::shadow_stack_get(x_slot),
                        pyre_object::gc_roots::shadow_stack_get(y_slot),
                    )? {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return Ok(false);
            }
        }
        for i in 0..nb {
            let Some(x) =
                w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(bb_slot), i as i64)
            else {
                return Ok(false);
            };
            let x_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(x);
            let mut found = false;
            for j in 0..na {
                if let Some(y) =
                    w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(aa_slot), j as i64)
                {
                    let y_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(y);
                    if crate::baseobjspace::eq_w(
                        pyre_object::gc_roots::shadow_stack_get(x_slot),
                        pyre_object::gc_roots::shadow_stack_get(y_slot),
                    )? {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// CPython 3.14 `union_hash` over the stored construction-time partitions.
pub(crate) fn union_hash_value(union: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let union = pyre_object::gc_roots::pin_root(union);
        let union_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let unhashable = w_union_get_unhashable_args(union);
        if !unhashable.is_null() {
            let unhashable = pyre_object::gc_roots::pin_root(unhashable);
            let unhashable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let n = w_tuple_len(pyre_object::gc_roots::shadow_stack_get(unhashable_slot));
            for i in 0..n {
                if let Some(member) = w_tuple_getitem(
                    pyre_object::gc_roots::shadow_stack_get(unhashable_slot),
                    i as i64,
                ) {
                    crate::builtins::try_hash_value(member)?;
                }
            }
            return Err(crate::PyError::type_error(format!(
                "union contains {n} unhashable elements"
            )));
        }
        crate::baseobjspace::hash_w_strict(w_union_get_hashable_args(
            pyre_object::gc_roots::shadow_stack_get(union_slot),
        ))
    }
}

/// `GenericAlias.__or__` (`_pypy_generic_alias.py`) — `X[...] | Y`.
fn ga_or(args: &[PyObjectRef]) -> crate::PyResult {
    let a = args.first().copied().unwrap_or_else(w_none);
    let b = args.get(1).copied().unwrap_or_else(w_none);
    create_union(a, b)
}

/// `GenericAlias.__ror__` (`_pypy_generic_alias.py`) — `Y | X[...]`.
fn ga_ror(args: &[PyObjectRef]) -> crate::PyResult {
    let a = args.first().copied().unwrap_or_else(w_none);
    let b = args.get(1).copied().unwrap_or_else(w_none);
    create_union(b, a)
}

/// `GenericAlias.__instancecheck__` (`_pypy_generic_alias.py`).
fn ga_instancecheck(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error(
        "isinstance() argument 2 cannot be a parameterized generic",
    ))
}

/// `GenericAlias.__subclasscheck__` (`_pypy_generic_alias.py`).
fn ga_subclasscheck(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error(
        "issubclass() argument 2 cannot be a parameterized generic",
    ))
}

/// `GenericAlias.__new__(cls, origin, args)` (`_pypy_generic_alias.py`)
/// — the public `types.GenericAlias(list, int)` constructor.
fn ga_new(args: &[PyObjectRef]) -> crate::PyResult {
    if args.len() != 3 {
        return Err(crate::PyError::type_error(format!(
            "GenericAlias expected 2 arguments, got {}",
            args.len().saturating_sub(1)
        )));
    }
    let mut cls = args[0];
    let mut generic_alias_type = crate::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE);
    // `_pypy_generic_alias.py GenericAlias.__new__` allocates through
    // `super(GenericAlias, cls).__new__(cls)`. `objspace.py`
    // `allocate_instance` keeps the builtin on the base layout and a subclass
    // on `typedef.py` `_getusercls`, including the `__del__` enqueue.
    crate::typedef::check_user_subclass(generic_alias_type, cls)?;
    pyre_object::with_roots!(cls, generic_alias_type => make_generic_alias(args[1], args[2], cls))
}

/// Build the `types.GenericAlias` namespace.
pub(crate) fn init_generic_alias_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    unsafe { crate::__pyre_put_new!(ns_slot, "__new__", crate::typedef::make_new_descr(ga_new)) };
    {
        let getter = make_builtin_function_with_arity("__origin__", ga_get_origin, 2);
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        unsafe {
            crate::__pyre_put_new!(
                ns_slot,
                "__origin__",
                crate::typedef::make_getset_descriptor_named(
                    pyre_object::gc_roots::shadow_stack_get(getter_slot),
                    "__origin__",
                )
            )
        };
    }
    {
        let getter = make_builtin_function_with_arity("__args__", ga_get_args, 2);
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        unsafe {
            crate::__pyre_put_new!(
                ns_slot,
                "__args__",
                crate::typedef::make_getset_descriptor_named(
                    pyre_object::gc_roots::shadow_stack_get(getter_slot),
                    "__args__",
                )
            )
        };
    }
    {
        let getter = make_builtin_function_with_arity("__parameters__", ga_get_parameters, 2);
        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(getter);
        unsafe {
            crate::__pyre_put_new!(
                ns_slot,
                "__parameters__",
                crate::typedef::make_getset_descriptor_named(
                    pyre_object::gc_roots::shadow_stack_get(getter_slot),
                    "__parameters__",
                )
            )
        };
    }
    unsafe { crate::__pyre_put_new!(ns_slot, "__eq__", make_builtin_function("__eq__", ga_eq)) };
    for (name, method) in [
        ("__ne__", ga_ne as fn(&[PyObjectRef]) -> crate::PyResult),
        ("__lt__", ga_ordering),
        ("__le__", ga_ordering),
        ("__gt__", ga_ordering),
        ("__ge__", ga_ordering),
    ] {
        unsafe {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                make_builtin_function_with_arity(name, method, 2)
            )
        };
    }
    // Pyre's ordinary operation dispatch has native fast paths for these
    // slots, but CPython 3.14 and PyPy also expose the methods through the
    // GenericAlias type dictionary for unbound calls.
    for (name, method, arity) in [
        (
            "__repr__",
            ga_repr as fn(&[PyObjectRef]) -> crate::PyResult,
            Some(1),
        ),
        ("__hash__", ga_hash, Some(1)),
        ("__call__", ga_call, None),
        ("__getattribute__", ga_getattribute, Some(2)),
        ("__iter__", ga_iter, Some(1)),
        ("__dir__", ga_dir, Some(1)),
    ] {
        unsafe {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                match arity {
                    Some(arity) => make_builtin_function_with_arity(name, method, arity),
                    None => make_builtin_function(name, method),
                }
            )
        };
    }
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getitem__",
            make_builtin_function("__getitem__", ga_getitem)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__mro_entries__",
            make_builtin_function("__mro_entries__", ga_mro_entries)
        )
    };
    unsafe { crate::__pyre_put_new!(ns_slot, "__or__", make_builtin_function("__or__", ga_or)) };
    unsafe { crate::__pyre_put_new!(ns_slot, "__ror__", make_builtin_function("__ror__", ga_ror)) };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__instancecheck__",
            make_builtin_function("__instancecheck__", ga_instancecheck)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__subclasscheck__",
            make_builtin_function("__subclasscheck__", ga_subclasscheck)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__reduce__",
            make_builtin_function("__reduce__", ga_reduce)
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__unpacked__",
            crate::typedef::make_getset_descriptor_named(
                make_builtin_function_with_arity("__unpacked__", ga_get_unpacked, 2),
                "__unpacked__",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__typing_unpacked_tuple_args__",
            crate::typedef::make_getset_descriptor_named(
                make_builtin_function_with_arity(
                    "__typing_unpacked_tuple_args__",
                    ga_get_typing_unpacked_tuple_args,
                    2,
                ),
                "__typing_unpacked_tuple_args__",
            )
        )
    };
    // Instance attribute access for `ga.__iter__`/`ga.__dir__` still delegates
    // to `__origin__` because they are not in `_ATTR_EXCEPTIONS`; the typedef
    // entries above provide CPython's `types.GenericAlias.__iter__(ga)` and
    // `types.GenericAlias.__dir__(ga)` unbound-call surface.
}

/// Render a GenericAlias for `repr()` (`GenericAlias.__repr__`,
/// `_pypy_generic_alias.py`).  Implemented here (not as a typedef
/// `__repr__`) so it matches the builtin-W_Root repr architecture, where
/// `display::py_repr` owns the rendering and explicit `.__repr__` access
/// still delegates to `__origin__`.
///
/// # Safety
/// `obj` must point to a valid `GenericAlias`.
pub(crate) unsafe fn repr(obj: PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    use rustpython_wtf8::Wtf8Buf;
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let origin = w_generic_alias_get_origin(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let _ = pyre_object::gc_roots::pin_root(origin);
    let origin_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let args = w_generic_alias_get_args(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let _ = pyre_object::gc_roots::pin_root(args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let current_args = || pyre_object::gc_roots::shadow_stack_get(args_slot);
    let n = w_tuple_len(current_args());
    // CPython's GenericAlias formatter compares against the actual
    // `collections.abc.Callable` object.  Spoofable `__module__` /
    // `__qualname__` metadata must not select Callable's flattened layout.
    let origin_is_callable =
        is_collections_abc_callable(pyre_object::gc_roots::shadow_stack_get(origin_slot))?;
    let inner = if origin_is_callable && n >= 1 {
        let result = w_tuple_getitem(current_args(), (n - 1) as i64).unwrap();
        let result_repr = repr_item(result)?;
        if n == 1 {
            crate::display::wtf8_format!("[], ", result_repr)
        } else {
            let first_slot =
                pyre_object::gc_roots::pin_roots(&[w_tuple_getitem(current_args(), 0).unwrap()]);
            let first = || pyre_object::gc_roots::shadow_stack_get(first_slot);
            if is_ellipsis(first()) {
                crate::display::wtf8_format!("..., ", result_repr)
            } else if n == 2 && (is_param_spec(first())? || is_typing_generic_alias(first())?) {
                crate::display::wtf8_format!(repr_item(first())?, ", ", result_repr)
            } else {
                let mut params = Vec::with_capacity(n - 1);
                for i in 0..n - 1 {
                    if let Some(item) = w_tuple_getitem(current_args(), i as i64) {
                        params.push(repr_item(item)?);
                    }
                }
                crate::display::wtf8_format!("[", join_wtf8(&params, ", "), "], ", result_repr)
            }
        }
    } else if n == 0 {
        Wtf8Buf::from_string("()".to_string())
    } else {
        let mut parts = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(current_args(), i as i64) {
                parts.push(if is_list(item) {
                    repr_items_list(item)?
                } else {
                    repr_item(item)?
                });
            }
        }
        join_wtf8(&parts, ", ")
    };
    let star = if w_generic_alias_get_unpacked(pyre_object::gc_roots::shadow_stack_get(obj_slot)) {
        "*"
    } else {
        ""
    };
    Ok(crate::display::wtf8_format!(
        star,
        repr_item(pyre_object::gc_roots::shadow_stack_get(origin_slot))?,
        "[",
        inner,
        "]"
    ))
}

/// `", ".join(parts)` for pieces that may hold a lone surrogate, which
/// `[Wtf8Buf]` has no `join` for.
fn join_wtf8(parts: &[rustpython_wtf8::Wtf8Buf], sep: &str) -> rustpython_wtf8::Wtf8Buf {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut index = 0;
    while index < parts.len() {
        if index > 0 {
            out.push_str(sep);
        }
        out.push_wtf8(&parts[index]);
        index += 1;
    }
    out
}

/// CPython 3.14 `ga_repr_items_list` — ParamSpec substitutions retain a
/// list, whose type items use typing-style rendering.  Fetch each element
/// after its predecessor's repr so mutation during a callback raises
/// `IndexError` rather than reading stale storage.
unsafe fn repr_items_list(list: PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    // An element's repr runs Python, and a `W_ListObject` header moves, so the
    // list is re-read from the shadow stack before every fetch.
    let _roots = pyre_object::gc_roots::push_roots();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(list);
    let list = || pyre_object::gc_roots::shadow_stack_get(list_slot);
    let n = w_list_len(list());
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let item = w_list_getitem(list(), i as i64)
            .ok_or_else(|| crate::PyError::index_error("list index out of range"))?;
        parts.push(repr_item(item)?);
    }
    Ok(crate::display::wtf8_format!(
        "[",
        join_wtf8(&parts, ", "),
        "]"
    ))
}

/// `_repr_item(it)` (`_pypy_generic_alias.py`) — a class renders as its
/// qualname (prefixed with the module when it is not `builtins`); anything
/// else falls back to `repr`.
pub(crate) unsafe fn repr_item(
    it: PyObjectRef,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(it);
    let item_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let current_item = || pyre_object::gc_roots::shadow_stack_get(item_slot);
    if is_ellipsis(current_item()) {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string("...".to_string()));
    }
    if is_generic_alias(current_item()) {
        return repr(current_item());
    }
    // `_pypy_generic_alias.py:129-130` checks
    // `isinstance(it, typing._GenericAlias)` before consulting
    // `__qualname__`; otherwise aliases such as
    // `typing.Concatenate[int, P]` collapse to `typing.Concatenate`.
    if is_typing_generic_alias(current_item())? {
        return unsafe { crate::display::py_repr_wtf8(current_item()) };
    }
    // `getattr(it, "__qualname__")` / `getattr(it, "__module__")`.
    if let Ok(w_qualname) = crate::baseobjspace::getattr_str(current_item(), "__qualname__")
        && let Ok(qualname) = crate::baseobjspace::text_w(w_qualname)
    {
        let qualname = qualname.to_string();
        let module = crate::baseobjspace::getattr_str(current_item(), "__module__")
            .ok()
            .and_then(|w| crate::baseobjspace::text_w(w).ok().map(str::to_string));
        return Ok(rustpython_wtf8::Wtf8Buf::from_string(match module {
            Some(m) if m != "builtins" => format!("{m}.{qualname}"),
            _ => qualname,
        }));
    }
    unsafe { crate::display::py_repr_wtf8(current_item()) }
}

fn is_collections_abc_callable(origin: PyObjectRef) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(origin);
    let origin_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let module = crate::importing::check_sys_modules("collections.abc")
        .or_else(|| crate::importing::check_sys_modules("_collections_abc"));
    let Some(module) = module else {
        return Ok(false);
    };
    let callable = crate::baseobjspace::getattr_str(module, "Callable")?;
    Ok(callable == pyre_object::gc_roots::shadow_stack_get(origin_slot))
}

fn is_typing_generic_alias(obj: PyObjectRef) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let Some(typing) = crate::importing::check_sys_modules("typing") else {
        return Ok(false);
    };
    let alias_type = crate::baseobjspace::getattr_str(typing, "_GenericAlias")?;
    Ok(unsafe {
        crate::baseobjspace::isinstance_w(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            alias_type,
        )
    })
}

/// `typing._is_param_expr`: the only non-alias object that remains as the
/// first element of a two-item `collections.abc.Callable.__args__` tuple is a
/// `ParamSpec`.  Concrete argument lists are flattened into the surrounding
/// tuple, while `Concatenate` is covered by `is_typing_generic_alias`.
fn is_param_spec(obj: PyObjectRef) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let Some(typing) = crate::importing::check_sys_modules("_typing") else {
        return Ok(false);
    };
    let param_spec_type = crate::baseobjspace::getattr_str(typing, "ParamSpec")?;
    Ok(unsafe {
        crate::baseobjspace::isinstance_w(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            param_spec_type,
        )
    })
}
