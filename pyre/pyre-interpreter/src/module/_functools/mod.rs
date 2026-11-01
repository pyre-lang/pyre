//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
//! `cmp_to_key` follows the stdlib fallback structurally: each invocation
//! creates a lexical `K`, capturing `mycmp` instead of exposing it on K.
//! `W_KeyWrapper` below carries the `keyobject` payload shape but is not what
//! `cmp_to_key` hands out — its `__call__` is generated positional-only, while
//! `keyobject` takes `obj` by keyword and reports `(obj)` to `inspect`.
//! `partial` follows the pure-Python `functools.partial` of
//! `lib-python/3/functools.py`, which PyPy runs (PyPy has no `_functools`):
//! its public state is exposed through read-only properties backed by private
//! slots. Placeholder argument merging follows the Python 3.14 `functools.py`
//! implementation. Its `__call__` is interp-level so the call snapshots the
//! state first.

use pyre_object::*;

/// `_functoolsmodule.c keyobject`.  The object returned by `cmp_to_key` is
/// itself the one-argument key factory; calling it produces another wrapper
/// carrying the value to compare.  Keeping both references on the payload
/// mirrors `keyobject.cmp` / `keyobject.object` and, unlike an app-level
/// function, leaves the module callable descriptor-neutral when a user stores
/// it on a class.
// CPython 3.14 Modules/_functoolsmodule.c:functools_exec creates
// keyobject_type_spec as an immutable module heap type.
#[pyre_interpreter::pyre_class("functools.KeyWrapper", cpython_heaptype)]
pub struct W_KeyWrapper {
    cmp: PyObjectRef,
    object: PyObjectRef,
}

fn key_wrapper_new(cmp: PyObjectRef, object: PyObjectRef) -> PyObjectRef {
    // The factory is called directly (rather than through `type.__call__`),
    // so force the TypeDef/typed-layout binding before allocating its payload.
    let _ = type_object();
    let _roots = pyre_object::gc_roots::push_roots();
    // Publish both livevars in one pass and read each back from its slot: a
    // slot is what a promotion rewrites, the Rust local naming the same object
    // is not.
    let slot = pyre_object::gc_roots::pin_roots(&[cmp, object]);
    W_KeyWrapper::allocate_stable(W_KeyWrapper {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        cmp: pyre_object::gc_roots::shadow_stack_get(slot),
        object: pyre_object::gc_roots::shadow_stack_get(slot + 1),
    })
}

fn key_wrapper_compare(
    this: &W_KeyWrapper,
    other: PyObjectRef,
    op: pyre_interpreter::baseobjspace::CompareOp,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let Some(other) = W_KeyWrapper::from_obj(other) else {
        return Err(pyre_interpreter::PyError::type_error(
            "other argument must be K instance",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    // `this` and `other` name payloads in the managed heap, so every operand
    // has to reach the shadow stack before the first forwarding query.
    let slot = pyre_object::gc_roots::pin_roots(&[this.cmp, this.object, other.object]);
    let result = pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(slot + 1),
            pyre_object::gc_roots::shadow_stack_get(slot + 2),
        ],
    )?;
    let _ = pyre_object::gc_roots::pin_root(result);
    // Build the comparison operand first: a custom numeric result makes
    // `w_int_new` allocate, and reading the result slot before that would leave
    // the argument naming a pre-move address.
    let w_zero = w_int_new(0);
    pyre_interpreter::baseobjspace::compare(
        pyre_object::gc_roots::shadow_stack_get(slot + 3),
        w_zero,
        op,
    )
}

#[pyre_interpreter::pyre_methods(unhashable)]
impl W_KeyWrapper {
    fn __call__(&self, obj: PyObjectRef) -> PyObjectRef {
        key_wrapper_new(self.cmp, obj)
    }

    #[getter]
    fn obj(&self) -> PyObjectRef {
        self.object
    }

    fn __lt__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Lt)
    }
    fn __le__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Le)
    }
    fn __eq__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Eq)
    }
    fn __ne__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Ne)
    }
    fn __gt__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Gt)
    }
    fn __ge__(&self, other: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        key_wrapper_compare(self, other, pyre_interpreter::baseobjspace::CompareOp::Ge)
    }
}

/// PyPy `module/_functools/interp_functools.py:reduce`.
///
/// Keep the accumulator in a GC-walked one-element list: `next()` and the
/// reduction callback may both collect, so a Rust local would not track a
/// relocated object between iterations.
fn reduce(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    // Bound scope: pos-only `function`, `iterable`, optional `initial`
    // (`PY_NULL` omitted).
    let function = args.first().copied().filter(|o| !o.is_null());
    let iterable = args.get(1).copied().filter(|o| !o.is_null());
    let given = usize::from(function.is_some()) + usize::from(iterable.is_some());
    let (Some(_function), Some(_iterable)) = (function, iterable) else {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "reduce() takes at least 2 positional arguments ({given} given)",
        )));
    };
    let _ = (_function, _iterable);

    let _roots = pyre_object::gc_roots::push_roots();
    // Publish every bound slot in one pass and read each back from its slot:
    // a slot is what a promotion rewrites.
    let base = pyre_object::gc_roots::pin_roots(args);
    let function_slot = base;
    let sequence_slot = base + 1;
    // `_functoolsmodule.c:reduce` reports the failed second argument itself
    // rather than letting the iteration protocol's own message surface.
    let w_iter = match pyre_interpreter::baseobjspace::iter(
        pyre_object::gc_roots::shadow_stack_get(sequence_slot),
    ) {
        Ok(w_iter) => w_iter,
        Err(err) if err.kind == pyre_interpreter::PyErrorKind::TypeError => {
            return Err(pyre_interpreter::PyError::type_error(
                "reduce() arg 2 must support iteration",
            ));
        }
        Err(err) => return Err(err),
    };
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let initial = if args.len() > 2 {
        let w = pyre_object::gc_roots::shadow_stack_get(base + 2);
        if w.is_null() { None } else { Some(w) }
    } else {
        None
    };
    let initial = if let Some(initial) = initial {
        initial
    } else {
        match pyre_interpreter::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(
            iter_slot,
        )) {
            Ok(value) => value,
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    return Err(pyre_interpreter::PyError::type_error(
                        "reduce() of empty iterable with no initial value",
                    ));
                } else {
                    return Err(err);
                }
            }
        }
    };
    let _ = pyre_object::gc_roots::pin_root(initial);
    let initial_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let accumulator =
        pyre_object::listobject::w_list_new(vec![pyre_object::gc_roots::shadow_stack_get(
            initial_slot,
        )]);
    let _ = pyre_object::gc_roots::pin_root(accumulator);
    let accumulator_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    loop {
        let item = match pyre_interpreter::baseobjspace::next(
            pyre_object::gc_roots::shadow_stack_get(iter_slot),
        ) {
            Ok(value) => value,
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(err);
            }
        };
        // `next()` returns a raw object reference.  Keep this iteration's
        // transient values in their own root scope: the reducer is arbitrary
        // Python and `w_list_setitem` may switch list strategy and allocate.
        let _iteration_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(item);
        let item_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let current = unsafe {
            pyre_object::listobject::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
                0,
            )
            .unwrap()
        };
        let result = pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(function_slot),
            &[current, pyre_object::gc_roots::shadow_stack_get(item_slot)],
        )?;
        let _ = pyre_object::gc_roots::pin_root(result);
        let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::listobject::w_list_setitem(
                pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
                0,
                pyre_object::gc_roots::shadow_stack_get(result_slot),
            );
        }
    }
    Ok(unsafe {
        pyre_object::listobject::w_list_getitem(
            pyre_object::gc_roots::shadow_stack_get(accumulator_slot),
            0,
        )
        .unwrap()
    })
}

fn partial_call(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "descriptor '__call__' of 'functools.partial' object needs an argument",
        ));
    }
    partial_call_impl(args[0], &args[1..], &[], &[])
}

fn partial_call_args(
    w_self: PyObjectRef,
    args: &pyre_interpreter::argument::Arguments,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    partial_call_impl(
        w_self,
        &args.arguments_w,
        args.keyword_names_w.as_deref().unwrap_or(&[]),
        args.keywords_w.as_deref().unwrap_or(&[]),
    )
}

/// `partial_vectorcall`.
///
/// [3.15-spec] `partial_vectorcall` reads `fn`, `args` and `kw` before it merges
/// the call's keywords, so a key whose `__hash__` re-enters `__setstate__`
/// cannot redirect the call.  PyPy has no `_functools` and calls the
/// app-level `functools.partial`, whose `**keywords` frame setup hashes
/// first.  Evidence: `test_functools.TestPartial.test_call_safety_against_reentrant_mutation`.
fn partial_call_impl(
    w_self: PyObjectRef,
    args_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    keywords_w: &[PyObjectRef],
) -> pyre_interpreter::PyResult {
    debug_assert_eq!(keyword_names_w.len(), keywords_w.len());
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::publish_roots(&[w_self]);
    let args_base = pyre_object::gc_roots::publish_roots(args_w);
    let names_base = pyre_object::gc_roots::publish_roots(keyword_names_w);
    let values_base = pyre_object::gc_roots::publish_roots(keywords_w);
    pyre_object::gc_roots::normalize_roots(
        self_slot,
        1 + args_w.len() + keyword_names_w.len() + keywords_w.len(),
    );
    let nargs = args_w.len();
    let n_call_kw = keyword_names_w.len();

    let pin_new = |obj: PyObjectRef| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        slot
    };
    let copy_range = |base: usize, n: usize| {
        let mut out = vec![pyre_object::PY_NULL; n];
        pyre_object::gc_roots::shadow_stack_copy_range(base, &mut out);
        out
    };

    let func_slot = pin_new(pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_func",
    )?);
    let pto_args_slot = pin_new(pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_args",
    )?);
    let pto_kw_slot = pin_new(pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_keywords",
    )?);
    let phcount_slot = pin_new(pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_phcount",
    )?);
    let merger_slot = pin_new(pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_merger",
    )?);
    let pto_args = pyre_object::gc_roots::shadow_stack_get(pto_args_slot);
    let pto_kw = pyre_object::gc_roots::shadow_stack_get(pto_kw_slot);
    let phcount_obj = pyre_object::gc_roots::shadow_stack_get(phcount_slot);
    if unsafe {
        !pyre_object::is_tuple(pto_args)
            || !pyre_object::is_dict(pto_kw)
            || !pyre_object::is_int(phcount_obj)
    } {
        return Err(pyre_interpreter::PyError::type_error(
            "invalid partial state",
        ));
    }
    let pto_args_len = unsafe { pyre_object::w_tuple_len(pto_args) };
    let phcount = pyre_interpreter::baseobjspace::int_w(phcount_obj)?;
    if phcount < 0 || (phcount as u64) > (pto_args_len as u64) {
        return Err(pyre_interpreter::PyError::type_error(
            "invalid partial state",
        ));
    }
    if (nargs as i64) < phcount {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "missing positional arguments in 'partial' call; expected at least {phcount}, got {nargs}"
        )));
    }
    let phcount = phcount as usize;

    let pto_items = pyre_interpreter::baseobjspace::fixedview(
        pyre_object::gc_roots::shadow_stack_get(pto_args_slot),
        -1,
    )?;
    let pto_n = pto_items.len();
    let pto_items_base = pyre_object::gc_roots::pin_roots(&pto_items);
    let mut concat = vec![pyre_object::PY_NULL; pto_n + nargs];
    pyre_object::gc_roots::shadow_stack_copy_range(pto_items_base, &mut concat[..pto_n]);
    if nargs > 0 {
        pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut concat[pto_n..]);
    }
    let concat_n = concat.len();
    let concat_base = pyre_object::gc_roots::pin_roots(&concat);
    let concat_live = copy_range(concat_base, concat_n);
    let positional = if phcount == 0 {
        concat_live
    } else {
        let concat_tuple_slot = pin_new(pyre_object::w_tuple_new(concat_live));
        let merged = pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(merger_slot),
            &[pyre_object::gc_roots::shadow_stack_get(concat_tuple_slot)],
        )?;
        let merged_slot = pin_new(merged);
        let merged_items = pyre_interpreter::baseobjspace::fixedview(
            pyre_object::gc_roots::shadow_stack_get(merged_slot),
            -1,
        )?;
        let merged_n = merged_items.len();
        let merged_items_base = pyre_object::gc_roots::pin_roots(&merged_items);
        let rest_n = nargs - phcount;
        let mut positional = vec![pyre_object::PY_NULL; merged_n + rest_n];
        pyre_object::gc_roots::shadow_stack_copy_range(
            merged_items_base,
            &mut positional[..merged_n],
        );
        if rest_n > 0 {
            pyre_object::gc_roots::shadow_stack_copy_range(
                args_base + phcount,
                &mut positional[merged_n..],
            );
        }
        positional
    };
    let pos_base = pyre_object::gc_roots::pin_roots(&positional);
    let pos_n = positional.len();

    let pto_nkw =
        unsafe { pyre_object::w_dict_len(pyre_object::gc_roots::shadow_stack_get(pto_kw_slot)) };
    let (kw_names, kw_values) = if pto_nkw == 0 {
        (
            copy_range(names_base, n_call_kw),
            copy_range(values_base, n_call_kw),
        )
    } else {
        let merged_slot = pin_new(unsafe {
            pyre_object::w_dict_copy(pyre_object::gc_roots::shadow_stack_get(pto_kw_slot))
        });
        for i in 0..n_call_kw {
            pyre_interpreter::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(merged_slot),
                pyre_object::gc_roots::shadow_stack_get(names_base + i),
                pyre_object::gc_roots::shadow_stack_get(values_base + i),
            )?;
        }
        let items = unsafe {
            pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(merged_slot))
        };
        let mut names = Vec::with_capacity(items.len());
        let mut values = Vec::with_capacity(items.len());
        for (k, v) in items {
            names.push(k);
            values.push(v);
        }
        (names, values)
    };
    let kw_n = kw_names.len();
    let kw_names_base = pyre_object::gc_roots::pin_roots(&kw_names);
    let kw_values_base = pyre_object::gc_roots::pin_roots(&kw_values);
    let pos_now = copy_range(pos_base, pos_n);
    if kw_n == 0 {
        pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(func_slot),
            &pos_now,
        )
    } else {
        let names_now = copy_range(kw_names_base, kw_n);
        let values_now = copy_range(kw_values_base, kw_n);
        pyre_interpreter::call::call_args(
            pyre_object::gc_roots::shadow_stack_get(func_slot),
            &pyre_interpreter::argument::Arguments::with_kw(&pos_now, &names_now, &values_now),
        )
    }
}

/// True when `object` is an instance of a `partial` class this module minted:
/// some class in `type(object)`'s MRO carries this `__call__` builtin in its
/// own type dict.  The class object itself is not stored; `_functools` can be
/// initialised more than once and each run produces a new class.
fn is_partial_instance(object: PyObjectRef) -> bool {
    let Some(w_type) = pyre_interpreter::typedef::r#type(object) else {
        return false;
    };
    let mro_ptr = unsafe { pyre_object::typeobject::w_type_get_mro(w_type.as_ptr()) };
    if mro_ptr.is_null() {
        return false;
    }
    for &cls in unsafe { (*mro_ptr).as_slice() } {
        let Some(descr) = pyre_interpreter::type_dict_lookup(cls, "__call__") else {
            continue;
        };
        if unsafe { !pyre_interpreter::is_function(descr) } {
            continue;
        }
        let code = unsafe { pyre_interpreter::function::getcode(descr) };
        if code.is_null() || unsafe { !pyre_interpreter::is_builtin_code(code) } {
            continue;
        }
        let func = unsafe { pyre_interpreter::builtin_code_get(code) };
        if pyre_interpreter::gateway::builtin_code_fn_eq(
            func,
            partial_call as pyre_interpreter::BuiltinCodeFn,
        ) {
            return true;
        }
    }
    false
}

static PARTIAL_CALL_OWNER: pyre_interpreter::gateway::MethodOwner =
    pyre_interpreter::gateway::MethodOwner {
        type_name: "functools.partial",
        is_instance: Some(is_partial_instance),
    };

pyre_interpreter::py_module! {
    "_functools",
    inline_app: {
        r#"
from operator import itemgetter as _partial_itemgetter
from reprlib import recursive_repr as _partial_recursive_repr
from types import GenericAlias as _PartialGenericAlias
from types import MethodType as _PartialMethodType


def _placeholder_immutable(name):
    return TypeError(
        f"cannot set {name!r} attribute of immutable type "
        "'functools._PlaceholderType'"
    )


class _PlaceholderMeta(type):
    # The accelerator's type is static, so `type_setattro` refuses both an
    # assignment and a deletion, with the same wording.  The pure-Python
    # fallback in `functools.py` is an ordinary class and accepts them; this
    # module stands in for the accelerator.
    def __setattr__(cls, name, value):
        raise _placeholder_immutable(name)

    def __delattr__(cls, name):
        raise _placeholder_immutable(name)


# The singleton cannot live on the class, whose attributes are now read-only.
_placeholder_instance = None


class _PlaceholderType(metaclass=_PlaceholderMeta):
    """The type of the Placeholder singleton."""

    __module__ = "functools"
    __slots__ = ()

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "type 'functools._PlaceholderType' is not an acceptable base type"
        )

    def __new__(cls):
        global _placeholder_instance
        if _placeholder_instance is None:
            _placeholder_instance = object.__new__(cls)
        return _placeholder_instance

    def __repr__(self):
        return "Placeholder"

    def __reduce__(self):
        return "Placeholder"


Placeholder = _PlaceholderType()


def _partial_prepare_merger(args):
    if not args:
        return 0, None
    nargs = len(args)
    order = []
    j = nargs
    for i, arg in enumerate(args):
        if arg is Placeholder:
            order.append(j)
            j += 1
        else:
            order.append(i)
    phcount = j - nargs
    merger = _partial_itemgetter(*order) if phcount else None
    return phcount, merger


class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    # CPython's accelerator keeps these three fields in the partial object and
    # exposes them through read-only member descriptors.  Private slots plus
    # getter-only properties preserve the same ownership and mutability shape.
    __slots__ = (
        "_func", "_args", "_keywords", "_phcount", "_merger",
        "__dict__", "__weakref__",
    )

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")
        if args and args[-1] is Placeholder:
            raise TypeError("trailing Placeholders are not allowed")
        for value in keywords.values():
            if value is Placeholder:
                raise TypeError("Placeholder cannot be passed as a keyword argument")

        # `partial_new` unpacks a wrapped partial only while it carries no
        # instance state (`part->dict == NULL`); placeholder positions and bound
        # arguments are merged for subclasses too, since subclassing alone does
        # not block the optimization.  The slots above keep `__dict__` empty until
        # user code assigns an attribute, so a partial holding its own state stays
        # wrapped and keeps running its own behaviour.
        if isinstance(func, partial) and not func.__dict__:
            pto_phcount = func._phcount
            tot_args = func.args
            if args:
                tot_args += args
                if pto_phcount:
                    nargs = len(args)
                    if nargs < pto_phcount:
                        tot_args += (Placeholder,) * (pto_phcount - nargs)
                    tot_args = func._merger(tot_args)
                    if nargs > pto_phcount:
                        tot_args += args[pto_phcount:]
                phcount, merger = _partial_prepare_merger(tot_args)
            else:
                phcount, merger = pto_phcount, func._merger
            keywords = {**func.keywords, **keywords}
            func = func.func
        else:
            tot_args = args
            phcount, merger = _partial_prepare_merger(tot_args)

        self = object.__new__(cls)
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_args", tot_args)
        object.__setattr__(self, "_keywords", keywords)
        object.__setattr__(self, "_phcount", phcount)
        object.__setattr__(self, "_merger", merger)
        return self

    @_partial_recursive_repr()
    def __repr__(self):
        cls = type(self)
        func, p_args, keywords = self.func, self.args, self.keywords
        args = [repr(func)]
        args.extend(map(repr, p_args))
        args.extend(f"{key}={value!r}" for key, value in keywords.items())
        return f"{cls.__module__}.{cls.__qualname__}({', '.join(args)})"


    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def keywords(self):
        return self._keywords

    def __delattr__(self, name):
        if name == "__dict__":
            raise TypeError("a partial object's dictionary may not be deleted")
        object.__delattr__(self, name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _PartialMethodType(self, obj)

    def __reduce__(self):
        return (
            type(self),
            (self.func,),
            (self.func, self.args, self.keywords or None, self.__dict__ or None),
        )

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (
            not callable(func)
            or not isinstance(args, tuple)
            or (kwds is not None and not isinstance(kwds, dict))
            or (namespace is not None and not isinstance(namespace, dict))
        ):
            raise TypeError("invalid partial state")
        if args and args[-1] is Placeholder:
            raise TypeError("trailing Placeholders are not allowed")

        phcount, merger = _partial_prepare_merger(args)
        args = tuple(args)
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_keywords", kwds)
        object.__setattr__(self, "_phcount", phcount)
        object.__setattr__(self, "_merger", merger)

    __class_getitem__ = classmethod(_PartialGenericAlias)


partial.__module__ = "functools"


def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

# `_functools.cmp_to_key` is an interp-level builtin in CPython.  Unlike an
# app-level function, it therefore does not acquire an instance when a caller
# stores it on a class (the CPython functools tests do exactly that).  A
# callable staticmethod preserves the app-level implementation while giving
# the exported object the same non-binding descriptor behavior.
cmp_to_key = staticmethod(cmp_to_key)
"# => ["cmp_to_key", "partial", "Placeholder", "_PlaceholderType"],
    },
    functions: {
        "reduce" / * = reduce; pyre_interpreter::Signature::new(vec!["function", "iterable", "initial"], None, None, 0, 2),
    },
    extra_init: |ns| {
        if let Some(mut function) = unsafe { pyre_object::w_dict_getitem_str(ns, "reduce") } {
            function = pyre_interpreter::gateway::function_set_null_positional_defaults(function, 1);
            pyre_interpreter::__pyre_store!(ns, "reduce", function);
        }
        let Some(partial_cls) = (unsafe { pyre_object::w_dict_getitem_str(ns, "partial") }) else {
            return Err(pyre_interpreter::PyError::type_error(
                "_functools.partial is missing",
            ));
        };
        let roots = pyre_object::gc_roots::push_roots();
        let partial_slot = roots.base();
        let _ = roots.pin_root(partial_cls);
        let wrapper = pyre_interpreter::gateway::make_method_descriptor_passthrough1_with_doc(
            "__call__",
            partial_call,
            partial_call_args,
            "Call self as a function.",
        );
        let wrapper_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(wrapper);
        let code = unsafe { pyre_interpreter::function::getcode(roots.get(wrapper_slot)) };
        unsafe {
            pyre_interpreter::gateway::builtin_code_set_owner(code, &PARTIAL_CALL_OWNER);
        }
        let qualname_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new("partial.__call__"));
        unsafe {
            pyre_interpreter::function::function_set_qualname(
                roots.get(wrapper_slot),
                roots.get(qualname_slot),
            );
            pyre_interpreter::function::function_set_objclass(
                roots.get(wrapper_slot),
                roots.get(partial_slot),
            );
        }
        pyre_interpreter::baseobjspace::setattr_str(
            roots.get(partial_slot),
            "__call__",
            roots.get(wrapper_slot),
        )?;
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `_functools.keyobject`: the comparator and wrapped object are both
    // managed edges.
    types.push(ModuleGcType {
        descriptor: <W_KeyWrapper as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
