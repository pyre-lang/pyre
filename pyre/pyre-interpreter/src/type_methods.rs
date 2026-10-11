//! Builtin type method implementations.
//!
//! PyPy equivalents:
//!   pypy/objspace/std/listobject.py  (list methods)
//!   pypy/objspace/std/unicodeobject.py  (str methods)
//!   pypy/objspace/std/dictmultiobject.py  (dict methods)
//!   pypy/objspace/std/tupleobject.py  (tuple methods)
//!
//! Separated from space.rs to avoid bloating the hot-path compilation
//! unit. Method functions are registered into TypeDef at startup.

use majit_rlib::rbigint::RBigInt as BigInt;
use num_traits::ToPrimitive;
use pyre_object::*;
// `classify`/`case`/`identifier` back the `str` predicate and `casefold`
// methods with the runtime-independent Unicode tables shared with the
// `unicodedata` module.  Those tables track Unicode 17.0.0, one release ahead
// of Python 3.14's 16.0.0, so predicates read a property on the ~500 code
// points release 17 newly assigns (unassigned under 16.0.0) — the same
// data-version skew already noted in `module/unicodedata`, kept consistent
// across `str` and `unicodedata` here.
use rustpython_unicode::{case, classify, identifier};
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};
// ── Arity checks for builtin methods ─────────────────────────────────
// A builtin method registered with `make_builtin_function_with_arity`
// records an arity only as a fast-path dispatch hint — some stubs register
// a hint that differs from their real signature, so the call machinery
// cannot treat it as a contract. Methods therefore validate their own
// argument count and raise instead of asserting. `args[0]` is the bound
// receiver, so the counts in these messages exclude it.

/// The argument count `args` carries once its receiver slot is discounted —
/// `len() - 1`, floored at zero for the malformed empty slice.
///
/// Spelled as an explicit branch rather than `saturating_sub`: the front end
/// lowers a plain subtraction (`Sub` plus an overflow `Assert` it strips), but
/// `core::num::<Impl>::saturating_sub` is an unregistered foreign leaf that
/// stops the enclosing graph before it can be lifted.
#[inline]
pub(crate) fn args_given(args: &[PyObjectRef]) -> usize {
    if args.is_empty() { 0 } else { args.len() - 1 }
}

/// The `i`th entry of a gateway argument slice, or `None` when the caller did
/// not pass it — the optional-positional-argument default.
///
/// Spelled as an explicit bounds test rather than `args.get(i)`: `slice::get`
/// yields an `Option<&T>` whose payload is valid on only one arm, and the front
/// end has no lowering that can build one.  Its single `Option`-synthesizing
/// path, `emit_tagged_pair_aggregate`, writes the payload unconditionally
/// before the consumer's discriminant switch runs (`front/mir.rs`), so an
/// out-of-bounds read would execute on the absent arm.  A plain index lowers to
/// a native read plus an overflow `Assert` the front end strips.
#[inline]
pub(crate) fn arg_or_none(args: &[PyObjectRef], i: usize) -> PyObjectRef {
    if i < args.len() { args[i] } else { w_none() }
}

/// `TYPE.NAME`, the name a bound builtin method reports itself under.  TYPE
/// comes from the receiver rather than from the type the method is declared
/// on, so a call on a subclass instance names the subclass
/// (`class L(list)` — `L().copy` reports `L.copy`); a receiver that IS a type
/// reports under its own name, which is what a class method binds to
/// (`dict.fromkeys`, not `type.fromkeys`).  With no receiver at all the bare
/// name is used; `require_receiver` normally rejects that first.
fn method_qualname(args: &[PyObjectRef], name: &str) -> String {
    match args.first() {
        Some(&receiver) => {
            let ty = if unsafe { pyre_object::typeobject::is_type(receiver) } {
                unsafe { pyre_object::w_type_get_name(receiver) }.to_string()
            } else {
                crate::baseobjspace::object_functionstr_type_name(receiver)
            };
            format!("{ty}.{name}")
        }
        None => name.to_string(),
    }
}

/// The builtin type a method reached through `receiver` is declared on — the
/// first non-heap type in the receiver's MRO.  Whether a name is filled by a
/// `wrapper_descriptor` or a `method_descriptor` is a property of the
/// declaring type, not of the subclass the call came in on, so the wording
/// split reads this while the qualified name reads the receiver's own type.
fn declaring_builtin_type_name(receiver: PyObjectRef) -> String {
    unsafe {
        let Some(w_type) = crate::typedef::r#type(receiver) else {
            return String::new();
        };
        let mro = pyre_object::w_type_get_mro(w_type.as_ptr());
        if !mro.is_null() {
            for &entry in (*mro).as_slice() {
                if !pyre_object::typeobject::w_type_is_heaptype(entry) {
                    return pyre_object::w_type_get_name(entry).to_string();
                }
            }
        }
        pyre_object::w_type_get_name(w_type.as_ptr()).to_string()
    }
}

/// TypeError for a keyword handed to a builtin method that takes a fixed
/// number of positional arguments.
///
/// Every arity helper here counts the raw slice length, so the body it guards
/// has no parameter a keyword could name — a keyword-aware builtin reads the
/// trailing marker dict itself (`split_builtin_kwargs`) and never validates by
/// slice length.  Without this the marker would be counted as a positional
/// value and reach the implementation as one.  A slot wrapper reports itself
/// as `wrapper NAME()`, everything else under its qualified name.
pub(crate) fn reject_kwargs(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    reject_kwargs_of(None, args, name)
}

/// [`reject_kwargs`] for a method whose declaring class is fixed rather than
/// read off the receiver — see [`arity_no_args_of`].
pub fn reject_kwargs_of(
    owner: Option<&str>,
    args: &[PyObjectRef],
    name: &str,
) -> Result<(), crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if !crate::builtins::has_real_kwargs(kwargs) {
        return Ok(());
    }
    let slot = positional.first().is_some_and(|&receiver| {
        crate::gateway::is_slot_wrapper(&declaring_builtin_type_name(receiver), name)
    });
    let subject = match (slot, owner) {
        (true, _) => format!("wrapper {name}"),
        (false, Some(owner)) => format!("{owner}.{name}"),
        (false, None) => method_qualname(positional, name),
    };
    Err(crate::PyError::type_error(format!(
        "{subject}() takes no keyword arguments"
    )))
}

/// TypeError for a method requiring exactly `n` positional arguments after
/// the receiver, called with a different count.  `name` is the bare method
/// name; the message qualifies it with the receiver's type.
pub(crate) fn arity_exact(
    args: &[PyObjectRef],
    name: &str,
    n: usize,
) -> Result<(), crate::PyError> {
    reject_kwargs(args, name)?;
    if args.len() != n + 1 {
        let expected = match n {
            0 => "no arguments".to_string(),
            1 => "exactly one argument".to_string(),
            k => format!("exactly {k} arguments"),
        };
        return Err(crate::PyError::type_error(format!(
            "{}() takes {expected} ({} given)",
            method_qualname(args, name),
            args_given(args),
        )));
    }
    Ok(())
}

/// TypeError for a method accepting at least `min` positional arguments
/// after the receiver, called with fewer — the PyArg_UnpackTuple
/// "X expected at least N arguments, got M" form (`str.index`, `dict.get`).
pub fn arity_at_least(args: &[PyObjectRef], name: &str, min: usize) -> Result<(), crate::PyError> {
    reject_kwargs(args, name)?;
    if args.len() < min + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at least {min} argument{}, got {}",
            if min == 1 { "" } else { "s" },
            args_given(args),
        )));
    }
    Ok(())
}

/// Validate that a str search method's substring (`args[1]`) is a `str`,
/// else raise `TypeError("{method}() argument 1 must be str, not {type}")`.
/// The search path reads `args[1]` as a `W_UnicodeObject`; a non-str would
/// be dereferenced as one. `args[1]` is guaranteed present by the caller's
/// preceding `arity_at_least(_, _, 1)`.
pub(crate) fn require_str_sub(args: &[PyObjectRef], method: &str) -> Result<(), crate::PyError> {
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 1 must be str, not {}",
            clinic_arg_type_name(args[1])
        )));
    }
    Ok(())
}

/// TypeError for a method accepting at most `max` positional arguments after
/// the receiver, called with more — the METH_VARARGS "X expected at most N
/// arguments, got M" form (`list.index`, `dict.pop`).
pub fn arity_at_most(args: &[PyObjectRef], name: &str, max: usize) -> Result<(), crate::PyError> {
    reject_kwargs(args, name)?;
    if args.len() > max + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at most {max} argument{}, got {}",
            if max == 1 { "" } else { "s" },
            args_given(args),
        )));
    }
    Ok(())
}

/// TypeError for a method whose positional count after the receiver has to
/// land in `min..=max` — the PyArg_UnpackTuple form with both bounds set
/// (`bytes.center`, `bytes.ljust`).  Reports whichever bound the call missed.
pub fn arity_between(
    args: &[PyObjectRef],
    name: &str,
    min: usize,
    max: usize,
) -> Result<(), crate::PyError> {
    arity_at_least(args, name, min)?;
    arity_at_most(args, name, max)
}

/// TypeError for a method requiring exactly `n` positional arguments after
/// the receiver, called with a different count — the PyArg_UnpackTuple
/// min==max form "X expected N arguments, got M" (`list.insert`,
/// `list.__setitem__`).  Unlike `arity_exact`, the message carries neither a
/// trailing "()" nor the "exactly" wording; "argument" is singular when
/// `n == 1`.
pub fn arity_exact_unpack(
    args: &[PyObjectRef],
    name: &str,
    n: usize,
) -> Result<(), crate::PyError> {
    reject_kwargs(args, name)?;
    if args.len() != n + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected {n} argument{}, got {}",
            if n == 1 { "" } else { "s" },
            args_given(args),
        )));
    }
    Ok(())
}

/// TypeError for a slot wrapper requiring exactly `n` positional arguments
/// after the receiver — the `check_num_args` (typeobject.c) form
/// "expected N argument(s), got M", with no method name.  "argument" is
/// singular when `n == 1`.
pub(crate) fn arity_slot(args: &[PyObjectRef], n: usize) -> Result<(), crate::PyError> {
    if args.len() != n + 1 {
        return Err(crate::PyError::type_error(format!(
            "expected {n} argument{}, got {}",
            if n == 1 { "" } else { "s" },
            args_given(args),
        )));
    }
    Ok(())
}

/// TypeError for a METH_NOARGS method called with positional arguments —
/// the "X() takes no arguments (M given)" form (`list.__reversed__`).  `name`
/// is the bare method name; the message qualifies it with the receiver's type.
pub(crate) fn arity_no_args(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    arity_no_args_of(None, args, name)
}

/// [`arity_no_args`] with an optional explicit declaring class for methods
/// whose owner cannot be recovered from the receiver.
pub(crate) fn arity_no_args_of(
    owner: Option<&str>,
    args: &[PyObjectRef],
    name: &str,
) -> Result<(), crate::PyError> {
    reject_kwargs_of(owner, args, name)?;
    if args.len() != 1 {
        let qualname = match owner {
            Some(owner) => format!("{owner}.{name}"),
            None => method_qualname(args, name),
        };
        return Err(crate::PyError::type_error(format!(
            "{qualname}() takes no arguments ({} given)",
            args_given(args),
        )));
    }
    Ok(())
}

/// TypeError for the ternary-power slot (`__pow__` / `__rpow__`), which
/// accepts one or two positional arguments after the receiver — the
/// "expected 1 or 2 arguments, got M" form with no method name.  `name` is
/// carried only for the keyword rejection, which does name the wrapper.
pub(crate) fn arity_pow(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    reject_kwargs(args, name)?;
    let extra = args_given(args);
    if !(1..=2).contains(&extra) {
        return Err(crate::PyError::type_error(format!(
            "expected 1 or 2 arguments, got {extra}"
        )));
    }
    Ok(())
}

/// `int.__round__` / `float.__round__`, which share `round()`'s body but are
/// methods: the receiver is the number, `ndigits` the one optional argument,
/// and the count in a mismatch excludes the receiver
/// (`__round__ expected at most 1 argument, got 2`).
pub fn number_dunder_round(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "__round__")?;
    arity_at_most(args, "__round__", 1)?;
    crate::builtins::builtin_round_dunder(args)
}

/// TypeError for an unbound method descriptor invoked with no receiver
/// (`list.append()` with zero arguments) — `args` is empty.
pub(crate) fn require_receiver(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of object needs an argument"
        )));
    }
    Ok(())
}

/// The receiver check normally supplied by PyPy's
/// `interp2app(W_ListObject.descr_*)` gateway.  CPython 3.14 exposes list's
/// slot wrappers and ordinary methods as two descriptor kinds with distinct
/// public mismatch messages, so callers identify which surface they expose.
pub(crate) fn require_list_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method list.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'list' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_list(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!("descriptor '{name}' for 'list' objects doesn't apply to a '{received}' object")
        } else {
            format!("descriptor '{name}' requires a 'list' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_AbstractTupleObject.descr_*)` gateway.  As with list,
/// CPython 3.14 distinguishes slot-wrapper and method-descriptor mismatch
/// messages on the public surface.
pub(crate) fn require_tuple_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method tuple.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'tuple' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_tuple(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'tuple' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a 'tuple' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// Receiver check supplied by PyPy's
/// `interp2app(W_DictMultiObject.descr_*)` gateway.
pub(crate) fn require_dict_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method dict.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'dict' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe {
        crate::baseobjspace::isinstance_w(
            receiver,
            crate::typedef::gettypeobject(&pyre_object::DICT_TYPE),
        )
    } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!("descriptor '{name}' for 'dict' objects doesn't apply to a '{received}' object")
        } else {
            format!("descriptor '{name}' requires a 'dict' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_SetObject.descr_*)` gateway.  CPython 3.14 exposes set's
/// slot wrappers and ordinary methods as two descriptor kinds with distinct
/// public mismatch messages.
pub(crate) fn require_set_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method set.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'set' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_set(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!("descriptor '{name}' for 'set' objects doesn't apply to a '{received}' object")
        } else {
            format!("descriptor '{name}' requires a 'set' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_FrozensetObject.descr_*)` gateway.  The inherited
/// `W_BaseSetObject` implementation is shared with set, while the gateway
/// still requires a frozenset receiver.
pub(crate) fn require_frozenset_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method frozenset.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'frozenset' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_frozenset(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'frozenset' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a 'frozenset' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_SetIterObject.descr_*)` gateway.  Python 3.14 exposes
/// `__iter__`/`__next__` as slot wrappers and the remaining operations as
/// method descriptors.
pub(crate) fn require_set_iterator_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method set_iterator.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'set_iterator' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_set_iterator(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'set_iterator' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!(
                "descriptor '{name}' requires a 'set_iterator' object but received a '{received}'"
            )
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// Receiver validation supplied by `W_AbstractRangeIterator.typedef`'s
/// gateways. Python 3.14 gives the machine-word and arbitrary-precision
/// implementations distinct public owner names even though PyPy shares the
/// abstract typedef implementation.
pub(crate) fn require_range_iterator_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
    long: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let owner = if long {
        "longrange_iterator"
    } else {
        "range_iterator"
    };
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method {owner}.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of '{owner}' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    let matches = unsafe {
        if long {
            pyre_object::is_long_range_iter(receiver)
        } else {
            pyre_object::is_range_iter(receiver)
        }
    };
    if !matches {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for '{owner}' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a '{owner}' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// Receiver-only arity for `str` methods that take no arguments (`isspace`,
/// `lower`, …).  Rejects a missing receiver and any extra positional argument,
/// matching `str.{name}() takes no arguments (N given)`.
pub(crate) fn require_no_args(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of 'str' object needs an argument"
        )));
    }
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "str.{name}() takes no arguments ({} given)",
            args.len() - 1
        )));
    }
    Ok(())
}

// ── List methods ─────────────────────────────────────────────────────
// All take self (list) as first arg.

pub fn list_method_append(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "append", true)?;
    arity_exact(args, "append", 1)?;
    unsafe { w_list_append(args[0], args[1]) };
    Ok(w_none())
}

/// `listobject.py ListStrategy.extend` — dispatcher only.  Each arm's loop
/// lives in its own function so `contains_loop` cannot decline this graph
/// and turn `list(iterable)` / `sorted(x)` into one residual CALL.
pub fn list_method_extend(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "extend", true)?;
    arity_exact(args, "extend", 1)?;
    list_extend_items(args[0], args[1])?;
    Ok(w_none())
}

/// `ListStrategy.extend` without the gateway arity/receiver checks.
/// `descr_init` calls this after its own receiver check so the descent
/// scan of `list((i, 3, 1))` does not inherit `args.first` / `format!`
/// from `require_list_receiver`.
pub(crate) fn list_extend_items(
    mut list: PyObjectRef,
    mut other: PyObjectRef,
) -> Result<(), crate::PyError> {
    // listobject.py extend only takes the storage-copy path when a
    // list/tuple uses its inherited iterator.  An overridden subclass
    // must use the generic incremental iterator path below.
    if unsafe { is_exact_list(other) } {
        extend_from_list(list, other)?;
    } else if unsafe { is_exact_tuple(other) } {
        extend_from_tuple(list, other)?;
    } else if unsafe {
        pyre_object::is_set_or_frozenset(other)
            && (*other).w_class == pyre_object::get_instantiate(&*(*other).ob_type)
    } {
        extend_from_set(list, other)?;
    } else if unsafe { crate::pyframe::frame_locals_proxy::is_frame_locals_proxy(other) } {
        // `framelocalsproxy_iter` is `iter(self.keys())`.  The key scan is
        // `pyframe.py fast2locals`'s positional loops
        // (`extend_from_frame_locals_proxy`), appended onto this list.
        // `false` means `other` was not a proxy; drain it as a generic
        // iterable.
        if !pyre_object::with_roots!(list, other => extend_from_frame_locals_proxy(list, other)) {
            extend_from_iterable(list, other)?;
        }
    } else {
        extend_from_iterable(list, other)?;
    }
    Ok(())
}

/// `listobject.py ListStrategy._extend_from_list`, a per-strategy storage
/// copy (`l += ...`) the JIT calls rather than inlines.  The
/// `dont_look_inside` call target gives that residual call a real fnaddr,
/// which the descent scan of `descr_init` requires of every helper it can
/// reach.
#[majit_macros::dont_look_inside]
fn extend_from_list(mut list: PyObjectRef, mut other: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        // BaseRangeListStrategy.extend switches its receiver before even
        // an empty donor is examined. Keep both operands rooted across
        // that materialisation and the append loop it feeds.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::pin_roots(&[list, other]);
        list = pyre_object::listobject::w_list_materialize_range(list);
        other = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let n = w_list_len(other);
        if w_list_len(list) == 0 && pyre_object::listobject::w_list_is_range_strategy(other) {
            // EmptyListStrategy._extend_from_list delegates to the
            // donor's copy_into; BaseRangeListStrategy shares its
            // immutable erased tuple rather than appending boxed ints.
            pyre_object::listobject::w_list_setslice_mode(
                list,
                0,
                0,
                other,
                majit_metainterp::jit::we_are_jitted(),
            )
            .expect("range copy_into an empty exact list");
            return Ok(());
        }
        pyre_object::listobject::w_list_reserve_for_extend(list, n);
        for i in 0..n {
            list = pyre_object::gc_roots::shadow_stack_get(root_base);
            other = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
            if let Some(item) = w_list_getitem(other, i as i64) {
                pyre_object::listobject::w_list_append_preallocated(list, item);
            }
        }
    }
    Ok(())
}

/// `listobject.py ListStrategy._extend_from_tuple`:
/// `@jit.look_inside_iff(lambda self, w_list, tup_w:
/// jit.loop_unrolling_heuristic(tup_w, len(tup_w), UNROLL_CUTOFF))`.
/// `tup_w` is `w_any.tolist()`; the tuple owns those items in every
/// variant, so its `isvirtual` is the probe.
fn extend_from_tuple_iff(_list: PyObjectRef, other: PyObjectRef) -> bool {
    let len = unsafe { w_tuple_len(other) };
    majit_rlib::jit::loop_unrolling_heuristic(unsafe { &*other }, len, LIST_UNROLL_CUTOFF)
}

/// `listobject.py UNROLL_CUTOFF`.
const LIST_UNROLL_CUTOFF: usize = 5;

#[majit_macros::look_inside_iff(extend_from_tuple_iff)]
fn extend_from_tuple(list: PyObjectRef, other: PyObjectRef) -> Result<(), crate::PyError> {
    // An integer- or float-specialised tuple boxes each item as it is read,
    // so `w_tuple_getitem` allocates and both operands can move under the
    // loop.  Keep them on the shadow stack and re-read them per item.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[list, other]);
    unsafe {
        let n = w_tuple_len(pyre_object::gc_roots::shadow_stack_get(base + 1));
        // `w_list._resize_hint(w_list.length() + len(tup_w))`, sized as the
        // `list_extend` reservation `__sizeof__` reports.
        pyre_object::listobject::w_list_reserve_for_extend(
            pyre_object::gc_roots::shadow_stack_get(base),
            n,
        );
        for i in 0..n {
            let other = pyre_object::gc_roots::shadow_stack_get(base + 1);
            if let Some(item) = w_tuple_getitem(other, i as i64) {
                pyre_object::listobject::w_list_append_preallocated(
                    pyre_object::gc_roots::shadow_stack_get(base),
                    item,
                );
            }
        }
    }
    Ok(())
}

/// `ListStrategy.extend` has no `_extend_from_set`. An empty receiver,
/// including `SizeListStrategy`, takes
/// `EmptyListStrategy._extend_from_iterable`: a non-empty
/// `unpackiterable_int`, or `listview_bytes` / `listview_ascii` including
/// an empty view. Anything else keeps the snapshot below — a non-empty
/// receiver, or a set whose listview is `None`. That snapshot is the
/// operation boundary that keeps `list(set)` from observing a concurrent
/// size transition between iterator steps. Residual for the same fnaddr
/// reason as `_extend_from_list`.
#[majit_macros::dont_look_inside]
fn extend_from_set(list: PyObjectRef, other: PyObjectRef) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[list, other]);
    if unsafe {
        pyre_object::w_list_try_extend_empty_from_set(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
        )
    } {
        return Ok(());
    }
    let items =
        unsafe { pyre_object::w_set_items(pyre_object::gc_roots::shadow_stack_get(base + 1)) };
    let items_base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(items_base, items.len());
    unsafe {
        pyre_object::listobject::w_list_resize_for_extend(
            pyre_object::gc_roots::shadow_stack_get(base),
            items.len(),
        );
        for index in 0..items.len() {
            pyre_object::listobject::w_list_append_preallocated(
                pyre_object::gc_roots::shadow_stack_get(base),
                pyre_object::gc_roots::shadow_stack_get(items_base + index),
            );
        }
    }
    Ok(())
}

/// `list(FrameLocalsProxy)` — append each bound name onto `list`.
///
/// `pyframe.py fast2locals` walks the slots with `@jit.unroll_safe`. The
/// same walk is `append_bound_entries`. `bool`, not
/// `Result`: a `Result` residual is may-force and `descr_init` is a
/// transparent helper. `false` asks the caller to use the generic iterable
/// drain. The names go straight onto `list`; an intermediate key list would
/// then call `extend_from_list`, and that may-force residual is not
/// recordable here.
#[majit_macros::unroll_safe]
fn extend_from_frame_locals_proxy(list: PyObjectRef, other: PyObjectRef) -> bool {
    crate::pyframe::frame_locals_proxy::extend_bound_entries(list, other, 0)
}

/// `listobject.py ListStrategy._extend_from_iterable`.  Upstream drains
/// through the `_do_extend_jitdriver` portal, so the caller's trace never
/// inlines the drain; here the whole arm is one residual call with a real
/// fnaddr.  `FrameLocalsProxy` does not take this arm.
#[majit_macros::dont_look_inside]
fn extend_from_iterable(list: PyObjectRef, other: PyObjectRef) -> Result<(), crate::PyError> {
    // Hint failures other than TypeError/AttributeError are observable and
    // propagate without appending a prefix.  Append each yielded value
    // before asking for the next one.  An exception from a later `next()`
    // does not roll back the prefix.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_roots(&[list, other]);
    let iterator =
        crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(root_base + 1))?;
    let _ = pyre_object::gc_roots::pin_root(iterator);
    // listobject.py _extend_from_iterable — consult the length hint
    // before iterating.  A `__length_hint__` returning a negative value
    // raises ValueError, and one exceeding a C ssize_t raises OverflowError
    // (via `int_w`), rather than being silently ignored.
    let hint = crate::baseobjspace::length_hint(
        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
        8,
    )?;
    // `length_hint` has already rejected negative and non-i64 values.
    let hint = hint as usize;
    let target = pyre_object::gc_roots::shadow_stack_get(root_base);
    let source = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let current_size = unsafe { pyre_object::w_list_len(target) };
    // An overflowing `m + n` hint is ignored on the chance that the iterator
    // lied, then growth comes from yielded items. Otherwise the pointer-array
    // byte size must be representable before reserving.
    if current_size <= (isize::MAX as usize).saturating_sub(hint) {
        if current_size + hint > (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>() {
            return Err(crate::PyError::memory_error(""));
        }
        let exact_dict = unsafe {
            pyre_object::is_dict(source)
                && (*source).w_class == pyre_object::get_instantiate(&*(*source).ob_type)
        };
        // Sibling `list_extend_dict{,items}` branches for the three exact
        // dict-view types; all use ordinary resize.
        if exact_dict || unsafe { pyre_object::dictmultiobject::is_dict_view(source) } {
            unsafe { pyre_object::listobject::w_list_resize_for_extend(target, hint) };
        } else if !unsafe { pyre_object::listobject::w_list_try_reserve_for_extend(target, hint) } {
            // `list_extend_iter_lock_held` reserves the hint with the
            // ordinary `list_resize`, so a hint no allocation can serve
            // is refused before the iterator is walked.
            return Err(crate::PyError::memory_error(""));
        }
    }
    do_extend_from_iterable(
        pyre_object::gc_roots::shadow_stack_get(root_base),
        pyre_object::gc_roots::shadow_stack_get(root_base + 2),
    )?;
    unsafe {
        pyre_object::listobject::w_list_finish_extend(pyre_object::gc_roots::shadow_stack_get(
            root_base,
        ));
    }
    Ok(())
}

/// `listobject.py _do_extend_from_iterable`.  Upstream drains through the
/// `_do_extend_jitdriver` portal (`greens=['strategy_type', 'greenkey']`),
/// which is not ported; here the drain runs inside the residual
/// [`extend_from_iterable`].  A length hint is advisory, so it cannot bound
/// an unrolled drain either.
fn do_extend_from_iterable(
    list: PyObjectRef,
    w_iterator: PyObjectRef,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::pin_roots(&[list, w_iterator]);
    loop {
        let item =
            match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(root_base + 1))
            {
                Ok(item) => item,
                Err(err) => {
                    let (stop, err) = err.matches_stop_iteration_keep();
                    if stop {
                        break;
                    }
                    return Err(err);
                }
            };
        let _item_roots = pyre_object::gc_roots::push_roots();
        let item_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(item);
        unsafe {
            pyre_object::listobject::w_list_append_preallocated(
                pyre_object::gc_roots::shadow_stack_get(root_base),
                pyre_object::gc_roots::shadow_stack_get(item_slot),
            );
        }
    }
    Ok(())
}

/// PyPy: listobject.py descr_insert — list.insert(index, item)
pub fn descr_insert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "insert", true)?;
    arity_exact_unpack(args, "insert", 2)?;
    // `@unwrap_spec(index='index')` → getindex_w(index, OverflowError): coerce
    // through `__index__`; `get_positive_index` then clamps to `[0, len]`
    // inside `w_list_insert`.
    // `args` is the copy the gateway built on the stack; a minor rewrites the
    // shadow slots and not that copy, so the receiver and the item read out of
    // it after `__index__` has run are pre-move addresses.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let index = unsafe {
        crate::baseobjspace::getindex_w_index(pyre_object::gc_roots::shadow_stack_get(base + 1))?
    };
    unsafe {
        pyre_object::listobject::w_list_insert(
            pyre_object::gc_roots::shadow_stack_get(base),
            index,
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        )
    };
    Ok(w_none())
}

/// PyPy: listobject.py descr_pop — list.pop([index])
/// listobject.py — empty list raises "pop from empty list",
/// otherwise out-of-range raises "pop index out of range".
pub fn list_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "pop", true)?;
    // `descr_pop` checks arity before touching the list, so `pop(1, 2)` on an
    // empty list reports the surplus argument rather than "pop from empty list".
    arity_at_most(args, "pop", 1)?;
    // `@unwrap_spec(index='index')` → getindex_w(index, OverflowError): coerce
    // through `__index__`.
    // Rooted as `descr_insert`: the receiver read out of `args` after
    // `__index__` has run is a pre-move address.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let index = if args.len() > 1 {
        unsafe {
            crate::baseobjspace::getindex_w_index(pyre_object::gc_roots::shadow_stack_get(
                base + 1,
            ))?
        }
    } else {
        -1
    };
    let list = pyre_object::gc_roots::shadow_stack_get(base);
    let length = unsafe { pyre_object::w_list_len(list) } as i64;
    if length == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop from empty list",
        ));
    }
    // listobject.py — clearly distinguish list.pop() from the
    // general list.pop(index) path.
    if index == -1 {
        return match unsafe { pyre_object::listobject::w_list_pop_end(list) } {
            Some(v) => Ok(v),
            None => Err(crate::PyError::new(
                crate::PyErrorKind::IndexError,
                "pop index out of range",
            )),
        };
    }
    match unsafe { pyre_object::listobject::w_list_pop(list, index) } {
        Some(v) => Ok(v),
        None => Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop index out of range",
        )),
    }
}

/// PyPy: listobject.py descr_clear — list.clear()
pub fn descr_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "clear", true)?;
    arity_no_args(args, "clear")?;
    unsafe { pyre_object::listobject::w_list_clear(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_copy — list.copy()
pub fn descr_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "copy", true)?;
    arity_no_args(args, "copy")?;
    let mut list = args[0];
    unsafe {
        if let Some(clone) = pyre_object::with_roots!(list =>
            pyre_object::listobject::w_list_clone_if_shared_strategy(list)
        ) {
            return Ok(clone);
        }
        let n = w_list_len(list);
        let _roots = pyre_object::gc_roots::push_roots();
        let list_slot = pyre_object::gc_roots::pin_roots(&[list]);
        let items_base = pyre_object::gc_roots::shadow_stack_len();
        let mut fetched = 0usize;
        for i in 0..n {
            if let Some(item) =
                w_list_getitem(pyre_object::gc_roots::shadow_stack_get(list_slot), i as i64)
            {
                let _ = pyre_object::gc_roots::pin_root(item);
                fetched += 1;
            }
        }
        let mut items = Vec::with_capacity(fetched);
        for i in 0..fetched {
            items.push(pyre_object::gc_roots::shadow_stack_get(items_base + i));
        }
        Ok(w_list_new(items))
    }
}

/// PyPy: listobject.py descr_reverse — list.reverse()
pub fn descr_reverse(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "reverse", true)?;
    arity_no_args(args, "reverse")?;
    unsafe { pyre_object::listobject::w_list_reverse(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_sort — list.sort().  The body is the
/// `interp2app` leaf [`crate::typedef::__majit_wrap_list_descr_sort`].
pub fn list_method_sort(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::typedef::__majit_wrap_list_descr_sort(args)
}

/// listobject.py `descr_index` — list.index(value[, start[, stop]]).
pub fn list_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "index", true)?;
    arity_at_least(args, "index", 1)?;
    arity_at_most(args, "index", 3)?;
    let list = args[0];
    let value = args[1];
    // listobject.py:799 unwrap_spec defaults: w_start=0 / w_stop=sys.maxint.
    // `W_ListObject.descr_index` hands `unwrap_start_stop` the length it read
    // first, and that call is what folds the negative bounds; the two halves
    // are called separately below so the length can be read after the
    // coercion instead.
    // An `__index__` on start/stop allocates and may move the list, the
    // search value, or the other bound. Publish the four words first so
    // no constructor runs before the live operands are visible.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::publish_roots(&[list, value]);
    let start_obj = if args.len() >= 3 {
        args[2]
    } else {
        w_int_new(0)
    };
    let _ = pyre_object::gc_roots::publish_roots(&[start_obj]);
    let stop_obj = if args.len() >= 4 {
        args[3]
    } else {
        w_int_new(i64::MAX)
    };
    let _ = pyre_object::gc_roots::publish_roots(&[stop_obj]);
    pyre_object::gc_roots::normalize_roots(sp, 4);
    let (raw_start, raw_stop) = crate::sliceobject::index_bounds_not_none(
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
        pyre_object::gc_roots::shadow_stack_get(sp + 3),
    )?;
    let list = pyre_object::gc_roots::shadow_stack_get(sp);
    let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
    // The length that folds a negative bound is read after the bounds have
    // been converted: their `__index__` may have resized this very list.
    let size = unsafe { pyre_object::w_list_len(list) } as i64;
    let (start, stop) = crate::sliceobject::adapt_start_stop(size, raw_start, raw_stop);
    match crate::listobject::w_list_find_or_count(list, value, start, stop, false)? {
        crate::listobject::FindOrCountResult::Index(i) => Ok(w_int_new(i)),
        crate::listobject::FindOrCountResult::NotFound => {
            // listobject.c list_index_impl (3.14): a fixed
            // "list.index(x): x not in list" message that does NOT format the
            // value.  gh-100242 dropped the older "%R is not in list" form
            // (which called `repr` on the missing value); PyPy still uses that
            // older form, so this message is the 3.14 target and cannot be
            // asserted against the PyPy check.py oracle.
            Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "list.index(x): x not in list".to_string(),
            ))
        }
        crate::listobject::FindOrCountResult::Count(_) => {
            unreachable!("find_or_count with count=false never returns Count")
        }
    }
}

/// listobject.py `descr_count` — list.count(value)
pub fn list_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "count", true)?;
    arity_exact(args, "count", 1)?;
    let list = args[0];
    let value = args[1];
    match crate::listobject::w_list_find_or_count(list, value, 0, i64::MAX, true)? {
        crate::listobject::FindOrCountResult::Count(n) => Ok(w_int_new(n)),
        crate::listobject::FindOrCountResult::NotFound => Ok(w_int_new(0)),
        crate::listobject::FindOrCountResult::Index(_) => {
            unreachable!("find_or_count with count=true never returns Index")
        }
    }
}

/// listobject.py `descr_remove` — list.remove(value).
pub fn descr_remove(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "remove", true)?;
    arity_exact(args, "remove", 1)?;
    crate::listobject::w_list_remove(args[0], args[1])?;
    Ok(w_none())
}

// ── String methods ───────────────────────────────────────────────────

pub fn str_method_join(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "join", 1)?;
    // `sequence_fast` and the type-error path allocate, so pin the separator
    // and iterable first and copy the separator payload off the object.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let sep = unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }
        .to_wtf8_buf();
    let items_base = pyre_object::gc_roots::shadow_stack_len();
    let n_items = unsafe {
        let iterable = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if is_list(iterable) {
            let n = w_list_len(iterable);
            let mut fetched = 0usize;
            for i in 0..n {
                if let Some(item) =
                    w_list_getitem(pyre_object::gc_roots::shadow_stack_get(base + 1), i as i64)
                {
                    let _ = pyre_object::gc_roots::pin_root(item);
                    fetched += 1;
                }
            }
            fetched
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            let mut fetched = 0usize;
            for i in 0..n {
                if let Some(item) =
                    w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(base + 1), i as i64)
                {
                    let _ = pyre_object::gc_roots::pin_root(item);
                    fetched += 1;
                }
            }
            fetched
        } else {
            // `PySequence_Fast(seq, "can only join an iterable")` — the
            // separator's own error, not the iteration protocol's.
            let items = crate::builtins::sequence_fast(iterable, "can only join an iterable")?;
            for item in &items {
                let _ = pyre_object::gc_roots::pin_root(*item);
            }
            items.len()
        }
    };
    let item = |i: usize| pyre_object::gc_roots::shadow_stack_get(items_base + i);
    // pypy/objspace/std/unicodeobject.py descr_join — each
    // element must be a str; otherwise TypeError("sequence item N:
    // expected str instance, <T> found"). Silently dropping non-str
    // items lost the error and produced an empty join.
    //
    // A single-element join returns that element (unicode_result_unchanged):
    // an exact str unchanged, a str subclass copied to a base str.
    if n_items == 1 {
        let item = item(0);
        if unsafe { !is_str(item) } {
            return Err(crate::PyError::type_error(format!(
                "sequence item 0: expected str instance, {} found",
                arg_type_name(item)
            )));
        }
        return Ok(str_result_unchanged(item));
    }
    let reloaded: Vec<_> = (0..n_items).map(item).collect();
    str_join_many_items(&sep, &reloaded)
}

/// `stringmethods.py _str_join_many_items`:
/// `@jit.look_inside_iff(lambda self, space, list_w, size: jit.loop_unrolling_heuristic(list_w, size))`.
fn str_join_many_items_iff(_sep: &rustpython_wtf8::Wtf8, items: &[PyObjectRef]) -> bool {
    majit_rlib::jit::loop_unrolling_heuristic(items, items.len(), 2)
}

#[majit_macros::look_inside_iff(str_join_many_items_iff)]
fn str_join_many_items(
    sep: &rustpython_wtf8::Wtf8,
    items: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    for (i, item) in items.iter().enumerate() {
        if unsafe { !is_str(*item) } {
            return Err(crate::PyError::type_error(format!(
                "sequence item {i}: expected str instance, {} found",
                arg_type_name(*item)
            )));
        }
        if i > 0 {
            out.push_wtf8(sep);
        }
        out.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(*item) });
    }
    // `str.join` is a dominant dynamic-churn producer; make the result collectable.
    Ok(pyre_object::w_str_from_wtf8_managed(out))
}

/// `str.split` / `str.rsplit` take `sep` and `maxsplit` positionally or by
/// keyword.  Builtin kwargs arrive as a trailing `__pyre_kw__` dict, so
/// resolve each argument from its positional slot (after the receiver),
/// falling back to the matching keyword.
fn resolve_split_args(
    args: &[PyObjectRef],
    fn_name: &str,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        fn_name,
        pos.len() - 1,
        crate::builtins::real_kwarg_count(kwargs),
        0,
        2,
        0,
    )?;
    crate::builtins::kwarg_reject_unknown(kwargs, &["sep", "maxsplit"], fn_name)?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "sep", pos.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "maxsplit", pos.get(2).is_some())?;
    let sep = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"))
        .unwrap_or(pyre_object::PY_NULL);
    let maxsplit = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "maxsplit"))
        .unwrap_or(pyre_object::PY_NULL);
    Ok((sep, maxsplit))
}

/// Box a run of `recv`'s own code points into a str object, so a run covering
/// all of them yields `recv` itself — see [`pyre_object::w_str_cut`].  The runs
/// a splitter emits are disjoint pieces of the receiver, so an equal WTF-8 byte
/// count means this one spans the whole string.
fn cps_to_str_cut(recv_slot: usize, cps: &[CodePoint]) -> PyObjectRef {
    let mut buf = Wtf8Buf::with_capacity(cps.len());
    for &cp in cps {
        buf.push(cp);
    }
    unsafe { pyre_object::w_str_cut(pyre_object::gc_roots::shadow_stack_get(recv_slot), &buf) }
}

/// A lone surrogate is not whitespace.
fn cp_is_whitespace(cp: CodePoint) -> bool {
    match cp.to_char() {
        Some(c) => classify::is_space(c),
        None => false,
    }
}

/// `str.split()` with no separator: split on runs of whitespace,
/// dropping leading/trailing runs.  When `maxsplit >= 0`, after that
/// many splits the rest (leading whitespace stripped) is one tail token.
fn wtf8_split_whitespace(
    recv_slot: usize,
    s: &Wtf8,
    maxsplit: i64,
    out: &mut pyre_object::gc_roots::RootedItems,
) {
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut i = 0usize;
    loop {
        if maxsplit >= 0 && out.len() as i64 >= maxsplit {
            break;
        }
        while i < cps.len() && cp_is_whitespace(cps[i]) {
            i += 1;
        }
        if i == cps.len() {
            break;
        }
        let start = i;
        while i < cps.len() && !cp_is_whitespace(cps[i]) {
            i += 1;
        }
        out.push(cps_to_str_cut(recv_slot, &cps[start..i]));
    }
    while i < cps.len() && cp_is_whitespace(cps[i]) {
        i += 1;
    }
    if i < cps.len() {
        out.push(cps_to_str_cut(recv_slot, &cps[i..]));
    }
}

/// `str.rsplit()` with no separator: like `wtf8_split_whitespace` but
/// scanning from the right, so the tail token is the leading remainder.
fn wtf8_rsplit_whitespace(
    recv_slot: usize,
    s: &Wtf8,
    maxsplit: i64,
    out: &mut pyre_object::gc_roots::RootedItems,
) {
    let cps: Vec<CodePoint> = s.code_points().collect();
    // The scan runs right to left but the result reads left to right.  Collect
    // the ranges first -- they borrow `cps` and hold no GC reference, so the
    // reversal costs nothing -- and cut in final order afterwards, so every
    // allocation lands in the rooted set with the earlier cuts already pinned.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = cps.len();
    loop {
        if maxsplit >= 0 && ranges.len() as i64 >= maxsplit {
            break;
        }
        while i > 0 && cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        let end = i;
        while i > 0 && !cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        ranges.push((i, end));
    }
    ranges.reverse();
    let mut prefix_end = i;
    while prefix_end > 0 && cp_is_whitespace(cps[prefix_end - 1]) {
        prefix_end -= 1;
    }
    if prefix_end > 0 {
        out.push(cps_to_str_cut(recv_slot, &cps[..prefix_end]));
    }
    for (start, end) in ranges {
        out.push(cps_to_str_cut(recv_slot, &cps[start..end]));
    }
}

pub fn str_method_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "split")?;
    // `parse_split_maxsplit` can run `__index__`, and every result cut can
    // allocate.  Keep the receiver below the result-set bracket and read its
    // rewritten slot for each identity-preserving `w_str_cut`.  The text is an
    // owned snapshot so no `&Wtf8` points into a receiver that may move.
    // Sep and maxsplit sit on the same set: the payload copy can collect
    // before `resolve_split_args` rereads them.
    let recv_roots = pyre_object::gc_roots::push_roots();
    let recv_slot = recv_roots.publish(args);
    recv_roots.normalize(recv_slot, args.len());
    let s = unsafe { w_str_get_wtf8(recv_roots.get(recv_slot)) }.to_wtf8_buf();
    let reloaded: Vec<_> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(recv_slot + i))
        .collect();
    let (sep_arg, maxsplit_arg) = resolve_split_args(&reloaded, "split")?;
    let sep = parse_split_sep(sep_arg)?;
    // `unicodeobject.py @unwrap_spec(maxsplit=int) descr_split` —
    // `space.int_w(w_maxsplit)` routes through `__index__`, so any
    // int-like object (subclass, numpy int, etc.) is accepted.
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    // Every cut allocates over the cuts already made, and a plain `Vec` is not
    // a root area, so each is pinned as it is produced -- `build_list_storage`
    // requires the elements to be pinned before it is handed them.
    let mut parts = pyre_object::gc_roots::RootedItems::new();
    match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            // Each piece is a `value[start:end]` cut (rstring.py split
            // `split`), so a separator-free receiver yields itself back — see
            // [`pyre_object::w_str_cut`].
            if maxsplit < 0 {
                for p in s.split(sep) {
                    parts.push(unsafe {
                        pyre_object::w_str_cut(
                            pyre_object::gc_roots::shadow_stack_get(recv_slot),
                            p,
                        )
                    });
                }
            } else {
                for p in s.splitn((maxsplit as usize) + 1, sep) {
                    parts.push(unsafe {
                        pyre_object::w_str_cut(
                            pyre_object::gc_roots::shadow_stack_get(recv_slot),
                            p,
                        )
                    });
                }
            }
        }
        None => wtf8_split_whitespace(recv_slot, &s, maxsplit, &mut parts),
    }
    Ok(w_list_new(parts.take()))
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject
/// .convert_arg_to_w_unicode` parity — `sep` must be `None` or a
/// `str`; anything else surfaces a TypeError at the same call site
/// where PyPy's `space.unicode_w` would.
fn parse_split_sep(value: PyObjectRef) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if value.is_null() || unsafe { is_none(value) } {
        return Ok(None);
    }
    if unsafe { is_str(value) } {
        return Ok(Some(unsafe { w_str_get_wtf8(value) }.to_wtf8_buf()));
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(value) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "must be str or None, not {tp_name}"
    )))
}

/// `unicodeobject.py:972 @unwrap_spec(maxsplit=int)` parity —
/// `space.int_w(w_maxsplit)` routes through `__index__`, so any
/// int-like object is accepted; an absent maxsplit defaults to -1
/// (unlimited).  An explicit `None` is not int-like and has no
/// `__index__`, so it raises `TypeError` like any other non-integer.
fn parse_split_maxsplit(value: PyObjectRef) -> Result<i64, crate::PyError> {
    if value.is_null() {
        return Ok(-1);
    }
    crate::builtins::space_index_w(value)
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject
/// .descr_rsplit`.  Mirrors `split` semantics in reverse — when
/// `maxsplit` is positive, only the rightmost `maxsplit` separators
/// participate.  Argument validation follows the same
/// `@unwrap_spec(maxsplit=int)` + `convert_arg_to_w_unicode` shape
/// as `descr_split`.
pub fn str_method_rsplit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "rsplit")?;
    let recv_roots = pyre_object::gc_roots::push_roots();
    let recv_slot = recv_roots.publish(args);
    recv_roots.normalize(recv_slot, args.len());
    let s = unsafe { w_str_get_wtf8(recv_roots.get(recv_slot)) }.to_wtf8_buf();
    let reloaded: Vec<_> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(recv_slot + i))
        .collect();
    let (sep_arg, maxsplit_arg) = resolve_split_args(&reloaded, "rsplit")?;
    let sep = parse_split_sep(sep_arg)?;
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    // As in `str_method_split`: the cuts are pinned as they are produced.
    let mut parts = pyre_object::gc_roots::RootedItems::new();
    match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path —
            // mirrors the forward `split` rejection.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            let mut out: Vec<&Wtf8> = if maxsplit < 0 {
                s.rsplit(sep).collect()
            } else {
                s.rsplitn((maxsplit as usize) + 1, sep).collect()
            };
            out.reverse();
            for p in out {
                parts.push(unsafe {
                    pyre_object::w_str_cut(pyre_object::gc_roots::shadow_stack_get(recv_slot), p)
                });
            }
        }
        None => wtf8_rsplit_whitespace(recv_slot, &s, maxsplit, &mut parts),
    }
    Ok(w_list_new(parts.take()))
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject.descr_casefold`:
///
/// ```python
/// def descr_casefold(self, space):
///     value = self._value
///     return space.newutf8(unicode_casefold(value), -1)
/// ```
///
/// PyPy delegates to `rpython.rlib.runicode.unicode_casefold` which
/// applies the full Unicode `CaseFolding.txt` mapping (status C +
/// status F: ß → ss, ﬁ → fi, İ → i + combining dot,
/// Lithuanian Į → i̇ǫ, the Greek sigma, etc.).  Pyre routes through
/// `case::casefold_wtf8`, which applies the same
/// `CaseFolding.txt` status-C+F mapping to each scalar code point and
/// passes lone surrogates through unchanged.
/// `unicodeobject.py W_UnicodeObject._casefold_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn casefold_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(case::casefold_wtf8(&s))
}

pub fn str_method_casefold(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "casefold")?;
    Ok(unsafe { casefold_unicode(args[0]) })
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject
/// .descr_format_map` parity:
///
/// ```python
/// def descr_format_map(self, space, w_mapping):
///     return newformat.format_method(space, self, None, w_mapping, True)
/// ```
///
/// PyPy passes the mapping straight through to `format_method`; each
/// `{name}` field is then resolved by `space.getitem(mapping, w_key)`
/// at format-time, so the mapping is consulted *lazily*.  This
/// matters for mappings with side-effecting `__getitem__`, a
/// `__missing__` hook, or no `keys()` — pre-materialising via
/// `keys()` (the previous implementation) breaks `defaultdict`,
/// custom `Mapping` subclasses, and any object that only implements
/// `__getitem__`.
pub fn descr_format_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "format_map", 1)?;
    str_method_format_core(args[0], &[], None, Some(args[1]))
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject._strip` —
/// `s.strip([chars])`.  When `chars` is missing or None, defaults to
/// the same Unicode whitespace predicate as `str.isspace()`.  When provided, removes
/// any character contained in `chars` from each end (NOT a substring
/// match — `'aabaa'.strip('a') == 'b'`).
///
/// Returns the cut itself, not a copy: both `_strip` arms cut
/// `value[lpos:rpos]` out of the utf8 storage (`_utf8_sliced`,
/// unicodeobject.py; `_strip_unboxed`, unicodeobject.py), so a strip
/// that removes nothing shares its operand's storage — see
/// [`pyre_object::w_str_cut`].
fn strip_chars<'a>(s: &'a Wtf8, chars: Option<&Wtf8>, left: bool, right: bool) -> &'a Wtf8 {
    let chars_set: Option<Vec<CodePoint>> = chars.map(|c| c.code_points().collect());
    let mut current: &Wtf8 = s;
    if left {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_start_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_start_matches(cp_is_whitespace),
        };
    }
    if right {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_end_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_end_matches(cp_is_whitespace),
        };
    }
    current
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject
/// ._strip` — extract the optional `chars` argument as a `&str`,
/// raising TypeError on non-str non-None arguments rather than
/// silently falling through to the whitespace default.
fn extract_strip_chars(arg: PyObjectRef, fn_name: &str) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if arg.is_null() || unsafe { pyre_object::is_none(arg) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_str(arg) } {
        return Ok(Some(
            unsafe { pyre_object::w_str_get_wtf8(arg) }.to_wtf8_buf(),
        ));
    }
    Err(crate::PyError::type_error(format!(
        "{fn_name} arg must be None or str"
    )))
}

fn str_strip_impl(
    args: &[PyObjectRef],
    method: &str,
    left: bool,
    right: bool,
) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, method)?;
    arity_at_most(args, method, 1)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = pyre_object::gc_roots::pin_roots(args);
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let chars = if args.len() > 1 {
        extract_strip_chars(
            pyre_object::gc_roots::shadow_stack_get(recv_slot + 1),
            method,
        )?
    } else {
        None
    };
    // Copy the receiver payload before `strip_chars` allocates its
    // code-point set. `w_str_cut` then reads the owned cut (same
    // shape as `str.split`).
    let s = unsafe { w_str_get_wtf8(recv()) }.to_wtf8_buf();
    Ok(unsafe { pyre_object::w_str_cut(recv(), strip_chars(&s, chars.as_deref(), left, right)) })
}

pub fn str_method_strip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_strip_impl(args, "strip", true, true)
}

pub fn str_method_lstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_strip_impl(args, "lstrip", true, false)
}

pub fn str_method_rstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_strip_impl(args, "rstrip", false, true)
}

/// `unicodeobject.py _unwrap_and_compute_idx_params` — unwrap the optional
/// `start` / `end` code-point bounds and return them as byte offsets into
/// `_utf8`.  `sliceobject.py unwrap_start_stop` runs each bound through
/// `adapt_lower_bound(_eval_slice_index(...))`, so a non-index bound raises a
/// TypeError and one carrying `__index__` is coerced.
///
/// A `start` past the end becomes `end_index + 1` rather than being clamped,
/// and `end` is lowered only when it is short of the end, so the window
/// inverts instead of emptying.  That is why `'abc'.startswith('', 5, 10)` is
/// `False` even for an empty needle, while `'abc'.startswith('', 3, 10)` — a
/// start exactly at the end — is `True`.
///
/// `__index__` can collect, so the receiver is re-read from its shadow-stack
/// slot after each bound rather than held across the conversions.
fn str_unwrap_and_compute_idx_params(
    recv_slot: usize,
    start_slot: Option<usize>,
    end_slot: Option<usize>,
) -> Result<(i64, i64), crate::PyError> {
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let length = unsafe { pyre_object::w_str_len(recv()) } as i64;
    // An absent bound, and a `None` in its place, mean "this side unbounded":
    // start is 0 and end is the code point count.
    let mut start = 0i64;
    if let Some(slot) = start_slot {
        let bound = pyre_object::gc_roots::shadow_stack_get(slot);
        if !unsafe { pyre_object::is_none(bound) } {
            start = crate::sliceobject::adapt_lower_bound(length, bound)?;
        }
    }
    let mut end = length;
    if let Some(slot) = end_slot {
        let bound = pyre_object::gc_roots::shadow_stack_get(slot);
        if !unsafe { pyre_object::is_none(bound) } {
            end = crate::sliceobject::adapt_lower_bound(length, bound)?;
        }
    }
    let byte_len = unsafe { pyre_object::w_str_byte_len(recv()) } as i64;
    let start_byte = if start > 0 && start <= length {
        unsafe { pyre_object::w_str_index_to_byte(recv(), start as usize) as i64 }
    } else {
        0
    };
    let end_byte = if end < length {
        unsafe { pyre_object::w_str_index_to_byte(recv(), end as usize) as i64 }
    } else {
        0
    };
    Ok(prefix_byte_window(
        length, byte_len, start, end, start_byte, end_byte,
    ))
}

/// The byte window of `_unwrap_and_compute_idx_params` once `start` and
/// `end` are code points. `start_byte` and `end_byte` are `_index_to_byte`
/// for a bound that sits strictly inside the string; the other arms do not
/// read them. A `start` past the end becomes `end_index + 1`, and `end` is
/// lowered only when it is short of the end, so the window inverts instead
/// of emptying.
#[inline(always)]
fn prefix_byte_window(
    length: i64,
    byte_len: i64,
    start: i64,
    end: i64,
    start_byte: i64,
    end_byte: i64,
) -> (i64, i64) {
    let mut start_index = 0i64;
    let mut end_index = byte_len;
    if start > 0 {
        start_index = if start > length {
            end_index + 1
        } else {
            start_byte
        };
    }
    if end < length {
        end_index = end_byte;
    }
    (start_index, end_index)
}

/// `unicodeobject.py _startswith` / `_endswith` for one `str` needle whose
/// bounds are already byte offsets: an inverted window matches nothing, an
/// empty needle matches anything else, and the rest is `rstring.py
/// startswith` / `endswith`, which are `@jit.elidable` — one pure call over
/// the two `_utf8` payloads, not a walked byte loop.
///
/// WTF-8 is self-synchronizing, so the byte-level match those perform
/// coincides with the code-point-level one.
fn str_prefix_match_one(
    w_self: PyObjectRef,
    w_needle: PyObjectRef,
    start: i64,
    end: i64,
    is_start: bool,
) -> bool {
    if start > end {
        return false;
    }
    if unsafe { pyre_object::w_str_get_wtf8(w_needle) }
        .as_bytes()
        .is_empty()
    {
        return true;
    }
    unsafe {
        if is_start {
            pyre_object::unicodeobject::startswith(w_self, w_needle, start, end)
        } else {
            pyre_object::unicodeobject::endswith(w_self, w_needle, start, end)
        }
    }
}

/// `stringmethods.py _startswith_tuple` / `_endswith_tuple` — the first
/// member that matches wins.  A non-`str` member is reported against the
/// tuple rather than against the argument as a whole.
fn str_prefix_match_tuple(
    w_self: PyObjectRef,
    w_needles: PyObjectRef,
    start: i64,
    end: i64,
    method: &str,
    is_start: bool,
) -> Result<bool, crate::PyError> {
    let n = unsafe { pyre_object::w_tuple_len(w_needles) } as i64;
    for i in 0..n {
        let item =
            unsafe { pyre_object::w_tuple_getitem(w_needles, i) }.expect("index is in range");
        if !unsafe { pyre_object::is_str(item) } {
            return Err(crate::PyError::type_error(format!(
                "tuple for {method} must only contain str, not {}",
                arg_type_name(item)
            )));
        }
        if str_prefix_match_one(w_self, item, start, end, is_start) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `unicodeobject.py descr_startswith` / `descr_endswith` — the bounds, then
/// either the tuple arm or the single needle.  `interp2app` unwraps
/// `(w_prefix, w_start=None, w_end=None)`; the flat `BuiltinCodeFn` slice
/// carries them positionally behind the receiver.
///
/// The whole method is one graph: the bounds arithmetic and the elidable
/// match are what the descent records, so a bounded call is not a residual.
fn str_descr_prefix_match(
    args: &[PyObjectRef],
    method: &str,
    is_start: bool,
) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, method, 1)?;
    arity_at_most(args, method, 3)?;
    // `__index__` on a bound can collect, so the receiver, the needle and
    // both bounds are pinned before any of them is unwrapped.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let (start, end) = str_unwrap_and_compute_idx_params(
        base,
        (args.len() >= 3).then_some(base + 2),
        (args.len() >= 4).then_some(base + 3),
    )?;
    let w_self = pyre_object::gc_roots::shadow_stack_get(base);
    let w_needle = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if unsafe { pyre_object::is_tuple(w_needle) } {
        return str_prefix_match_tuple(w_self, w_needle, start, end, method, is_start)
            .map(w_bool_from);
    }
    if !unsafe { pyre_object::is_str(w_needle) } {
        return Err(crate::PyError::type_error(format!(
            "{method} first arg must be str or a tuple of str, not {}",
            arg_type_name(w_needle)
        )));
    }
    Ok(w_bool_from(str_prefix_match_one(
        w_self, w_needle, start, end, is_start,
    )))
}

/// `unicodeobject.py descr_startswith`.
pub fn str_method_startswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_descr_prefix_match(args, "startswith", true)
}

/// `unicodeobject.py descr_endswith`.
pub fn str_method_endswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_descr_prefix_match(args, "endswith", false)
}

/// `unicodeobject.py _unwrap_and_compute_idx_params` for the operand shapes
/// that reach the match without running a user slot, returning the receiver,
/// the needle and the two byte offsets.
///
/// Bounds are converted before the needle is inspected, as
/// `descr_startswith` does. `int` or an absent/`None` bound is
/// `getindex_w`'s `isinstance_w(w_int)` arm plus `adapt_lower_bound`; the
/// `__index__` arm stays on the residual because that call is a whole-body
/// blocker. The byte window is [`prefix_byte_window`]. `_index_to_byte` is
/// needed only for a bound strictly inside the string, and its `is_ascii()`
/// arm is the identity. A default window never calls `_index_to_byte` —
/// `start` stays 0 and `end` stays `len(_utf8)` — so it is traced for every
/// encoding. A non-ASCII interior bound goes through the same look-inside
/// `_index_to_byte`: `getfield index_storage` plus `conditional_call_value`
/// (miss residual `_compute_index_storage`) then the elidable
/// `codepoint_position_at_index`.
fn str_idx_params_unrooted(args: &[PyObjectRef]) -> Option<(PyObjectRef, PyObjectRef, i64, i64)> {
    // `startswith(prefix[, start[, end]])` behind the receiver.  A keyword
    // argument rides the same slice as a trailing marker dict, which is
    // neither `int` nor `None`, so the bound tests below decline it.
    if args.len() < 2 || args.len() > 4 {
        return None;
    }
    let w_self = args[0];
    if unsafe { !pyre_object::is_str(w_self) } {
        return None;
    }
    let length = unsafe { pyre_object::w_str_len(w_self) } as i64;
    let byte_len = unsafe { pyre_object::w_str_byte_len(w_self) } as i64;
    let start = str_traced_codepoint_bound(args, 2, 0, length)?;
    let end = str_traced_codepoint_bound(args, 3, length, length)?;
    let start_byte = if start > 0 && start <= length {
        unsafe { pyre_object::w_str_index_to_byte(w_self, start as usize) as i64 }
    } else {
        0
    };
    let end_byte = if end < length {
        unsafe { pyre_object::w_str_index_to_byte(w_self, end as usize) as i64 }
    } else {
        0
    };
    let (start_index, end_index) =
        prefix_byte_window(length, byte_len, start, end, start_byte, end_byte);
    let w_needle = args[1];
    if unsafe { !pyre_object::is_str(w_needle) } {
        return None;
    }
    Some((w_self, w_needle, start_index, end_index))
}

/// One bound of [`str_idx_params_unrooted`]: absent or `None` is `default`,
/// `getindex_w`'s `is_int` arm is `w_int_get_value` plus `adapt_bound`, and
/// anything else declines so the residual runs `__index__`.
fn str_traced_codepoint_bound(
    args: &[PyObjectRef],
    index: usize,
    default: i64,
    length: i64,
) -> Option<i64> {
    if args.len() <= index {
        return Some(default);
    }
    let bound = args[index];
    if unsafe { pyre_object::is_none(bound) } {
        return Some(default);
    }
    if unsafe { !pyre_object::is_int(bound) } {
        return None;
    }
    let index = unsafe { pyre_object::w_int_get_value(bound) };
    Some(crate::sliceobject::adapt_bound(length, index))
}

/// `BuiltinCode.func` PBC member for `str.startswith` — the wrapper
/// `interp2app` would generate.  The descent walker keys the args-array
/// heap-cache off the element reads, the same shape
/// `__majit_wrap_builtin_len` uses.
///
/// The arm in this graph is `descr_startswith`'s own body over the operand
/// shapes [`str_idx_params_unrooted`] admits, and the match it ends in is
/// `rstring.py startswith`, which is `@jit.elidable`: one pure call, which
/// the optimizer folds when the operands are constant.  Every other shape
/// runs the *same* method through the residual below, so there is one
/// implementation and not two.
pub fn __majit_wrap_str_descr_startswith(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    if let Some((w_self, w_prefix, start, end)) = str_idx_params_unrooted(args) {
        return Ok(w_bool_from(str_prefix_match_one(
            w_self, w_prefix, start, end, true,
        )));
    }
    str_descr_startswith_residual(args)
}

/// `BuiltinCode.func` PBC member for `str.endswith`.
pub fn __majit_wrap_str_descr_endswith(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    if let Some((w_self, w_suffix, start, end)) = str_idx_params_unrooted(args) {
        return Ok(w_bool_from(str_prefix_match_one(
            w_self, w_suffix, start, end, false,
        )));
    }
    str_descr_endswith_residual(args)
}

/// `descr_startswith` for the arms that run a user slot, build an error, or
/// walk a tuple.  `dont_look_inside` for the reason `argument.py` keeps
/// `ArgErr.getmsg` out of `_match_signature`: the message and the generic
/// `__index__` conversion pull the whole attribute-lookup and codec surface
/// into the graph, and the descent's blocker scan is whole-body, so a
/// runtime branch does not keep them out.
#[majit_macros::dont_look_inside]
fn str_descr_startswith_residual(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_method_startswith(args)
}

/// `descr_endswith`'s residual.  See [`str_descr_startswith_residual`].
#[majit_macros::dont_look_inside]
fn str_descr_endswith_residual(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_method_endswith(args)
}

pub fn str_method_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `old` / `new` are positional-only; `count` is positional-or-keyword.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() < 3 {
        return Err(crate::PyError::type_error(format!(
            "replace() takes at least 2 positional arguments ({} given)",
            args_given(pos)
        )));
    }
    if pos.len() > 4 {
        return Err(crate::PyError::type_error(format!(
            "replace() takes at most 3 arguments ({} given)",
            args_given(pos)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["count"], "replace")?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "replace", "count", pos.get(3).is_some())?;
    // pypy/objspace/std/unicodeobject.py descr_replace —
    // both `old` and `new` must be str / W_UnicodeObject; otherwise
    // TypeError("replace() argument N must be str, not ...").
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 1 must be str, not {}",
            clinic_arg_type_name(pos[1])
        )));
    }
    if !unsafe { pyre_object::is_str(pos[2]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 2 must be str, not {}",
            clinic_arg_type_name(pos[2])
        )));
    }
    // `__index__` on `count` can collect, so pin the three strings and the
    // optional count together and copy their payloads off the objects first.
    let w_count = pos
        .get(3)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "count"));
    let _roots = pyre_object::gc_roots::push_roots();
    let base = if let Some(c) = w_count {
        pyre_object::gc_roots::pin_roots(&[pos[0], pos[1], pos[2], c])
    } else {
        pyre_object::gc_roots::pin_roots(&[pos[0], pos[1], pos[2]])
    };
    let recv = || pyre_object::gc_roots::shadow_stack_get(base);
    let s = unsafe { pyre_object::w_str_get_wtf8(recv()) }.to_wtf8_buf();
    let old =
        unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }
            .to_wtf8_buf();
    let new =
        unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 2)) }
            .to_wtf8_buf();
    // Optional `count`: a negative count means "no limit"; 0 leaves the
    // string untouched. Resolved through `__index__`.
    let maxcount = match w_count {
        Some(_) => {
            crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(base + 3))?
        }
        None => -1,
    };
    let (out, replacements) = wtf8_replace(&s, &old, &new, maxcount);
    // `descr_replace` (unicodeobject.py) returns `self` when nothing
    // was replaced — keyed on the count, so `s.replace('a', 'a')` still builds
    // a new string.  Exact `str` only: a subclass yields a fresh base `str`,
    // and `is_w` rejects a `user_overridden_class` operand
    // (unicodeobject.py:106).
    if replacements == 0 && unsafe { pyre_object::is_exact_type(recv(), &pyre_object::STR_TYPE) } {
        return Ok(recv());
    }
    Ok(w_str_from_wtf8_managed(out))
}

/// WTF-8 window for the optional `start` / `end` search args: resolve them
/// (PyPy slice semantics via `unwrap_start_stop`) into a byte-offset window
/// `(byte_start, byte_end)` into the WTF-8 backing, indexing by code point so
/// a surrogate-bearing string does not panic in `w_str_get_value`.  Returns
/// `None` when the codepoint window is empty because `start` is past the end
/// or past `end` (the search-miss case shared by count).
fn wtf8_idx_window(
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<Option<(usize, usize)>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let cp_len = unsafe { pyre_object::w_str_len(obj()) } as i64;
    let w_start = if args.len() >= 3 {
        pyre_object::gc_roots::shadow_stack_get(args_base + 2)
    } else {
        w_none()
    };
    let w_end = if args.len() >= 4 {
        pyre_object::gc_roots::shadow_stack_get(args_base + 3)
    } else {
        w_none()
    };
    let (start, end) = crate::sliceobject::unwrap_start_stop(cp_len, w_start, w_end)?;
    if start > cp_len {
        return Ok(None);
    }
    let end = end.min(cp_len);
    if start > end {
        return Ok(None);
    }
    let obj = obj();
    let byte_start = unsafe { pyre_object::w_str_index_to_byte(obj, start as usize) };
    let byte_end = unsafe { pyre_object::w_str_index_to_byte(obj, end as usize) };
    Ok(Some((byte_start, byte_end)))
}

#[inline(always)]
fn bound_is_none_or_exact_int(w: PyObjectRef) -> bool {
    if w.is_null() {
        return true;
    }
    // `W_LongObject` carries the `int` class, so `is_exact_type(INT)` is
    // true while the payload at `intval` is a bigint pointer.
    // `w_int_get_value` would read that pointer. A long takes the slow arm,
    // which converts through the slice index.
    unsafe {
        pyre_object::is_none(w)
            || (pyre_object::is_exact_type(w, &pyre_object::INT_TYPE) && !pyre_object::is_long(w))
    }
}

/// An omitted bound is `None`, the `w_start=None, w_end=None` defaults of
/// `descr_find`. `None` and an exact `int` stay on the elidable search;
/// anything else (`__index__`, a subclass) is the slow arm.
/// A plain index, not `slice::get`: that `Option` does not lower.
#[inline(always)]
fn str_search_bound(args: &[PyObjectRef], i: usize) -> PyObjectRef {
    if i < args.len() { args[i] } else { w_none() }
}

/// `_convert_idx_params` (`stringmethods.py`) unwraps each bound to a
/// machine int before `ll_find` / `ll_rfind` / `ll_count`. Omitted or
/// `None` is the default; an exact int is `w_int_get_value`.
#[inline(always)]
fn str_search_bound_int(w: PyObjectRef, default: i64) -> i64 {
    if w.is_null() || unsafe { pyre_object::is_none(w) } {
        default
    } else {
        unsafe { pyre_object::w_int_get_value(w) }
    }
}

/// Bounds that are not `None` or an exact `int` (`__index__`, a subclass)
/// stay in the interpreter body. `dont_look_inside` so that arm does not
/// pull its helpers into the generated wrapper.
#[inline(never)]
#[majit_macros::dont_look_inside]
fn str_descr_find_slow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_method_find(args)
}

#[inline(never)]
#[majit_macros::dont_look_inside]
fn str_descr_rfind_slow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_method_rfind(args)
}

#[inline(never)]
#[majit_macros::dont_look_inside]
fn str_descr_count_slow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    str_method_count(args)
}

/// `unicodeobject.py descr_find`. Bounds are unwrapped here, then `ll_find`
/// (`jit_str_find_bounds`) sees two GC refs and two machine ints.
pub fn __majit_wrap_str_descr_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Arity and kwargs stay on the slow arm. `arity_at_least` pulls the
    // kwargs-marker read into this graph, and that effect makes every later
    // helper a descent blocker.
    if args.len() < 2 || args.len() > 4 {
        return str_descr_find_slow(args);
    }
    if unsafe { !pyre_object::is_str(args[0]) || !pyre_object::is_str(args[1]) } {
        return str_descr_find_slow(args);
    }
    let w_start = str_search_bound(args, 2);
    let w_end = str_search_bound(args, 3);
    if !bound_is_none_or_exact_int(w_start) || !bound_is_none_or_exact_int(w_end) {
        return str_descr_find_slow(args);
    }
    let start = str_search_bound_int(w_start, 0);
    let end = str_search_bound_int(w_end, i64::MAX);
    let n = pyre_object::unicodeobject::jit_str_find_bounds(args[0], args[1], start, end);
    Ok(w_int_new(n))
}

/// `unicodeobject.py descr_rfind`. Bounds are unwrapped here, then `ll_rfind`.
pub fn __majit_wrap_str_descr_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 || args.len() > 4 {
        return str_descr_rfind_slow(args);
    }
    if unsafe { !pyre_object::is_str(args[0]) || !pyre_object::is_str(args[1]) } {
        return str_descr_rfind_slow(args);
    }
    let w_start = str_search_bound(args, 2);
    let w_end = str_search_bound(args, 3);
    if !bound_is_none_or_exact_int(w_start) || !bound_is_none_or_exact_int(w_end) {
        return str_descr_rfind_slow(args);
    }
    let start = str_search_bound_int(w_start, 0);
    let end = str_search_bound_int(w_end, i64::MAX);
    let n = pyre_object::unicodeobject::jit_str_rfind_bounds(args[0], args[1], start, end);
    Ok(w_int_new(n))
}

/// `unicodeobject.py descr_count`. Bounds are unwrapped here, then `ll_count`.
pub fn __majit_wrap_str_descr_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 || args.len() > 4 {
        return str_descr_count_slow(args);
    }
    if unsafe { !pyre_object::is_str(args[0]) || !pyre_object::is_str(args[1]) } {
        return str_descr_count_slow(args);
    }
    let w_start = str_search_bound(args, 2);
    let w_end = str_search_bound(args, 3);
    if !bound_is_none_or_exact_int(w_start) || !bound_is_none_or_exact_int(w_end) {
        return str_descr_count_slow(args);
    }
    let start = str_search_bound_int(w_start, 0);
    let end = str_search_bound_int(w_end, i64::MAX);
    let n = pyre_object::unicodeobject::jit_str_count_bounds(args[0], args[1], start, end);
    Ok(w_int_new(n))
}

pub fn str_method_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "find", 1)?;
    arity_at_most(args, "find", 3)?;
    require_str_sub(args, "find")?;
    Ok(w_int_new(str_unwrap_and_search(args, true)?.unwrap_or(-1)))
}

pub fn str_method_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rfind", 1)?;
    arity_at_most(args, "rfind", 3)?;
    require_str_sub(args, "rfind")?;
    Ok(w_int_new(str_unwrap_and_search(args, false)?.unwrap_or(-1)))
}

/// `unicodeobject.py W_UnicodeObject._upper_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn upper_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(wtf8_map_str_runs(&s, str::to_uppercase))
}

pub fn str_method_upper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "upper")?;
    Ok(unsafe { upper_unicode(args[0]) })
}

/// `unicodeobject.py W_UnicodeObject._lower_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn lower_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(wtf8_map_str_runs(&s, str::to_lowercase))
}

pub fn str_method_lower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "lower")?;
    Ok(unsafe { lower_unicode(args[0]) })
}

/// PyPy: unicodeobject.py descr_format
/// Requires format spec parser — correct for no-arg case only.
/// `str.format(*args)` — PyPy: unicodeobject.py descr_format → newformat.py
pub fn descr_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "format")?;
    // `pypy/objspace/std/newformat.py Formatter.format` —
    // positional args are slots 1.. of the receiver; keyword args
    // (`{name}` lookups) live in the trailing CALL_KW dict.
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(&args[1..]);
    str_method_format_core(args[0], positional, kwargs_dict, None)
}

/// Shared core for `str.format` (`{name}` looks up the trailing
/// CALL_KW dict) and `str.format_map` (`{name}` looks up the
/// mapping via `space.getitem(mapping, w_key)`).  PyPy folds both
/// into one `newformat.format_method(space, fmt, args_w, w_kwds, ...)`
/// entry point per `unicodeobject.py:422-430`; pyre splits the
/// keyword-source into "dict snapshot" vs "lazy mapping" so the
/// mapping path stays line-by-line lazy (no pre-materialisation).
fn str_method_format_core(
    fmt_obj: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    // Copy the template off the object before field lookup / rendering
    // allocate. Publish the template and every operand first so the copy
    // cannot move them out from under the words handed to `format_render`.
    // A lone surrogate in a literal run (or a surrogate arg spliced in)
    // still survives instead of panicking.
    let (fmt, positional, kwargs_dict, mapping) = {
        let _roots = pyre_object::gc_roots::push_roots();
        let fmt_base = pyre_object::gc_roots::publish_roots(&[fmt_obj]);
        let pos_base = pyre_object::gc_roots::publish_roots(positional);
        let kw_base = pyre_object::gc_roots::publish_roots(&[
            kwargs_dict.unwrap_or(pyre_object::PY_NULL),
            mapping.unwrap_or(pyre_object::PY_NULL),
        ]);
        pyre_object::gc_roots::normalize_roots(fmt_base, 1 + positional.len() + 2);
        let fmt = unsafe {
            pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(fmt_base))
        }
        .to_wtf8_buf();
        let positional: Vec<_> = (0..positional.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(pos_base + i))
            .collect();
        let kwargs = pyre_object::gc_roots::shadow_stack_get(kw_base);
        let mapping = pyre_object::gc_roots::shadow_stack_get(kw_base + 1);
        (
            fmt,
            positional,
            if kwargs.is_null() { None } else { Some(kwargs) },
            if mapping.is_null() {
                None
            } else {
                Some(mapping)
            },
        )
    };
    let mut auto_idx = 0usize;
    // `newformat.py` auto_numbering_state — `None` = ANS_INIT, `Some(true)`
    // = ANS_AUTO (empty `{}` fields), `Some(false)` = ANS_MANUAL (numbered
    // `{0}` fields).  Mixing the two raises ValueError.
    let mut numbering: Option<bool> = None;
    match format_render(
        &fmt,
        &positional,
        kwargs_dict,
        mapping,
        &mut auto_idx,
        &mut numbering,
        2,
    )? {
        TemplateRender::Object(obj) => Ok(obj),
        // `str.format` / `%` rendering is a dominant dynamic-churn producer.
        TemplateRender::Text(rendered) => Ok(pyre_object::w_str_from_wtf8_managed(rendered)),
    }
}

/// What a template rendered to.
///
/// `build_string` (`unicode_format.h`) renders through a writer that adopts a
/// lone written `str` as its own buffer instead of copying it, so a template
/// that is exactly one replacement field yields that field's own formatted
/// object — its identity and a `str` subclass's type both survive
/// `"{}".format(x)` and `"{k}".format_map(m)`.  Every other template yields
/// text, and a caller that cannot use an object (the nested-spec recursion
/// splices its result into the surrounding template) reads it back with
/// [`TemplateRender::text`].
enum TemplateRender {
    Text(Wtf8Buf),
    Object(PyObjectRef),
}

impl TemplateRender {
    fn text(self) -> Wtf8Buf {
        match self {
            TemplateRender::Text(text) => text,
            TemplateRender::Object(obj) => {
                unsafe { pyre_object::w_str_get_wtf8(obj) }.to_wtf8_buf()
            }
        }
    }
}

/// `newformat.py Formatter.format` rendering pass.  Renders the
/// template `fmt`, threading the auto-/manual-numbering state through
/// the recursive evaluation of nested `{...}` format specs so that
/// `"{:{}}".format(42, ">5")` consumes positional args 0 then 1.
/// `depth` bounds that recursion (the markup recursion limit is 2).
fn format_render(
    fmt: &Wtf8,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
    auto_idx: &mut usize,
    numbering: &mut Option<bool>,
    depth: u32,
) -> Result<TemplateRender, crate::PyError> {
    use rustpython_common::format::{
        FieldName, FieldNamePart, FieldType, FormatParseError, FromTemplate,
    };
    // A gateway hands a builtin its arguments as a native array rebuilt from
    // root slots immediately before the indirect call
    // (`call_builtin_code_positional`), so that array is right once and then
    // frozen. Every field after the first is resolved once this loop has run
    // `__getattr__` / `__repr__` / `__format__`, and a minor collection there
    // relocates a young argument while leaving the array holding its pre-move
    // address. Publish the run and both keyword sources here and read each one
    // back at its use; the absent source takes a slot too, so the two indices
    // stay fixed whichever spelling the caller used.
    let npositional = positional.len();
    let has_kwargs = kwargs_dict.is_some();
    let has_mapping = mapping.is_some();
    let _roots = pyre_object::gc_roots::push_roots();
    let positional_base = pyre_object::gc_roots::publish_roots(positional);
    let kwargs_slot =
        pyre_object::gc_roots::publish_roots(&[kwargs_dict.unwrap_or(pyre_object::PY_NULL)]);
    let mapping_slot =
        pyre_object::gc_roots::publish_roots(&[mapping.unwrap_or(pyre_object::PY_NULL)]);
    pyre_object::gc_roots::normalize_roots(positional_base, npositional + 2);
    let lookup_kwarg = |name: &str| -> Result<Option<PyObjectRef>, crate::PyError> {
        if has_mapping {
            // `newformat.format_method(... w_mapping, True)` resolves
            // `{name}` via `space.getitem(mapping, w_key)` per
            // `newformat.py:Template.get_value`; KeyError propagates
            // to the caller (no silent default).
            let _key_roots = pyre_object::gc_roots::push_roots();
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
            // `w_str_new_managed` / `pin_root` may collect; reload the
            // mapping only after the key is published.
            return crate::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(mapping_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
            )
            .map(Some);
        }
        if has_kwargs {
            let dict = pyre_object::gc_roots::shadow_stack_get(kwargs_slot);
            let v = unsafe { pyre_object::w_dict_getitem_str(dict, name) };
            return Ok(v);
        }
        Ok(None)
    };

    if depth == 0 {
        return Err(crate::PyError::value_error("Max string recursion exceeded"));
    }
    let parsed = parse_format_parts(fmt)?;
    // Read once, before the loop the field emission below tests it in.
    let sole_field = parsed.len() == 1 && matches!(parsed[0], PyPyFormatPart::Field { .. });
    let mut result = Wtf8Buf::new();
    for part in &parsed {
        let (field_name, conversion_spec, format_spec) = match part {
            PyPyFormatPart::Literal(literal) => {
                result.push_wtf8(literal);
                continue;
            }
            PyPyFormatPart::Field {
                field_name,
                conversion_spec,
                format_spec,
            } => (field_name, conversion_spec, format_spec),
        };

        // `_get_argument` — resolve the base argument named by the field,
        // threading the auto-/manual-numbering state (`None` = uncommitted,
        // `Some(true)` = automatic `{}`, `Some(false)` = manual `{0}`).
        //
        // A missing `]` terminates the field, so it precedes base-argument
        // resolution; every other chain error (empty attribute, empty/`!`
        // item, stray char after `]`) is a resolution error that follows the
        // base lookup. Classify only the head (up to the first `.`/`[`) here,
        // so a missing/out-of-range base raises IndexError/KeyError before a
        // malformed attribute or item chain raises ValueError.
        let full = FieldName::parse(field_name);
        if matches!(full, Err(FormatParseError::MissingRightBracket)) {
            return Err(format_parse_err(FormatParseError::MissingRightBracket, fmt));
        }
        let mut head = Wtf8Buf::new();
        for ch in field_name.code_points() {
            if ch == '.' || ch == '[' {
                break;
            }
            head.push(ch);
        }
        let FieldName { field_type, .. } =
            FieldName::parse(&head).map_err(|e| format_parse_err(e, fmt))?;
        if has_mapping && matches!(field_type, FieldType::Auto | FieldType::Index(_)) {
            return Err(crate::PyError::value_error(
                "Format string contains positional fields",
            ));
        }
        let mut val = match field_type {
            FieldType::Auto => {
                if let Some(false) = *numbering {
                    return Err(crate::PyError::value_error(
                        "cannot switch from manual field specification to automatic \
                         field numbering",
                    ));
                }
                *numbering = Some(true);
                let idx = *auto_idx;
                *auto_idx += 1;
                index_positional(positional_base, npositional, idx)?
            }
            FieldType::Index(idx) => {
                if let Some(true) = *numbering {
                    return Err(crate::PyError::value_error(
                        "cannot switch from automatic field numbering to manual \
                         field specification",
                    ));
                }
                *numbering = Some(false);
                index_positional(positional_base, npositional, idx)?
            }
            FieldType::Keyword(name) => {
                let name_str = name.as_str().unwrap_or("");
                match lookup_kwarg(name_str)? {
                    Some(v) => v,
                    None => {
                        return Err(crate::PyError::key_error_with_key(
                            pyre_object::w_str_from_wtf8_managed(name.clone()),
                        ));
                    }
                }
            }
        };

        // Surface any remaining chain parse error now that the base argument
        // has been resolved, then walk the attribute/item chain.
        let FieldName { parts, .. } = full.map_err(|e| format_parse_err(e, fmt))?;

        // `_resolve_lookups` — walk the `.attr` / `[element]` chain; a
        // bracketed all-digit element is an integer index, anything else a
        // string key (already classified by `FieldNamePart`).
        //
        // Each hop can collect. Keep the current value in one shadow-stack
        // slot and reload it before the next getattr/getitem, the same way
        // the nested-spec arm already republishes `val`. The scope is this
        // field only (`newformat.py` `_render_field` locals).
        let field_roots = pyre_object::gc_roots::push_roots();
        let val_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(val);
        for name_part in &parts {
            let receiver = pyre_object::gc_roots::shadow_stack_get(val_slot);
            let next = match name_part {
                FieldNamePart::Attribute(attr) => {
                    crate::baseobjspace::getattr_str(receiver, attr.as_str().unwrap_or(""))?
                }
                FieldNamePart::Index(idx) => {
                    let idx_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(*idx as i64));
                    crate::baseobjspace::getitem(
                        pyre_object::gc_roots::shadow_stack_get(val_slot),
                        pyre_object::gc_roots::shadow_stack_get(idx_slot),
                    )?
                }
                FieldNamePart::StringIndex(key) => {
                    // An all-digit bracket element that reached here overflowed
                    // the integer index — a value that fits is already
                    // classified as `Index` — so reject it like the head does.
                    if key
                        .as_str()
                        .ok()
                        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    {
                        return Err(crate::PyError::value_error(
                            "Too many decimal digits in format string",
                        ));
                    }
                    let key_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
                        key.clone(),
                    ));
                    crate::baseobjspace::getitem(
                        pyre_object::gc_roots::shadow_stack_get(val_slot),
                        pyre_object::gc_roots::shadow_stack_get(key_slot),
                    )?
                }
            };
            pyre_object::gc_roots::shadow_stack_set(val_slot, next);
        }
        let mut val = pyre_object::gc_roots::shadow_stack_get(val_slot);

        // A spec containing `{` is itself a template: render it (sharing the
        // numbering state and the recursion budget) before applying it.
        let resolved_spec = if format_spec.as_bytes().contains(&b'{') {
            // The chain walk above ran Python, so every argument reachable
            // only through this frame's native slice is a pre-move copy —
            // including `val`, which the conversion below still needs. Hand
            // the nested render the reloaded run (it publishes what it is
            // given, so passing the stale slice would launder the stale words
            // into its own roots) and read `val` back once it returns.
            let inner = pyre_object::gc_roots::push_roots();
            let val_slot = inner.base();
            let _ = inner.pin_root(val);
            let reloaded: Vec<PyObjectRef> = (0..npositional)
                .map(|i| pyre_object::gc_roots::shadow_stack_get(positional_base + i))
                .collect();
            let kwargs_now =
                has_kwargs.then(|| pyre_object::gc_roots::shadow_stack_get(kwargs_slot));
            let mapping_now =
                has_mapping.then(|| pyre_object::gc_roots::shadow_stack_get(mapping_slot));
            let spec = format_render(
                format_spec,
                &reloaded,
                kwargs_now,
                mapping_now,
                auto_idx,
                numbering,
                depth - 1,
            )?
            .text();
            val = inner.get(val_slot);
            spec
        } else {
            format_spec.clone()
        };

        // `do_conversion` — `!s`/`!r`/`!a` apply str / repr / ascii before
        // the format spec; any other conversion char is an error.  `!s`
        // preserves WTF-8 so a lone surrogate passes through unchanged.
        let converted = match conversion_spec {
            None => val,
            Some(cp) => match cp.to_char_lossy() {
                's' => pyre_object::w_str_from_wtf8_managed(unsafe { crate::py_str_wtf8(val)? }),
                'r' => pyre_object::w_str_from_wtf8_managed(unsafe { crate::py_repr_wtf8(val)? }),
                'a' => pyre_object::w_str_new_managed(&crate::builtins::py_ascii(val)?),
                // `\0` is the sentinel for a field with no conversion
                // requested, so a literal NUL conversion character formats the
                // field as if the `!` had not been written.
                '\0' => val,
                _ => {
                    return Err(crate::PyError::value_error(format!(
                        "Unknown conversion specifier {}",
                        unknown_conversion_display(*cp)
                    )));
                }
            },
        };
        if sole_field {
            // The whole template is this field, so hand back the object
            // `__format__` produced rather than a copy of its text.  An empty
            // spec crosses as `PY_NULL`, which `format_w` reads as empty:
            // building a `str` for it would allocate where the byte path
            // allocated nothing.
            let inner = pyre_object::gc_roots::push_roots();
            let converted_slot = inner.base();
            let _ = inner.pin_root(converted);
            let spec_obj = if resolved_spec.is_empty() {
                pyre_object::PY_NULL
            } else {
                pyre_object::w_str_from_wtf8_managed(resolved_spec)
            };
            return Ok(TemplateRender::Object(format_w(
                inner.get(converted_slot),
                spec_obj,
            )?));
        }
        let formatted = format_value_dispatch(converted, &resolved_spec)?;
        drop(field_roots);
        result.push_wtf8(&formatted);
    }
    Ok(TemplateRender::Text(result))
}

enum PyPyFormatPart {
    Literal(Wtf8Buf),
    Field {
        field_name: Wtf8Buf,
        conversion_spec: Option<CodePoint>,
        format_spec: Wtf8Buf,
    },
}

fn codepoint_slice(codepoints: &[CodePoint], start: usize, end: usize) -> Wtf8Buf {
    let mut out = Wtf8Buf::new();
    for &cp in &codepoints[start..end] {
        out.push(cp);
    }
    out
}

/// `newformat.py:TemplateFormatter._do_build_string`, splitting each field
/// the way `unicode_format.h parse_field` does.  Braces inside the
/// first-part item lookup (`{[{]}`) are ordinary key text; braces after `:`
/// are recursive format markup.
fn parse_format_parts(fmt: &Wtf8) -> Result<Vec<PyPyFormatPart>, crate::PyError> {
    let s: Vec<CodePoint> = fmt.code_points().collect();
    let end = s.len();
    let mut parts = Vec::new();
    let mut literal = Wtf8Buf::new();
    let mut i = 0usize;
    while i < end {
        let c = s[i];
        if c != '{' && c != '}' {
            literal.push(c);
            i += 1;
            continue;
        }
        if c == '}' {
            if i + 1 == end || s[i + 1] != '}' {
                return Err(crate::PyError::value_error(
                    "Single '}' encountered in format string",
                ));
            }
            literal.push(c);
            i += 2;
            continue;
        }
        if i + 1 == end {
            return Err(crate::PyError::value_error(
                "Single '{' encountered in format string",
            ));
        }
        if s[i + 1] == '{' {
            literal.push(c);
            i += 2;
            continue;
        }
        if !literal.is_empty() {
            parts.push(PyPyFormatPart::Literal(std::mem::take(&mut literal)));
        }

        i += 1;
        parts.push(parse_field(&s, &mut i, end)?);
    }
    if !literal.is_empty() {
        parts.push(PyPyFormatPart::Literal(literal));
    }
    Ok(parts)
}

/// `unicode_format.h parse_field`: split the replacement field beginning at
/// `*i` — already past its `{` — into name, conversion character and format
/// spec, leaving `*i` just past the field's closing `}`.
///
/// The name ends at the end of the string or at a `:` / `!`; a `[` suspends
/// that search until the matching `]`, so a `:` inside an item key is key
/// text.  The single code point after a `!` is the conversion character
/// whatever it is — a `{` or `}` there is consumed as one rather than read as
/// markup — and the spec that may follow it runs to the `}` that balances the
/// field, so a nested `{...}` stays inside the spec.
fn parse_field(
    s: &[CodePoint],
    i: &mut usize,
    end: usize,
) -> Result<PyPyFormatPart, crate::PyError> {
    let field_start = *i;
    let mut c = CodePoint::from_char('\0');
    while *i < end {
        c = s[*i];
        *i += 1;
        if c == '{' {
            return Err(crate::PyError::value_error("unexpected '{' in field name"));
        }
        if c == '[' {
            while *i < end && s[*i] != ']' {
                *i += 1;
            }
            continue;
        }
        if c == '}' || c == ':' || c == '!' {
            break;
        }
    }
    let field_name = codepoint_slice(s, field_start, *i - 1);
    let mut conversion_spec = None;
    let format_spec = 'field: {
        if c == '}' {
            break 'field Wtf8Buf::new();
        }
        if c != ':' && c != '!' {
            return Err(crate::PyError::value_error(
                "expected '}' before end of string",
            ));
        }
        if c == '!' {
            if *i == end {
                return Err(crate::PyError::value_error(
                    "end of string while looking for conversion specifier",
                ));
            }
            conversion_spec = Some(s[*i]);
            *i += 1;
            if *i < end {
                let c = s[*i];
                *i += 1;
                if c == '}' {
                    break 'field Wtf8Buf::new();
                }
                if c != ':' {
                    return Err(crate::PyError::value_error(
                        "expected ':' after conversion specifier",
                    ));
                }
            }
        }
        let spec_start = *i;
        let mut count = 1usize;
        while *i < end {
            let c = s[*i];
            *i += 1;
            if c == '{' {
                count += 1;
            } else if c == '}' {
                count -= 1;
                if count == 0 {
                    break 'field codepoint_slice(s, spec_start, *i - 1);
                }
            }
        }
        return Err(crate::PyError::value_error("unmatched '{' in format spec"));
    };
    Ok(PyPyFormatPart::Field {
        field_name,
        conversion_spec,
        format_spec,
    })
}

/// Fetch positional argument `idx`, raising the `str.format` IndexError for
/// an out-of-range replacement index.
///
/// The read goes through the run `format_render` published, so the value is
/// the one the last collection left behind rather than the address the
/// gateway's native argument array froze.
fn index_positional(base: usize, len: usize, idx: usize) -> Result<PyObjectRef, crate::PyError> {
    if idx < len {
        return Ok(pyre_object::gc_roots::shadow_stack_get(base + idx));
    }
    Err(crate::PyError::index_error(format!(
        "Replacement index {idx} out of range for positional args tuple"
    )))
}

/// Map a template parse error to the matching `str.format` ValueError.
/// `FormatString` reports `UnmatchedBracket` for both a lone trailing `{`
/// and an opened-but-unclosed field, so the trailing byte disambiguates the
/// two CPython messages.
fn format_parse_err(
    err: rustpython_common::format::FormatParseError,
    fmt: &Wtf8,
) -> crate::PyError {
    use rustpython_common::format::FormatParseError as E;
    let msg = match err {
        E::UnmatchedBracket => {
            if fmt.as_bytes().last() == Some(&b'{') {
                "Single '{' encountered in format string"
            } else {
                "expected '}' before end of string"
            }
        }
        E::MissingStartBracket => "Single '}' encountered in format string",
        E::UnescapedStartBracketInLiteral => "Single '{' encountered in format string",
        E::MissingRightBracket => "expected '}' before end of string",
        E::EmptyAttribute => "Empty attribute in format string",
        E::UnknownConversion => "expected ':' after conversion specifier",
        // The template parser accepts only one bracket layer in a spec and
        // reports a second as `InvalidFormatSpecifier`; that is the markup
        // recursion limit being hit.
        E::InvalidFormatSpecifier => "Max string recursion exceeded",
        E::InvalidCharacterAfterRightBracket => "Only '[' and '.' may follow ']' in format string",
        E::TooManyDecimalDigits => "Too many decimal digits in format string",
    };
    crate::PyError::value_error(msg)
}

/// Geometry of a Python 3.14 format spec:
/// `[fill][align][sign][#][0][width][grouping][.precision[grouping]][type]`.
/// Only the presentation types the shared engine
/// cannot format correctly — integer `c` and non-finite floats — read
/// this back; every other case parses through `FormatSpec` directly.
struct ParsedSpec {
    /// The fill is the one position in a spec that may hold an arbitrary code
    /// point — including a lone surrogate, which has no `char` — so it is kept
    /// as a `CodePoint`.  Every other position is ASCII.
    fill: CodePoint,
    align: Option<char>,
    sign: Option<char>,
    alt_form: bool,
    width: usize,
    grouping: Option<char>,
    fractional_grouping: Option<char>,
    precision: Option<usize>,
    ty: char,
    /// The `z` option (PEP 682): coerce a negative-zero result to positive
    /// zero. Held only in the pyre-side parse — the engine spec drops it.
    no_neg_zero: bool,
    /// The equivalent pre-3.14 spec accepted by rustpython-common's format
    /// engine (the fractional grouping marker, its otherwise-empty dot, and
    /// the `z` option are removed).
    engine_spec: Wtf8Buf,
    width_start: usize,
    width_end: usize,
    /// More than one code point remained after the presentation type.  The
    /// local scanner retains the first one in `ty`; this bit preserves the
    /// parser's final `length - i > 1` validation from `newformat.py`.
    invalid_trailing: bool,
}

/// The lossy `char` view of a spec, for the scans that only test for ASCII
/// delimiters.  A code point with no `char` (a lone surrogate) matches none of
/// them, so substituting it changes no decision.
fn spec_chars(spec: &Wtf8) -> Vec<char> {
    spec.code_points().map(|cp| cp.to_char_lossy()).collect()
}

/// PyPy `newformat.py:_parse_int`: accumulate a decimal field only while it
/// still fits in `sys.maxint` (`Py_ssize_t` in CPython).  Checking before the
/// multiply is important: an overlong width or precision is a Python
/// `ValueError`, not a Rust integer-overflow panic.
fn parse_spec_int(chars: &[char], i: &mut usize) -> Result<usize, crate::PyError> {
    let mut result = 0usize;
    while *i < chars.len() && chars[*i].is_ascii_digit() {
        let digit = (chars[*i] as u8 - b'0') as usize;
        if result > ((isize::MAX as usize) - digit) / 10 {
            return Err(crate::PyError::value_error(
                "Too many decimal digits in format string",
            ));
        }
        result = result * 10 + digit;
        *i += 1;
    }
    Ok(result)
}

fn parse_spec(spec: &Wtf8) -> Result<ParsedSpec, crate::PyError> {
    let cps: Vec<CodePoint> = spec.code_points().collect();
    // The structural scan below tests for ASCII delimiters only, so it runs on
    // the lossy `char` view; a code point with no `char` matches none of them
    // either way.  `cps` keeps the true code points for the fill and for the
    // engine spec.
    let chars: Vec<char> = cps.iter().map(|cp| cp.to_char_lossy()).collect();
    let mut i = 0;
    let n = chars.len();
    let mut fill = CodePoint::from_char(' ');
    let mut align: Option<char> = None;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = cps[0];
        align = Some(chars[1]);
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = Some(chars[0]);
        i = 1;
    }
    let mut sign: Option<char> = None;
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        sign = Some(chars[i]);
        i += 1;
    }
    // PEP 682 `z`: sits after the sign and before `#`. The engine does not
    // accept it, so record it and drop its index from `engine_spec` below.
    let mut no_neg_zero = false;
    let mut z_index: Option<usize> = None;
    if i < n && chars[i] == 'z' {
        no_neg_zero = true;
        z_index = Some(i);
        i += 1;
    }
    let mut alt_form = false;
    if i < n && chars[i] == '#' {
        alt_form = true;
        i += 1;
    }
    if i < n && chars[i] == '0' {
        // The `0` flag implies `fill='0', align='='` when neither was given.
        if align.is_none() {
            align = Some('=');
        }
        if fill == ' ' {
            fill = CodePoint::from_char('0');
        }
        i += 1;
    }
    let width_start = i;
    let width = parse_spec_int(&chars, &mut i)?;
    let width_end = i;
    let mut grouping: Option<char> = None;
    if i < n && matches!(chars[i], ',' | '_') {
        grouping = Some(chars[i]);
        i += 1;
    }
    let mut precision: Option<usize> = None;
    let mut fractional_grouping: Option<char> = None;
    let mut empty_precision_dot: Option<usize> = None;
    let mut fractional_grouping_index: Option<usize> = None;
    if i < n && chars[i] == '.' {
        let dot = i;
        i += 1;
        let precision_start = i;
        let p = parse_spec_int(&chars, &mut i)?;
        if i < n && matches!(chars[i], ',' | '_') {
            fractional_grouping = Some(chars[i]);
            fractional_grouping_index = Some(i);
            if i == precision_start {
                // Python 3.14's `._f` means default precision plus
                // fractional grouping, not precision zero.
                empty_precision_dot = Some(dot);
                precision = None;
            } else {
                precision = Some(p);
            }
            i += 1;
        } else {
            // `parse_internal_render_format_spec` counts a grouping marker as
            // consumed precision, so a dot followed by neither digits nor one
            // is the error; it is reported while parsing, ahead of the
            // presentation code the spec ends with.
            if i == precision_start {
                return Err(crate::PyError::value_error(
                    "Format specifier missing precision",
                ));
            }
            precision = Some(p);
        }
    }
    let ty = if i < n { chars[i] } else { '\0' };
    let invalid_trailing = i.saturating_add(1) < n;
    let mut engine_spec = Wtf8Buf::with_capacity(spec.len());
    for (index, cp) in cps.iter().enumerate() {
        if Some(index) == fractional_grouping_index
            || Some(index) == empty_precision_dot
            || Some(index) == z_index
        {
            continue;
        }
        engine_spec.push(*cp);
    }
    // `z` precedes the width, so dropping it from `engine_spec` shifts the
    // width slice left by one; `float_engine_spec_with_width` indexes the
    // engine spec, so bias the recorded bounds to match.
    let z_shift = z_index.is_some() as usize;
    Ok(ParsedSpec {
        fill,
        align,
        sign,
        alt_form,
        width,
        grouping,
        fractional_grouping,
        precision,
        ty,
        no_neg_zero,
        engine_spec,
        width_start: width_start - z_shift,
        width_end: width_end - z_shift,
        invalid_trailing,
    })
}

/// Complete the Python 3.14 syntax checks that are intentionally outside the
/// pre-3.14 `rustpython_common::FormatSpec` parser.  PyPy's
/// `newformat.py:_parse_spec` rejects a second grouping marker before choosing
/// a presentation type, and rejects trailing code points as one invalid spec.
/// Python 3.14 permits a grouping marker after the precision dot as well, so
/// apply the same ordering to that new slot.  `default_type` is the
/// presentation code the lane falls back to when the spec names none.
fn validate_python314_spec_shape(
    p: &ParsedSpec,
    spec: &Wtf8,
    type_name: &str,
    default_type: char,
) -> Result<(), crate::PyError> {
    if let Some(first) = p.grouping
        && matches!(p.ty, ',' | '_')
    {
        if first != p.ty {
            return Err(crate::PyError::value_error(
                "Cannot specify both ',' and '_'.",
            ));
        }
        if !p.invalid_trailing {
            return Err(crate::PyError::value_error(format!(
                "Cannot specify '{first}' with '{}'.",
                p.ty
            )));
        }
    }
    if let Some(first) = p.fractional_grouping
        && matches!(p.ty, ',' | '_')
    {
        if first != p.ty {
            return Err(crate::PyError::value_error(
                "Cannot specify both ',' and '_'.",
            ));
        }
        if !p.invalid_trailing {
            return Err(crate::PyError::value_error(format!(
                "Unknown format code '{}' for object of type '{type_name}'",
                p.ty
            )));
        }
    }
    if p.invalid_trailing {
        return Err(crate::PyError::value_error(format!(
            "Invalid format specifier '{spec}' for object of type '{type_name}'"
        )));
    }
    // `_parse_spec` validates a requested grouping option against the
    // presentation code while parsing the spec, before the value's own type
    // decides which codes it can render.  PEP 378 allows both separators on
    // the decimal and floating codes; PEP 515 adds `_` to the bin/oct/hex
    // radices, where it groups every four digits rather than three.
    //
    // The presentation code the switch reads is the one the parse settled on,
    // so an omitted slot carries the lane's own default: `'d'` for an integer
    // and `'s'` for a string, both of which decide the pairing differently
    // from the float lanes' `'\0'`.
    let ty = if p.ty == '\0' { default_type } else { p.ty };
    if let Some(separator) = p.grouping
        && !matches!(ty, 'd' | 'e' | 'f' | 'g' | 'E' | 'G' | '%' | 'F' | '\0')
        && !(separator == '_' && matches!(ty, 'b' | 'o' | 'x' | 'X'))
    {
        return Err(invalid_thousands_separator_type(separator, ty));
    }
    if ty == 'n'
        && let Some(separator) = p.fractional_grouping
    {
        return Err(invalid_thousands_separator_type(separator, ty));
    }
    Ok(())
}

/// `invalid_thousands_separator_type`: report a grouping option the
/// presentation code does not accept.  A code outside printable ASCII is
/// spelled as its own escape, the way `unknown_presentation_type` spells one.
fn invalid_thousands_separator_type(specifier: char, presentation_type: char) -> crate::PyError {
    crate::PyError::value_error(format!(
        "Cannot specify '{specifier}' with '{}'.",
        unknown_code_display(presentation_type)
    ))
}

/// Component spec used by CPython's complex advanced formatter.  Width and
/// alignment belong to the joined complex value; every remaining numeric flag
/// is repeated for the real and imaginary float components.
fn complex_component_spec(p: &ParsedSpec, sign: char) -> String {
    let mut spec = String::new();
    if sign != '-' {
        spec.push(sign);
    }
    if p.no_neg_zero {
        spec.push('z');
    }
    if p.alt_form {
        spec.push('#');
    }
    if let Some(grouping) = p.grouping {
        spec.push(grouping);
    }
    if let Some(precision) = p.precision {
        spec.push('.');
        spec.push_str(&precision.to_string());
        if let Some(grouping) = p.fractional_grouping {
            spec.push(grouping);
        }
    } else if let Some(grouping) = p.fractional_grouping {
        spec.push('.');
        spec.push(grouping);
    }
    // `format_complex_internal` turns an omitted presentation type into `'r'`
    // and only then, when a precision was supplied, into `'g'`.  The float
    // lane's own omitted type carries `Py_DTSF_ADD_DOT_0`, which shortens the
    // point at which `'g'` flips to exponent form by one significant digit, so
    // naming `'g'` here is what keeps the lanes off that flag.
    let ty = if p.ty == '\0' && p.precision.is_some() {
        'g'
    } else {
        p.ty
    };
    if ty != '\0' {
        spec.push(ty);
    }
    spec
}

/// Format one real/imaginary lane.  `repr_type` is `format_complex_internal`'s
/// `'r'` mode — an omitted presentation type with no precision — which renders
/// through the float lane's omitted type and then drops the trailing `.0` that
/// `Py_DTSF_ADD_DOT_0` appended.  An alternate form keeps the point the flag
/// asks for, but not the digit after it.
fn format_complex_component(
    value: f64,
    spec: &Wtf8,
    repr_type: bool,
    alternate: bool,
) -> Result<String, crate::PyError> {
    let rendered = if value.is_nan() || value.is_infinite() {
        format_nonfinite(value, spec)?
    } else {
        format_finite_float(value, spec)?
    };
    let mut text = rendered
        .as_str()
        .expect("numeric formatting always produces UTF-8")
        .to_string();
    if repr_type && let Some(head) = text.strip_suffix(".0") {
        text.truncate(head.len());
        if alternate {
            text.push('.');
        }
    }
    Ok(text)
}

/// Return the normalized engine spec with its width replaced.  Fractional
/// grouping is applied after rustpython-common has rounded the number; reducing
/// the engine width by the number of inserted separators keeps Python's width
/// measured against the final grouped result.
fn float_engine_spec_with_width(p: &ParsedSpec, width: Option<usize>) -> Wtf8Buf {
    let cps: Vec<CodePoint> = p.engine_spec.code_points().collect();
    let mut out = Wtf8Buf::new();
    for cp in &cps[..p.width_start] {
        out.push(*cp);
    }
    if let Some(width) = width
        && width != 0
    {
        out.push_str(&width.to_string());
    }
    for cp in &cps[p.width_end..] {
        out.push(*cp);
    }
    out
}

fn fractional_digit_count(s: &str) -> usize {
    let Some(dot) = s.find('.') else {
        return 0;
    };
    s[dot + 1..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count()
}

fn group_fractional_digits(s: String, separator: char, digits: usize) -> String {
    if digits < 4 {
        return s;
    }
    let Some(dot) = s.find('.') else {
        return s;
    };
    let fraction_start = dot + 1;
    let fraction_end = fraction_start + digits;
    let mut out = String::with_capacity(s.len() + (digits - 1) / 3);
    out.push_str(&s[..fraction_start]);
    for (index, byte) in s.as_bytes()[fraction_start..fraction_end]
        .iter()
        .enumerate()
    {
        if index != 0 && index % 3 == 0 {
            out.push(separator);
        }
        out.push(*byte as char);
    }
    out.push_str(&s[fraction_end..]);
    out
}

/// Pad `body` to `width` characters with `fill`, honouring the numeric
/// alignments.  `=` splits a leading sign (and `0x`/`0o`/`0b` base prefix)
/// from the digits and inserts the fill between them.
fn pad_to_width(
    body: String,
    fill: CodePoint,
    align: char,
    width: usize,
) -> Result<Wtf8Buf, crate::PyError> {
    if body.chars().count() >= width {
        return Ok(Wtf8Buf::from_string(body));
    }
    let need = width - body.chars().count();
    let capacity = padded_byte_len(body.len(), need, fill);
    Ok(match align {
        '<' => {
            let mut s = Wtf8Buf::from_string(body);
            push_cp_repeated(&mut s, fill, need)?;
            s
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            let mut s = crate::builtins::try_wtf8_with_capacity(capacity)?;
            push_cp_repeated(&mut s, fill, left)?;
            s.push_str(&body);
            push_cp_repeated(&mut s, fill, right)?;
            s
        }
        '=' => {
            let mut chars = body.chars().peekable();
            let mut prefix = String::new();
            if let Some(&c) = chars.peek()
                && (c == '-' || c == '+' || c == ' ')
            {
                prefix.push(c);
                chars.next();
            }
            let rest_so_far: String = chars.clone().collect();
            if rest_so_far.len() >= 2 && rest_so_far.as_bytes()[0] == b'0' {
                let next = rest_so_far.as_bytes()[1];
                if matches!(next, b'x' | b'X' | b'o' | b'b') {
                    prefix.push('0');
                    prefix.push(next as char);
                    chars.next();
                    chars.next();
                }
            }
            let digits: String = chars.collect();
            let mut s = crate::builtins::try_wtf8_with_capacity(capacity)?;
            s.push_str(&prefix);
            push_cp_repeated(&mut s, fill, need)?;
            s.push_str(&digits);
            s
        }
        _ => {
            let mut s = crate::builtins::try_wtf8_with_capacity(capacity)?;
            push_cp_repeated(&mut s, fill, need)?;
            s.push_str(&body);
            s
        }
    })
}

/// `newformat.py Formatter._get_locale`: the decimal point, thousands
/// separator and group sizes a numeric format renders with.
///
/// Three arms, in upstream's order.  `'n'` takes the current locale.  An
/// explicit `,` or `_` takes that separator at a fixed group size — four for
/// the power-of-two radices, three otherwise.  Everything else separates
/// nothing and carries the `0xFF` stop sentinel.
///
/// The locale bytes decode lossily because the body being assembled is a
/// `String`.  Upstream holds them as the bytes `charp2str` returned, RPython
/// strings being byte strings; a separator that is not valid UTF-8 is the one
/// input on which the two disagree.
fn get_locale(ty: char, thousands_sep: Option<char>) -> (String, String, Vec<u8>) {
    if ty == 'n' {
        let (dec, sep, grouping) = crate::module::_locale::rlocale::numeric_formatting();
        return (
            String::from_utf8_lossy(&dec).into_owned(),
            String::from_utf8_lossy(&sep).into_owned(),
            grouping,
        );
    }
    match thousands_sep {
        Some(separator) => {
            let group = if matches!(ty, 'b' | 'o' | 'x' | 'X') {
                4
            } else {
                3
            };
            (".".to_string(), separator.to_string(), vec![group])
        }
        None => (".".to_string(), String::new(), vec![0xFF]),
    }
}

/// `newformat.py Formatter._fill_digits`.  The separator is appended
/// ahead of the digits it precedes because `group_digits` reverses the whole
/// buffer once at the end.
///
/// Whole chunks go in where upstream appends one character at a time: the
/// final reverse puts the chunks in order and leaves each chunk's own contents
/// alone, which is also what keeps a multi-byte separator intact.
fn fill_digits(
    buf: &mut Vec<String>,
    digits: &str,
    d_state: i32,
    n_chars: i32,
    n_zeros: i32,
    thousands_sep: Option<&str>,
) {
    if let Some(separator) = thousands_sep
        && !separator.is_empty()
    {
        buf.push(separator.to_string());
    }
    if n_chars > 0 {
        let end = d_state as usize;
        buf.push(digits[end - n_chars as usize..end].to_string());
    }
    if n_zeros > 0 {
        buf.push("0".repeat(n_zeros as usize));
    }
}

/// `newformat.py Formatter._group_digits`: separate `digits` at the
/// group sizes `loc_grouping` names, zero-filling out to `min_width` while
/// doing so.
///
/// The zero fill belongs inside the grouping rather than after it: `0=` pads
/// to the width with digits, and those digits are separated like any other.
/// A group size past the end of the vector repeats the previous one, which is
/// how a C grouping string's last element means "and every group after this".
/// `0xFF` is the sentinel [`get_locale`] carries when nothing is to be
/// separated.  A locale that terminates its own string with `CHAR_MAX` needs no
/// arm of its own — `final_grouping` clamps any size to the digits left, so a
/// group of 127 consumes the rest and the loop ends.
fn group_digits(
    digits: &str,
    mut min_width: i32,
    mut left: i32,
    loc_grouping: &[u8],
    loc_thousands: &str,
) -> String {
    let mut buf: Vec<String> = Vec::new();
    let mut grouping_state = 0usize;
    let n_ts = loc_thousands.chars().count() as i32;
    let mut need_separator = false;
    let mut done = false;
    let mut previous = 0i32;
    loop {
        let group = if grouping_state >= loc_grouping.len() {
            previous
        } else {
            let group = i32::from(loc_grouping[grouping_state]);
            if group == 0xFF {
                break;
            }
            grouping_state += 1;
            previous = group;
            group
        };
        let final_grouping = group.min(left.max(min_width.max(1)));
        let n_zeros = (final_grouping - left).max(0);
        let n_chars = left.min(final_grouping).max(0);
        fill_digits(
            &mut buf,
            digits,
            left,
            n_chars,
            n_zeros,
            need_separator.then_some(loc_thousands),
        );
        need_separator = true;
        left -= n_chars;
        min_width -= final_grouping;
        if left <= 0 && min_width <= 0 {
            done = true;
            break;
        }
        min_width -= n_ts;
    }
    if !done {
        let group = left.max(min_width).max(1);
        let n_zeros = (group - left).max(0);
        let n_chars = left.min(group).max(0);
        fill_digits(
            &mut buf,
            digits,
            left,
            n_chars,
            n_zeros,
            need_separator.then_some(loc_thousands),
        );
    }
    buf.reverse();
    buf.concat()
}

#[cfg(test)]
mod group_digits_tests {
    use super::{get_locale, group_digits};

    /// Every expectation is what `newformat.py _group_digits` returns
    /// for the same arguments, taken from the vendored source rather than from
    /// a second implementation of the same rule.
    #[track_caller]
    fn check(digits: &str, min_width: i32, grouping: &[u8], separator: &str, expected: &str) {
        let got = group_digits(digits, min_width, digits.len() as i32, grouping, separator);
        assert_eq!(got, expected, "digits={digits:?} min_width={min_width}");
    }

    #[test]
    fn separates_at_the_group_size() {
        check("1234", 0, &[3], ",", "1,234");
        check("123456789", 0, &[3], ",", "123,456,789");
        check("12345678", 0, &[4], "_", "1234_5678");
    }

    #[test]
    fn zero_fills_to_min_width_inside_the_grouping() {
        // The zero fill of a `0=` spec is separated like any other digit, so
        // the result is longer than the width that asked for it — `format(1234,
        // '012n')` is thirteen characters.
        check("1234", 12, &[3], ",", "0,000,001,234");
        check("1234", 20, &[3], ",", "0,000,000,000,001,234");
        check("0", 12, &[3], ",", "0,000,000,000");
        check("7", 7, &[3], ",", "000,007");
        // A width the digits already exceed adds nothing.
        check("1234567", 5, &[3], ",", "1,234,567");
    }

    #[test]
    fn repeats_the_last_group_size_past_the_end_of_the_vector() {
        check("123456789", 0, &[3, 2], ",", "12,34,56,789");
    }

    #[test]
    fn stops_at_the_sentinel() {
        check("123456789", 0, &[3, 0xFF], ",", "123456,789");
        check("123456789", 0, &[0xFF], ",", "123456789");
    }

    #[test]
    fn keeps_a_multi_byte_separator_intact() {
        // The buffer is reversed once at the end, so a separator wider than a
        // byte has to travel as one chunk.
        check("1234", 0, &[3], "\u{202f}", "1\u{202f}234");
    }

    #[test]
    fn get_locale_sizes_the_group_by_presentation_code() {
        for ty in ['b', 'o', 'x', 'X'] {
            assert_eq!(get_locale(ty, Some('_')), (".".into(), "_".into(), vec![4]));
        }
        for ty in ['\0', 'd'] {
            assert_eq!(get_locale(ty, Some(',')), (".".into(), ",".into(), vec![3]));
        }
        // No separator asked for: nothing is separated.
        assert_eq!(
            get_locale('d', None),
            (".".into(), String::new(), vec![0xFF])
        );
    }
}

/// `FormatSpec::format_int`, with RPython rbigint as the value owner.
///
/// The shared parser remains the authority for format-spec syntax and error
/// ordering; radix conversion, grouping, signs, prefixes, and padding operate
/// directly on rbigint digits.
fn format_rbigint(num: &BigInt, spec: &Wtf8, type_name: &str) -> Result<Wtf8Buf, crate::PyError> {
    let parsed = rustpython_common::format::FormatSpec::parse(spec)
        .map_err(|e| format_spec_err(e, spec, type_name, true))?;
    let p = parse_spec(spec)?;
    // `format_long_internal` refuses a precision and nothing else: the
    // fractional grouping marker has no fraction to separate here, so
    // `.,`/`._` (which leave the precision unset) render like a bare spec.
    if p.precision.is_some() {
        return Err(crate::PyError::value_error(
            "Precision not allowed in integer format specifier",
        ));
    }
    // The group size a separator implies is not part of the radix: `_get_locale`
    // derives it from the presentation code, and for `'n'` it comes out of the
    // locale instead.
    let (radix, upper, prefix) = match p.ty {
        '\0' | 'd' => (10, false, ""),
        'b' => (2, false, if p.alt_form { "0b" } else { "" }),
        'o' => (8, false, if p.alt_form { "0o" } else { "" }),
        'x' => (16, false, if p.alt_form { "0x" } else { "" }),
        'X' => (16, true, if p.alt_form { "0X" } else { "" }),
        'n' => (10, false, ""),
        other => {
            return Err(crate::PyError::value_error(format!(
                "Unknown format code '{}' for object of type '{type_name}'",
                unknown_code_display(other)
            )));
        }
    };

    // Keep the parsed value live so syntax validation cannot be optimized
    // away independently from rendering.
    let _ = parsed;
    let digits = match radix {
        2 => "01",
        8 => majit_rlib::rbigint::BASE8,
        10 => majit_rlib::rbigint::BASE10,
        16 => majit_rlib::rbigint::BASE16,
        _ => unreachable!("integer format uses radix 2, 8, 10, or 16"),
    };
    let negative = num.int_lt(0);
    // Only the decimal conversion is quadratic, so only it carries the
    // `sys.set_int_max_str_digits` limit; the power-of-two radices are exempt.
    let maxdigits = if radix == 10 {
        crate::module::sys::state::int_max_str_digits()
    } else {
        0
    };
    let mut magnitude =
        num.format(digits, "", "", maxdigits as i64)
            .map_err(|error| match error {
                majit_rlib::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
                majit_rlib::rbigint::RBigIntError::MaxStrDigits => {
                    crate::builtins::int_max_str_digits_error(maxdigits)
                }
                _ => unreachable!("validated radix formatting returned an unrelated error"),
            })?;
    if negative {
        debug_assert!(magnitude.starts_with('-'));
        magnitude.remove(0);
    }
    if upper {
        magnitude.make_ascii_uppercase();
    }
    let sign = if negative {
        "-"
    } else {
        match p.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    let sign_prefix = format!("{sign}{prefix}");
    let (_loc_dec, loc_thousands, loc_grouping) = get_locale(p.ty, p.grouping);
    // `_calc_num_width:695-704`: the digits are separated only when there is a
    // separator to put between them, and the width a `0=` spec still owes after
    // the sign and the base prefix travels in as `n_min_width` so the zero fill
    // is separated along with them.
    if !loc_thousands.is_empty() && !magnitude.is_empty() {
        let n_min_width = if p.fill == '0' && p.align == Some('=') {
            p.width as i32 - sign_prefix.chars().count() as i32
        } else {
            0
        };
        magnitude = group_digits(
            &magnitude,
            n_min_width,
            magnitude.len() as i32,
            &loc_grouping,
            &loc_thousands,
        );
    }
    pad_to_width(
        format!("{sign_prefix}{magnitude}"),
        p.fill,
        p.align.unwrap_or('>'),
        p.width,
    )
}

/// A `&str` paired with its precomputed code-point count, adapting a
/// str body to the `CharLen + Deref<str>` bound `FormatSpec::format_string`
/// requires for width padding.
struct CharLenStr<'a>(&'a str, usize);

impl std::ops::Deref for CharLenStr<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}

impl rustpython_common::format::CharLen for CharLenStr<'_> {
    fn char_len(&self) -> usize {
        self.1
    }
}

/// Render a presentation-type code for the "Unknown format code" message:
/// printable ASCII (`0x21..=0x7f`) verbatim, anything else as `\x{hex}`.
fn unknown_code_display(c: char) -> String {
    unknown_code_point_display(CodePoint::from_char(c))
}

/// [`unknown_code_display`] for a code point that may have no `char` — a
/// presentation slot holding a lone surrogate prints as its own escape.
fn unknown_code_point_display(cp: CodePoint) -> String {
    match cp.to_char() {
        Some(c) if ('\u{21}'..='\u{7f}').contains(&c) => c.to_string(),
        _ => format!("\\x{:x}", cp.to_u32()),
    }
}

/// Render an unknown `!` conversion character for `do_conversion`'s message.
/// Its printable-ASCII window stops one code point earlier than
/// [`unknown_code_point_display`]'s: `do_conversion` escapes 0x7f, while
/// `unknown_presentation_type` prints it verbatim.
fn unknown_conversion_display(cp: CodePoint) -> String {
    match cp.to_char() {
        Some(c) if ('\u{21}'..'\u{7f}').contains(&c) => c.to_string(),
        _ => format!("\\x{:x}", cp.to_u32()),
    }
}

/// Map a `FormatSpec` engine error to the matching Python exception.
/// `spec` and `type_name` supply the value's format spec and type for the
/// messages that name them; `integer` selects the integer-specific text
/// for the `z` presentation code.
fn format_spec_err(
    err: rustpython_common::format::FormatSpecError,
    spec: &Wtf8,
    type_name: &str,
    integer: bool,
) -> crate::PyError {
    use rustpython_common::format::FormatSpecError as E;
    match err {
        E::DecimalDigitsTooMany => {
            crate::PyError::value_error("Too many decimal digits in format string")
        }
        // An integer never accepts a precision, so its presence is reported
        // before the value is range-checked — the size never surfaces.
        E::PrecisionTooBig if integer => {
            crate::PyError::value_error("Precision not allowed in integer format specifier")
        }
        E::PrecisionTooBig => crate::PyError::value_error("precision too big"),
        E::PrecisionMissing => crate::PyError::value_error("Format specifier missing precision"),
        E::InvalidFormatSpecifier => crate::PyError::value_error(format!(
            "Invalid format specifier '{spec}' for object of type '{type_name}'"
        )),
        E::UnspecifiedFormat(c1, c2) => {
            crate::PyError::value_error(format!("Cannot specify '{c1}' with '{c2}'."))
        }
        E::ExclusiveFormat(c1, c2) => {
            crate::PyError::value_error(format!("Cannot specify both '{c1}' and '{c2}'."))
        }
        E::UnknownFormatCode(c, _) if integer && c == 'z' => crate::PyError::value_error(
            "Negative zero coercion (z) not allowed in integer format specifier",
        ),
        E::UnknownFormatCode(c, _) => crate::PyError::value_error(format!(
            "Unknown format code '{}' for object of type '{type_name}'",
            unknown_code_display(c)
        )),
        E::PrecisionNotAllowed => {
            crate::PyError::value_error("Precision not allowed in integer format specifier")
        }
        E::NotAllowed(s) => crate::PyError::value_error(format!(
            "{s} not allowed with integer format specifier 'c'"
        )),
        E::UnableToConvert => crate::PyError::value_error("Unable to convert int to float"),
        E::CodeNotInRange => crate::PyError::overflow_error("%c arg not in range(0x110000)"),
        // A `c` code is read through `PyLong_AsLong`, so a value outside the C
        // long range is reported as that conversion failing rather than as the
        // code point range check.
        E::IntTooLargeForCLong => {
            crate::PyError::overflow_error("Python int too large to convert to C long")
        }
        E::ZeroPadding => {
            crate::PyError::value_error("Zero padding is not allowed in complex format specifier")
        }
        E::AlignmentFlag => crate::PyError::value_error(
            "'=' alignment flag is not allowed in complex format specifier",
        ),
        E::NegativeZeroCoercionNotAllowed(type_name) => crate::PyError::value_error(format!(
            "Negative zero coercion (z) not allowed in {type_name} format specifier"
        )),
        E::StringAlignmentFlag => {
            crate::PyError::value_error("'=' alignment not allowed in string format specifier")
        }
        E::StringSpecNotAllowed(s) => {
            crate::PyError::value_error(format!("{s} not allowed in string format specifier"))
        }
        E::NotImplemented(c, s) => crate::PyError::value_error(format!(
            "Format code '{c}' for object of type '{s}' not implemented yet"
        )),
    }
}

/// Public entry point for the f-string `FormatWithSpec` opcode in
/// `eval.rs::format_with_spec`. Forwards to the same parser used by
/// `str.format` so both surfaces share the spec semantics.
pub fn format_with_spec_public(val: PyObjectRef, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    format_with_spec(val, spec)
}

/// Bind the type-level `__format__` descriptor `meth` to `val` and call it
/// with `spec_obj`, requiring a `str` result.  Dispatching the looked-up
/// `meth` (rather than a fresh instance lookup) keeps `__format__` a
/// type-level special method — an instance-dict `__format__` is ignored —
/// and binds a `@staticmethod` / `@classmethod` / other descriptor override
/// through the descriptor protocol.  The spec is passed through untouched so
/// the type's own `__format__` runs its validation.
pub(crate) fn call_format_dispatch_w(
    val: PyObjectRef,
    meth: PyObjectRef,
    spec_obj: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let w_type = crate::typedef::r#type(val).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
        let result = crate::baseobjspace::get_and_call_function(meth, val, w_type, &[spec_obj])?;
        if !pyre_object::is_str(result) {
            return Err(crate::PyError::type_error(format!(
                "__format__ must return a str, not {}",
                arg_type_name(result)
            )));
        }
        // Returned as it came back: `format` (descroperation.py) checks
        // `isinstance_w(w_res, w_unicode)` and hands `w_res` itself on, so a
        // `str` subclass keeps its type and the result keeps its identity.
        Ok(result)
    }
}

/// [`call_format_dispatch_w`] for the callers that want the formatted bytes.
pub(crate) fn call_format_dispatch(
    val: PyObjectRef,
    meth: PyObjectRef,
    spec_obj: PyObjectRef,
) -> Result<Wtf8Buf, crate::PyError> {
    let result = call_format_dispatch_w(val, meth, spec_obj)?;
    Ok(unsafe { pyre_object::w_str_get_wtf8(result) }.to_wtf8_buf())
}

/// Whether the type-level `__format__` in `meth` is the shared builtin body
/// that `int`, `bool`, `float` and `str` publish, *and* applies to `val`.
///
/// The question the dispatch below has to answer is not how `__format__` is
/// spelled but whether calling it is the same thing as calling
/// [`format_with_spec_public`] -- which holds for [`builtin_value_format`] and
/// for nothing else, since that body is exactly `format_with_spec_public(self,
/// spec)` once its non-empty spec is read back out.  `object.__format__`
/// carries its own body and is not recognised, so a class that defines no
/// `__format__` of its own still reaches the TypeError it raises for a
/// non-empty spec.
///
/// The test this replaced named `BUILTIN_FUNCTION_TYPE`, a type a type-level
/// lookup never returns: `int.__format__` is a `method_descriptor`, so the
/// disjunct was true for every builtin and the fast path below it was
/// unreachable.  `f"{k:>5}"` on an exact `int` wrapped its spec in a `str`,
/// walked the generic call path, and read the same bytes back out.
///
/// # Safety
/// `val` and `meth` must be valid, non-null pointers to a `PyObject`.
unsafe fn is_shared_builtin_format(val: PyObjectRef, meth: PyObjectRef) -> bool {
    unsafe {
        let Some(found) = crate::jit_builtin_folds::builtin_code_fn_of(meth) else {
            return false;
        };
        if !crate::gateway::builtin_code_fn_eq(found, builtin_value_format) {
            return false;
        }
        // `descr_check` — the descriptor a type-level lookup finds need not be
        // the one that type publishes.  `class S(str): __format__ =
        // int.__format__` installs `int`'s, and calling it owes the descriptor
        // TypeError rather than formatting the receiver as a `str`.  Only a
        // descriptor whose `__objclass__` the receiver satisfies is the
        // inherited one whose call this path may stand in for.
        let Ok(owner) = crate::function::fget_func_objclass(meth) else {
            return false;
        };
        let Some(w_type) = crate::typedef::r#type(val) else {
            return false;
        };
        // The exact builtin -- the whole of what this path exists for -- is one
        // comparison; only a subclass pays the MRO scan.
        let w_type = w_type.as_ptr();
        std::ptr::eq(w_type, owner) || crate::baseobjspace::issubtype_w(w_type, owner)
    }
}

/// `PyObject_Format` — when `val` is a class instance whose type defines
/// `__format__`, dispatch to it (the result must be a `str`); otherwise
/// apply the shared builtin spec parser, with an empty spec collapsing to
/// `str(value)`.  Shared by `format()`, the `FormatSimple`/`FormatWithSpec`
/// f-string opcodes, and `str.format` field formatting.
pub fn format_value_dispatch(val: PyObjectRef, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    // Fast path for common types.  An exact `str` or `int` cannot carry a
    // `__format__` override, and with an empty spec its builtin `__format__`
    // is `str(value)` — so resolve and call nothing.  `bool` is excluded (it
    // is a distinct type with its own `__format__`), as is any subclass.
    if spec.is_empty()
        && unsafe {
            pyre_object::is_exact_type(val, &pyre_object::STR_TYPE)
                || pyre_object::is_exact_type(val, &pyre_object::INT_TYPE)
                || pyre_object::is_exact_type(val, &pyre_object::LONG_TYPE)
        }
    {
        return Ok(unsafe { crate::py_str_wtf8(val)? });
    }
    // A class instance always dispatches to its `__format__` (its own
    // override or the inherited `object.__format__`).  A builtin dispatches
    // whenever its `__format__` is anything other than the shared builtin
    // default — a `def`, `@staticmethod`, `@classmethod`, another type's own
    // builtin body (`object.__format__` among them), or any other descriptor;
    // the shared default takes the fast path below, which formats the
    // underlying value directly.  `__format__` is resolved on the type (not
    // the instance) so an instance-dict attribute does not shadow it.
    if let Some(meth) = unsafe { crate::baseobjspace::lookup(val, "__format__") }
        && (unsafe { is_instance(val) } || !unsafe { is_shared_builtin_format(val, meth) })
    {
        let _roots = pyre_object::gc_roots::push_roots();
        let val_slot = pyre_object::gc_roots::pin_roots(&[val, meth]);
        let meth_slot = val_slot + 1;
        let spec_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
            spec.to_wtf8_buf(),
        ));
        return call_format_dispatch(
            pyre_object::gc_roots::shadow_stack_get(val_slot),
            pyre_object::gc_roots::shadow_stack_get(meth_slot),
            pyre_object::gc_roots::shadow_stack_get(spec_slot),
        );
    }
    if spec.is_empty() {
        // Empty spec collapses to `str(value)`, preserved in WTF-8 so a
        // str — or an exception whose single argument is a str — keeps
        // its lone surrogates.
        Ok(unsafe { crate::py_str_wtf8(val)? })
    } else {
        Ok(format_with_spec_public(val, spec)?)
    }
}

/// `space.format(w_obj, w_format_spec)` (descroperation.py) — `PyObject_Format`
/// returning the formatted object rather than its bytes.
///
/// Upstream threads the spec from the caller to `__format__` and returns
/// `w_res` unchanged, so nothing is rebuilt at either end.  Going through a
/// `Wtf8Buf` instead costs a copy and a fresh `str` on the way in and again on
/// the way out, and the substitution is observable three ways: `spec is` the
/// argument that was passed, `format(x, s) is` what `__format__` returned, and
/// a `str` subclass result keeps its type.  Only a builtin `__format__` and
/// the fast paths below read the bytes.
///
/// A `PY_NULL` or non-`str` spec reads as the empty spec, matching
/// `format_simple`.  A user `__format__` reached that way is handed a real
/// empty `str`, since its second parameter is a string.
pub fn format_w(val: PyObjectRef, w_spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let spec_is_empty =
        w_spec.is_null() || unsafe { !is_str(w_spec) || pyre_object::w_str_len(w_spec) == 0 };

    // The empty-spec fast paths of `format_value_dispatch`: an exact `str` or
    // `int` cannot carry a `__format__` override, so resolve and call nothing.
    // `format_string` (newformat.py) hands the receiver's own utf8 storage to
    // `self.wrap(...)` once `_parse_spec("s", "<")` reports the defaults —
    // which only an empty spec does — so `format(s, "")` shares its operand's
    // storage and `is_w` (unicodeobject.py) reports the result identical to
    // `s`.  An exact `str` is the whole of that case: a subclass is converted
    // by `space.str(w_string)` (newformat.py) to a fresh base `str` first, and
    // `is_w` rejects a `user_overridden_class` operand anyway
    // (unicodeobject.py:106).
    if spec_is_empty {
        if unsafe { pyre_object::is_exact_type(val, &pyre_object::STR_TYPE) } {
            return Ok(val);
        }
        // `w_class` of a `W_LongObject` is `int`, so `is_exact_type(INT)`
        // also matches a long.  Machine int is the storage `ob_type`.
        // A subclass keeps that `ob_type` and may override `__str__`.
        if unsafe {
            pyre_object::py_type_check(val, &pyre_object::INT_TYPE)
                && pyre_object::is_exact_builtin_instance(val)
        } {
            // `format_int_or_long` empty spec is `space.str(w_num)` —
            // `intobject.py descr_str`.
            return Ok(unsafe { pyre_object::descr_str(val) });
        }
    }
    format_w_slow(val, w_spec)
}

/// `pyopcode.py FORMAT_VALUE` without a spec: `space.format(w_value,
/// space.newtext(''))`, the [`format_w`] of an empty spec.
///
/// FORMAT_SIMPLE lowers to an `inline_call` of this body, so the JIT traces
/// the empty-spec arms for an exact `str` / `int` and keeps every other shape
/// behind the `dont_look_inside` [`format_simple_slow`].
pub fn format_simple_w(val: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_exact_type(val, &pyre_object::STR_TYPE) } {
        return Ok(val);
    }
    if unsafe {
        pyre_object::py_type_check(val, &pyre_object::INT_TYPE)
            && pyre_object::is_exact_builtin_instance(val)
    } {
        return Ok(unsafe { pyre_object::descr_str(val) });
    }
    format_simple_slow(val)
}

/// [`format_simple_w`] for every shape its `str` / `int` arms do not answer.
#[majit_macros::dont_look_inside]
fn format_simple_slow(val: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    format_w_slow(val, pyre_object::PY_NULL)
}

/// [`format_w`] for every shape its empty-spec `str` / `int` arms do not
/// answer: a long, a `__format__` dispatch, and the spec parser.
fn format_w_slow(val: PyObjectRef, w_spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let spec_is_str = !w_spec.is_null() && unsafe { is_str(w_spec) };
    let spec_is_empty = !spec_is_str || unsafe { pyre_object::w_str_get_wtf8(w_spec) }.is_empty();

    if spec_is_empty {
        if unsafe {
            pyre_object::py_type_check(val, &pyre_object::LONG_TYPE)
                && pyre_object::is_exact_builtin_instance(val)
        } {
            return Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
                crate::py_str_wtf8(val)?
            }));
        }
    }

    if let Some(meth) = unsafe { crate::baseobjspace::lookup(val, "__format__") }
        && (unsafe { is_instance(val) } || !unsafe { is_shared_builtin_format(val, meth) })
    {
        let spec_obj = if spec_is_str {
            w_spec
        } else {
            pyre_object::w_str_new("")
        };
        return call_format_dispatch_w(val, meth, spec_obj);
    }

    if spec_is_empty {
        return Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
            crate::py_str_wtf8(val)?
        }));
    }
    // The shared spec parser wants the bytes.  Copied out rather than
    // borrowed: `format_with_spec_public` can run Python, and a borrow into
    // `w_spec`'s storage would not survive the heap moving under it.
    let spec = unsafe { pyre_object::w_str_get_wtf8(w_spec) }.to_wtf8_buf();
    Ok(pyre_object::w_str_from_wtf8_managed(
        format_with_spec_public(val, &spec)?,
    ))
}

/// The type name of `obj` for a TypeError message — the `w_class` name
/// for instances, else the storage type name.
/// `_PyArg_BadArgument`'s rendering of a rejected argument: `None` names
/// itself where every other value names its type.
pub fn clinic_arg_type_name(obj: PyObjectRef) -> String {
    if unsafe { pyre_object::is_none(obj) } {
        return "None".to_string();
    }
    arg_type_name(obj)
}

pub fn arg_type_name(obj: PyObjectRef) -> String {
    if obj.is_null() {
        return "object".to_string();
    }
    unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => w_type_get_name(tp.as_ptr()).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    }
}

/// Read a format spec's stored string value. The spec must be a `str`
/// (or subclass); its `__str__` is not consulted, so a raising override
/// does not leak out of formatting.  `arg_desc` names the argument in the
/// `TypeError` raised for a non-`str` spec (`format()` reports `format()
/// argument 2`, a type's `__format__` reports `__format__() argument`).
pub(crate) fn read_format_spec(
    spec_obj: PyObjectRef,
    arg_desc: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let spec_obj = check_format_spec(spec_obj, arg_desc)?;
    // Read through the raw buffer: the fill character may be any code
    // point, so `format(x, '\ud800<6')` pads with the lone surrogate it
    // was given rather than demanding a `&str` view of the buffer.
    Ok(unsafe { pyre_object::w_str_get_wtf8(spec_obj) }.to_wtf8_buf())
}

/// [`read_format_spec`]'s check alone, returning the spec object itself for
/// the callers that pass it onward rather than reading its bytes.
pub(crate) fn check_format_spec(
    spec_obj: PyObjectRef,
    arg_desc: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if !spec_obj.is_null() && unsafe { is_str(spec_obj) } {
        return Ok(spec_obj);
    }
    Err(crate::PyError::type_error(format!(
        "{arg_desc} must be str, not {}",
        clinic_arg_type_name(spec_obj)
    )))
}

/// `int/float/str/bool.__format__(self, format_spec)` — formats `self`
/// through the shared spec parser without re-dispatching to an instance
/// `__format__` (which `format_value_dispatch` would do for subclasses,
/// risking recursion).  An empty spec collapses to `str(self)`.
pub fn builtin_value_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let spec = if args.len() > 1 {
        read_format_spec(args[1], "__format__() argument")?
    } else {
        Wtf8Buf::new()
    };
    if spec.is_empty() {
        // `format_string` (newformat.py) wraps the receiver's own utf8
        // storage when the spec parses to the defaults, so an exact `str`
        // comes back identical (see [`format_w`]).
        if unsafe { pyre_object::is_exact_type(args[0], &pyre_object::STR_TYPE) } {
            return Ok(args[0]);
        }
        // `str(self)` — passes through as WTF-8 whatever the receiver is.
        // `py_str` is `py_str_wtf8` plus a lossy UTF-8 encode, so taking it
        // for a non-`str` receiver only folded that receiver's `__str__`
        // result to U+FFFD on the way back into a `str`.  Dynamic result:
        // collectable.
        return Ok(pyre_object::w_str_from_wtf8_managed(unsafe {
            crate::display::py_str_wtf8(args[0])?
        }));
    }
    Ok(pyre_object::w_str_from_wtf8_managed(
        format_with_spec_public(args[0], &spec)?,
    ))
}

fn format_with_spec(mut val: PyObjectRef, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    use rustpython_common::format::FormatSpec;
    unsafe {
        // `int` / `bool` / `long` share the integer formatter.  The `c`
        // character type is formatted here instead — the shared engine pads
        // by byte length rather than code point — and the float presentation
        // codes (`e`/`f`/`g`/`%`) format the `f64` conversion of the value.
        if pyre_object::is_int(val) || pyre_object::is_bool(val) || pyre_object::is_long(val) {
            let type_name = arg_type_name(val);
            let owned;
            let big = if pyre_object::is_bool(val) {
                owned = BigInt::from(pyre_object::w_bool_get_value(val) as i64);
                &owned
            } else if pyre_object::is_int(val) {
                owned = BigInt::from(pyre_object::w_int_get_value(val));
                &owned
            } else {
                // newformat.py's `w_num.asbigint()` borrows `W_LongObject.num`.
                pyre_object::w_long_get_value(val)
            };
            let p = parse_spec(spec)?;
            validate_python314_spec_shape(&p, spec, &type_name, 'd')?;
            let parsed = FormatSpec::parse(&p.engine_spec);
            if p.no_neg_zero && matches!(p.ty, '\0' | 'b' | 'c' | 'd' | 'o' | 'x' | 'X' | 'n') {
                // pypy/objspace/std/newformat.py format_int_or_long:
                // `z` is meaningful only after conversion to a floating
                // presentation type.  The sentence names the specifier it was
                // read out of, the way the string arm below names its own.
                return Err(crate::PyError::value_error(
                    "Negative zero coercion (z) not allowed in integer format specifier",
                ));
            }
            if spec.ends_with("c") && parsed.is_ok() {
                return format_char(big, spec);
            }
            // A float presentation code formats the `f64` conversion (`n` and
            // the integer bases keep full integer precision instead).  That
            // conversion is `PyNumber_Float`, so a subclass overriding
            // `__float__` decides the digits; reading the payload is the answer
            // only when the receiver's Python class is the builtin.
            if matches!(p.ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%') {
                if !pyre_object::is_exact_builtin_instance(val) {
                    let w_float = crate::builtins::builtin_float(&[val])?;
                    let f = pyre_object::floatobject::w_float_get_value(w_float);
                    if f.is_nan() || f.is_infinite() {
                        return format_nonfinite(f, spec);
                    }
                    return format_finite_float(f, spec);
                }
                let f = pyre_object::jit_bigint_to_f64_or_inf(big);
                if f.is_infinite() {
                    return Err(crate::PyError::overflow_error(
                        "int too large to convert to float",
                    ));
                }
                return format_finite_float(f, spec);
            }
            parsed.map_err(|e| format_spec_err(e, spec, &type_name, true))?;
            return format_rbigint(big, spec, &type_name);
        }
        if pyre_object::is_float(val) {
            let v = pyre_object::floatobject::w_float_get_value(val);
            // `inf` / `nan` go through the local formatter — the shared engine
            // inserts the digit-grouping separator into their padding.
            if v.is_nan() || v.is_infinite() {
                return format_nonfinite(v, spec);
            }
            return format_finite_float(v, spec);
        }
        if let Some((re, im)) =
            pyre_object::with_roots!(val => crate::objspace::descroperation::complex_val(val))
        {
            let p = parse_spec(spec)?;
            validate_python314_spec_shape(&p, spec, "complex", '\0')?;
            FormatSpec::parse(&p.engine_spec)
                .map_err(|e| format_spec_err(e, spec, "complex", false))?;
            if p.fill == '0' {
                return Err(crate::PyError::value_error(
                    "Zero padding is not allowed in complex format specifier",
                ));
            }
            if p.align == Some('=') {
                return Err(crate::PyError::value_error(
                    "'=' alignment flag is not allowed in complex format specifier",
                ));
            }
            // Only the float presentation types e/E/f/F/g/G/n are valid for
            // complex; reject any other code with a `complex`-typed message
            // rather than letting the per-part float formatter report `float`.
            if p.ty != '\0' && !matches!(p.ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'n') {
                return Err(crate::PyError::value_error(format!(
                    "Unknown format code '{}' for object of type 'complex'",
                    unknown_code_display(p.ty)
                )));
            }
            let align = p.align.unwrap_or('>');
            // With no component-affecting flags, the default presentation is
            // exactly str(self), padded as a single complex value.
            if p.ty == '\0'
                && p.precision.is_none()
                && p.fractional_grouping.is_none()
                && p.grouping.is_none()
                && !p.alt_form
                && !matches!(p.sign, Some('+') | Some(' '))
            {
                let body = unsafe { crate::display::py_str_wtf8(val)? };
                return pad_wtf8(&body, p.fill, align, p.width);
            }
            // CPython _PyComplex_FormatAdvancedWriter: component flags
            // (sign / alternate / grouping / precision / type) are applied
            // to each float, while alignment and width pad the joined value.
            let default_type = p.ty == '\0';
            // `format_complex_internal` keeps `'r'` only while no precision was
            // given; a precision demotes the lanes to plain `'g'`.  The parens
            // and the omitted real part are decided by the type alone.
            let repr_type = default_type && p.precision.is_none();
            let real_sign = p.sign.unwrap_or('-');
            let real_spec = complex_component_spec(&p, real_sign);
            let imag_sign = if default_type && re == 0.0 && re.is_sign_positive() {
                p.sign.unwrap_or('-')
            } else {
                '+'
            };
            let imag_spec = complex_component_spec(&p, imag_sign);
            let real_text =
                format_complex_component(re, real_spec.as_str().into(), repr_type, p.alt_form)?;
            let imag_text =
                format_complex_component(im, imag_spec.as_str().into(), repr_type, p.alt_form)?;
            let text = if default_type && re == 0.0 && re.is_sign_positive() {
                format!("{imag_text}j")
            } else if default_type {
                format!("({real_text}{imag_text}j)")
            } else {
                format!("{real_text}{imag_text}j")
            };
            let body = Wtf8Buf::from_string(text);
            return pad_wtf8(&body, p.fill, align, p.width);
        }
        if pyre_object::is_str(val) {
            let full = pyre_object::w_str_get_wtf8(val);
            let p = parse_spec(spec)?;
            validate_python314_spec_shape(&p, spec, "str", 's')?;
            // `format_string_internal` refuses these after the spec has been
            // parsed and validated as a whole.  The shared string formatter
            // rejects grouping and numeric types but not a sign or `=`
            // alignment, which are also disallowed for strings.
            reject_string_sign_align(spec)?;
            if p.no_neg_zero {
                return Err(crate::PyError::value_error(
                    "Negative zero coercion (z) not allowed in string format specifier",
                ));
            }
            // A `0` fill flag at the start of the width pads text with the
            // default-left alignment; the shared formatter treats it as a
            // numeric alignment, so handle it here.
            let sc = spec_chars(spec);
            let zero_fill = if sc.len() >= 2 && matches!(sc[1], '<' | '>' | '=' | '^') {
                sc.get(2) == Some(&'0')
            } else if sc
                .first()
                .is_some_and(|c| matches!(c, '<' | '>' | '=' | '^'))
            {
                sc.get(1) == Some(&'0')
            } else {
                sc.first() == Some(&'0')
            };
            if zero_fill {
                return format_surrogate_str(full, spec);
            }
            // A valid-UTF-8 body goes through the shared string formatter,
            // which pads by code point.  A spec carrying a lone surrogate —
            // its fill — has to be padded here instead, since the shared
            // formatter yields a `String`.
            if let (Ok(valid), Ok(_)) = (full.as_str(), spec.as_str()) {
                let parsed =
                    FormatSpec::parse(spec).map_err(|e| format_spec_err(e, spec, "str", false))?;
                let s = parsed
                    .format_string(&CharLenStr(valid, valid.chars().count()))
                    .map_err(|e| format_spec_err(e, spec, "str", false))?;
                return Ok(Wtf8Buf::from_string(s));
            }
            // A body carrying a lone surrogate cannot be handed to the
            // `&str`-typed formatter, so pad it by code point here.
            return format_surrogate_str(full, spec);
        }
        // Reached only for the rare builtin type whose `__format__` is the
        // inherited default yet still routes here with a non-empty spec;
        // format its `str()` through the shared string formatter.
        let full = pyre_object::with_roots!(val => unsafe { crate::display::py_str_wtf8(val) })?;
        if let Ok(s) = full.as_str() {
            let parsed = FormatSpec::parse(spec)
                .map_err(|e| format_spec_err(e, spec, &arg_type_name(val), false))?;
            let out = parsed
                .format_string(&CharLenStr(s, s.chars().count()))
                .map_err(|e| format_spec_err(e, spec, &arg_type_name(val), false))?;
            return Ok(Wtf8Buf::from_string(out));
        }
        // A body carrying a lone surrogate has no `&str` view for the shared
        // formatter, so pad it by code point.
        format_surrogate_str(&full, spec)
    }
}

/// Reject a sign flag, `=` alignment, or `#` alternate form in a string
/// format spec — all disallowed for `str` values but accepted by the shared
/// formatter.  Only the *explicit* alignment is inspected: a leading `0` flag
/// (which the shared parser normalises to `=`) is a zero fill and stays legal
/// for text, and a `#` used as a fill character precedes the alignment.
fn reject_string_sign_align(spec: &Wtf8) -> Result<(), crate::PyError> {
    let chars = spec_chars(spec);
    let n = chars.len();
    let mut i = 0;
    let mut align: Option<char> = None;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        align = Some(chars[1]);
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = Some(chars[0]);
        i = 1;
    }
    if align == Some('=') {
        return Err(crate::PyError::value_error(
            "'=' alignment not allowed in string format specifier",
        ));
    }
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        return Err(crate::PyError::value_error(if chars[i] == ' ' {
            "Space not allowed in string format specifier"
        } else {
            "Sign not allowed in string format specifier"
        }));
    }
    if i < n && chars[i] == '#' {
        return Err(crate::PyError::value_error(
            "Alternate form (#) not allowed in string format specifier",
        ));
    }
    Ok(())
}

/// Format a `str` value whose WTF-8 body carries a lone surrogate, which
/// the shared `&str`-typed formatter cannot accept.  Only the geometry
/// codes that make sense for text — `[fill][align][width][.precision]`
/// and an optional `s` type — are honoured; a numeric presentation type
/// raises the same "unknown format code" error the valid-UTF-8 path does.
/// A sign / `=` alignment is rejected by the caller.
fn format_surrogate_str(body: &Wtf8, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    let cps: Vec<CodePoint> = spec.code_points().collect();
    let chars: Vec<char> = cps.iter().map(|cp| cp.to_char_lossy()).collect();
    let n = chars.len();
    let mut i = 0;
    let mut fill = CodePoint::from_char(' ');
    let mut align = '<';
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = cps[0];
        align = chars[1];
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = chars[0];
        i = 1;
    }
    if i < n && chars[i] == '0' {
        fill = CodePoint::from_char('0');
        i += 1;
    }
    let width = parse_spec_int(&chars, &mut i)?;
    let mut precision: Option<usize> = None;
    if i < n && chars[i] == '.' {
        i += 1;
        let p = parse_spec_int(&chars, &mut i)?;
        precision = Some(p);
    }
    if i < n {
        if chars[i] != 's' {
            return Err(crate::PyError::value_error(format!(
                "Unknown format code '{}' for object of type 'str'",
                unknown_code_point_display(cps[i])
            )));
        }
        i += 1;
    }
    if i != n {
        return Err(crate::PyError::value_error(format!(
            "Invalid format specifier '{spec}' for object of type 'str'"
        )));
    }
    let body = if let Some(prec) = precision {
        let mut t = Wtf8Buf::new();
        for (k, cp) in body.code_points().enumerate() {
            if k >= prec {
                break;
            }
            t.push(cp);
        }
        t
    } else {
        body.to_wtf8_buf()
    };
    pad_wtf8(&body, fill, align, width)
}

/// Format an integer through the `c` presentation type: reject the flags
/// that make no sense for a single character, map the value to its code
/// point, then pad by code point.  The caller has already confirmed the
/// spec parses and ends in `c`.
fn format_char(num: &BigInt, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec)?;
    // Rejection order matches the reference: grouping, then precision, then
    // sign, then alternate form — each beats the value range check.
    if let Some(sep) = p.grouping {
        return Err(crate::PyError::value_error(format!(
            "Cannot specify '{sep}' with 'c'."
        )));
    }
    if p.precision.is_some() {
        return Err(crate::PyError::value_error(
            "Precision not allowed in integer format specifier",
        ));
    }
    if p.sign.is_some() {
        return Err(crate::PyError::value_error(
            "Sign not allowed with integer format specifier 'c'",
        ));
    }
    if p.alt_form {
        return Err(crate::PyError::value_error(
            "Alternate form (#) not allowed with integer format specifier 'c'",
        ));
    }
    // The code is read through `PyLong_AsLong`, so it is a C `long` -- 32 bits
    // on Windows -- that bounds it, and overflowing that is reported as the
    // conversion failing rather than as a code point out of range.
    let code = (pyre_object::jit_bigint_to_i64_fits(num) != 0)
        .then(|| pyre_object::jit_bigint_to_i64_value(num))
        .filter(|code| core::ffi::c_long::try_from(*code).is_ok());
    let Some(code) = code else {
        return Err(crate::PyError::overflow_error(
            "Python int too large to convert to C long",
        ));
    };
    let cp = match u32::try_from(code).ok().and_then(CodePoint::from_u32) {
        Some(cp) => cp,
        None => {
            return Err(crate::PyError::overflow_error(
                "%c arg not in range(0x110000)",
            ));
        }
    };
    let mut body = Wtf8Buf::new();
    body.push(cp);
    // `c` keeps the integer default alignment (right).
    pad_wtf8(&body, p.fill, p.align.unwrap_or('>'), p.width)
}

/// Format `inf` / `nan`: validate the presentation type, then emit the
/// signed word (`inf` / `nan`, upper-cased for `E`/`F`/`G`, `%`-suffixed
/// for the percentage type) padded to width.  Grouping only validates
/// here — a non-finite value has no digits to separate.
fn format_nonfinite(v: f64, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec)?;
    validate_float_spec(spec, &p)?;
    let upper = matches!(p.ty, 'E' | 'F' | 'G');
    let word = match (v.is_nan(), upper) {
        (true, false) => "nan",
        (true, true) => "NAN",
        (false, false) => "inf",
        (false, true) => "INF",
    };
    let mut magnitude = String::from(word);
    if p.ty == '%' {
        magnitude.push('%');
    }
    // A NaN carries no meaningful sign, so it takes only an explicit sign
    // flag; an infinity takes its own sign.
    let sign = if v.is_sign_negative() && !v.is_nan() {
        "-"
    } else {
        match p.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    let body = format!("{sign}{magnitude}");
    pad_to_width(body, p.fill, p.align.unwrap_or('>'), p.width)
}

/// Validate a float presentation spec against the reference rules,
/// independent of the value.  Structural errors (bad flag order, a
/// precision beyond `i32`, trailing garbage) come from the shared parser
/// for their exact messages; the type and grouping-with-`n` checks are
/// applied on top.
fn validate_float_spec(spec: &Wtf8, p: &ParsedSpec) -> Result<(), crate::PyError> {
    validate_python314_spec_shape(p, spec, "float", '\0')?;
    rustpython_common::format::FormatSpec::parse(&p.engine_spec)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    if !matches!(p.ty, '\0' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'n' | '%') {
        return Err(crate::PyError::value_error(format!(
            "Unknown format code '{}' for object of type 'float'",
            unknown_code_display(p.ty)
        )));
    }
    Ok(())
}

/// Probe whether `v` renders to a zero magnitude under `p` — every digit of
/// the unpadded rendering is `0` — so PEP 682's `z` option coerces its sign.
fn float_rounds_to_zero(v: f64, p: &ParsedSpec) -> bool {
    let probe_spec = float_engine_spec_with_width(p, Some(0));
    let Some(rendered) = rustpython_common::format::FormatSpec::parse(&probe_spec)
        .ok()
        .and_then(|f| f.format_float(v).ok())
    else {
        return false;
    };
    let mut saw_digit = false;
    for b in rendered.bytes() {
        if b.is_ascii_digit() {
            saw_digit = true;
            if b != b'0' {
                return false;
            }
        }
    }
    saw_digit
}

/// Format a finite `f64` through `spec`.  Every presentation type
/// (`\0`/`e`/`E`/`f`/`F`/`g`/`G`/`n`/`%`) pads, groups, and rounds through the
/// shared engine.  `validate_float_spec` still supplies the type and
/// grouping-with-`n` messages before delegating.
fn format_finite_float(v: f64, spec: &Wtf8) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec)?;
    validate_float_spec(spec, &p)?;
    // PEP 682 `z`: a value that rounds to a zero magnitude renders its sign
    // away. Format the positive zero in its place so the padded result the
    // engine produces below already carries the coerced sign.
    let v = if p.no_neg_zero && v.is_sign_negative() && float_rounds_to_zero(v, &p) {
        0.0
    } else {
        v
    };
    if p.ty == 'n' {
        // `_format_float:1004-1044`: the locale is read while the presentation
        // code is still `'n'` and the value only then renders as `'g'`.  The
        // integer digit run is separated on its own; the fraction and any
        // exponent are the remainder `_parse_number` splits off, and neither is
        // ever separated.
        let (loc_dec, loc_thousands, loc_grouping) = get_locale('n', None);
        let unpadded_spec = float_engine_spec_with_width(&p, None);
        let rendered = rustpython_common::format::FormatSpec::parse(&unpadded_spec)
            .map_err(|e| format_spec_err(e, spec, "float", false))?
            .format_float(v)
            .map_err(|e| format_spec_err(e, spec, "float", false))?;
        let bytes = rendered.as_bytes();
        let to_number = usize::from(matches!(bytes.first(), Some(b'-' | b'+' | b' ')));
        let mut to_dec = to_number;
        while bytes.get(to_dec).is_some_and(u8::is_ascii_digit) {
            to_dec += 1;
        }
        // `_parse_number:984-991`: the remainder begins past the decimal point
        // when there is one, leaving the point itself to be re-emitted as the
        // locale's.
        let has_dec = bytes.get(to_dec) == Some(&b'.');
        let to_remainder = to_dec + usize::from(has_dec);
        let sign = &rendered[..to_number];
        let int_digits = &rendered[to_number..to_dec];
        let remainder = &rendered[to_remainder..];
        let grouped = if !loc_thousands.is_empty() && !int_digits.is_empty() {
            // `_calc_num_width:696-704`: everything that is not padding or an
            // integer digit is `extra_length`, and a `0=` spec owes the rest of
            // the width in separated zeros.
            let extra_length = to_number + usize::from(has_dec) + remainder.chars().count();
            let n_min_width = if p.fill == '0' && p.align == Some('=') {
                p.width as i32 - extra_length as i32
            } else {
                0
            };
            group_digits(
                int_digits,
                n_min_width,
                int_digits.len() as i32,
                &loc_grouping,
                &loc_thousands,
            )
        } else {
            int_digits.to_string()
        };
        let mut body = String::with_capacity(rendered.len() + grouped.len() + loc_dec.len());
        body.push_str(sign);
        body.push_str(&grouped);
        if has_dec {
            body.push_str(&loc_dec);
        }
        body.push_str(remainder);
        return pad_to_width(body, p.fill, p.align.unwrap_or('>'), p.width);
    }
    if p.fill.to_char().is_none() {
        // A fill with no `char` cannot come back out of the `String`-typed
        // engine, so render the value unpadded and pad it by code point.
        let unpadded_spec = float_engine_spec_with_width(&p, None);
        let body = rustpython_common::format::FormatSpec::parse(&unpadded_spec)
            .map_err(|e| format_spec_err(e, spec, "float", false))?
            .format_float(v)
            .map_err(|e| format_spec_err(e, spec, "float", false))?;
        let body = match p.fractional_grouping {
            Some(separator) => {
                let digits = fractional_digit_count(&body);
                group_fractional_digits(body, separator, digits)
            }
            None => body,
        };
        return pad_to_width(body, p.fill, p.align.unwrap_or('>'), p.width);
    }
    if let Some(separator) = p.fractional_grouping {
        let unpadded_spec = float_engine_spec_with_width(&p, None);
        let unpadded = rustpython_common::format::FormatSpec::parse(&unpadded_spec)
            .map_err(|e| format_spec_err(e, spec, "float", false))?
            .format_float(v)
            .map_err(|e| format_spec_err(e, spec, "float", false))?;
        let digits = fractional_digit_count(&unpadded);
        let separators = digits.saturating_sub(1) / 3;
        let adjusted_width = p.width.saturating_sub(separators);
        let adjusted_spec = float_engine_spec_with_width(&p, Some(adjusted_width));
        let rendered = rustpython_common::format::FormatSpec::parse(&adjusted_spec)
            .map_err(|e| format_spec_err(e, spec, "float", false))?
            .format_float(v)
            .map_err(|e| format_spec_err(e, spec, "float", false))?;
        return Ok(Wtf8Buf::from_string(group_fractional_digits(
            rendered, separator, digits,
        )));
    }
    let parsed = rustpython_common::format::FormatSpec::parse(&p.engine_spec)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    let s = parsed
        .format_float(v)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    Ok(Wtf8Buf::from_string(s))
}

/// Pad a WTF-8 string body to `width` code points with `fill`,
/// honouring `<` / `^` / `>` alignment.  String bodies never use the
/// numeric `=` alignment, so any non-`<`/`^` alignment right-aligns.
fn pad_wtf8(
    body: &Wtf8,
    fill_cp: CodePoint,
    align: char,
    width: usize,
) -> Result<Wtf8Buf, crate::PyError> {
    let body_len = body.code_points().count();
    if body_len >= width {
        return Ok(body.to_wtf8_buf());
    }
    let need = width - body_len;
    let mut out =
        crate::builtins::try_wtf8_with_capacity(padded_byte_len(body.len(), need, fill_cp))?;
    match align {
        '<' => {
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, need)?;
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            push_cp_repeated(&mut out, fill_cp, left)?;
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, right)?;
        }
        _ => {
            push_cp_repeated(&mut out, fill_cp, need)?;
            out.push_wtf8(body);
        }
    }
    Ok(out)
}

/// runicode.py unicode_encode_utf_8 + interp_codecs.py
/// surrogatepass / surrogateescape encode branches.  The conversion loop
/// lives in `rustpython_common::encodings::utf8`; this only threads the
/// str being encoded so a strict failure can build a structured
/// UnicodeEncodeError carrying it.
pub(crate) fn encode_utf8_with_errors(
    s: &Wtf8,
    w_object: PyObjectRef,
    err_mode: &str,
) -> Result<Vec<u8>, crate::PyError> {
    crate::codec_engine::encode_utf8(s, w_object, err_mode)
}

/// PyPy: unicodeobject.py descr_encode → encode_object.
/// For the common 'utf-8' / 'ascii' fast paths, returns the UTF-8 bytes
/// of the string. Other codecs fall through to a best-effort UTF-8 encoding.
pub fn str_method_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "encode")?;
    // `encoding` and `errors` arrive positionally or by keyword; builtin
    // kwargs are packed in a trailing `__pyre_kw__` dict.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // Clinic's `encode` converter is `_PyArg_BadArgument`: a rejected
    // value is `None` or its type name.  PyPy's `get_encoding_and_errors`
    // goes through `space.text_w` / `_typed_unwrap_error` and says
    // "expected str, got %T object"; the clinic wording governs.
    let str_arg =
        |name: &str, obj: Option<PyObjectRef>, default: &str| -> Result<String, crate::PyError> {
            match obj {
                None => Ok(default.to_string()),
                Some(o) if o.is_null() => Ok(default.to_string()),
                Some(o) if unsafe { pyre_object::is_str(o) } => {
                    Ok(crate::baseobjspace::str_utf8_w(o)?.to_string())
                }
                Some(o) => Err(crate::PyError::type_error(format!(
                    "encode() argument '{name}' must be str, not {}",
                    clinic_arg_type_name(o)
                ))),
            }
        };
    // `encode(encoding=None, errors=None)` — both positional-or-keyword;
    // the gateway rejects unknown keywords and a value given both ways.
    crate::builtins::clinic_arity(
        "encode",
        pos.len() - 1,
        crate::builtins::real_kwarg_count(kwargs),
        0,
        2,
        0,
    )?;
    crate::builtins::kwarg_reject_unknown(kwargs, &["encoding", "errors"], "encode")?;
    // Only the counts are read off the native slices once they are pinned.
    let nargs = args.len();
    let npos = pos.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let arg_base = pyre_object::gc_roots::pin_roots(args);
    let reload = |i: usize| pyre_object::gc_roots::shadow_stack_get(arg_base + i);
    let kwargs = kwargs.map(|_| reload(nargs - 1));
    let dual =
        |name: &str, p: Option<PyObjectRef>| -> Result<Option<PyObjectRef>, crate::PyError> {
            let kw = crate::builtins::kwarg_get(kwargs, name);
            if p.is_some() && kw.is_some() {
                return Err(crate::PyError::type_error(format!(
                    "got multiple values for argument '{name}'"
                )));
            }
            Ok(p.or(kw))
        };
    let w_encoding = if npos > 1 { Some(reload(1)) } else { None };
    let w_errors = if npos > 2 { Some(reload(2)) } else { None };
    let has_w_errors = w_errors.is_some();
    let w_errors_slot = _roots.pin_roots(&[w_errors.unwrap_or(pyre_object::PY_NULL)]);
    let encoding_arg = dual("encoding", w_encoding)?;
    let encoding = str_arg("encoding", encoding_arg, "utf-8")?;
    let w_errors = has_w_errors.then(|| _roots.get(w_errors_slot));
    let errors = str_arg("errors", dual("errors", w_errors)?, "strict")?;
    Ok(pyre_object::w_bytes_from_bytes(&encode_object(
        reload(0),
        &encoding,
        &errors,
    )?))
}

/// `unicodeobject.py W_UnicodeObject.descr_encode` → `encode_object`.
/// Encodes a str (`w_object`) to bytes with the named codec and error
/// handler.  Shared by `str.encode`, `bytes(str, …)` and
/// `bytearray(str, …)` so all three honour the same codec set and error
/// handlers.  The whole path reads the surrogate-aware WTF-8 view, so a
/// lone surrogate is routed to the error handler rather than crashing.
pub fn encode_object(
    mut w_object: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    // The name is matched with case folded and `_` read as `-`.  Every caller
    // inside the runtime, and `str.encode`'s own default, already spells it
    // that way, so the rewrite is the exception and only it pays for a buffer
    // -- `decode_bytes_to_wtf8` reads its own name the same way.
    let enc_owned;
    let enc_lower: &str = if encoding
        .bytes()
        .any(|b| b.is_ascii_uppercase() || b == b'_')
    {
        enc_owned = encoding.to_ascii_lowercase().replace('_', "-");
        &enc_owned
    } else {
        encoding
    };
    if crate::importing::dev_mode_flag()
        && matches!(
            enc_lower,
            "utf-8"
                | "utf8"
                | "u8"
                | "ascii"
                | "us-ascii"
                | "646"
                | "latin-1"
                | "latin1"
                | "iso-8859-1"
                | "8859"
                | "raw-unicode-escape"
                | "utf-16"
                | "utf-16-le"
                | "utf-16-be"
                | "utf-32"
                | "utf-32-le"
                | "utf-32-be"
        )
    {
        pyre_object::with_roots!(w_object =>
            crate::module::_codecs::validate_error_handler(errors)
        )?;
    }
    // Name the encoder before reading the string.  `encode_text` hands
    // `w_object` to the registry untouched, so a name the built-ins do not
    // own must not pay for a copy of the whole input first.
    let Some(builtin) = builtin_encoder(enc_lower) else {
        let encoded = crate::module::_codecs::encode_text_codec(w_object, encoding, errors)?;
        return Ok(unsafe { pyre_object::bytesobject::bytes_like_data(encoded) }.to_vec());
    };
    let s = unsafe { w_str_get_wtf8(w_object) }.to_wtf8_buf();
    match builtin {
        BuiltinEncoder::Utf8 => encode_utf8_with_errors(&s, w_object, errors),
        BuiltinEncoder::Ascii => crate::codec_engine::encode_ascii(&s, w_object, errors),
        BuiltinEncoder::Latin1 => crate::codec_engine::encode_latin1(&s, w_object, errors),
        BuiltinEncoder::RawUnicodeEscape => Ok(encode_raw_unicode_escape(&s)),
        BuiltinEncoder::Utf16Or32(form) => encode_utf16_32(&s, form, w_object, errors),
    }
}

/// The built-in encoder a `lower`-normalized codec name selects, or `None`
/// for a name only the codec registry can answer.
enum BuiltinEncoder {
    Utf8,
    Ascii,
    Latin1,
    RawUnicodeEscape,
    Utf16Or32(Utf16Or32Form),
}

fn builtin_encoder(lower: &str) -> Option<BuiltinEncoder> {
    Some(match lower {
        "utf-8" | "utf8" | "u8" => BuiltinEncoder::Utf8,
        "ascii" | "us-ascii" | "646" => BuiltinEncoder::Ascii,
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => BuiltinEncoder::Latin1,
        "raw-unicode-escape" => BuiltinEncoder::RawUnicodeEscape,
        _ => BuiltinEncoder::Utf16Or32(utf16_32_form(lower)?),
    })
}

/// `unicodeobject.c:_PyUnicode_EncodeRawUnicodeEscape` — code points
/// below 0x100 map to a single Latin-1 byte; 0x100..0x10000 become the
/// 6-byte `\uXXXX` form; everything larger becomes `\UXXXXXXXX`.  Unlike
/// `unicode-escape`, the backslash and control characters are not
/// escaped.
pub fn encode_raw_unicode_escape(s: &Wtf8) -> Vec<u8> {
    crate::codec_engine::encode_raw_unicode_escape(s)
}

/// `unicodeobject.c:_PyUnicode_DecodeRawUnicodeEscape` — the inverse of
/// [`encode_raw_unicode_escape`].  A backslash starts a `\uXXXX` /
/// `\UXXXXXXXX` escape; any other byte (including a lone backslash or a
/// malformed escape) is taken as a Latin-1 code point.
pub fn decode_raw_unicode_escape(data: &[u8], errors: &str) -> Result<Wtf8Buf, crate::PyError> {
    Ok(decode_raw_unicode_escape_stateful(data, errors, true)?.0)
}

/// `unicodeobject.c:_PyUnicode_DecodeRawUnicodeEscapeStateful` — the form that
/// answers how many input bytes were consumed alongside the text.
///
/// While `final_` is false more input may still follow, so an escape that runs
/// off the end of this chunk is left unconsumed rather than reported: the
/// digits it is missing can arrive in the next call.  Only a sequence this
/// chunk already decides -- `\u12zz`, which has its four bytes and they are
/// not all hex -- reaches the error handler.
pub fn decode_raw_unicode_escape_stateful(
    data: &[u8],
    errors: &str,
    final_: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    crate::codec_engine::decode_raw_unicode_escape(data.to_vec(), errors, final_)
}

/// Which of the six utf-16 / utf-32 spellings a codec name selects.  The
/// bare `utf-16` / `utf-32` forms emit a native-endian BOM; the `-le` /
/// `-be` forms omit it.
pub struct Utf16Or32Form {
    is32: bool,
    order: rustpython_common::encodings::ByteOrder,
    bom: bool,
}

/// The width, byte order and BOM the `lower`-normalized name selects, or
/// `None` if it names neither codec.  Answered without reading the string so
/// a registry name never pays for one.
fn utf16_32_form(lower: &str) -> Option<Utf16Or32Form> {
    use rustpython_common::encodings::ByteOrder;
    // Callers rewrite `_` to `-` before this match. The compact spellings
    // stay so `utf16le` and `utf-16-le` select the same form.
    let (is32, order, bom) = match lower {
        "utf-16" | "utf16" | "u16" => (false, ByteOrder::Native, true),
        "utf-16-le" | "utf-16le" | "utf16-le" | "utf16le" => (false, ByteOrder::Little, false),
        "utf-16-be" | "utf-16be" | "utf16-be" | "utf16be" => (false, ByteOrder::Big, false),
        "utf-32" | "utf32" | "u32" => (true, ByteOrder::Native, true),
        "utf-32-le" | "utf-32le" | "utf32-le" | "utf32le" => (true, ByteOrder::Little, false),
        "utf-32-be" | "utf-32be" | "utf32-be" | "utf32be" => (true, ByteOrder::Big, false),
        _ => return None,
    };
    Some(Utf16Or32Form { is32, order, bom })
}

/// utf-16 / utf-32 encode in the named `form`.  A lone surrogate is routed
/// through `errors` (`surrogatepass` emits its raw code unit; `strict`
/// raises) rather than crashing.
pub fn encode_utf16_32(
    s: &Wtf8,
    form: Utf16Or32Form,
    w_object: PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let Utf16Or32Form { is32, order, bom } = form;
    if is32 {
        crate::codec_engine::encode_utf32(s, w_object, errors, order, bom)
    } else {
        crate::codec_engine::encode_utf16(s, w_object, errors, order, bom)
    }
}

/// utf-16 / utf-32 decode for the `lower`-normalized codec name, or
/// `None` if `lower` names neither.  The bare `utf-16` / `utf-32` forms
/// consume a leading BOM to choose endianness (defaulting to
/// little-endian); the `-le` / `-be` forms are fixed.  A lone surrogate
/// is routed through `err_mode` (`surrogatepass` keeps it as a code
/// point; `strict` raises), so the result is a `Wtf8Buf`.
pub fn decode_utf16_32(
    data: &[u8],
    lower: &str,
    err_mode: &str,
) -> Option<Result<Wtf8Buf, crate::PyError>> {
    // `codec` is the canonical name reported in a UnicodeDecodeError.
    let (is32, fixed_be, codec) = match lower {
        "utf-16" | "utf16" | "u16" => (false, None, "utf-16"),
        "utf-16-le" | "utf-16le" | "utf16-le" | "utf16le" => (false, Some(false), "utf-16-le"),
        "utf-16-be" | "utf-16be" | "utf16-be" | "utf16be" => (false, Some(true), "utf-16-be"),
        "utf-32" | "utf32" | "u32" => (true, None, "utf-32"),
        "utf-32-le" | "utf-32le" | "utf32-le" | "utf32le" => (true, Some(false), "utf-32-le"),
        "utf-32-be" | "utf-32be" | "utf32-be" | "utf32be" => (true, Some(true), "utf-32-be"),
        _ => return None,
    };
    Some(
        decode_utf16_32_helper(data, is32, fixed_be, codec, err_mode, true)
            .map(|(decoded, _, _)| decoded),
    )
}

/// PyPy `unicodehelper.str_decode_utf_{16,32}_helper`: decode one incremental
/// chunk and return `(text, consumed, byteorder)`.  `fixed_be == None` is the
/// native/BOM-detecting mode used by `_codecs.utf_{16,32}_ex_decode`;
/// `Some(false/true)` is forced little/big endian.
pub fn decode_utf16_32_helper(
    data: &[u8],
    is32: bool,
    fixed_be: Option<bool>,
    codec: &str,
    err_mode: &str,
    final_: bool,
) -> Result<(Wtf8Buf, usize, i32), crate::PyError> {
    use rustpython_common::encodings::ByteOrder;
    let order = match fixed_be {
        None => ByteOrder::Native,
        Some(false) => ByteOrder::Little,
        Some(true) => ByteOrder::Big,
    };
    // [3.14-spec] A bare utf-16/utf-32 decode reports the effective
    // `-le` / `-be` name once byte order is settled.
    let error_encoding = if matches!(codec, "utf16" | "utf-16" | "utf32" | "utf-32") {
        let big_endian = match order {
            ByteOrder::Big => true,
            ByteOrder::Little => false,
            ByteOrder::Native => {
                if is32 {
                    data.starts_with(&[0x00, 0x00, 0xFE, 0xFF])
                        || (!data.starts_with(&[0xFF, 0xFE, 0x00, 0x00])
                            && cfg!(target_endian = "big"))
                } else {
                    data.starts_with(&[0xFE, 0xFF])
                        || (!data.starts_with(&[0xFF, 0xFE]) && cfg!(target_endian = "big"))
                }
            }
        };
        if is32 {
            if big_endian { "utf-32-be" } else { "utf-32-le" }
        } else if big_endian {
            "utf-16-be"
        } else {
            "utf-16-le"
        }
    } else {
        codec
    };
    crate::codec_engine::decode_utf16_32(data, is32, order, err_mode, final_, error_encoding)
}

/// Which buffer a decode error handler's returned position is folded against,
/// and with it whether `exc.object` is reread at all.
enum DecodeResume {
    /// `unicodeobject.c` rereads `exc.object` once the handler returns and
    /// goes on decoding *that*, so a handler which replaced it redirects
    /// decoding, the position is bounded by what it left behind, and a
    /// non-bytes value there is an error.
    RereadObject,
    /// `multibytecodec_decerror` folds the position against the buffer
    /// decoding started on and goes on decoding that one.  It never reads
    /// `exc.object` back, so replacing it neither redirects decoding nor
    /// widens the range a position may name, and a non-bytes value is
    /// simply not looked at.
    OriginalInput,
}

/// interp_codecs.py `call_errorhandler` (decode branch): invoke a
/// custom handler registered through `_codecs.register_error` for a decode
/// error position.
/// Returns `(newpos, resume)`.  `newpos` is the byte position to resume
/// decoding at, folded against the length of the buffer decoding will
/// continue in.  `resume` is `Some(new_bytes)` when the handler replaced
/// `exc.object` with different bytes (the caller must rebind its working
/// buffer to them) and `None` when the object was left unchanged (the
/// caller keeps decoding the same slice, no allocation).
pub(crate) fn call_registered_decode_error_handler(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut Wtf8Buf,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    call_decode_error_handler(
        err_mode,
        codec,
        data,
        start,
        end,
        reason,
        out,
        DecodeResume::RereadObject,
    )
}

/// [`call_registered_decode_error_handler`] for a multibyte codec, which
/// resumes in the buffer it started on whatever the handler did to
/// `exc.object`.  Only the position comes back.
pub fn call_registered_multibyte_decode_error_handler(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut Wtf8Buf,
) -> Result<usize, crate::PyError> {
    let (newpos, _) = call_decode_error_handler(
        err_mode,
        codec,
        data,
        start,
        end,
        reason,
        out,
        DecodeResume::OriginalInput,
    )?;
    Ok(newpos)
}

#[allow(clippy::too_many_arguments)]
fn call_decode_error_handler(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut Wtf8Buf,
    resume_in: DecodeResume,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    let w_handler = crate::module::_codecs::lookup_registered_error(err_mode).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )
    })?;
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_handler);

    let w_exc =
        crate::typedef::unicode_decode_error(codec, data, start, end, reason).to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(w_exc);
    let w_res = crate::baseobjspace::call_function(
        pyre_object::gc_roots::shadow_stack_get(sp),
        &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
    );
    if w_res.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::type_error("error handler failed")));
    }
    let _ = pyre_object::gc_roots::pin_root(w_res);

    if !unsafe { pyre_object::is_tuple(pyre_object::gc_roots::shadow_stack_get(sp + 2)) }
        || unsafe { pyre_object::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(sp + 2)) } != 2
    {
        return Err(crate::PyError::type_error(
            "decoding error handler must return (str, int) tuple",
        ));
    }
    let w_replace = unsafe {
        pyre_object::w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(sp + 2), 0).unwrap()
    };
    let _ = pyre_object::gc_roots::pin_root(w_replace);
    let w_newpos = unsafe {
        pyre_object::w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(sp + 2), 1).unwrap()
    };
    let _ = pyre_object::gc_roots::pin_root(w_newpos);
    if !unsafe { pyre_object::is_str(pyre_object::gc_roots::shadow_stack_get(sp + 3)) } {
        return Err(crate::PyError::type_error(
            "decoding error handler must return (str, int) tuple",
        ));
    }

    // interp_codecs.py:94-104 — reread exc.object after the handler
    // returns.  The handler may have replaced it, in which case decoding
    // resumes from the new bytes.  A non-bytes object is rejected (the
    // decode C code checks PyBytes_Check on the reread object).
    let new_bytes = match resume_in {
        DecodeResume::RereadObject => {
            let w_obj = unsafe {
                pyre_object::interp_exceptions::w_exception_get_object(
                    pyre_object::gc_roots::shadow_stack_get(sp + 1),
                )
            };
            if !unsafe { pyre_object::bytesobject::is_bytes(w_obj) } {
                return Err(crate::PyError::type_error(
                    "UnicodeError 'object' attribute must be a bytes",
                ));
            }
            Some(unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) }.to_vec())
        }
        DecodeResume::OriginalInput => None,
    };

    // newpos folds against the length of whichever buffer decoding resumes
    // in: the reread object (unicodeobject.c insize), or the one it started
    // on (`multibytecodec_decerror`'s `size`).
    let length = new_bytes.as_ref().map_or(data.len(), Vec::len) as i64;
    // `multibytecodec_decerror` substitutes -1 for a position outside the
    // machine integer and lets the bounds test below report it.  [3.14-spec]
    // `unicode_decode_call_errorhandler` propagates the `PyLong_AsSsize_t`
    // failure instead: 3.14.2 answers `OverflowError` for ascii, utf-8,
    // latin-1 and charmap, and `IndexError: position -1 ...` for the
    // multibyte engines, while PyPy 7.3.20 folds on both.  The negative fold
    // stays in the conversion's `else` clause either way, so it runs only for
    // a position that converted: folding a failure would land on 0 for a
    // one-unit input and hand the same span back to the handler without end.
    let folds_overflow = !matches!(resume_in, DecodeResume::RereadObject);
    let newpos = match crate::baseobjspace::int_w(pyre_object::gc_roots::shadow_stack_get(sp + 4)) {
        Ok(n) if n < 0 => n + length,
        Ok(n) => n,
        Err(error) if folds_overflow && error.kind == crate::PyErrorKind::OverflowError => -1,
        Err(error) => return Err(error),
    };
    if newpos < 0 || newpos > length {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            format!("position {newpos} from error handler out of bounds"),
        ));
    }

    out.push_wtf8(unsafe {
        pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(sp + 3))
    });
    let resume = match new_bytes {
        Some(bytes) if bytes.as_slice() != data => Some(bytes),
        _ => None,
    };
    Ok((newpos as usize, resume))
}

/// Which upstream owns the encode error-handler call, mirroring
/// [`DecodeResume`] on the decode side.  The two disagree over a handler
/// position that does not fit the machine integer.
pub enum EncodeErrorOwner {
    /// `unicode_encode_call_errorhandler` — every codec in `unicodeobject.c`,
    /// including `charmap_encode` and the win32 code-page encoder.
    UnicodeObject,
    /// `multibytecodec_encerror` — the CJK engines.
    MultibyteCodec,
}

/// Replacement returned by a custom encode error handler: either a str
/// (its code points, re-encoded by the codec) or raw bytes (copied
/// verbatim). interp_codecs.py rettype 'u' vs 'b'.
pub enum EncodeReplacement {
    Str(Vec<u32>),
    Bytes(Vec<u8>),
}

/// interp_codecs.py `call_errorhandler` (encode branch): invoke a
/// custom handler registered through `_codecs.register_error` for an
/// encode error span. `start`/`end`/`char_len` are CHARACTER (code-point)
/// indices into the source; the returned position is a character index to
/// resume at. The caller re-encodes an `EncodeReplacement::Str` through
/// its own codec — the utf-* ones raise the ORIGINAL error when a
/// replacement code point is not encodable — and copies
/// `EncodeReplacement::Bytes` verbatim.
pub fn call_registered_encode_error_handler(
    err_mode: &str,
    codec: &str,
    mut source: PyObjectRef,
    char_len: usize,
    start: usize,
    end: usize,
    reason: &str,
    owner: EncodeErrorOwner,
) -> Result<(EncodeReplacement, usize), crate::PyError> {
    let mut w_handler = pyre_object::with_roots!(source =>
        crate::module::_codecs::lookup_registered_error(err_mode)
    )
    .ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )
    })?;

    let mut w_err = pyre_object::with_roots!(w_handler =>
        crate::typedef::unicode_encode_error(codec, source, start as i64, end as i64, reason)
    );
    let w_exc = pyre_object::with_roots!(w_handler => w_err.to_exc_object());
    let w_res = crate::baseobjspace::call_function(w_handler, &[w_exc]);
    if w_res.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::type_error("error handler failed")));
    }

    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        return Err(crate::PyError::type_error(
            "encoding error handler must return (str/bytes, int) tuple",
        ));
    }
    let w_replace = unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap() };
    let w_newpos = unsafe { pyre_object::w_tuple_getitem(w_res, 1).unwrap() };

    let replacement = if unsafe { pyre_object::is_str(w_replace) } {
        EncodeReplacement::Str(
            unsafe { pyre_object::w_str_get_wtf8(w_replace) }
                .code_points()
                .map(|c| c.to_u32())
                .collect(),
        )
    } else if unsafe { pyre_object::bytesobject::is_bytes(w_replace) } {
        EncodeReplacement::Bytes(
            unsafe { pyre_object::bytesobject::bytes_like_data(w_replace) }.to_vec(),
        )
    } else {
        return Err(crate::PyError::type_error(
            "encoding error handler must return (str/bytes, int) tuple",
        ));
    };

    // newpos folds against the source CHARACTER length and resumes into the
    // original code-point sequence.
    let length = char_len as i64;
    // The same split as the decode branch above: `multibytecodec_encerror`
    // folds an unrepresentable position to -1, `unicode_encode_call_errorhandler`
    // refuses it with its own `OverflowError`.
    let folds_overflow = !matches!(owner, EncodeErrorOwner::UnicodeObject);
    let newpos = match crate::baseobjspace::int_w(w_newpos) {
        Ok(n) if n < 0 => n + length,
        Ok(n) => n,
        Err(error) if folds_overflow && error.kind == crate::PyErrorKind::OverflowError => -1,
        Err(error) => return Err(error),
    };
    if newpos < 0 || newpos > length {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            format!("position {newpos} from error handler out of bounds"),
        ));
    }

    Ok((replacement, newpos as usize))
}

fn wtf8_map_str_runs(s: &Wtf8, f: impl Fn(&str) -> String) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(s.len());
    let mut run = String::new();
    for cp in s.code_points() {
        match cp.to_char() {
            Some(c) => run.push(c),
            None => {
                if !run.is_empty() {
                    out.push_str(&f(&run));
                    run.clear();
                }
                out.push(cp);
            }
        }
    }
    if !run.is_empty() {
        out.push_str(&f(&run));
    }
    out
}

pub fn str_method_isdigit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isdigit")?;
    // A lone surrogate satisfies no character class, so a non-UTF-8
    // backing is never all-digit (and the empty string is false too).
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_digit),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isdecimal(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isdecimal")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_decimal),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isnumeric(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isnumeric")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_numeric),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_istitle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "istitle")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let mut cased = false;
    let mut prev_cased = false;
    for cp in s.code_points() {
        // A lone surrogate is uncased, so it resets `prev_cased` like
        // any other non-cased code point (the `None` arm / `else`).
        match cp.to_char() {
            Some(c) => {
                if c.is_uppercase() || case::is_titlecase(c) {
                    if prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else if c.is_lowercase() {
                    if !prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else {
                    prev_cased = false;
                }
            }
            None => prev_cased = false,
        }
    }
    Ok(w_bool_from(cased))
}

pub fn str_method_isalpha(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isalpha")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_alpha),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isidentifier
pub fn descr_isidentifier(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isidentifier")?;
    // An identifier cannot contain a lone surrogate, so a non-UTF-8
    // backing is never an identifier.
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => is_identifier(v),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Check if a string is a valid Python identifier: an `XID_Start` code point
/// (or underscore) followed by `XID_Continue` code points.
/// PyPy: unicodeobject.py `_isidentifier` via `unicodedb.isxidstart`/`isxidcontinue`.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let valid_start = chars.next().is_some_and(identifier::is_start);
    valid_start && chars.all(identifier::is_continue)
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject.descr_zfill`.
/// Pads with leading zeros up to `width`; when the string starts with
/// a sign character (`+`/`-`), the sign stays at the front and zeros
/// fill between it and the digits (`'-42'.zfill(5) == '-0042'`).
pub fn str_method_zfill(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "zfill", 1)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let width =
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(recv_slot + 1))?
            .max(0) as usize;
    let s = unsafe { w_str_get_wtf8(recv()) }.to_wtf8_buf();
    let len = unsafe { pyre_object::w_str_len(recv()) };
    if len >= width {
        return Ok(str_result_unchanged(recv()));
    }
    let need = width - len;
    let zero = CodePoint::from_char('0');
    let mut cps = s.code_points();
    let mut out = crate::builtins::try_wtf8_with_capacity(padded_byte_len(s.len(), need, zero))?;
    let first = cps.clone().next();
    if first == Some(CodePoint::from_char('+')) || first == Some(CodePoint::from_char('-')) {
        out.push(first.unwrap());
        cps.next();
    }
    push_cp_repeated(&mut out, zero, need)?;
    for cp in cps {
        out.push(cp);
    }
    Ok(w_str_from_wtf8_managed(out))
}

/// Number of non-overlapping occurrences of `needle` in `haystack`,
/// scanning over the WTF-8 bytes. PyPy's `unicodeobject.py:1116` delegates
/// to `_utf8.count`; the RPython translation path for non-host strings is
/// `rstring.py:_search_normal(..., SEARCH_COUNT)`.
///
/// The empty needle is answered here rather than through that path. Any
/// other needle is a whole encoded sequence, so counting matches over bytes
/// and counting them over code points give the same number; the empty
/// needle alone degenerates to `end - start + 1` (`rstring.py`), which
/// is a length, and the bounds `descr_count` hands down are byte offsets
/// (`unicodeobject.py:1322` says so). PyPy therefore reports the byte
/// length plus one for a non-ASCII receiver: `"\xe9\xe8\xe9\xe8\xe9"
/// .count("")` is 11 there against 6 in 3.14. `descr_find` and its
/// siblings escape this because `_unwrap_and_search` maps the byte result
/// back with `_byte_to_index`, a step a count has nowhere to put. 3.14
/// counts code points, so that is what this counts.
fn wtf8_count(haystack: &Wtf8, needle: &Wtf8) -> usize {
    if needle.is_empty() {
        return pyre_object::rutf8::codepoints_in_utf8(haystack, 0, haystack.len()) + 1;
    }
    pyre_object::rstring::search_normal(
        haystack.as_bytes(),
        needle.as_bytes(),
        0,
        haystack.len(),
        pyre_object::rstring::SearchMode::Count,
    ) as usize
}

/// First byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. PyPy `unicodeobject.py:_unwrap_and_search` searches the
/// `_utf8` storage after converting codepoint bounds to byte bounds.
fn wtf8_find_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    let res = pyre_object::rstring::search_normal(
        haystack,
        needle,
        lo,
        hi,
        pyre_object::rstring::SearchMode::Find,
    );
    if res == -1 { None } else { Some(res as usize) }
}

/// Last byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. Mirrors `rstring.py:_search_normal(..., SEARCH_RFIND)`.
fn wtf8_rfind_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    let res = pyre_object::rstring::search_normal(
        haystack,
        needle,
        lo,
        hi,
        pyre_object::rstring::SearchMode::RFind,
    );
    if res == -1 { None } else { Some(res as usize) }
}

/// `rstring.py _search_elidable` is `@jit.elidable`.  The walk is
/// `_search_normal` after the code-point window is already byte offsets.
#[majit_macros::elidable]
unsafe fn search_elidable(
    value: PyObjectRef,
    other: PyObjectRef,
    start: usize,
    end: usize,
    mode: pyre_object::rstring::SearchMode,
) -> isize {
    pyre_object::rstring::search_normal(
        unsafe { w_str_get_wtf8(value) }.as_bytes(),
        unsafe { w_str_get_wtf8(other) }.as_bytes(),
        start,
        end,
        mode,
    )
}

/// `unicodeobject.py _unwrap_and_search` — the shared path for
/// find/rfind/index/rindex. `start`/`end` (args[2]/args[3]) flow through
/// `unwrap_start_stop`, so `None` / omitted arguments default, any
/// `__index__`-bearing object is accepted, and a `TypeError` propagates.
/// The search runs over the WTF-8 bytes inside the codepoint window and
/// the matching byte offset is mapped back to a codepoint index. Returns
/// `Ok(None)` when not found.
fn str_unwrap_and_search(
    args: &[PyObjectRef],
    forward: bool,
) -> Result<Option<i64>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let obj = || pyre_object::gc_roots::shadow_stack_get(base);
    let length = unsafe { pyre_object::w_str_len(obj()) } as i64;

    let w_start = if args.len() >= 3 {
        pyre_object::gc_roots::shadow_stack_get(base + 2)
    } else {
        w_none()
    };
    let w_end = if args.len() >= 4 {
        pyre_object::gc_roots::shadow_stack_get(base + 3)
    } else {
        w_none()
    };
    let (start, end) = crate::sliceobject::unwrap_start_stop(length, w_start, w_end)?;

    // `_search` (`unicodeobject.py:1290`): the two code point bounds become
    // byte offsets through `_index_to_byte`, and the byte offset the search
    // lands on comes back through `_byte_to_index`.
    let obj = obj();
    let start_index = if start == 0 {
        0
    } else if start > length {
        return Ok(None);
    } else {
        unsafe { pyre_object::w_str_index_to_byte(obj, start as usize) }
    };
    let end_index = if end >= length {
        unsafe { pyre_object::w_str_get_wtf8(obj) }.len()
    } else {
        unsafe { pyre_object::w_str_index_to_byte(obj, end as usize) }
    };
    if start_index > end_index {
        return Ok(None);
    }

    let mode = if forward {
        pyre_object::rstring::SearchMode::Find
    } else {
        pyre_object::rstring::SearchMode::RFind
    };
    let res = unsafe {
        search_elidable(
            obj,
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            start_index,
            end_index,
            mode,
        )
    };
    Ok(if res < 0 {
        None
    } else {
        Some(unsafe { pyre_object::w_str_byte_to_index(obj, res as usize) } as i64)
    })
}

/// PyPy: unicodeobject.py descr_count
pub fn str_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "count", 1)?;
    arity_at_most(args, "count", 3)?;
    require_str_sub(args, "count")?;
    // Operands read as WTF-8 so lone surrogates do not panic; the optional
    // start / end arguments bound the count window over the code points.
    // Bound conversion runs `__index__` and can collect, so copy the
    // payloads after the window is known rather than holding interior
    // `&str` views across that allocation.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let mut w_args = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        w_args.push(pyre_object::gc_roots::shadow_stack_get(base + i));
    }
    let Some((byte_start, byte_end)) =
        wtf8_idx_window(pyre_object::gc_roots::shadow_stack_get(base), &w_args)?
    else {
        return Ok(w_int_new(0));
    };
    let s = unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }
        .to_wtf8_buf();
    let sub =
        unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }
            .to_wtf8_buf();
    let window = rustpython_wtf8::Wtf8::from_bytes(&s.as_bytes()[byte_start..byte_end])
        .expect("code-point boundary slice is valid WTF-8");
    Ok(w_int_new(wtf8_count(window, &sub) as i64))
}

/// PyPy: unicodeobject.py descr_index
/// `unicodeobject.py:1006-1010 _descr_index` — missing substring raises
/// "substring not found" (ValueError).
pub fn str_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "index", 1)?;
    arity_at_most(args, "index", 3)?;
    require_str_sub(args, "index")?;
    match str_unwrap_and_search(args, true)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// `unicodeobject.py descr_rindex` — like rfind, but raises ValueError
/// when the substring is absent.
/// unicodeobject.py descr_rindex
pub fn descr_rindex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rindex", 1)?;
    arity_at_most(args, "rindex", 3)?;
    require_str_sub(args, "rindex")?;
    match str_unwrap_and_search(args, false)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// `unicodeobject.py W_UnicodeObject._title_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn title_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(case::title_wtf8(&s))
}

/// PyPy: unicodeobject.py descr_title
pub fn descr_title(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "title")?;
    Ok(unsafe { title_unicode(args[0]) })
}

/// `unicodeobject.py W_UnicodeObject._capitalize_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn capitalize_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(case::capitalize_wtf8(&s))
}

/// PyPy: unicodeobject.py descr_capitalize
pub fn descr_capitalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "capitalize")?;
    Ok(unsafe { capitalize_unicode(args[0]) })
}

/// `unicodeobject.py W_UnicodeObject._swapcase_unicode` is `@jit.elidable`.
#[majit_macros::elidable]
unsafe fn swapcase_unicode(obj: PyObjectRef) -> PyObjectRef {
    let s = unsafe { w_str_get_wtf8(obj) }.to_wtf8_buf();
    w_str_from_wtf8_managed(case::swapcase_wtf8(&s))
}

/// PyPy: unicodeobject.py descr_swapcase
pub fn descr_swapcase(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "swapcase")?;
    Ok(unsafe { swapcase_unicode(args[0]) })
}

/// Resolve the fillchar arg for `center`/`ljust`/`rjust`. Defaults to `' '`
/// for an *omitted* argument (`w_fillchar=WrappedDefault(u' ')`); a supplied
/// `None` is an ordinary value and is refused here like any other non-str.
///
/// The operand is converted by two different helpers, and each refuses in its
/// own words. `descr_center` uses `space.utf8_w` (unicodeobject.py),
/// which takes a str and nothing else. `descr_ljust`/`descr_rjust` use
/// `convert_arg_to_w_unicode` (unicodeobject.py): it declines `bytes`
/// with the message reproduced below, and hands every other non-str to
/// `decode_object`, which turns a buffer operand into a fill char rather than
/// refusing it.
///
/// `decode_object` reports a failed conversion as `"decoding to str: %S"` over
/// the buffer error (unicodeobject.py:1727-1739). apptest_unicode.py  allow-line-citation
/// pins that in either of two wordings; the arm below is the one pypy prints,
/// down to `None` rendering unquoted where a type name is quoted.
///
/// The decode is strict, so an operand that exports bytes which are not valid
/// UTF-8 raises from the decode rather than being refused by type.
///
/// A fill of the wrong length is a separate refusal, and the one upstream words
/// per method; it applies to the decoded operand, so a multi-byte buffer is
/// rejected by its code-point count and not by its byte count.
fn pad_fillchar(args: &[PyObjectRef], method: &str) -> Result<CodePoint, crate::PyError> {
    if args.len() <= 2 {
        return Ok(CodePoint::from_char(' '));
    }
    let mut w_fill = args[2];
    let decoded;
    let raw = if unsafe { pyre_object::is_str(w_fill) } {
        unsafe { w_str_get_wtf8(w_fill) }
    } else {
        let type_name = arg_type_name(w_fill);
        if method == "center" {
            return Err(crate::PyError::type_error(format!(
                "expected str, got {type_name} object"
            )));
        } else if unsafe { pyre_object::is_bytes(w_fill) } {
            return Err(crate::PyError::type_error(format!(
                "Can't convert '{type_name}' object to str implicitly"
            )));
        } else {
            // Decoded and released inside the bracket rooting `w_fill`, which
            // the export's own bracket must not outlive.
            let result = pyre_object::with_roots!(w_fill =>
            crate::baseobjspace::simple_buffer_bytes(w_fill).map(|buffer| {
                buffer.map(|buffer| {
                    let result = crate::typedef::decode_bytes_to_wtf8(
                        buffer.as_bytes(),
                        "utf-8",
                        "strict",
                    );
                    buffer.release();
                    result
                })
            }))?;
            let Some(result) = result else {
                let operand = if unsafe { pyre_object::is_none(w_fill) } {
                    "None".to_string()
                } else {
                    format!("'{type_name}'")
                };
                return Err(crate::PyError::type_error(format!(
                    "decoding to str: a bytes-like object is required, not {operand}"
                )));
            };
            decoded = result?;
            decoded.as_ref()
        }
    };
    let mut iter = raw.code_points();
    let first = iter.next();
    if first.is_none() || iter.next().is_some() {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 2 must be a single character"
        )));
    }
    Ok(first.unwrap())
}

/// Byte length of a `body_len`-byte string once `run` copies of `fill` are
/// added to it.
///
/// Saturating throughout: a `width` near `sys.maxsize` would otherwise wrap
/// the product and hand the reservation a small, satisfiable capacity.
fn padded_byte_len(body_len: usize, run: usize, fill: CodePoint) -> usize {
    run.saturating_mul(fill.len_wtf8()).saturating_add(body_len)
}

/// Append `cp` to `out`, `n` times.
///
/// `n` is a padding run derived from a caller-supplied `width`, so the space
/// for it is reserved fallibly up front; growing the buffer one code point at
/// a time would abort the process on an unsatisfiable run.
fn push_cp_repeated(out: &mut Wtf8Buf, cp: CodePoint, n: usize) -> Result<(), crate::PyError> {
    out.try_reserve_exact(n.saturating_mul(cp.len_wtf8()))
        .map_err(|_| crate::builtins::reservation_failed())?;
    for _ in 0..n {
        out.push(cp);
    }
    Ok(())
}

/// `unicode_result_unchanged`: a str method whose result equals the
/// receiver returns the receiver itself only when it is an exact `str`; a
/// `str` subclass is copied to a fresh base `str`, since str methods never
/// return a subclass instance.
pub(crate) fn str_result_unchanged(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { is_exact_type(obj, &STR_TYPE) } {
        obj
    } else {
        w_str_from_wtf8_managed(unsafe { w_str_get_wtf8(obj) }.to_owned())
    }
}

pub fn str_method_center(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "center", 1)?;
    arity_at_most(args, "center", 2)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = pyre_object::gc_roots::pin_roots(args);
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let width =
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(recv_slot + 1))?
            .max(0) as usize;
    let reloaded: Vec<_> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(recv_slot + i))
        .collect();
    let fillchar = pad_fillchar(&reloaded, "center")?;
    let s = unsafe { w_str_get_wtf8(recv()) }.to_wtf8_buf();
    let s_len = unsafe { pyre_object::w_str_len(recv()) };
    if s_len >= width {
        return Ok(str_result_unchanged(recv()));
    }
    // unicodeobject.py:1098 d = (width - len) ; lpad = d//2 + (d & width & 1)
    let d = width - s_len;
    let left = d / 2 + (d & width & 1);
    let right = d - left;
    let mut out = crate::builtins::try_wtf8_with_capacity(padded_byte_len(s.len(), d, fillchar))?;
    push_cp_repeated(&mut out, fillchar, left)?;
    out.push_wtf8(&s);
    push_cp_repeated(&mut out, fillchar, right)?;
    Ok(w_str_from_wtf8_managed(out))
}

/// PyPy: unicodeobject.py descr_ljust
pub fn descr_ljust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "ljust", 1)?;
    arity_at_most(args, "ljust", 2)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = pyre_object::gc_roots::pin_roots(args);
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let width =
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(recv_slot + 1))?
            .max(0) as usize;
    let reloaded: Vec<_> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(recv_slot + i))
        .collect();
    let fillchar = pad_fillchar(&reloaded, "ljust")?;
    let s = unsafe { w_str_get_wtf8(recv()) }.to_wtf8_buf();
    let s_len = unsafe { pyre_object::w_str_len(recv()) };
    if s_len >= width {
        return Ok(str_result_unchanged(recv()));
    }
    let mut out =
        crate::builtins::try_wtf8_with_capacity(padded_byte_len(s.len(), width - s_len, fillchar))?;
    out.push_wtf8(&s);
    push_cp_repeated(&mut out, fillchar, width - s_len)?;
    Ok(w_str_from_wtf8_managed(out))
}

/// PyPy: unicodeobject.py descr_rjust
pub fn descr_rjust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rjust", 1)?;
    arity_at_most(args, "rjust", 2)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = pyre_object::gc_roots::pin_roots(args);
    let recv = || pyre_object::gc_roots::shadow_stack_get(recv_slot);
    let width =
        crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(recv_slot + 1))?
            .max(0) as usize;
    let reloaded: Vec<_> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(recv_slot + i))
        .collect();
    let fillchar = pad_fillchar(&reloaded, "rjust")?;
    let s = unsafe { w_str_get_wtf8(recv()) }.to_wtf8_buf();
    let s_len = unsafe { pyre_object::w_str_len(recv()) };
    if s_len >= width {
        return Ok(str_result_unchanged(recv()));
    }
    let mut out =
        crate::builtins::try_wtf8_with_capacity(padded_byte_len(s.len(), width - s_len, fillchar))?;
    push_cp_repeated(&mut out, fillchar, width - s_len)?;
    out.push_wtf8(&s);
    Ok(w_str_from_wtf8_managed(out))
}

/// `pypy/objspace/std/unicodeobject.py descr_isprintable` —
///
/// ```python
/// def descr_isprintable(self, space):
///     for ch in self._utf8:
///         if not unicodedb.isprintable(ord(ch)):
///             return space.w_False
///     return space.w_True
/// ```
///
/// Empty string returns True per CPython.  Delegates the per-character
/// category check to `classify::is_printable`.
pub fn descr_isprintable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isprintable")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    // Empty returns True (vacuous); a lone surrogate is not printable.
    let result = match s.as_str() {
        Ok(v) => v.chars().all(classify::is_printable),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isspace
pub fn descr_isspace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isspace")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_space),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py `descr_islower` / `descr_isupper` —
///
/// ```python
/// cased = False
/// for uchar in rutf8.Utf8StringIterator(self._utf8):
///     if unicodedb.isupper(uchar) or unicodedb.istitle(uchar):
///         return space.w_False
///     if not cased and unicodedb.islower(uchar):
///         cased = True
/// return space.newbool(cased)
/// ```
///
/// Only the three cased categories participate: an uncased code point —
/// a digit, a space, a lone surrogate, or an uncased *letter* such as a
/// Hangul syllable — is skipped rather than counted, so `'가나다a'.islower()`
/// is True.  `is_alphabetic` is the wrong test for `cased` because it holds
/// for those uncased letters.
fn wtf8_cased_all(s: &Wtf8, want_upper: bool) -> bool {
    let mut cased = false;
    for cp in s.code_points() {
        let Some(c) = cp.to_char() else { continue };
        // `unicodedb.istitle` is category Lt, which is neither `is_uppercase`
        // (Lu + Other_Uppercase) nor `is_lowercase` (Ll + Other_Lowercase);
        // it disqualifies both predicates.
        let (rejects, qualifies) = if want_upper {
            (c.is_lowercase() || case::is_titlecase(c), c.is_uppercase())
        } else {
            (c.is_uppercase() || case::is_titlecase(c), c.is_lowercase())
        };
        if rejects {
            return false;
        }
        if !cased && qualifies {
            cased = true;
        }
    }
    cased
}

/// PyPy: unicodeobject.py descr_isupper
pub fn descr_isupper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isupper")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, true)))
}

/// PyPy: unicodeobject.py descr_islower
pub fn descr_islower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "islower")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, false)))
}

/// PyPy: unicodeobject.py descr_isalnum
pub fn descr_isalnum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isalnum")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_alnum),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isascii
pub fn descr_isascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isascii")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => v.is_ascii(),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Builds an owned str from a byte sub-slice of a WTF-8 string. The
/// slice must start and end on code-point boundaries, which holds for
/// the partition cuts below (the separator aligns on boundaries).
fn wtf8_slice_str(bytes: &[u8]) -> PyObjectRef {
    let part = unsafe { Wtf8::from_bytes_unchecked(bytes) };
    w_str_from_wtf8_managed(part.to_wtf8_buf())
}

/// Replaces up to `maxcount` occurrences of `sub` with `by` over the
/// WTF-8 bytes (rstring.py `replace_count` isutf8 path). A
/// negative `maxcount` means no limit; an empty `sub` inserts `by` at
/// every code-point boundary, including the ends.
///
/// Returns the `(res, replacements)` pair `replace_count` does — `descr_replace`
/// (unicodeobject.py) keys its identity shortcut on the count, not on
/// the resulting bytes.
fn wtf8_replace(input: &Wtf8, sub: &Wtf8, by: &Wtf8, maxcount: i64) -> (Wtf8Buf, usize) {
    if maxcount == 0 {
        return (input.to_wtf8_buf(), 0);
    }
    let inp = input.as_bytes();
    let sub_b = sub.as_bytes();
    let mut out = Wtf8Buf::new();
    let mut start = 0usize;
    let mut maxcount = maxcount;
    let mut replacements = 0usize;
    if sub_b.is_empty() {
        let mut indices = input.code_point_indices().map(|(i, _)| i);
        // Skip the leading boundary at 0; it is handled by the first
        // `by` insertion before each code point.
        indices.next();
        loop {
            out.push_wtf8(by);
            maxcount -= 1;
            replacements += 1;
            if start == inp.len() || maxcount == 0 {
                break;
            }
            let next = indices.next().unwrap_or(inp.len());
            out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
            start = next;
        }
    } else {
        while maxcount != 0 {
            match wtf8_find_bounded(inp, sub_b, start, inp.len()) {
                Some(next) => {
                    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
                    out.push_wtf8(by);
                    start = next + sub_b.len();
                    maxcount -= 1;
                    replacements += 1;
                }
                None => break,
            }
        }
    }
    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..]) });
    (out, replacements)
}

/// PyPy: unicodeobject.py descr_partition
pub fn descr_partition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "partition", 1)?;
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "must be str, not {}",
            arg_type_name(args[1])
        )));
    }
    // Cuts allocate, so pin the receiver and separator first and copy
    // their payloads off the objects before any mint.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let s = unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }
        .as_bytes()
        .to_vec();
    let sep =
        unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }
            .as_bytes()
            .to_vec();
    if sep.is_empty() {
        return Err(crate::PyError::value_error("empty separator"));
    }
    match wtf8_find_bounded(&s, &sep, 0, s.len()) {
        Some(i) => {
            let left_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(wtf8_slice_str(&s[..i]));
            let right_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(wtf8_slice_str(&s[i + sep.len()..]));
            Ok(w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(left_slot),
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                pyre_object::gc_roots::shadow_stack_get(right_slot),
            ]))
        }
        None => {
            let empty_a = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new(""));
            let empty_b = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new(""));
            Ok(w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(base),
                pyre_object::gc_roots::shadow_stack_get(empty_a),
                pyre_object::gc_roots::shadow_stack_get(empty_b),
            ]))
        }
    }
}

/// PyPy: unicodeobject.py descr_rpartition
pub fn descr_rpartition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "rpartition", 1)?;
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "must be str, not {}",
            arg_type_name(args[1])
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let s = unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }
        .as_bytes()
        .to_vec();
    let sep =
        unsafe { pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }
            .as_bytes()
            .to_vec();
    if sep.is_empty() {
        return Err(crate::PyError::value_error("empty separator"));
    }
    match wtf8_rfind_bounded(&s, &sep, 0, s.len()) {
        Some(i) => {
            let left_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(wtf8_slice_str(&s[..i]));
            let right_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(wtf8_slice_str(&s[i + sep.len()..]));
            Ok(w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(left_slot),
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                pyre_object::gc_roots::shadow_stack_get(right_slot),
            ]))
        }
        None => {
            let empty_a = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new(""));
            let empty_b = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new(""));
            Ok(w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(empty_a),
                pyre_object::gc_roots::shadow_stack_get(empty_b),
                pyre_object::gc_roots::shadow_stack_get(base),
            ]))
        }
    }
}

/// PyPy: unicodeobject.py descr_splitlines.
/// Walks the Unicode line-boundary set explicitly so that
/// `keepends=True` retains the terminator on each emitted line and a
/// trailing `\n` does NOT produce an extra empty entry — matching
/// `'a\nb\n'.splitlines() == ['a', 'b']`.
fn cp_is_linebreak(cp: CodePoint) -> bool {
    matches!(
        cp.to_char(),
        Some(
            '\n' | '\r'
                | '\u{000b}'
                | '\u{000c}'
                | '\u{001c}'
                | '\u{001d}'
                | '\u{001e}'
                | '\u{0085}'
                | '\u{2028}'
                | '\u{2029}'
        )
    )
}

pub fn str_method_splitlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "splitlines")?;
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "splitlines() takes at most 1 argument ({} given)",
            args_given(pos)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["keepends"], "splitlines")?;
    crate::builtins::kwarg_reject_duplicate(
        kwargs,
        "splitlines",
        "keepends",
        pos.get(1).is_some(),
    )?;
    let keepends_obj =
        crate::builtins::kwarg_get(kwargs, "keepends").or_else(|| pos.get(1).copied());
    let recv_roots = pyre_object::gc_roots::push_roots();
    let recv_slot = if let Some(keepends_obj) = keepends_obj {
        let recv_slot = recv_roots.publish(&[pos[0], keepends_obj]);
        recv_roots.normalize(recv_slot, 2);
        recv_slot
    } else {
        let recv_slot = recv_roots.publish(&[pos[0]]);
        recv_roots.normalize(recv_slot, 1);
        recv_slot
    };
    // keepends is positional-or-keyword. Resolve it before copying the
    // receiver: `__bool__` can collect.
    let keepends = if keepends_obj.is_some() {
        crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(recv_slot + 1))?
    } else {
        false
    };
    let cps: Vec<CodePoint> = unsafe { w_str_get_wtf8(recv_roots.get(recv_slot)) }
        .code_points()
        .collect();
    // Each `cps_to_str_cut` allocates over the pieces already cut.
    let mut parts = pyre_object::gc_roots::RootedItems::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < cps.len() {
        if cp_is_linebreak(cps[i]) {
            let mut term_end = i + 1;
            if cps[i].to_char() == Some('\r')
                && term_end < cps.len()
                && cps[term_end].to_char() == Some('\n')
            {
                term_end += 1;
            }
            let end = if keepends { term_end } else { i };
            parts.push(cps_to_str_cut(recv_slot, &cps[start..end]));
            start = term_end;
            i = term_end;
        } else {
            i += 1;
        }
    }
    if start < cps.len() {
        parts.push(cps_to_str_cut(recv_slot, &cps[start..]));
    }
    Ok(w_list_new(parts.take()))
}

/// PyPy: unicodeobject.py descr_removeprefix (Python 3.9+)
pub fn descr_removeprefix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removeprefix() takes exactly one argument ({} given)",
            args_given(pos)
        )));
    }
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "removeprefix() argument must be str, not {}",
            clinic_arg_type_name(pos[1])
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[pos[0], pos[1]]);
    let s = unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }.to_wtf8_buf();
    let prefix =
        unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }.to_wtf8_buf();
    match s.strip_prefix(prefix.as_ref()) {
        Some(rest) => Ok(w_str_from_wtf8_managed(rest.to_wtf8_buf())),
        None => Ok(str_result_unchanged(
            pyre_object::gc_roots::shadow_stack_get(base),
        )),
    }
}

/// PyPy: unicodeobject.py descr_removesuffix (Python 3.9+)
pub fn descr_removesuffix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removesuffix() takes exactly one argument ({} given)",
            args_given(pos)
        )));
    }
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "removesuffix() argument must be str, not {}",
            clinic_arg_type_name(pos[1])
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[pos[0], pos[1]]);
    let s = unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }.to_wtf8_buf();
    let suffix =
        unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base + 1)) }.to_wtf8_buf();
    // `descr_removesuffix` (unicodeobject.py) guards the slice arm
    // with `if suffix and ...`, so an empty suffix falls through to the arm
    // that rewraps the receiver's own storage — `s.removesuffix("") is s` for
    // an exact `str`.  `removeprefix` has no such guard: its slice arm goes
    // through `ll_stringslice_startonly` (rstr.py), which has no
    // whole-span shortcut, so an empty prefix still builds a fresh object.
    if !suffix.is_empty()
        && let Some(rest) = s.strip_suffix(suffix.as_ref())
    {
        return Ok(w_str_from_wtf8_managed(rest.to_wtf8_buf()));
    }
    Ok(str_result_unchanged(
        pyre_object::gc_roots::shadow_stack_get(base),
    ))
}

/// PyPy: unicodeobject.py descr_expandtabs
pub fn descr_expandtabs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `tabsize` is positional-or-keyword (default 8).
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "expandtabs() takes at most 1 argument ({} given)",
            args_given(pos)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["tabsize"], "expandtabs")?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "expandtabs", "tabsize", pos.get(1).is_some())?;
    // [3.14-spec] PyPy `W_UnicodeObject.descr_expandtabs` unwraps a machine
    // `int`; CPython `unicode_expandtabs` declares an Argument Clinic `int`
    // and therefore calls `PyLong_AsInt` before inspecting even an empty
    // receiver.  Keep PyPy's method body below, but narrow this observable
    // argument boundary to a C int.
    let w_tabsize = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "tabsize"));
    let _roots = pyre_object::gc_roots::push_roots();
    let recv_slot = if let Some(t) = w_tabsize {
        pyre_object::gc_roots::pin_roots(&[pos[0], t])
    } else {
        pyre_object::gc_roots::pin_roots(&[pos[0]])
    };
    let tabsize = i64::from(match w_tabsize {
        Some(_) => crate::baseobjspace::index_c_int_w(pyre_object::gc_roots::shadow_stack_get(
            recv_slot + 1,
        ))?,
        None => 8,
    });
    let s =
        unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(recv_slot)) }.to_wtf8_buf();
    // Tabs advance to the next multiple of `tabsize` measured from the
    // start of the current line (the column resets on `\n` / `\r`); a
    // non-positive `tabsize` drops tabs entirely. The expanded length is
    // tracked with checked arithmetic so a pathological `tabsize` raises
    // OverflowError instead of attempting an unbounded allocation.
    let overflow = || crate::PyError::overflow_error("result is too long");
    let mut total: i64 = 0;
    let mut col: i64 = 0;
    for cp in s.code_points() {
        match cp.to_char() {
            Some('\t') => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col = col.checked_add(incr).ok_or_else(overflow)?;
                    total = total.checked_add(incr).ok_or_else(overflow)?;
                }
            }
            Some('\n') | Some('\r') => {
                total = total.checked_add(1).ok_or_else(overflow)?;
                col = 0;
            }
            _ => {
                total = total.checked_add(1).ok_or_else(overflow)?;
                col += 1;
            }
        }
    }
    let cap = usize::try_from(total).map_err(|_| overflow())?;
    let mut result = Wtf8Buf::with_capacity(cap);
    let mut col: i64 = 0;
    for cp in s.code_points() {
        match cp.to_char() {
            Some('\t') => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col += incr;
                    for _ in 0..incr {
                        result.push_char(' ');
                    }
                }
            }
            Some('\n') | Some('\r') => {
                result.push(cp);
                col = 0;
            }
            _ => {
                result.push(cp);
                col += 1;
            }
        }
    }
    Ok(w_str_from_wtf8_managed(result))
}

/// str.translate(table) — table is a mapping from ordinals (int) to
/// ordinals (int), strings (str), or None (delete).
pub fn str_method_translate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "translate", 1)?;
    // Each lookup can collect, so pin the receiver and table and copy the
    // source payload off the object before the loop.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let s = unsafe { w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(base)) }.to_wtf8_buf();
    let table = || pyre_object::gc_roots::shadow_stack_get(base + 1);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    let mut result = Wtf8Buf::with_capacity(s.len());
    unsafe {
        for cp in s.code_points() {
            pyre_object::gc_roots::shadow_stack_set(key_slot, w_int_new(cp.to_u32() as i64));
            // `unicode_translate` clears any LookupError the lookup raises
            // and keeps the character, so a table that indexes rather than
            // maps (a str, a list) leaves unmapped code points alone.
            let found = match crate::baseobjspace::finditem(
                table(),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
            ) {
                Ok(found) => found,
                Err(e)
                    if matches!(
                        e.kind,
                        crate::PyErrorKind::LookupError
                            | crate::PyErrorKind::KeyError
                            | crate::PyErrorKind::IndexError
                    ) =>
                {
                    None
                }
                Err(e) => return Err(e),
            };
            match found {
                None => result.push(cp),
                Some(val) if is_none(val) => {}
                Some(val) if is_int(val) => {
                    let code = w_int_get_value(val);
                    if let Some(c) = u32::try_from(code).ok().and_then(CodePoint::from_u32) {
                        result.push(c);
                    } else {
                        return Err(crate::PyError::value_error(
                            "character mapping must be in range(0x110000)",
                        ));
                    }
                }
                Some(val) if is_str(val) => {
                    result.push_wtf8(w_str_get_wtf8(val));
                }
                Some(_) => {
                    return Err(crate::PyError::type_error(
                        "character mapping must return integer, None or str",
                    ));
                }
            }
        }
    }
    Ok(w_str_from_wtf8_managed(result))
}

// ── Dict methods ─────────────────────────────────────────────────────

pub use pyre_object::typeobject::DICT_DATA_SLOT;

/// Whether a type built on `w_base` owes its instances a reserved
/// [`DICT_DATA_SLOT`] of their own: the base's instances are dicts, and no
/// layout along its chain reserves the slot already.
///
/// Both routes that build a type owe this — `create_all_slots` for a class
/// written in Python and the builtin layout mint for one readied from C — so
/// the question is asked here rather than answered twice.  The base's layout
/// typedef settles "are these dicts" without consulting the `dict` type
/// object, which the builtin route cannot look up while it is still being
/// built.
///
/// # Safety
/// `w_base` must be null or a valid `W_TypeObject` pointer.
pub unsafe fn base_owes_dict_backing(w_base: PyObjectRef) -> bool {
    unsafe {
        if w_base.is_null() {
            return false;
        }
        let layout = pyre_object::w_type_get_layout_ptr(w_base);
        let dict_typedef = &pyre_object::pyobject::DICT_TYPE as *const pyre_object::PyType;
        if layout.is_null() || !std::ptr::eq((*(*layout).typedef).instance_type, dict_typedef) {
            return false;
        }
        // `Layout::dict_data_slot` already answers for the whole chain — it
        // is sealed with the base's index when a layout reserves none itself.
        (*layout).dict_data_slot == pyre_object::typeobject::DICT_DATA_SLOT_UNRESOLVED
    }
}

/// True when `obj` can be read as a dict — an exact dict, a dict subclass
/// carrying a mapping payload, or a mapping proxy wrapping one.  This is the
/// receiver test a `dict` method descriptor applies, so an object that merely
/// claims `__class__ is dict` never reaches the backing lookup.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
pub unsafe fn has_dict_backing(obj: PyObjectRef) -> bool {
    !resolve_dict_backing(obj).is_null()
}

/// Absolute layout-slot index of the reserved [`DICT_DATA_SLOT`] payload for
/// `obj`, walking the layout chain from the object's type toward its base.
/// `None` when no layout in the chain reserves the slot.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
unsafe fn dict_data_slot(obj: PyObjectRef) -> Option<u32> {
    unsafe {
        let w_type = crate::typedef::r#type(obj).map_or(PY_NULL, |p| p.as_ptr());
        if w_type.is_null() {
            return None;
        }
        let layout = pyre_object::w_type_get_layout_ptr(w_type);
        if layout.is_null() {
            return None;
        }
        // `typeobject.py create_all_slots` hands each `Member` the index it
        // resolved while building the layout; the index is never re-derived
        // from the slot's name afterwards.  `leak_layout` seals this one the
        // same way, walking `base_layout` once instead of on every read.
        let slot = (*layout).dict_data_slot;
        if slot == pyre_object::typeobject::DICT_DATA_SLOT_UNRESOLVED {
            return None;
        }
        Some(slot as u32)
    }
}

/// Write the mapping payload of a dict subclass instance.  This is a
/// layout-slot operation, not Python attribute assignment: a subclass may
/// override `__setattr__` with a dict method before its payload exists.
pub fn set_dict_backing(obj: PyObjectRef, backing: PyObjectRef) -> bool {
    unsafe {
        if !is_instance(obj) || !is_dict(backing) {
            return false;
        }
        if let Some(slot) = dict_data_slot(obj) {
            crate::objspace::std::mapdict::setslotvalue(obj, slot, backing);
            return true;
        }
    }
    false
}

pub fn resolve_dict_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_dict(obj) {
            return obj;
        }
        // `pypy/objspace/std/dictproxyobject.py keys_w/values_w/
        // items_w` forward through `space.call_method(self.w_mapping,
        // ...)` — the mapping is unwrapped before any dict-method
        // dispatch.  Surface the same shape here so
        // `dict_method_{keys,values,items,get,copy,update,...}` work
        // on `type.__dict__` without per-method proxy plumbing.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        if pyre_object::is_dict_proxy(pyre_object::gc_roots::shadow_stack_get(obj_slot)) {
            let inner = pyre_object::w_dict_proxy_get_mapping(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            );
            // The wrapped mapping may itself be a dict subclass (e.g. a
            // class whose namespace is an `OrderedDict`), which keeps its
            // entries in a native backing dict — resolve through to that
            // rather than requiring `inner` to be an exact dict.
            if !inner.is_null() && inner != pyre_object::gc_roots::shadow_stack_get(obj_slot) {
                let backing = resolve_dict_backing(inner);
                if !backing.is_null() {
                    return backing;
                }
            }
        }
        if is_instance(pyre_object::gc_roots::shadow_stack_get(obj_slot)) {
            // Read the reserved layout slot directly: going through
            // `getattr_str` would incorrectly expose internal dict operations
            // to a subclass's Python-level `__getattribute__` hook.
            if let Some(slot) = dict_data_slot(pyre_object::gc_roots::shadow_stack_get(obj_slot))
                && let Some(backing) = crate::objspace::std::mapdict::getslotvalue(
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    slot,
                )
                && is_dict(backing)
            {
                return backing;
            }
        }
    }
    pyre_object::PY_NULL
}

fn dict_lookup_checked(
    dict: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[dict, key]);
        pyre_object::dictmultiobject::w_dict_lookup_checked(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
        )
        .map_err(|_| {
            crate::baseobjspace::take_pending_dict_key_error(
                pyre_object::gc_roots::shadow_stack_get(base + 1),
            )
        })
    }
}

pub(crate) fn dict_store_checked(
    dict: PyObjectRef,
    mut key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        pyre_object::with_roots!(key =>
            pyre_object::dictmultiobject::w_dict_store_checked(dict, key, value)
        )
        .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(key))
    }
}

/// Hash `item` with it rooted, then look it up with `probe`.
///
/// A user `__hash__` is a collection point and [`ObjectKey`] caches the element
/// pointer, so hashing inside the key build (`object_key_for_checked`) captures
/// the pre-move pointer — the reason `object_key_hashed` exists. The element is
/// rooted across the hash and reloaded, matching the add path
/// (`builtin_set_add_items`).
///
/// The set is rooted too: `CONTAINS_OP` pops the container off the operand
/// stack before dispatching here, so on `x in {...}` the hash below runs with
/// nothing else referring to the set at all.  The body relocates, so the
/// probe reloads it after `__hash__`.
unsafe fn set_lookup_checked(
    set: PyObjectRef,
    item: PyObjectRef,
    probe: unsafe fn(
        PyObjectRef,
        pyre_object::dictmultiobject::ObjectKey,
    ) -> Result<bool, pyre_object::dictmultiobject::DictKeyError>,
) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::pin_roots(&[item, set]);
    let hash = crate::builtins::try_hash_value(pyre_object::gc_roots::shadow_stack_get(sp))
        .map_err(|err| {
            crate::baseobjspace::wrap_set_element_hash_error(
                pyre_object::gc_roots::shadow_stack_get(sp),
                err,
            )
        })?;
    let key = pyre_object::dictmultiobject::object_key_hashed(
        pyre_object::gc_roots::shadow_stack_get(sp),
        hash,
    );
    probe(pyre_object::gc_roots::shadow_stack_get(sp + 1), key)
        .map_err(|_| crate::baseobjspace::take_pending_hash_error())
}

/// Remove an element from a set, hashing it through the protocol.
pub(crate) fn set_discard_checked(
    set: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe { set_lookup_checked(set, item, pyre_object::setobject::w_set_discard_key_checked) }
}

/// Test membership in a set, hashing the element through the protocol.
pub(crate) fn set_contains_checked(
    set: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe {
        set_lookup_checked(
            set,
            item,
            pyre_object::setobject::w_set_contains_key_checked,
        )
    }
}

/// `dictmultiobject.py descr_get`.  The body is this leaf so
/// `BuiltinCode.func`'s PBC family has a jitcode for `d.get`; a closure
/// registered under another name has no member and the call stays
/// `no jitcode for address`.
pub fn __majit_wrap_dict_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `d.get(key)` / `d.get(key, default)` of an exact dict and an exact
    // `str` key.  `dict_method_get` hashes through user code and its graph
    // reaches `w_dict_str_entries_wtf8`, which the descent scan refuses.
    // Those two residuals stay out of this graph; the slow arm is one real
    // fnaddr.
    if (args.len() == 2 || args.len() == 3) && !args[0].is_null() {
        // A three-word call can be `d.get(k, **kw)`: the third word is then
        // the marker dict, not a default.
        let tail = if args.len() == 3 {
            args[2]
        } else {
            pyre_object::PY_NULL
        };
        if dict_get_plain_applies(args[0], args[1], tail) {
            let default = if args.len() == 3 {
                tail
            } else {
                pyre_object::w_none()
            };
            let found = dict_get_plain(args[0], args[1], default);
            if !found.is_null() {
                return Ok(found);
            }
        }
    }
    let dict = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let key = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    // Three words cover a call of length 0..=3. A longer slice's last
    // element rides in the third word so a trailing keyword dict is still
    // visible when the real arity is rejected.
    let default = if args.len() > 3 {
        args[args.len() - 1]
    } else {
        args.get(2).copied().unwrap_or(pyre_object::PY_NULL)
    };
    dict_get_slow(dict, key, default, args.len() as i64)
}

/// Exact `dict`, an exact `str` key, and a third word that is not the
/// trailing keyword dict.
///
/// Exactness is the requirement, not layout: `py_type_check` passes a `str`
/// subclass, whose `__hash__` can raise, and the unchecked probe reports that
/// raise as a miss.  `tail` is `PY_NULL` for a two-word call; anything else is
/// the caller's third word, and a keyword marker there means the call is
/// `d.get(k, **kw)`, which `dict_method_get` rejects.
///
/// The dict's strategy is deliberately not tested here.  Reading it in this
/// call and probing in the next is a time-of-check window, so
/// [`dict_get_plain`] decides it under the lock that probes.
#[majit_macros::dont_look_inside_cannot_raise]
fn dict_get_plain_applies(dict: PyObjectRef, key: PyObjectRef, tail: PyObjectRef) -> bool {
    unsafe {
        pyre_object::is_exact_type(dict, &pyre_object::DICT_TYPE)
            && pyre_object::is_exact_type(key, &pyre_object::STR_TYPE)
            && (tail.is_null() || !crate::builtins::builtin_kwargs_marker_tail(tail))
    }
}

/// Lookup for a key [`dict_get_plain_applies`] already accepted.
///
/// `PY_NULL` says the answer is not this path's to give and the caller owes
/// the checked arm.  A stored value is never `PY_NULL`, so the only other
/// reading is a miss whose `default` was itself `PY_NULL` — the padded shape
/// of a two-argument call — and the checked arm answers that one identically.
#[majit_macros::dont_look_inside_cannot_raise]
fn dict_get_plain(dict: PyObjectRef, key: PyObjectRef, default: PyObjectRef) -> PyObjectRef {
    unsafe {
        pyre_object::dictmultiobject::w_dict_lookup_str_keyed(dict, key, default)
            .unwrap_or(pyre_object::PY_NULL)
    }
}

/// Word ABI so the residual has a real fnaddr.  A `&[PyObjectRef]`
/// parameter is a fat pointer and gets no trampoline, which the descent
/// scan reports as an un-lowered helper.
#[majit_macros::dont_look_inside]
fn dict_get_slow(
    dict: PyObjectRef,
    key: PyObjectRef,
    default: PyObjectRef,
    nargs: i64,
) -> Result<PyObjectRef, crate::PyError> {
    // `nargs` is the flat length, receiver included. Collapsing anything
    // above three into a three-argument call drops the too-many-arguments
    // TypeError `dict_method_get` raises from the real slice.
    if nargs == 3 {
        dict_method_get(&[dict, key, default])
    } else if nargs == 2 {
        dict_method_get(&[dict, key])
    } else if nargs == 1 {
        dict_method_get(&[dict])
    } else if nargs <= 0 {
        dict_method_get(&[])
    } else {
        let n = nargs as usize;
        let mut buf = vec![pyre_object::PY_NULL; n];
        buf[0] = dict;
        buf[1] = key;
        buf[n - 1] = default;
        dict_method_get(&buf)
    }
}

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_dict_descr_get,
    __majit_wrap_dict_descr_get
);

pub fn dict_method_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "get", 1)?;
    arity_at_most(args, "get", 2)?;
    let nargs = args.len();
    // The lookup hashes and compares the key, which is user code. Pin the
    // receiver before resolving its backing so the slice is not live across
    // that call, then reload every operand from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let dict = resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(base));
    if dict.is_null() {
        return Ok(if nargs >= 3 {
            pyre_object::gc_roots::shadow_stack_get(base + 2)
        } else {
            w_none()
        });
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    let found = dict_lookup_checked(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        pyre_object::gc_roots::shadow_stack_get(base + 1),
    )?;
    Ok(found.unwrap_or_else(|| {
        if nargs >= 3 {
            pyre_object::gc_roots::shadow_stack_get(base + 2)
        } else {
            w_none()
        }
    }))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_keys` parity — returns
/// a live `dict_keys` view bound to the source dict, not a snapshot
/// list.  The view's iter / len / contains semantics dispatch back
/// through the source dict (see baseobjspace getattr arm) so
/// mutations on the dict are visible through the view, matching
/// `W_DictViewKeysObject`'s behaviour.
///
/// Named `__majit_wrap_*` and published below because `BuiltinCode.func` is a
/// PBC whose family `builtin_wrapper_indirect_graphs` builds out of exactly the
/// wrapper paths that carry a registered graph. A method registered under any
/// other name has no member in it, so `bytecode_for_address` finds no jitcode
/// for its address and a traced `d.keys()` declines
/// `try_walker_inline_builtin_call` with `no jitcode for address` — it stays a
/// `bh_call_fn` blackhole residual, and the dict-view allocation and the
/// `w_dict_len` behind it never reach the trace. Upstream has no such split:
/// `interp2app` turns every builtin method into one of these graphs.
pub fn __majit_wrap_dict_descr_keys(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "keys", 0)?;
    require_receiver(args, "keys")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        // Type-erased fallback: the receiver isn't a dict, surface
        // an empty view rather than fabricating a foreign-shaped
        // list (the view's source-dict slot tolerates PY_NULL via
        // the read-side guards).
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Keys,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Keys,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_values` parity — same
/// shape as `__majit_wrap_dict_descr_keys`, kind tag `Values`.
pub fn __majit_wrap_dict_descr_values(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "values", 0)?;
    require_receiver(args, "values")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Values,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Values,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_items` parity — same
/// shape as `__majit_wrap_dict_descr_keys`, kind tag `Items`.
pub fn __majit_wrap_dict_descr_items(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "items", 0)?;
    require_receiver(args, "items")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Items,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Items,
    ))
}

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_str_descr_startswith,
    __majit_wrap_str_descr_startswith
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_str_descr_endswith,
    __majit_wrap_str_descr_endswith
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_str_descr_find,
    __majit_wrap_str_descr_find
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_str_descr_rfind,
    __majit_wrap_str_descr_rfind
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_str_descr_count,
    __majit_wrap_str_descr_count
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_dict_descr_keys,
    __majit_wrap_dict_descr_keys
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_dict_descr_values,
    __majit_wrap_dict_descr_values
);

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_dict_descr_items,
    __majit_wrap_dict_descr_items
);

/// Materialise a dict_keys / values / items view's current snapshot
/// as a list of items.  Mirrors the view iteration bodies on
/// `W_DictViewKeysObject` / values / items — pyre's `repr` /
/// `len` / `compare` / set-op paths call this to produce the
/// kind-appropriate list eagerly.
///
/// `__iter__` no longer routes through this helper: it allocates a
/// live `W_BaseDictMultiIterObject` that walks the source dict's entries
/// directly and trips on the dictversion counter, raising
/// `RuntimeError("dictionary changed size during iteration")` when
/// the source mutates mid-iteration.
pub fn dict_view_snapshot(view: PyObjectRef) -> Vec<PyObjectRef> {
    let kind = unsafe { pyre_object::dictmultiobject::w_dict_view_get_kind(view) };
    let dict = unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
    if dict.is_null() {
        return Vec::new();
    }
    let items = unsafe { pyre_object::w_dict_items(dict) };
    let pairs = items.as_slice();
    let n = pairs.len();
    match kind {
        pyre_object::dictmultiobject::DictViewKind::Keys => {
            let mut out = Vec::with_capacity(n);
            let mut i = 0usize;
            while i < n {
                out.push(pairs[i].0);
                i += 1;
            }
            out
        }
        pyre_object::dictmultiobject::DictViewKind::Values => {
            let mut out = Vec::with_capacity(n);
            let mut i = 0usize;
            while i < n {
                out.push(pairs[i].1);
                i += 1;
            }
            out
        }
        pyre_object::dictmultiobject::DictViewKind::Items => {
            // `w_tuple_new` allocates, so a pair still waiting in this native
            // Vec moves under the loop.  A pair already built into a tuple is
            // just as exposed: that tuple's header is old-gen, but until
            // something roots it the collector does not walk it and its two
            // slots keep pre-move addresses.  Pin both sides of the loop.
            let _roots = pyre_object::gc_roots::push_roots();
            let mut flat = Vec::with_capacity(n * 2);
            let mut i = 0usize;
            while i < n {
                flat.push(pairs[i].0);
                flat.push(pairs[i].1);
                i += 1;
            }
            let pair_base = pyre_object::gc_roots::pin_roots(&flat);
            let mut built = Vec::with_capacity(n);
            i = 0;
            while i < n {
                let k = pyre_object::gc_roots::shadow_stack_get(pair_base + i * 2);
                let v = pyre_object::gc_roots::shadow_stack_get(pair_base + i * 2 + 1);
                let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::jit_w_tuple2(k, v));
                built.push(pyre_object::gc_roots::shadow_stack_get(tuple_slot));
                i += 1;
            }
            built
        }
    }
}

/// `pypy/objspace/std/dictmultiobject.py descr_copy` —
/// `return w_dict.copy()` which delegates to `strategy.copy(w_dict)`
/// (`:1152 AbstractTypedStrategy.copy`).  Typed strategies preserve
/// their backing shape by cloning the typed storage box and wrapping
/// it in a fresh W_DictObject with the same strategy.  Used by
/// `dict.copy()` and (via `resolve_dict_backing` proxy unwrap) by
/// `mappingproxy.copy()` (`dictproxyobject.py copy_w`).
pub fn dict_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "copy", 0)?;
    let src = resolve_dict_backing(args[0]);
    if src.is_null() {
        return Ok(pyre_object::w_dict_new());
    }
    unsafe { Ok(pyre_object::dictmultiobject::w_dict_copy(src)) }
}

/// PyPy: dictmultiobject.py descr_update — dict.update([other], **kwargs).
///
/// CPython 3.x signature accepts a single optional positional that is
/// either a mapping (uses keys()) or an iterable of (key, value) pairs,
/// followed by arbitrary kwargs that are merged on top.  The trailing
/// `dictmultiobject.py update1` — merge `w_data` into
/// `w_dict`.  Shared by `dict.__init__` and `dict.update`.
pub(crate) fn dict_update1(w_dict: PyObjectRef, w_data: PyObjectRef) -> Result<(), crate::PyError> {
    // RPython's shadowstack transform keeps both operands live across
    // `keys`, iteration, `__getitem__`, hashing, and equality calls.  Those
    // are arbitrary Python collection points; without the explicit roots an
    // otherwise-unreachable destination dict can be swept while this Rust
    // frame still holds its raw pointer. Resolve the backing from the pinned
    // destination so `w_data` is not live across that call.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::pin_roots(&[w_dict, w_data]);
    let dict = resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(root_base));
    if dict.is_null() {
        return Ok(());
    }
    let backing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    let dict = || pyre_object::gc_roots::shadow_stack_get(backing_slot);
    let data = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    unsafe {
        let fast_path_eligible = !resolve_dict_backing(data()).is_null()
            && pyre_object::is_dict(resolve_dict_backing(data()))
            && dict_subclass_uses_default_iter(data());
        if fast_path_eligible {
            // `dictmultiobject.py update1_dict_dict`
            let dst_is_empty = pyre_object::dictmultiobject::w_dict_is_regular_empty(dict());
            if dst_is_empty {
                let w_copy =
                    pyre_object::dictmultiobject::w_dict_copy(resolve_dict_backing(data()));
                pyre_object::dictmultiobject::w_dict_adopt_regular_copy_for_empty_update(
                    dict(),
                    w_copy,
                );
            } else {
                // PyPy `create_iterator_classes.rev_update1_dict_dict`
                // (`dictmultiobject.py`) supplies the live
                // `getiteritems_with_hash` walk.  Python 3.14 adds
                // `dict_dict_merge`'s `orig_size` check after every
                // destination insertion (`Objects/dictobject.c:3947-3985`):
                // destination key equality can re-enter Python and mutate
                // the source, which must raise instead of silently ending
                // the next `nth_item` lookup.
                let source = resolve_dict_backing(data());
                let orig_size = pyre_object::dictmultiobject::w_dict_len(source);
                let mut i = 0;
                loop {
                    // `getiteritems_with_hash` (`dictmultiobject.py`) — an
                    // object- or unicode-shaped source already stores each key's
                    // hash, so reusing it keeps `__hash__` at the single call
                    // that filled the source.  Re-deriving it here is observable
                    // for a key whose `__hash__` has side effects, and it puts
                    // an extra Python call inside this walk.  Typed strategies
                    // store no hash and `w_dict_nth_hashed_key` returns `None`
                    // for them, which selects the plain walk below.
                    let source_now = resolve_dict_backing(data());
                    // `i` names an entry slot, and a delete leaves a tombstone
                    // between the live ones, so step to the next live slot
                    // rather than assuming the numbering is dense.
                    let Some(live) = pyre_object::dictmultiobject::w_dict_next_slot(source_now, i)
                    else {
                        break;
                    };
                    i = live;
                    let hashed = pyre_object::is_exact_type(source_now, &pyre_object::DICT_TYPE)
                        .then(|| pyre_object::dictmultiobject::w_dict_nth_hashed_key(source_now, i))
                        .flatten();
                    let Some((k, v)) = (match hashed {
                        Some(key) => pyre_object::dictmultiobject::w_dict_nth_value(source_now, i)
                            .map(|v| (key.obj, v)),
                        None => pyre_object::dictmultiobject::w_dict_nth_item(source_now, i),
                    }) else {
                        break;
                    };
                    let _iteration_roots = pyre_object::gc_roots::push_roots();
                    let iteration_roots = pyre_object::gc_roots::pin_roots(&[k, v]);
                    let k = pyre_object::gc_roots::shadow_stack_get(iteration_roots);
                    let v = pyre_object::gc_roots::shadow_stack_get(iteration_roots + 1);
                    match hashed {
                        Some(key) => pyre_object::dictmultiobject::w_dict_store_hashed_checked(
                            dict(),
                            k,
                            v,
                            key.hash,
                        )
                        .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(k))?,
                        None => dict_store_checked(dict(), k, v)?,
                    }
                    if orig_size
                        != pyre_object::dictmultiobject::w_dict_len(resolve_dict_backing(data()))
                    {
                        return Err(crate::PyError::runtime_error("dict mutated during update"));
                    }
                    i += 1;
                }
            }
        } else {
            // `dictmultiobject.py update1`
            let w_keys_method = match crate::baseobjspace::getattr_str(data(), "keys") {
                Ok(value) => Some(value),
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
                Err(e) => return Err(e),
            };
            if let Some(w_method) = w_keys_method {
                // `dictmultiobject.py update1_keys`
                let w_keys_view = crate::call::call_function_impl_result(w_method, &[])?;
                let keys = crate::builtins::collect_iterable(w_keys_view)?;
                let keys_base = pyre_object::gc_roots::pin_roots(&keys);
                for i in 0..keys.len() {
                    let k = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
                    let v = crate::baseobjspace::getitem(data(), k)?;
                    let value_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(v);
                    let k = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
                    let v = pyre_object::gc_roots::shadow_stack_get(value_slot);
                    dict_store_checked(dict(), k, v)?;
                }
            } else {
                // PyPy `dictmultiobject.py:update1_pairs`, with Python 3.14
                // `merge_from_seq2_lock_held`'s PySequence_Fast error/note
                // semantics where the versions differ.
                let pairs = crate::builtins::collect_iterable(data())?;
                let pairs_base = pyre_object::gc_roots::pin_roots(&pairs);
                for idx in 0..pairs.len() {
                    let pair = pyre_object::gc_roots::shadow_stack_get(pairs_base + idx);
                    let entries = dict_update_pair_entries(pair, idx)?;
                    if entries.len() != 2 {
                        return Err(crate::PyError::value_error(format!(
                            "dictionary update sequence element #{idx} has length {}; 2 is required",
                            entries.len()
                        )));
                    }
                    let entry_base = pyre_object::gc_roots::pin_roots(&[entries[0], entries[1]]);
                    let k = pyre_object::gc_roots::shadow_stack_get(entry_base);
                    let v = pyre_object::gc_roots::shadow_stack_get(entry_base + 1);
                    dict_store_checked(dict(), k, v)?;
                }
            }
        }
    }
    Ok(())
}

/// CPython 3.14 `merge_from_seq2_lock_held`'s
/// `PySequence_Fast(item, "object is not iterable")` conversion.  An exact
/// list/tuple is already a fast sequence; for other objects, only a TypeError
/// from obtaining the iterator is replaced.  A TypeError raised later by the
/// iterator keeps its original message.
fn dict_update_pair_entries(
    pair: PyObjectRef,
    idx: usize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let pair_slot = pyre_object::gc_roots::shadow_stack_len();
    let pair = pyre_object::gc_roots::pin_root(pair);
    unsafe {
        if is_exact_list(pair) {
            let len = w_list_len(pair);
            return Ok((0..len)
                .filter_map(|i| w_list_getitem(pair, i as i64))
                .collect());
        }
        if is_exact_tuple(pair) {
            let len = w_tuple_len(pair);
            return Ok((0..len)
                .filter_map(|i| w_tuple_getitem(pair, i as i64))
                .collect());
        }
    }
    let iter = match crate::baseobjspace::iter(pair) {
        Ok(iter) => iter,
        Err(err) if err.kind == crate::PyErrorKind::TypeError => {
            return Err(dict_update_pair_note(
                crate::PyError::type_error("object is not iterable"),
                idx,
            ));
        }
        Err(err) => return Err(err),
    };
    crate::builtins::collect_iterator(iter).map_err(|err| dict_update_pair_note(err, idx))
}

/// `_PyErr_FormatNote` on a failed dict sequence-pair conversion.
/// `merge_from_seq2_lock_held` attaches the note only when the conversion
/// error matches `TypeError`, through `_PyException_AddNote`.
fn dict_update_pair_note(mut err: crate::PyError, idx: usize) -> crate::PyError {
    if err.kind != crate::PyErrorKind::TypeError {
        return err;
    }
    let note = format!("Cannot convert dictionary update sequence element #{idx} to a sequence");
    let _roots = pyre_object::gc_roots::push_roots();
    let mut err = err;
    let err_slot = err.pin(&_roots);
    match crate::baseobjspace::add_internal_exception_note(&mut err, &note) {
        Ok(()) => {
            err.reload(&_roots, err_slot);
            err
        }
        Err(note_err) => note_err,
    }
}

/// `dictmultiobject.py init_or_update` — shared by `dict.__init__`
/// and `dict.update`; `name` selects the error-message verb (`"dict"` vs
/// `"update"`). Stores resolve the subclass backing before writing, so a dict
/// subclass instance is updated through its backing dict rather than its own
/// (uninitialised) strategy slot.
///
/// `__pyre_kw__`-marked dict is the kwargs vehicle pyre's CALL_KW
/// emits for builtin callees (`call.rs`'s `pack_pyre_kwargs`).
pub fn dict_init_or_update(
    args: &[PyObjectRef],
    name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    // `dictmultiobject.py init_or_update` receives an `Arguments`
    // object whose `arguments_w` entries are roots in translated RPython.
    // Keep the flat Rust ABI's equivalent slots live before inspecting the
    // trailing kwargs vehicle or passing an operand to `update1`.
    let nargs = args.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::pin_roots(args);
    let rooted_args = (0..nargs)
        .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + i))
        .collect::<Vec<_>>();
    require_receiver(&rooted_args, "update")?;
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(&rooted_args);
    let has_kwargs = kwargs_dict.is_some();
    let kwargs_slot = _roots.pin_roots(&[kwargs_dict.unwrap_or(pyre_object::PY_NULL)]);
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at most 1 argument, got {}",
            positional.len() - 1
        )));
    }
    if positional.len() > 1 {
        dict_update1(
            pyre_object::gc_roots::shadow_stack_get(root_base),
            pyre_object::gc_roots::shadow_stack_get(root_base + 1),
        )?;
    }
    let backing = resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(root_base));
    if backing.is_null() {
        // A dict subclass declared with `__slots__` has no attribute storage
        // for its item backing (pyre keeps a dict subclass's items in an
        // instance attribute), so there is nowhere to merge into. Full
        // slotted-dict-subclass support needs intrinsic dict backing.
        return Ok(w_none());
    }
    let backing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(backing);
    if has_kwargs {
        let kwargs = _roots.get(kwargs_slot);
        unsafe {
            // Every pair stays rooted while an earlier store runs a key's
            // `__hash__` / `__eq__`.
            let items = pyre_object::w_dict_items(kwargs);
            let flat: Vec<PyObjectRef> = items.iter().flat_map(|&(k, v)| [k, v]).collect();
            let items_base = pyre_object::gc_roots::pin_roots(&flat);
            for i in 0..items.len() {
                let k = pyre_object::gc_roots::shadow_stack_get(items_base + 2 * i);
                if pyre_object::is_str(k)
                    && pyre_object::w_str_get_wtf8(k).as_str() == Ok("__pyre_kw__")
                {
                    continue;
                }
                dict_store_checked(
                    pyre_object::gc_roots::shadow_stack_get(backing_slot),
                    k,
                    pyre_object::gc_roots::shadow_stack_get(items_base + 2 * i + 1),
                )?;
            }
        }
    }
    Ok(w_none())
}

/// `dictmultiobject.py descr_update` → `init_or_update`; the verb in
/// the arity error is `update`.
pub fn dict_method_update(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "update")?;
    dict_init_or_update(args, "update")
}

/// `dictmultiobject.py update1` —
///
/// ```python
/// if (isinstance(w_data, W_DictMultiObject) and
///         space.is_w(
///             space.findattr(space.type(w_data), w_st_iter),
///             space.findattr(space.w_dict, w_st_iter))):
///     update1_dict_dict(space, w_dict, w_data)
/// ```
///
/// Returns True when `other` is either a real `dict` (no subclass)
/// or a `dict` subclass that hasn't shadowed `__iter__`.  Pyre's
/// `is_dict` already established the W_DictMultiObject side; this
/// helper performs the `findattr` identity check to keep
/// `__iter__`-overriding subclasses on the slow `keys()` path.
fn dict_subclass_uses_default_iter(other: PyObjectRef) -> bool {
    let Some(other_type) = crate::typedef::r#type(other) else {
        return false;
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
    if dict_type.is_null() {
        // No registered dict typeobject yet (init order quirk) —
        // degrade to "fast path" (treat as plain dict) to preserve
        // current behaviour.
        return true;
    }
    // Real `dict` type — no subclass at all, so __iter__ is by
    // definition unshadowed.
    if std::ptr::eq(other_type.as_ptr() as *const _, dict_type as *const _) {
        return true;
    }
    let other_iter =
        unsafe { crate::baseobjspace::lookup_in_type(other_type.as_ptr(), "__iter__") };
    let dict_iter = unsafe { crate::baseobjspace::lookup_in_type(dict_type, "__iter__") };
    match (other_iter, dict_iter) {
        (Some(a), Some(b)) => std::ptr::eq(a, b),
        _ => false,
    }
}

/// `dictmultiobject.py descr_pop` →
/// `strategy.pop(self, w_key, w_default)` — single-operation pop
/// via strategy dispatch (one hash).
pub fn dict_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "pop", 1)?;
    arity_at_most(args, "pop", 2)?;
    let nargs = args.len();
    // Rooted as `dict_method_get`: the removal hashes the key, and the default
    // is only consumed once that has run. Resolve the backing from the pinned
    // receiver so `args` is not live across that call.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let dict = resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(base));
    if !dict.is_null() {
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(dict);
        unsafe {
            match pyre_object::dictmultiobject::w_dict_pop_checked(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(base + 1),
            ) {
                Ok(Some(val)) => return Ok(val),
                Ok(None) => {}
                Err(_) => {
                    return Err(crate::baseobjspace::take_pending_dict_key_error(
                        pyre_object::gc_roots::shadow_stack_get(base + 1),
                    ));
                }
            }
        }
    }
    if nargs >= 3 {
        return Ok(pyre_object::gc_roots::shadow_stack_get(base + 2));
    }
    Err(crate::PyError::key_error_with_key(
        pyre_object::gc_roots::shadow_stack_get(base + 1),
    ))
}

/// `dictmultiobject.py` `W_DictMultiObject.descr_popitem`:
///
/// ```python
/// def descr_popitem(self, space):
///     try:
///         w_key, w_value = self.popitem()
///     except KeyError:
///         raise oefmt(space.w_KeyError, "dictionary is empty")
///     return space.newtuple([w_key, w_value])
/// ```
///
/// Keep the pre-dispatch empty guard: a receiver whose backing is null, or an
/// empty mapdict/module strategy, raises before strategy code can initialise or
/// touch backing storage.
pub fn dict_method_popitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "popitem")?;
    arity_no_args(args, "popitem")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
    }
    unsafe {
        if pyre_object::w_dict_len(dict) == 0 {
            return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
        }
        let (k, v) = pyre_object::dictmultiobject::w_dict_popitem(dict)
            .ok_or_else(|| crate::PyError::key_error("popitem(): dictionary is empty"))?;
        let _roots = pyre_object::gc_roots::push_roots();
        let pair_slot = pyre_object::gc_roots::pin_roots(&[k, v]);
        let k = pyre_object::gc_roots::shadow_stack_get(pair_slot);
        let v = pyre_object::gc_roots::shadow_stack_get(pair_slot + 1);
        Ok(pyre_object::w_tuple_new(vec![k, v]))
    }
}

/// `dictmultiobject.py descr_setdefault` →
/// `self.setdefault(w_key, w_default)` — delegates to
/// `strategy.setdefault` as a single atomic operation (one hash).
pub fn dict_method_setdefault(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "setdefault", 1)?;
    arity_at_most(args, "setdefault", 2)?;
    let nargs = args.len();
    // Hashing the key can collect; pin the receiver before resolving its
    // backing, then reload the key and default from those slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let dict = resolve_dict_backing(pyre_object::gc_roots::shadow_stack_get(base));
    if dict.is_null() {
        return Ok(if nargs >= 3 {
            pyre_object::gc_roots::shadow_stack_get(base + 2)
        } else {
            w_none()
        });
    }
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setdefault_checked(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            if nargs >= 3 {
                pyre_object::gc_roots::shadow_stack_get(base + 2)
            } else {
                w_none()
            },
        )
        .map_err(|_| {
            crate::baseobjspace::take_pending_dict_key_error(
                pyre_object::gc_roots::shadow_stack_get(base + 1),
            )
        })
    }
}

#[cfg(test)]
mod dict_method_tests {
    use super::*;

    use crate::test_hooks::install_hash_hook;

    fn init_dict_test_runtime() {
        crate::typedef::init_typeobjects();
        install_hash_hook();
    }

    fn assert_type_error<T: std::fmt::Debug>(result: Result<T, crate::PyError>) {
        let err = result.expect_err("operation should reject unhashable dict key");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(
            err.message_text(),
            "cannot use 'list' as a dict key (unhashable type: 'list')"
        );
    }

    #[test]
    fn dict_get_rejects_unhashable_key() {
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_get(&[dict, key]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setitem_rejects_unhashable_key_without_inserting() {
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(crate::baseobjspace::setitem(dict, key, w_int_new(1)));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setdefault_rejects_unhashable_key() {
        // `dictmultiobject.py EmptyDictStrategy.setdefault`:
        //   self.switch_to_correct_strategy(w_dict, w_key)
        //   w_dict.setitem(w_key, w_default)
        // `w_dict.setitem` hashes the key via the object strategy's
        // `space.hash_w`, so an unhashable key raises TypeError before
        // anything is stored — the dict stays empty.
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_setdefault(&[dict, key, w_int_new(1)]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setdefault_rejects_unhashable_key_on_a_typed_strategy() {
        // A dict that already holds one entry is on the strategy that entry's
        // key type chose, and `DictStrategy.setdefault` there is `getitem`
        // then `setitem`.  Both answer for a key of another type by switching
        // to the object strategy and asking again, so the key is hashed on the
        // way -- and an unhashable one has to be refused there rather than
        // stored under no hash at all.
        for (label, key, value) in [
            ("unicode", w_str_new("a"), w_int_new(1)),
            ("int", w_int_new(7), w_int_new(1)),
            (
                "bytes",
                pyre_object::bytesobject::w_bytes_from_bytes(b"a"),
                w_int_new(1),
            ),
        ] {
            init_dict_test_runtime();
            let dict = w_dict_new();
            crate::baseobjspace::setitem(dict, key, value).expect("seed the strategy");
            let unhashable = w_list_new(vec![]);

            assert_type_error(dict_method_setdefault(&[dict, unhashable, w_int_new(9)]));
            assert_eq!(unsafe { w_dict_len(dict) }, 1, "{label} strategy");
        }
    }

    #[test]
    fn dict_pop_empty_returns_default_without_hashing_key() {
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let default = w_int_new(42);

        let result = dict_method_pop(&[dict, key, default]).expect("default should be returned");
        assert_eq!(result, default);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_pop_empty_without_default_raises_keyerror_not_typeerror() {
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        let err = dict_method_pop(&[dict, key]).expect_err("missing key should raise KeyError");
        assert_eq!(err.kind, crate::PyErrorKind::KeyError);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_update_pairs_rejects_unhashable_key() {
        init_dict_test_runtime();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let pair = w_tuple_new(vec![key, w_int_new(1)]);
        let pairs = w_list_new(vec![pair]);

        assert_type_error(dict_method_update(&[dict, pairs]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }
}

// ── Tuple methods ────────────────────────────────────────────────────

/// tupleobject.py descr_index — tuple.index(value[, start[, stop]]).
pub fn tuple_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_tuple_receiver(args, "index", true)?;
    arity_at_least(args, "index", 1)?;
    arity_at_most(args, "index", 3)?;
    let tup = args[0];
    let value = args[1];
    // descr_index defaults: w_start=0, w_stop=maxint; unwrap_start_stop does
    // negative normalization and __index__ coercion.
    let size = unsafe { w_tuple_len(tup) } as i64;
    // A user __eq__ (or an __index__ on start/stop) may allocate and move the
    // tuple, search value, or the other bound. Publish all four first.
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::publish_roots(&[tup, value]);
        let start_obj = if args.len() >= 3 {
            args[2]
        } else {
            w_int_new(0)
        };
        let _ = pyre_object::gc_roots::publish_roots(&[start_obj]);
        let stop_obj = if args.len() >= 4 {
            args[3]
        } else {
            w_int_new(i64::MAX)
        };
        let _ = pyre_object::gc_roots::publish_roots(&[stop_obj]);
        pyre_object::gc_roots::normalize_roots(sp, 4);
        let (start, stop) = crate::sliceobject::unwrap_start_stop_not_none(
            size,
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
            pyre_object::gc_roots::shadow_stack_get(sp + 3),
        )?;
        let mut i = start.max(0);
        while i < stop {
            let tup = pyre_object::gc_roots::shadow_stack_get(sp);
            if i >= w_tuple_len(tup) as i64 {
                break;
            }
            if let Some(item) = w_tuple_getitem(tup, i) {
                let _elem_roots = pyre_object::gc_roots::push_roots();
                let item = pyre_object::gc_roots::pin_root(item);
                let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                if crate::baseobjspace::eq_w(item, value)? {
                    return Ok(w_int_new(i));
                }
            }
            i += 1;
        }
    }
    Err(crate::PyError::value_error(
        "tuple.index(x): x not in tuple",
    ))
}

/// PyPy: tupleobject.py descr_count — tuple.count(value)
pub fn tuple_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_tuple_receiver(args, "count", true)?;
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "tuple.count() takes exactly one argument ({} given)",
            args_given(args)
        )));
    }
    let tup = args[0];
    let value = args[1];
    let mut count: i64 = 0;
    // A user __eq__ may allocate and move the tuple / value across a minor
    // collection; root both and reload after each comparison.
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::pin_roots(&[tup, value]);
        let mut i = 0i64;
        loop {
            let tup = pyre_object::gc_roots::shadow_stack_get(sp);
            if i >= w_tuple_len(tup) as i64 {
                break;
            }
            if let Some(item) = w_tuple_getitem(tup, i) {
                let _elem_roots = pyre_object::gc_roots::push_roots();
                let item = pyre_object::gc_roots::pin_root(item);
                let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                if crate::baseobjspace::eq_w(item, value)? {
                    count += 1;
                }
            }
            i += 1;
        }
    }
    Ok(w_int_new(count))
}
