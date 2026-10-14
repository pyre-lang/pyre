//! Arguments objects.
//!
//! Line-by-line port of `pypy/interpreter/argument.py:Arguments`.
//! Surface ported: struct + `new` (PyPy `__init__`) + `firstarg` +
//! `_combine_wrapped` + `_jit_few_keywords` + `fixedunpack` +
//! `unpack` + `replace_arguments` + `prepend` + `_match_signature` +
//! `_match_keywords` + `_collect_keyword_args` + `_parse` +
//! `parse_into_scope` + `parse_obj` + `frompacked` + `topacked`.
//!
//! Pyre's legacy `call::call_callable` surface still takes a flat
//! `&[PyObjectRef]`.  Callers that know the keyword layout route
//! through `Arguments::with_kw` before entering the profiled-builtin
//! path; callers with only a flat positional slice use
//! `Arguments::positional_only`.  Both shortcuts now delegate to `new`
//! so the PyPy `__init__` invariant chain runs once.

use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8Buf;

/// The `%T` conversion `oefmt` applies, for the `*args` / `**kwargs` parity
/// messages below ("argument after * must be an iterable, not %T", "argument
/// after ** must be a mapping, not %T", "keywords must be strings, not '%T'")
/// — which are `oefmt(..., "%T", w_obj)` upstream.
use crate::error::type_name_of;

/// pypy/interpreter/argument.py `raise_type_error`.
///
/// ```python
/// @specialize.arg(2)
/// def raise_type_error(space, w_function, msg, *args):
///     if w_function is None:
///         raise oefmt(space.w_TypeError, msg, *args)
///     msg = "%s " + msg
///     raise oefmt(space.w_TypeError, msg, space.object_functionstr(w_function), *args)
/// ```
///
/// Returns a `PyError` of kind `TypeError` rather than raising via
/// `oefmt(space.w_TypeError, ...)`.  When `w_function` is `PY_NULL`
/// (PyPy's `None` sentinel) the message lands without prefix; otherwise
/// `object_functionstr` resolves a leading `"<qualname>() "` prefix.
///
/// Caller sites in `argument.py` (`_combine_starargs_wrapped`:99,
/// `_check_not_duplicate_kwargs`:415, `_do_combine_starstarargs_wrapped`:434
/// + 443) all use this helper to format diagnostic context.
///
/// PyPy's `oefmt(space.w_TypeError, "%s ..." , space.object_functionstr(...))`
/// instantiates the `%s` substitution at raise time; if
/// `object_functionstr` itself raises an async OperationError
/// (SystemExit / KeyboardInterrupt via `findattr`'s
/// `baseobjspace.py` re-raise), the async error preempts the
/// TypeError and propagates as the actual exception PyPy raises.
/// Pyre mirrors this by returning the async `PyError` verbatim
/// instead of building the TypeError prefix.
pub fn raise_type_error(
    w_function: PyObjectRef,
    msg: impl Into<rustpython_wtf8::Wtf8Buf>,
) -> crate::PyError {
    let msg = msg.into();
    if w_function.is_null() {
        return crate::PyError::type_error(msg);
    }
    match crate::baseobjspace::object_functionstr(w_function) {
        Ok(mut prefix) => {
            prefix.push_str(" ");
            prefix.push_wtf8(&msg);
            crate::PyError::type_error(prefix)
        }
        // Async findattr propagates as the actual raise (matches
        // PyPy `oefmt`'s argument-evaluation order: side effects of
        // `%s` formatting fire before the TypeError is constructed).
        Err(async_err) => async_err,
    }
}

/// pypy/interpreter/argument.py `contains_w_names`.
///
/// ```python
/// def contains_w_names(w_key, keys_w):
///     for w_other in keys_w:
///         if w_other.eq_w(w_key):
///             return True
///     return False
/// ```
///
/// PyPy: `w_other.eq_w(w_key)` is the typed unicode method
/// `pypy/objspace/std/unicodeobject.py`:
///
/// ```python
/// def eq_w(self, w_other):
///     # shortcut for UnicodeDictStrategy
///     assert isinstance(w_other, W_UnicodeObject)
///     return self._utf8 == w_other._utf8
/// ```
///
/// Both arms in `argument.py` only feed string keys into this helper
/// (`_do_combine_starstarargs_wrapped` stamps `assert isinstance(w_key,
/// space.UnicodeObjectCls)` after `text_w`, and `keyword_names_w` is a
/// parallel string-typed list), so the typed dispatch reduces to byte
/// equality of the underlying utf8.
///
/// The identity fast path `is_w` is preserved as the cheap shortcut PyPy gets
/// from CPython-style interned-string identity.  The typed fallback routes
/// through `W_UnicodeObject.eq_w`; using generic `space.eq_w` here would invoke
/// Python-level `__eq__` on hand-built Arguments and silently drift.
pub fn contains_w_names(w_key: PyObjectRef, keys_w: &[PyObjectRef]) -> bool {
    // `is_w` can build a bigint, so the key and the names are read from their
    // slots on every turn.
    let roots = pyre_object::gc_roots::push_roots();
    let key_slot = roots.pin_roots(&[w_key]);
    let keys_base = roots.pin_roots(keys_w);
    for i in 0..keys_w.len() {
        if crate::baseobjspace::is_w(roots.get(keys_base + i), roots.get(key_slot)) {
            return true;
        }
        let (w_other, w_key) = (roots.get(keys_base + i), roots.get(key_slot));
        unsafe {
            if pyre_object::is_str(w_other)
                && pyre_object::is_str(w_key)
                && pyre_object::w_str_eq_w(w_other, w_key)
            {
                return true;
            }
        }
    }
    false
}

/// pypy/interpreter/argument.py `_check_not_duplicate_kwargs`.
///
/// ```python
/// def _check_not_duplicate_kwargs(space, existingkeywords_w, keyword_names_w, keywords_w, w_function):
///     # looks quadratic, but the JIT should remove all of it nicely.
///     # Also, all the lists should be small
///     for w_key in keyword_names_w:
///         if contains_w_names(w_key, existingkeywords_w):
///             raise_type_error(space, w_function,
///                         "got multiple values for keyword argument '%S'",
///                         w_key)
/// ```
///
/// `keywords_w` is the parallel value list, kept in the signature for
/// upstream parity but unused in the duplicate check (only the names
/// matter).
pub fn check_not_duplicate_kwargs(
    existingkeywords_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    _keywords_w: &[PyObjectRef],
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    let Some((mut w_key, mut w_function)) =
        find_duplicate_kwarg(existingkeywords_w, keyword_names_w, w_function)
    else {
        return Ok(());
    };
    let key_repr = unsafe {
        if pyre_object::is_str(w_key) {
            pyre_object::w_str_get_wtf8(w_key).to_owned()
        } else {
            pyre_object::with_roots!(w_key, w_function => crate::display::py_str_wtf8(w_key))?
        }
    };
    Err(raise_type_error(
        w_function,
        crate::display::wtf8_format!("got multiple values for keyword argument '", key_repr, "'"),
    ))
}

/// The loop of `_check_not_duplicate_kwargs`: the first of `keyword_names_w`
/// that `existingkeywords_w` already holds, returned with `w_function`.
/// `contains_w_names` can collect, so both lists and the callable are read
/// from their slots.
fn find_duplicate_kwarg(
    existingkeywords_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    w_function: PyObjectRef,
) -> Option<(PyObjectRef, PyObjectRef)> {
    let existing_len = existingkeywords_w.len();
    let names_len = keyword_names_w.len();
    let roots = pyre_object::gc_roots::push_roots();
    let function_slot = roots.pin_roots(&[w_function]);
    let existing_base = roots.pin_roots(existingkeywords_w);
    let names_base = roots.pin_roots(keyword_names_w);
    let mut existing_now = vec![pyre_object::PY_NULL; existing_len];
    for i in 0..names_len {
        pyre_object::gc_roots::shadow_stack_copy_range(existing_base, &mut existing_now);
        if contains_w_names(roots.get(names_base + i), &existing_now) {
            return Some((roots.get(names_base + i), roots.get(function_slot)));
        }
    }
    None
}

/// pypy/interpreter/argument.py `_do_combine_starstarargs_wrapped`.
///
/// ```python
/// def _do_combine_starstarargs_wrapped(space, keys_w, w_starstararg, keyword_names_w,
///         keywords_w, existingkeywords_w, is_dict, w_function):
///     i = 0
///     seen = {}
///     for w_key in keys_w:
///         try:
///             key = space.text_w(w_key)
///         except OperationError as e:
///             if e.match(space, space.w_TypeError):
///                 raise_type_error(space, w_function,
///                             "keywords must be strings, not '%T'",
///                             w_key)
///             raise
///         else:
///             if ((existingkeywords_w and
///                  contains_w_names(w_key, existingkeywords_w)) or
///                 key in seen
///             ):
///                 raise_type_error(space, w_function,
///                             "got multiple values for keyword argument '%S'",
///                             w_key)
///         seen[key] = None
///         assert isinstance(w_key, space.UnicodeObjectCls)
///         keyword_names_w[i] = w_key
///         if is_dict:
///             # issue 2435: bug-to-bug compatibility with cpython.
///             from pypy.objspace.descroperation import dict_getitem
///             w_descr = dict_getitem(space)
///             w_value = space.get_and_call_function(w_descr, w_starstararg, w_key)
///         else:
///             w_value = space.getitem(w_starstararg, w_key)
///         keywords_w[i] = w_value
///         i += 1
/// ```
///
/// `_do_combine_starstarargs_wrapped` — when `is_dict` is true (an exact
/// `dict`, or a `dict` subclass whose `__iter__` is still `dict.__iter__`),
/// the value is read through `dict_getitem` on the original `w_key`.
/// The seen-set still records `text_w` (`key in seen`). The other arm is
/// `space.getitem`, so a mapping's `__getitem__` override runs.
pub fn do_combine_starstarargs_wrapped(
    keys_w: &[PyObjectRef],
    w_starstararg: PyObjectRef,
    keyword_names_w: &mut [PyObjectRef],
    keywords_w: &mut [PyObjectRef],
    existingkeywords_w: Option<&[PyObjectRef]>,
    is_dict: bool,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // argument.py `seen = {}` — Python dict with None values.  Map
    // to `HashMap<String, ()>` rather than `HashSet` so the shape
    // matches PyPy's `dict` (key membership tested via `in`, value slot
    // present-but-unread per `seen[key] = None` at line 446).
    let mut seen: std::collections::HashMap<rustpython_wtf8::Wtf8Buf, ()> =
        std::collections::HashMap::new();
    // The per-key value lookup below runs `__getitem__` (or the dict
    // descriptor, which hashes the key), so every turn is a collection point.
    // The key list, the mapping, the callable, the names the caller already
    // accepted and the pairs filled so far are native locals no root walker
    // updates, so publish the whole set and read each one back on the turn
    // that uses it.  The two output slices are written from the published
    // pairs once the loop is done.
    let _roots = pyre_object::gc_roots::push_roots();
    let existing_len = existingkeywords_w.map_or(0, |existing| existing.len());
    let mut live = Vec::with_capacity(2 + existing_len + keys_w.len());
    live.push(w_starstararg);
    live.push(w_function);
    if let Some(existing) = existingkeywords_w {
        live.extend_from_slice(existing);
    }
    live.extend_from_slice(keys_w);
    let root_base = pyre_object::gc_roots::pin_roots(&live);
    let existing_base = root_base + 2;
    let keys_base = existing_base + existing_len;
    let pairs_base = pyre_object::gc_roots::shadow_stack_len();
    for _ in 0..2 * keys_w.len() {
        let _ = pyre_object::gc_roots::pin_root(pyre_object::PY_NULL);
    }
    let w_starstararg = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let w_function = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    for i in 0..keys_w.len() {
        let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
        // argument.py — `key = space.text_w(w_key)`; raise TypeError
        // if w_key is not a string.  `_PyStack_UnpackDict` reports this
        // without the callable's qualname or the offending key's type, where
        // argument.py formats `"keywords must be strings, not '%T'"`
        // through `raise_type_error`.
        let key = unsafe {
            if !pyre_object::is_str(w_key) {
                return Err(crate::PyError::type_error("keywords must be strings"));
            }
            // `space.text_w` preserves lone surrogates as WTF-8 bytes; keep the
            // key in WTF-8 so a surrogate keyword name survives the seen-set and
            // the dict value read.
            pyre_object::w_str_get_wtf8(w_key).to_owned()
        };
        // argument.py:439-445 — duplicate check against existing kwargs +
        // already-seen names in this iteration.
        let already_in_existing = existing_len != 0 && {
            let existing_now: Vec<PyObjectRef> = (0..existing_len)
                .map(|j| pyre_object::gc_roots::shadow_stack_get(existing_base + j))
                .collect();
            contains_w_names(w_key, &existing_now)
        };
        if already_in_existing || seen.contains_key(&key) {
            return Err(raise_type_error(
                w_function(),
                format!("got multiple values for keyword argument '{key}'"),
            ));
        }
        // `contains_w_names` can collect; the key is read back from its slot.
        let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
        // argument.py — `keyword_names_w[i] = w_key`.
        pyre_object::gc_roots::shadow_stack_set(pairs_base + 2 * i, w_key);
        // argument.py:449-457 — value lookup.
        let w_value = if is_dict {
            // `_do_combine_starstarargs_wrapped` calls `dict_getitem` with
            // the original `w_key`. A str subclass is probed as that object:
            // its `__hash__` / `__eq__` may differ from `str`, so the WTF-8
            // text recorded for `key in seen` is the wrong probe. A dict
            // subclass keeps its entries in a backing dict; exact `dict` is
            // that backing already. A miss is KeyError, the same answer
            // `dict.__getitem__` gives when the key disappeared mid-iteration.
            let backing = crate::type_methods::resolve_dict_backing(w_starstararg());
            let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
            let direct = if backing.is_null() {
                Ok(None)
            } else {
                unsafe { pyre_object::dictmultiobject::w_dict_lookup_checked(backing, w_key) }
            };
            // The lookup hashes `w_key` and can collect. Read the key back
            // before a hash error is wrapped around it.
            let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
            match direct {
                Ok(Some(v)) => v,
                Ok(None) => return Err(crate::PyError::key_error(format!("'{key}'"))),
                Err(_) => return Err(crate::baseobjspace::take_pending_dict_key_error(w_key)),
            }
        } else {
            // argument.py:457 — `w_value = space.getitem(...)`.
            crate::baseobjspace::getitem(w_starstararg(), w_key)?
        };
        pyre_object::gc_roots::shadow_stack_set(pairs_base + 2 * i + 1, w_value);
        // argument.py `seen[key] = None` — record key, value slot
        // is unread.
        seen.insert(key, ());
    }
    for i in 0..keys_w.len() {
        keyword_names_w[i] = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i);
        keywords_w[i] = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i + 1);
    }
    Ok(())
}

/// pypy/interpreter/argument.py `_combine_starargs_wrapped`.
///
/// ```python
/// def _combine_starargs_wrapped(self, w_stararg, w_function=None):
///     # unpack the * arguments
///     space = self.space
///     try:
///         args_w = space.fixedview(w_stararg)
///     except OperationError as e:
///         if (e.match(space, space.w_TypeError) and
///                 not space.is_iterable(w_stararg)):
///             raise_type_error(space, w_function,
///                         "argument after * must be an iterable, not %T",
///                         w_stararg)
///         raise
///     self.arguments_w = self.arguments_w + args_w
/// ```
///
/// Operates on a borrowed `&mut Vec<PyObjectRef>` so the
/// `Arguments::_combine_wrapped` constructor flow can extend the
/// owned `self.arguments_w` Vec in place.
pub fn combine_starargs_wrapped(
    arguments_w: &mut Vec<PyObjectRef>,
    w_stararg: PyObjectRef,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // `fixedview` iterates the star argument, which runs Python — a generator
    // body or a subtype `__iter__`.  The words the caller already put in
    // `arguments_w` and these by-value parameters are native copies no root
    // walker updates, so they are published before the unpack and read back
    // after it.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::publish_roots(&arguments_w[..]);
    let star_slot = pyre_object::gc_roots::publish_roots(&[w_stararg, w_function]);
    pyre_object::gc_roots::normalize_roots(args_base, arguments_w.len() + 2);
    let w_stararg = || pyre_object::gc_roots::shadow_stack_get(star_slot);
    let w_function = || pyre_object::gc_roots::shadow_stack_get(star_slot + 1);
    match crate::baseobjspace::fixedview(w_stararg(), -1) {
        Ok(args_w) => {
            pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut arguments_w[..]);
            // argument.py — `self.arguments_w = self.arguments_w + args_w`.
            arguments_w.extend(args_w);
            Ok(())
        }
        Err(e) => {
            // argument.py:97-103 — TypeError + non-iterable surfaces
            // `"argument after * must be an iterable, not %T"`; other
            // errors propagate.
            if e.kind == crate::PyErrorKind::TypeError
                && !crate::baseobjspace::is_iterable(w_stararg())
            {
                let tp = type_name_of(w_stararg());
                Err(raise_type_error(
                    w_function(),
                    format!("argument after * must be an iterable, not {tp}"),
                ))
            } else {
                Err(e)
            }
        }
    }
}

/// pypy/interpreter/argument.py `_combine_starstarargs_wrapped`.
///
/// ```python
/// def _combine_starstarargs_wrapped(self, w_starstararg, w_function=None):
///     # unpack the ** arguments
///     space = self.space
///     keyword_names_w, values_w = space.view_as_kwargs(w_starstararg)
///     if keyword_names_w is not None: # this path also taken for empty dicts
///         if self.keyword_names_w is None:
///             self.keyword_names_w = keyword_names_w
///             self.keywords_w = values_w
///         else:
///             _check_not_duplicate_kwargs(
///                 self.space, self.keyword_names_w, keyword_names_w, values_w,
///                 w_function)
///             self.keyword_names_w = self.keyword_names_w + keyword_names_w
///             self.keywords_w = self.keywords_w + values_w
///         return
///     is_dict = False
///     if space.isinstance_w(w_starstararg, space.w_dict):
///         w_st_iter = space.newtext("__iter__")
///         is_dict = space.is_w(
///                 space.findattr(space.type(w_starstararg), w_st_iter),
///                 space.findattr(space.w_dict, w_st_iter))
///         keys_w = space.unpackiterable(w_starstararg)
///     else:
///         try:
///             w_keys = space.call_method(w_starstararg, "keys")
///         except OperationError as e:
///             if e.match(space, space.w_AttributeError):
///                 raise_type_error(space, w_function,
///                             "argument after ** must be a mapping, not %T",
///                             w_starstararg)
///             raise
///         keys_w = space.unpackiterable(w_keys)
///     keywords_w = [None] * len(keys_w)
///     keyword_names_w = [None] * len(keys_w)
///     _do_combine_starstarargs_wrapped(...)
///     ...merge into self.keyword_names_w / self.keywords_w...
/// ```
///
/// `baseobjspace::view_as_kwargs` supplies a direct names-and-values view for
/// exact dicts whose keys are all strings. Other mappings use the `keys()` and
/// item-lookup path below, preserving mapping overrides and their exceptions.
pub fn combine_starstarargs_wrapped(
    keyword_names_out: &mut Vec<PyObjectRef>,
    keywords_out: &mut Vec<PyObjectRef>,
    w_starstararg: PyObjectRef,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // Deriving the key list reaches `__getattribute__` and the mapping's own
    // `keys()`, both of which run Python, and `w_starstararg` is read again
    // afterwards to fetch each value.  Publish the mapping and the callable so
    // that later read is of the live object, not the address they had on entry.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::publish_roots(&[w_starstararg, w_function]);
    let names_base = pyre_object::gc_roots::publish_roots(&keyword_names_out[..]);
    let values_base = pyre_object::gc_roots::publish_roots(&keywords_out[..]);
    pyre_object::gc_roots::normalize_roots(
        root_base,
        2 + keyword_names_out.len() + keywords_out.len(),
    );
    let w_starstararg = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let w_function = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let refresh_out = |names: &mut Vec<PyObjectRef>, values: &mut Vec<PyObjectRef>| {
        pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut names[..]);
        pyre_object::gc_roots::shadow_stack_copy_range(values_base, &mut values[..]);
    };
    // argument.py:109 — fast path via view_as_kwargs.  Both halves of
    // the tuple are `Option`; PyPy's base default is `(None, None)`
    // while the kwargsdict-aware override returns `(Some, Some)`.
    let (fast_names, fast_values) = crate::baseobjspace::view_as_kwargs(w_starstararg());
    if let Some(names) = fast_names {
        let values = fast_values
            .expect("baseobjspace.py:1159 view_as_kwargs returns matching Some/Some or None/None");
        let mut names = names;
        let mut values = values;
        refresh_out(keyword_names_out, keywords_out);
        // argument.py:111-119 — merge with optional duplicate check.
        if !keyword_names_out.is_empty() {
            // The duplicate check compares through `is_w`, which can collect.
            let fast_names_base = pyre_object::gc_roots::pin_roots(&names);
            let fast_values_base = pyre_object::gc_roots::pin_roots(&values);
            check_not_duplicate_kwargs(keyword_names_out, &names, &values, w_function())?;
            refresh_out(keyword_names_out, keywords_out);
            pyre_object::gc_roots::shadow_stack_copy_range(fast_names_base, &mut names[..]);
            pyre_object::gc_roots::shadow_stack_copy_range(fast_values_base, &mut values[..]);
        }
        keyword_names_out.extend(names);
        keywords_out.extend(values);
        return Ok(());
    }
    // argument.py:121-128 — slow path: derive keys list, dispatch on
    // dict-fast-path vs generic mapping.
    //
    // ```python
    // is_dict = False
    // if space.isinstance_w(w_starstararg, space.w_dict):
    //     w_st_iter = space.newtext("__iter__")
    //     # bug-to-bug compatibility: CPython ignores __getitem__
    //     # overrides in dict subclasses if there is no __iter__
    //     is_dict = space.is_w(
    //             space.findattr(space.type(w_starstararg), w_st_iter),
    //             space.findattr(space.w_dict, w_st_iter))
    // ```
    //
    // The double check (isinstance + `__iter__` identity) ensures a
    // dict subclass that overrides `__iter__` falls through to the
    // generic `__getitem__` path, while subclasses that don't override
    // it use the direct dict-storage bypass.
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    // `gettypeobject` returns null before `init_typeobjects()` runs
    // (early bootstrap or unit tests).  Fall back to the exact-type
    // `pyre_object::is_dict` check in that case — pyre has no dict
    // subclass yet, so the answer matches PyPy's
    // `isinstance_w(w_starstararg, w_dict)` for every reachable
    // input.  Once `init_typeobjects` always runs before this code
    // path, the fallback collapses to dead code.
    let is_dict = unsafe {
        let isinst = if w_dict_type.is_null() {
            pyre_object::is_dict(w_starstararg())
        } else {
            crate::baseobjspace::isinstance_w(w_starstararg(), w_dict_type)
        };
        if !isinst {
            false
        } else {
            // `argument.py` — `space.findattr(space.type(...),
            // w_st_iter)`: the `getattr` is issued on the type object
            // itself, going through the metaclass / descriptor lookup
            // chain, so a raising `__getattribute__` on the metaclass
            // surfaces here instead of reading as "no `__iter__`".
            let w_obj_type = crate::typedef::r#type(w_starstararg())
                .map_or(pyre_object::PY_NULL, |p| p.as_ptr());
            let lhs = if w_obj_type.is_null() {
                None
            } else {
                crate::baseobjspace::findattr(w_obj_type, "__iter__")?
            };
            // `findattr` runs `__getattribute__`: the dict type is fetched
            // again and `lhs` is compared by identity after it.
            let mut lhs = lhs.unwrap_or(pyre_object::PY_NULL);
            let mut w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
            let rhs = if w_dict_type.is_null() {
                None
            } else {
                pyre_object::with_roots!(lhs, w_dict_type =>
                    crate::baseobjspace::findattr(w_dict_type, "__iter__"))?
            };
            // `space.is_w` is pointer identity.
            match (!lhs.is_null(), rhs) {
                (true, Some(b)) => crate::baseobjspace::is_w(lhs, b),
                (false, None) => true,
                _ => false,
            }
        }
    };
    let keys_w: Vec<PyObjectRef> = if is_dict {
        crate::baseobjspace::unpackiterable(w_starstararg(), -1)?
    } else {
        // argument.py:131-138 — the `try` covers the whole
        // `space.call_method(w_starstararg, "keys")`, so an
        // AttributeError raised either while finding `keys` OR inside
        // the keys() call is converted to the mapping TypeError below.
        let w_keys_meth = match crate::baseobjspace::getattr_str(w_starstararg(), "keys") {
            Ok(m) => m,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                let tp = type_name_of(w_starstararg());
                return Err(raise_type_error(
                    w_function(),
                    format!("argument after ** must be a mapping, not {tp}"),
                ));
            }
            Err(e) => return Err(e),
        };
        let w_keys = match crate::call::call_function_impl_result(w_keys_meth, &[]) {
            Ok(w) => w,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                let tp = type_name_of(w_starstararg());
                return Err(raise_type_error(
                    w_function(),
                    format!("argument after ** must be a mapping, not {tp}"),
                ));
            }
            Err(e) => return Err(e),
        };
        crate::baseobjspace::unpackiterable(w_keys, -1)?
    };
    // argument.py:140-141 — pre-allocate parallel name/value buffers.
    let n = keys_w.len();
    let mut keyword_names_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; n];
    let mut keywords_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; n];
    // argument.py:142-144 — slow-path body fills name/value buffers,
    // checking for duplicates against keyword_names_out (existing).
    refresh_out(keyword_names_out, keywords_out);
    let existing: Option<&[PyObjectRef]> = if keyword_names_out.is_empty() {
        None
    } else {
        Some(keyword_names_out.as_slice())
    };
    do_combine_starstarargs_wrapped(
        &keys_w,
        w_starstararg(),
        &mut keyword_names_w,
        &mut keywords_w,
        existing,
        is_dict,
        w_function(),
    )?;
    // argument.py:145-150 — merge into the running output buffers.
    refresh_out(keyword_names_out, keywords_out);
    keyword_names_out.extend(keyword_names_w);
    keywords_out.extend(keywords_w);
    Ok(())
}

/// `pypy/interpreter/argument.py class Arguments`.
///
/// PyPy fields (argument.py:34-53):
/// ```text
/// self.space            -- always available; pyre passes context implicitly
/// self.arguments_w      -- list[w_obj]
/// self.keyword_names_w  -- list[w_text] or None
/// self.keywords_w       -- list[w_obj]   or None
/// self._jit_few_keywords -- bool, JIT unroll hint (argument.py:50)
/// self.methodcall       -- bool flag (argument.py:53)
/// ```
///
/// `w_stararg`, `w_starstararg`, `w_function` are constructor inputs
/// that PyPy's `_combine_wrapped` (argument.py) expands into
/// `arguments_w` / `keyword_names_w` / `keywords_w` at construction
/// time.  They are NOT stored as instance state in PyPy — only their
/// expanded form is.
///
/// Owns its `arguments_w` / `keyword_names_w` / `keywords_w` Vec
/// storage line-by-line with PyPy: `__init__` assigns `self.arguments_w =
/// args_w` (a list), and `_combine_starargs_wrapped` /
/// `_combine_starstarargs_wrapped` mutate the list via `+= args_w`.
/// Pyre's earlier borrowed-slice shape forced star expansion to
/// happen at the call site (because the slice could not grow);
/// owning the Vec lets `_combine_wrapped` invoke
/// `combine_starargs_wrapped` / `combine_starstarargs_wrapped`
/// in-place, matching the upstream constructor flow.
///
/// TODO(justified): PyPy stores RPython list
/// references — the same mutable list object can be aliased between
/// the caller's `args` and `Arguments.arguments_w`.  Pyre owns
/// `Vec<PyObjectRef>` and clones at construction.  Rust's borrow
/// checker forbids `replace_arguments` (argument.py) and
/// `_combine_starargs_wrapped` (argument.py) from building a
/// new `Arguments` that aliases `&self.arguments_w` while the
/// existing instance still holds a `&mut` to it.  Wrapping in
/// `Rc<RefCell<Vec<PyObjectRef>>>` would restore the alias shape
/// but adds runtime refcount + per-access borrow check overhead and
/// would diverge from pyre's owned-Vec convention used elsewhere
/// (`PyFrame.locals_w`, `Signature.argnames`, etc.).  Functional
/// consequence is identical — PyPy's call paths consume `Arguments`
/// before mutating the source, so post-construction mutations to
/// either side never propagate either way.
pub struct Arguments {
    /// argument.py:36 `self.arguments_w = args_w`.
    pub arguments_w: Vec<PyObjectRef>,
    /// argument.py:38 `self.keyword_names_w = keyword_names_w` (`None` allowed).
    pub keyword_names_w: Option<Vec<PyObjectRef>>,
    /// argument.py:39 `self.keywords_w = keywords_w` (`None` allowed,
    /// must be parallel to `keyword_names_w` when present —
    /// argument.py `assert len(keywords_w) == len(keyword_names_w)`).
    pub keywords_w: Option<Vec<PyObjectRef>>,
    /// argument.py `__init__`: `self._jit_few_keywords = self.keyword_names_w
    /// is None or jit.isconstant(len(self.keyword_names_w))`.
    /// `unpack` reads it through `look_inside_iff`; `_match_keywords` /
    /// `_collect_keyword_args` take the same flag as `jiton` upstream.
    pub jit_few_keywords: bool,
    /// argument.py `self.methodcall = methodcall`.  Default `false`
    /// for the `positional_only` / `with_kw` shortcuts; the future
    /// CALL_METHOD opcode port should set it `true`.
    pub methodcall: bool,
}

/// argument.py `unpack`: `@jit.look_inside_iff(lambda self: self._jit_few_keywords)`.
fn unpack_iff(arguments: &Arguments) -> bool {
    arguments.jit_few_keywords
}

impl Arguments {
    /// pypy/interpreter/argument.py `__init__` (full port).
    ///
    /// ```python
    /// def __init__(self, space, args_w, keyword_names_w=None,
    ///              keywords_w=None, w_stararg=None, w_starstararg=None,
    ///              methodcall=False, w_function=None):
    ///     self.space = space
    ///     assert isinstance(args_w, list)
    ///     self.arguments_w = args_w
    ///     self.keyword_names_w = keyword_names_w
    ///     self.keywords_w = keywords_w
    ///     if keyword_names_w is not None:
    ///         assert keywords_w is not None
    ///         assert len(keywords_w) == len(keyword_names_w)
    ///         make_sure_not_resized(self.keyword_names_w)
    ///         make_sure_not_resized(self.keywords_w)
    ///     make_sure_not_resized(self.arguments_w)
    ///     self._combine_wrapped(w_stararg, w_starstararg, w_function)
    ///     self._jit_few_keywords = self.keyword_names_w is None or jit.isconstant(len(self.keyword_names_w))
    ///     self.methodcall = methodcall
    /// ```
    ///
    /// `_jit_few_keywords` is a JIT-time elidability hint —
    /// `jit.isconstant(...)` is true when the trace recorder sees a
    /// fixed length.  `unpack`'s `look_inside_iff` reads that field.
    ///
    /// `space` is implicit in pyre (carried by call sites).
    /// `_combine_wrapped(w_stararg, w_starstararg, w_function)` runs
    /// the star expansion in-place against the freshly-cloned Vec
    /// storage so that `Arguments::new(.., w_stararg=Some(w), ..)`
    /// behaves like PyPy's `Arguments(space, args_w, ..., w_stararg=w)`
    /// constructor (the helpers it depends on — `space.fixedview`,
    /// `space.view_as_kwargs`, `space.unpackiterable`,
    /// `space.is_iterable`, `space.call_method` — landed in
    /// Pre-B.1/Pre-B.2/Pre-B.3).  Returns a `PyError` propagated from
    /// the helpers when the star expansion fails (e.g. non-iterable
    /// `*` arg).
    pub fn new(
        args_w: &[PyObjectRef],
        keyword_names_w: Option<&[PyObjectRef]>,
        keywords_w: Option<&[PyObjectRef]>,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
        methodcall: bool,
        w_function: Option<PyObjectRef>,
    ) -> Result<Self, crate::PyError> {
        // argument.py:40-44 — keyword_names_w / keywords_w invariant.
        if let (Some(names), Some(values)) = (keyword_names_w, keywords_w) {
            debug_assert_eq!(names.len(), values.len());
        } else {
            debug_assert!(
                keyword_names_w.is_none() && keywords_w.is_none(),
                "keyword_names_w and keywords_w must agree on Some/None"
            );
        }
        let mut arguments = Self {
            arguments_w: args_w.to_vec(),
            keyword_names_w: keyword_names_w.map(|s| s.to_vec()),
            keywords_w: keywords_w.map(|s| s.to_vec()),
            // argument.py `__init__` writes `_jit_few_keywords` after
            // `_combine_wrapped`; seed the same formula so the field is
            // defined before that helper runs.
            jit_few_keywords: match keyword_names_w {
                None => true,
                Some(names) => majit_rlib::jit::isconstant(&names.len()),
            },
            methodcall,
        };
        let w_function = w_function.unwrap_or(pyre_object::PY_NULL);
        arguments._combine_wrapped(w_stararg, w_starstararg, w_function)?;
        // argument.py — recompute after `_combine_wrapped`, since
        // that helper may have grown `keyword_names_w`.
        arguments.jit_few_keywords = match arguments.keyword_names_w.as_deref() {
            None => true,
            Some(names) => majit_rlib::jit::isconstant(&names.len()),
        };
        Ok(arguments)
    }

    /// pypy/interpreter/argument.py `_combine_wrapped`.
    ///
    /// ```python
    /// def _combine_wrapped(self, w_stararg, w_starstararg, w_function=None):
    ///     "unpack the *arg and **kwd into arguments_w and keywords_w"
    ///     if w_stararg is not None:
    ///         self._combine_starargs_wrapped(w_stararg, w_function)
    ///     if w_starstararg is not None:
    ///         self._combine_starstarargs_wrapped(w_starstararg, w_function)
    /// ```
    ///
    /// Routes through the free-fn helpers `combine_starargs_wrapped` /
    /// `combine_starstarargs_wrapped` defined above (Phase A.1) which
    /// operate on `&mut Vec<PyObjectRef>` storage in-place.
    fn _combine_wrapped(
        &mut self,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
        w_function: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // Unpacking `*` runs Python, so the mapping the `**` arm has yet to
        // read, the callable and the positional words the first arm leaves in
        // `arguments_w` are published here and read back where the second arm
        // uses them; none of the three sits in storage a root walker updates.
        let _roots = pyre_object::gc_roots::push_roots();
        let has_starstar = w_starstararg.is_some();
        let root_base = pyre_object::gc_roots::pin_roots(&[
            w_starstararg.unwrap_or(pyre_object::PY_NULL),
            w_function,
        ]);
        let w_function = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        // argument.py — `if w_stararg is not None: self._combine_starargs_wrapped(...)`.
        if let Some(w_star) = w_stararg {
            let w_function = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
            combine_starargs_wrapped(&mut self.arguments_w, w_star, w_function)?;
        }
        // argument.py — `if w_starstararg is not None: self._combine_starstarargs_wrapped(...)`.
        // Materialize keyword_names_w / keywords_w as empty Vecs if the
        // field is currently None so the helper has a buffer to extend
        // (mirrors argument.py:111-112 `if self.keyword_names_w is None:
        // self.keyword_names_w = keyword_names_w`).  Reassign Some after
        // the helper runs whether or not it produced any keys, mirroring
        // PyPy's unconditional `self.keyword_names_w = ...` at lines
        // 145-150.
        if has_starstar {
            let args_base = pyre_object::gc_roots::pin_roots(&self.arguments_w[..]);
            let mut names = self.keyword_names_w.take().unwrap_or_default();
            let mut values = self.keywords_w.take().unwrap_or_default();
            combine_starstarargs_wrapped(
                &mut names,
                &mut values,
                pyre_object::gc_roots::shadow_stack_get(root_base),
                pyre_object::gc_roots::shadow_stack_get(root_base + 1),
            )?;
            pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut self.arguments_w[..]);
            self.keyword_names_w = Some(names);
            self.keywords_w = Some(values);
        }
        Ok(())
    }

    /// pypy/interpreter/argument.py `__init__` (positional-only shortcut).
    ///
    /// Used for call surfaces that have only positional args (no
    /// kwargs, no star expansion).  Cannot fail because there are no
    /// star args to unpack.
    #[inline]
    pub fn positional_only(args_w: &[PyObjectRef]) -> Self {
        Self::new(args_w, None, None, None, None, false, None)
            .expect("positional_only cannot fail without star args")
    }

    /// pypy/interpreter/argument.py `__init__` (positional + kwargs shortcut).
    ///
    /// `keyword_names_w` and `keywords_w` are parallel slices
    /// (argument.py `assert len(keywords_w) == len(keyword_names_w)`).
    /// Callers with both positional and kwargs (e.g. the
    /// `call.rs:call_with_kwargs` builtin path) use this to keep
    /// the kwargs separated from `arguments_w`, so `firstarg()`
    /// returns `arguments_w[0]` (or `None`) rather than surfacing
    /// the trailing kwargs dict that pyre's flat call surface
    /// otherwise appends to the merged slice.  Cannot fail because
    /// there are no star args to unpack.
    #[inline]
    pub fn with_kw(
        args_w: &[PyObjectRef],
        keyword_names_w: &[PyObjectRef],
        keywords_w: &[PyObjectRef],
    ) -> Self {
        Self::new(
            args_w,
            Some(keyword_names_w),
            Some(keywords_w),
            None,
            None,
            false,
            None,
        )
        .expect("with_kw cannot fail without star args")
    }

    /// pypy/interpreter/argument.py `__init__` shortcut with the
    /// `methodcall` flag explicit.  Use when the caller has both kwargs
    /// and the `methodcall` flag (e.g. CALL_METHOD lowering); when
    /// methodcall is false, `with_kw` is the lighter alternative.
    /// Cannot fail because there are no star args to unpack.
    #[inline]
    pub fn full(
        args_w: &[PyObjectRef],
        keyword_names_w: Option<&[PyObjectRef]>,
        keywords_w: Option<&[PyObjectRef]>,
        methodcall: bool,
    ) -> Self {
        Self::new(
            args_w,
            keyword_names_w,
            keywords_w,
            None,
            None,
            methodcall,
            None,
        )
        .expect("full cannot fail without star args")
    }

    /// pypy/interpreter/argument.py `fixedunpack`.
    ///
    /// ```python
    /// def fixedunpack(self, argcount):
    ///     """The simplest argument parsing: get the 'argcount' arguments,
    ///     or raise a real ValueError if the length is wrong."""
    ///     if self.keyword_names_w:
    ///         raise ValueError("no keyword arguments expected")
    ///     if len(self.arguments_w) > argcount:
    ///         raise ValueError("too many arguments (%d expected)" % argcount)
    ///     elif len(self.arguments_w) < argcount:
    ///         raise ValueError("not enough arguments (%d expected)" % argcount)
    ///     return self.arguments_w
    /// ```
    ///
    /// Returns a borrowed slice instead of cloning (PyPy returns the
    /// list directly too — same borrow semantics as Python's list
    /// reference return).  Error variant carries the formatted PyPy
    /// message including `argcount` per argument.py:159-161.
    pub fn fixedunpack(&self, argcount: usize) -> Result<&[PyObjectRef], String> {
        if self
            .keyword_names_w
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            return Err("no keyword arguments expected".to_string());
        }
        if self.arguments_w.len() > argcount {
            return Err(format!("too many arguments ({argcount} expected)"));
        }
        if self.arguments_w.len() < argcount {
            return Err(format!("not enough arguments ({argcount} expected)"));
        }
        Ok(&self.arguments_w)
    }

    /// argument.py:164-168 — line-by-line port:
    /// ```python
    /// def firstarg(self):
    ///     "Return the first argument for inspection."
    ///     if self.arguments_w:
    ///         return self.arguments_w[0]
    ///     return None
    /// ```
    #[inline]
    pub fn firstarg(&self) -> Option<PyObjectRef> {
        match self.arguments_w.first() {
            Some(&w) if !w.is_null() => Some(w),
            _ => None,
        }
    }

    /// pypy/interpreter/argument.py `unpack`.
    ///
    /// ```python
    /// @jit.look_inside_iff(lambda self: self._jit_few_keywords)
    /// def unpack(self): # slowish
    ///     "Return a ([w1,w2...], {'kw':w3...}) pair."
    ///     kwds_w = {}
    ///     if self.keyword_names_w:
    ///         for i in range(len(self.keyword_names_w)):
    ///             kwds_w[self.space.text_w(self.keyword_names_w[i])] = self.keywords_w[i]
    ///     return self.arguments_w, kwds_w
    /// ```
    ///
    /// PyPy returns a Python interpreter-level dict (RPython `{}`),
    /// whose semantics are: keys overwrite on duplicate (last-write-
    /// wins), no insertion-order guarantee.  Pyre returns a
    /// `HashMap<Wtf8Buf, PyObjectRef>` to preserve the overwrite
    /// behaviour exactly — abnormal/hand-built duplicate keyword
    /// names collapse to the last value as PyPy does.  `arguments_w`
    /// is cloned because the storage is owned (PyPy returns the list
    /// reference; Rust's borrow checker forbids returning `&Vec`
    /// alongside a freshly-built second value without lifetime
    /// gymnastics, and `unpack` is documented as "slowish").
    ///
    /// `space.text_w` / `W_UnicodeObject.text_w` returns `self._utf8`,
    /// so a lone surrogate stays in the buffer.  Routing the name
    /// through `str_utf8_w` would raise `UnicodeEncodeError` before
    /// any caller that unpacks then ignores `kwds_w`
    /// (`descr_new_base_exception`) or only tests emptiness
    /// (`W_OSError.descr_new`) could run.
    ///
    /// Returns `Err(PyError(TypeError))` if any keyword name is not a
    /// string — PyPy's `space.text_w(w_name)` raises TypeError in that
    /// case, and `unpack`'s upstream consumers propagate the error via
    /// `OperationError`.
    #[majit_macros::look_inside_iff(unpack_iff)]
    pub fn unpack(
        &self,
    ) -> Result<
        (
            Vec<PyObjectRef>,
            std::collections::HashMap<Wtf8Buf, PyObjectRef>,
        ),
        crate::PyError,
    > {
        // `text_w` / the result clone can collect. Publish every live list
        // before the first `normalize_roots` so a safepoint cannot rewrite
        // an unpinned sibling slice (`gc_roots::publish_roots`).
        let _roots = pyre_object::gc_roots::push_roots();
        let args_n = self.arguments_w.len();
        let args_base = pyre_object::gc_roots::publish_roots(&self.arguments_w);
        let (names_n, names_base, values_base) =
            match (self.keyword_names_w.as_deref(), self.keywords_w.as_deref()) {
                (Some(names), Some(values)) => {
                    let names_base = pyre_object::gc_roots::publish_roots(names);
                    let values_base = pyre_object::gc_roots::publish_roots(values);
                    (names.len(), Some(names_base), Some(values_base))
                }
                _ => (0, None, None),
            };
        pyre_object::gc_roots::normalize_roots(args_base, args_n);
        if let (Some(names_base), Some(values_base)) = (names_base, values_base) {
            pyre_object::gc_roots::normalize_roots(names_base, names_n);
            pyre_object::gc_roots::normalize_roots(values_base, names_n);
        }
        let mut kwds_w: std::collections::HashMap<Wtf8Buf, PyObjectRef> =
            std::collections::HashMap::new();
        if let (Some(names_base), Some(values_base)) = (names_base, values_base) {
            for i in 0..names_n {
                let w_name = pyre_object::gc_roots::shadow_stack_get(names_base + i);
                let w_value = pyre_object::gc_roots::shadow_stack_get(values_base + i);
                let key = unsafe {
                    if pyre_object::is_str(w_name) {
                        // argument.py `unpack` — `space.text_w(...)`.
                        // `W_UnicodeObject.text_w` returns `self._utf8`.
                        pyre_object::w_str_get_wtf8(w_name).to_owned()
                    } else {
                        // argument.py `unpack` — `space.text_w(...)`. PyPy's
                        // `_typed_unwrap_error` (baseobjspace.py)
                        // raises `TypeError("expected str, got %T object")`.
                        let tp = type_name_of(w_name);
                        return Err(crate::PyError::type_error(format!(
                            "expected str, got {tp} object",
                        )));
                    }
                };
                // argument.py `unpack` — `kwds_w[key] = ...` overwrites on duplicate.
                kwds_w.insert(key, w_value);
            }
        }
        let arguments_w = (0..args_n)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(args_base + i))
            .collect();
        Ok((arguments_w, kwds_w))
    }

    /// pypy/interpreter/argument.py `replace_arguments`.
    ///
    /// ```python
    /// def replace_arguments(self, args_w):
    ///     "Return a new Arguments with a args_w as positional arguments."
    ///     return Arguments(self.space, args_w, self.keyword_names_w, self.keywords_w)
    /// ```
    ///
    /// PyPy invokes the bare 4-arg `Arguments(space, args_w,
    /// keyword_names_w, keywords_w)` constructor — `methodcall`
    /// defaults to `False` (argument.py:33).  Carrying
    /// `self.methodcall` forward would surface a stale "Did you
    /// forget 'self'" hint at the next signature mismatch.
    /// PyPy passes the existing keyword lists straight through (they
    /// are immutable list references in RPython); pyre clones the
    /// owned Vecs because the new instance needs its own storage.
    /// Cannot fail because no star args are passed.
    pub fn replace_arguments(&self, args_w: Vec<PyObjectRef>) -> Self {
        let keyword_names_w = self.keyword_names_w.clone();
        let keywords_w = self.keywords_w.clone();
        Self {
            arguments_w: args_w,
            // argument.py `replace_arguments` builds a new `Arguments`
            // through `__init__`, which recomputes `_jit_few_keywords`.
            jit_few_keywords: match keyword_names_w.as_deref() {
                None => true,
                Some(names) => majit_rlib::jit::isconstant(&names.len()),
            },
            methodcall: false,
            keyword_names_w,
            keywords_w,
        }
    }

    /// pypy/interpreter/argument.py `prepend`.
    ///
    /// ```python
    /// def prepend(self, w_firstarg):
    ///     "Return a new Arguments with a new argument inserted first."
    ///     return self.replace_arguments([w_firstarg] + self.arguments_w)
    /// ```
    pub fn prepend(&self, w_firstarg: PyObjectRef) -> Self {
        let mut args_w = Vec::with_capacity(self.arguments_w.len() + 1);
        args_w.push(w_firstarg);
        args_w.extend_from_slice(&self.arguments_w);
        self.replace_arguments(args_w)
    }

    /// pypy/interpreter/argument.py `_match_signature`.
    ///
    /// Parse positional + kwargs arguments against `signature`, filling
    /// `scope_w` (caller-provided buffer of length `signature.scope_length()`).
    /// Returns `Err(ArgErr::*)` on shape mismatch.  PyPy raises ArgErr
    /// from the call site and converts to `oefmt(space.w_TypeError, ...)`
    /// at `parse_obj` / `parse_into_scope`.
    ///
    /// Inputs (matching the upstream parameter names):
    /// - `w_firstarg`: optional implicit first arg (e.g. `self` for
    ///   bound methods); `PY_NULL` means absent.
    /// - `scope_w`: output buffer, must be `signature.scope_length()`
    ///   long, pre-filled with `PY_NULL`.
    /// - `signature`: callee signature.
    /// - `defaults_w`: positional default values, `None` if absent.
    /// - `w_kw_defs`: keyword-only defaults dict, `PY_NULL` if absent.
    /// - `blindargs`: number of "blind" positional args from `prepend()`
    ///   (mostly 0; non-zero only for the bound-method dispatcher).
    /// `argument.py _match_signature` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    pub fn match_signature(
        &self,
        w_firstarg: PyObjectRef,
        scope_w: &mut [PyObjectRef],
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<(), MatchSignatureError> {
        let co_posonlyargcount = signature.posonlyargcount;
        let co_argcount = signature.num_argnames();
        let co_kwonlyargcount = signature.num_kwonlyargnames();
        let mut too_many_args = false;

        // argument.py:191-201 — put `w_firstarg` into the scope.
        let mut upfront: usize = 0;
        // PyPy can prepend `[w_firstarg] + args_w` (line 201) when the
        // callee shape is `def meth(*args)` and a method-call adds
        // `self`; pyre clones into a Vec for that rare path.
        let mut args_w_owned: Option<Vec<PyObjectRef>> = None;
        let args_w: &[PyObjectRef] = if !w_firstarg.is_null() {
            if co_argcount > 0 {
                scope_w[0] = w_firstarg;
                upfront = 1;
                &self.arguments_w
            } else {
                // argument.py:198-201 fall-back.
                let mut v = Vec::with_capacity(self.arguments_w.len() + 1);
                v.push(w_firstarg);
                v.extend_from_slice(&self.arguments_w);
                args_w_owned = Some(v);
                args_w_owned.as_deref().unwrap()
            }
        } else {
            &self.arguments_w
        };
        let num_args = args_w.len();
        let avail = num_args + upfront;

        // argument.py:206-209 — kwds count.
        let keyword_names_w: Option<&[PyObjectRef]> = self.keyword_names_w.as_deref();
        let num_kwds = keyword_names_w.map(|k| k.len()).unwrap_or(0);

        // argument.py — positional copy.
        let mut input_argcount = upfront;
        if input_argcount < co_argcount {
            let take = num_args.min(co_argcount - upfront);
            for i in 0..take {
                scope_w[i + input_argcount] = args_w[i];
            }
            input_argcount += take;
        }

        // `w_kw_defs`, the `**kwargs` dict, the keyword values, the leftover
        // positionals that become `*args`, the positional defaults and the
        // words already copied into `scope_w` are all read after the
        // allocations below — the vararg tuple, the dict itself, the keyword
        // collection, the kwonly-default lookups — and a `list` or `dict`
        // among them moves under any of those.  A by-value parameter copy is
        // no more a slot the collector updates than the caller's scope buffer
        // or `Arguments`' own vectors are, so each word is pinned here, ahead
        // of the first allocation, and read back where it is used.
        let _roots = pyre_object::gc_roots::push_roots();
        let keywords_w = self.keywords_w.as_deref().unwrap_or(&[]);
        let names_w = self.keyword_names_w.as_deref().unwrap_or(&[]);
        let mut live = Vec::with_capacity(1 + keywords_w.len() + names_w.len() + args_w.len());
        live.push(w_kw_defs);
        live.extend_from_slice(keywords_w);
        live.extend_from_slice(names_w);
        live.extend_from_slice(args_w);
        let kw_defs_slot = pyre_object::gc_roots::pin_roots(&live);
        let w_kw_defs = pyre_object::gc_roots::shadow_stack_get(kw_defs_slot);
        let keywords_base = kw_defs_slot + 1;
        let names_base = keywords_base + keywords_w.len();
        let args_base = names_base + names_w.len();
        // The defaults arrive as a caller-owned slice — `Function`'s `defs_w`
        // reaches here through a borrow, and a gateway builds its own array —
        // so the collector rewrites neither, while the run is read out one
        // entry at a time in the defaults loop far below.
        let defaults_base = pyre_object::gc_roots::pin_roots(defaults_w.unwrap_or_default());
        // The scope buffer keeps its own run of slots: `w_firstarg` and the
        // positional copy above are in it already, and every later store goes
        // through `store` so the refresh before `Ok(())` reads the whole scope
        // back at its current addresses.
        let scope_base = pyre_object::gc_roots::pin_roots(scope_w);
        let store = |scope_w: &mut [PyObjectRef], index: usize, w_value: PyObjectRef| {
            scope_w[index] = w_value;
            pyre_object::gc_roots::shadow_stack_set(scope_base + index, w_value);
        };

        // argument.py:222-236 — *vararg collection.
        if signature.has_vararg() {
            let args_left = co_argcount - upfront;
            // argument.py — `assert args_left >= 0` always holds in
            // pyre (usize subtraction would have panicked above).
            let starargs_w: Vec<PyObjectRef> = if num_args > args_left {
                let extra_base = args_base + args_left;
                (0..(num_args - args_left))
                    .map(|i| pyre_object::gc_roots::shadow_stack_get(extra_base + i))
                    .collect()
            } else {
                Vec::new()
            };
            let loc = co_argcount + co_kwonlyargcount;
            let w_stararg = pyre_object::w_tuple_new(starargs_w);
            store(scope_w, loc, w_stararg);
        } else if avail > co_argcount {
            too_many_args = true;
        }

        // argument.py:238-242 — **kwargs dict allocation.
        let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
        let kwarg_loc = co_argcount + co_kwonlyargcount + (signature.has_vararg() as usize);
        let mut w_kwds: PyObjectRef = pyre_object::PY_NULL;
        if signature.has_kwarg() {
            // `_match_signature` — `space.newdict(kwargs=True)`
            // (EmptyKwargsDictStrategy). An exact str key stays on
            // KwargsDictStrategy; a str subclass is not an exact str, so
            // `switch_to_correct_strategy` promotes to the object strategy
            // and `_collect_keyword_args` `setitem`s that same key.
            let _ =
                pyre_object::gc_roots::pin_root(pyre_object::dictmultiobject::w_dict_new_kwargs());
            w_kwds = pyre_object::gc_roots::shadow_stack_get(kwds_slot);
            store(scope_w, kwarg_loc, w_kwds);
        }

        // argument.py:244-271 — keyword arg matching.
        let mut num_remainingkwds: usize = 0;
        let keywords_w: Option<&[PyObjectRef]> = self.keywords_w.as_deref();
        let mut kwds_mapping: Option<Vec<isize>> = None;
        if num_kwds > 0 {
            // argument.py — pre-init `kwds_mapping[i] = -1`.
            let mapping_len = co_argcount + co_kwonlyargcount - input_argcount;
            let mut mapping = vec![-1isize; mapping_len];
            // argument.py:259-262 — match keyword names to argnames.
            let names_w: Vec<PyObjectRef> = (0..keyword_names_w.unwrap().len())
                .map(|i| pyre_object::gc_roots::shadow_stack_get(names_base + i))
                .collect();
            num_remainingkwds = match_keywords(
                signature,
                blindargs,
                co_posonlyargcount,
                input_argcount,
                &names_w,
                &mut mapping,
            )?;
            if num_remainingkwds > 0 {
                if !w_kwds.is_null() {
                    // argument.py:266-268 — collect overflow into **kwarg.
                    // The helper pins whatever slice it is handed, so it takes
                    // the published words rather than the ones the vector has
                    // been holding since the two allocations above.
                    let values_w: Vec<PyObjectRef> = (0..keywords_w.unwrap().len())
                        .map(|i| pyre_object::gc_roots::shadow_stack_get(keywords_base + i))
                        .collect();
                    collect_keyword_args(
                        &names_w,
                        &values_w,
                        pyre_object::gc_roots::shadow_stack_get(kwds_slot),
                        &mapping,
                    )?;
                } else {
                    // argument.py:270-271 — ArgErrUnknownKwds.  PyPy's
                    // `__init__` (argument.py:609-617) calls
                    // `space.text_w(keyword_names_w[i])` on the single
                    // offending name when `num_remainingkwds == 1`; if
                    // that text-conversion fails (non-str key), the
                    // OperationError propagates back to the caller as a
                    // TypeError.  Pyre mirrors that propagation rather
                    // than silently substituting an empty string.
                    let mut name = Wtf8Buf::new();
                    if num_remainingkwds == 1 {
                        for (i, &w_n) in names_w.iter().enumerate() {
                            if !mapping_contains(&mapping, i as isize) {
                                if unsafe { pyre_object::is_str(w_n) } {
                                    // `text_w` returns the raw buffer. A lone
                                    // surrogate is still the unexpected name.
                                    name =
                                        unsafe { pyre_object::w_str_get_wtf8(w_n) }.to_wtf8_buf();
                                } else {
                                    // baseobjspace.py `_typed_unwrap_error`
                                    // → `TypeError("expected str, got %T object")`.
                                    let tp = type_name_of(w_n);
                                    return Err(MatchSignatureError::Py(
                                        crate::PyError::type_error(format!(
                                            "expected str, got {tp} object",
                                        )),
                                    ));
                                }
                                break;
                            }
                        }
                    }
                    return Err(MatchSignatureError::Shape(ArgErr::UnknownKwds {
                        num_kwds: num_remainingkwds,
                        kwd_name: name,
                    }));
                }
            }
            kwds_mapping = Some(mapping);
        }

        // argument.py:273-287 — fill missing args from kwds.
        let more_filling = input_argcount < co_argcount + co_kwonlyargcount;
        // `argument.py:274` `def_first = co_argcount - len(defaults_w)` is
        // signed: `defaults_w` longer than the positional parameters makes it
        // negative and `defnum = i - def_first` then names the tuple's tail.
        let mut def_first: isize = 0;
        if more_filling {
            let defaults_len = defaults_w.map(|d| d.len()).unwrap_or(0);
            def_first = co_argcount as isize - defaults_len as isize;
            let mut j = 0usize;
            for i in input_argcount..(co_argcount + co_kwonlyargcount) {
                if let Some(ref mapping) = kwds_mapping {
                    let kwds_index = mapping[j];
                    j += 1;
                    if kwds_index >= 0 {
                        let w_value = pyre_object::gc_roots::shadow_stack_get(
                            keywords_base + kwds_index as usize,
                        );
                        store(scope_w, i, w_value);
                    }
                }
            }
        }

        // argument.py:289-300 — too-many-args ArgErr.
        if too_many_args {
            let mut kwonly_given: usize = 0;
            for i in co_argcount..(co_argcount + co_kwonlyargcount) {
                if !pyre_object::gc_roots::shadow_stack_get(scope_base + i).is_null() {
                    kwonly_given += 1;
                }
            }
            let num_defaults = defaults_w.map(|d| d.len()).unwrap_or(0);
            if self.methodcall {
                return Err(MatchSignatureError::Shape(ArgErr::TooManyMethod {
                    signature: signature.clone(),
                    num_defaults,
                    given: avail,
                    kwonly_given,
                }));
            } else {
                return Err(MatchSignatureError::Shape(ArgErr::TooMany {
                    signature: signature.clone(),
                    num_defaults,
                    given: avail,
                    kwonly_given,
                }));
            }
        }

        // argument.py:302-338 — fill defaults + collect missing names.
        if more_filling {
            let mut missing_positional: Option<Vec<String>> = None;
            let mut missing_kwonly: Option<Vec<String>> = None;
            // argument.py:306-315 — posonly defaults.
            for i in input_argcount..co_argcount {
                if !pyre_object::gc_roots::shadow_stack_get(scope_base + i).is_null() {
                    continue;
                }
                let defnum = (i as isize) - def_first;
                if defnum >= 0 {
                    let w_default =
                        pyre_object::gc_roots::shadow_stack_get(defaults_base + defnum as usize);
                    store(scope_w, i, w_default);
                } else if let Some(list) = missing_positional.as_mut() {
                    list.push(signature.argnames[i].to_string());
                } else {
                    missing_positional = Some(vec![signature.argnames[i].to_string()]);
                }
            }
            // argument.py:317-333 — kwonly defaults via w_kw_defs dict.
            for i in co_argcount..(co_argcount + co_kwonlyargcount) {
                if !pyre_object::gc_roots::shadow_stack_get(scope_base + i).is_null() {
                    continue;
                }
                let name = signature.argnames[i];
                if w_kw_defs.is_null() {
                    if let Some(list) = missing_kwonly.as_mut() {
                        list.push(name.to_string());
                    } else {
                        missing_kwonly = Some(vec![name.to_string()]);
                    }
                    continue;
                }
                // PyPy `baseobjspace.py finditem` re-raises any
                // `OperationError` other than KeyError, so a kwonly-defaults
                // dict with a subclass `__getitem__` raising e.g.
                // `RuntimeError` surfaces here instead of being
                // mis-classified as "default missing".
                match crate::baseobjspace::finditem_str(
                    pyre_object::gc_roots::shadow_stack_get(kw_defs_slot),
                    name,
                )? {
                    Some(w_def) => store(scope_w, i, w_def),
                    None => {
                        if let Some(list) = missing_kwonly.as_mut() {
                            list.push(name.to_string());
                        } else {
                            missing_kwonly = Some(vec![name.to_string()]);
                        }
                    }
                }
            }
            // argument.py — surface ArgErrMissing.
            if let Some(missing) = missing_positional {
                return Err(MatchSignatureError::Shape(ArgErr::Missing {
                    missing,
                    positional: true,
                }));
            }
            if let Some(missing) = missing_kwonly {
                return Err(MatchSignatureError::Shape(ArgErr::Missing {
                    missing,
                    positional: false,
                }));
            }
        }

        // The collection points above have all run by now, so the buffer takes
        // the scope's current addresses instead of keeping the ones it was
        // filled with.  The early `return Err(...)` paths skip this refresh and
        // leave pre-move words in the buffer; that is sound only because every
        // caller drops `scope_w` unread on an error.
        pyre_object::gc_roots::shadow_stack_copy_range(scope_base, scope_w);
        Ok(())
    }

    /// pypy/interpreter/argument.py `_parse`.
    ///
    /// ```python
    /// def _parse(self, w_firstarg, signature, defaults_w, w_kw_defs, blindargs=0):
    ///     scopelen = signature.scope_length()
    ///     scope_w = [None] * scopelen
    ///     self._match_signature(w_firstarg, scope_w, signature, defaults_w,
    ///                           w_kw_defs, blindargs)
    ///     return scope_w
    /// ```
    pub(crate) fn _parse(
        &self,
        w_firstarg: PyObjectRef,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<Vec<PyObjectRef>, MatchSignatureError> {
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; scopelen];
        self.match_signature(
            w_firstarg,
            &mut scope_w,
            signature,
            defaults_w,
            w_kw_defs,
            blindargs,
        )?;
        Ok(scope_w)
    }

    /// pypy/interpreter/argument.py `parse_into_scope`.
    ///
    /// ```python
    /// def parse_into_scope(self, w_firstarg,
    ///                      scope_w, fnname, signature, defaults_w=None,
    ///                      w_kw_defs=None):
    ///     try:
    ///         self._match_signature(w_firstarg,
    ///                               scope_w, signature, defaults_w,
    ///                               w_kw_defs, 0)
    ///     except ArgErr as e:
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    ///     return signature.scope_length()
    /// ```
    pub fn parse_into_scope(
        &self,
        w_firstarg: PyObjectRef,
        scope_w: &mut [PyObjectRef],
        fnname: &str,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        match self.match_signature(w_firstarg, scope_w, signature, defaults_w, w_kw_defs, 0) {
            Ok(()) => Ok(signature.scope_length()),
            Err(MatchSignatureError::Shape(e)) => {
                let msg = e.getmsg_wtf8();
                Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    format!("{fnname}() "),
                    msg
                )))
            }
            Err(MatchSignatureError::Py(e)) => Err(e),
        }
    }

    /// pypy/interpreter/argument.py `parse_obj`.
    ///
    /// ```python
    /// def parse_obj(self, w_firstarg,
    ///               fnname, signature, defaults_w=None, w_kw_defs=None,
    ///               blindargs=0):
    ///     try:
    ///         return self._parse(w_firstarg, signature, defaults_w, w_kw_defs,
    ///                            blindargs)
    ///     except ArgErrUnknownKwds as e:
    ///         if not signature.has_kwarg() and signature.kwonlyargcount == 0:
    ///             raise oefmt(self.space.w_TypeError,
    ///                         "%s() takes no keyword arguments", fnname)
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    ///     except ArgErr as e:
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    /// ```
    pub fn parse_obj(
        &self,
        w_firstarg: PyObjectRef,
        fnname: &str,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<Vec<PyObjectRef>, crate::PyError> {
        match self._parse(w_firstarg, signature, defaults_w, w_kw_defs, blindargs) {
            Ok(scope_w) => Ok(scope_w),
            Err(MatchSignatureError::Shape(ArgErr::UnknownKwds { num_kwds, kwd_name })) => {
                // argument.py:378-380 — special-case "takes no keyword
                // arguments" when the signature has no **kwarg AND no
                // keyword-only args.
                if !signature.has_kwarg() && signature.num_kwonlyargnames() == 0 {
                    return Err(crate::PyError::type_error(format!(
                        "{}() takes no keyword arguments",
                        fnname,
                    )));
                }
                let msg = ArgErr::UnknownKwds { num_kwds, kwd_name }.getmsg_wtf8();
                Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    format!("{fnname}() "),
                    msg
                )))
            }
            Err(MatchSignatureError::Shape(e)) => {
                let msg = e.getmsg_wtf8();
                Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    format!("{fnname}() "),
                    msg
                )))
            }
            Err(MatchSignatureError::Py(e)) => Err(e),
        }
    }

    /// pypy/interpreter/argument.py `frompacked`.
    ///
    /// ```python
    /// @staticmethod
    /// def frompacked(space, w_args=None, w_kwds=None):
    ///     """Convenience static method to build an Arguments
    ///        from a wrapped sequence and a wrapped dictionary."""
    ///     return Arguments(space, [], w_stararg=w_args, w_starstararg=w_kwds)
    /// ```
    pub fn frompacked(
        w_args: Option<PyObjectRef>,
        w_kwds: Option<PyObjectRef>,
    ) -> Result<Self, crate::PyError> {
        Self::new(&[], None, None, w_args, w_kwds, false, None)
    }

    /// pypy/interpreter/argument.py `topacked`.
    ///
    /// ```python
    /// def topacked(self):
    ///     """Express the Argument object as a pair of wrapped w_args, w_kwds."""
    ///     space = self.space
    ///     w_args = space.newtuple(self.arguments_w)
    ///     w_kwds = space.newdict()
    ///     if self.keyword_names_w is not None:
    ///         for i in range(len(self.keyword_names_w)):
    ///             w_key = self.keyword_names_w[i]
    ///             space.setitem(w_kwds, w_key, self.keywords_w[i])
    ///     return w_args, w_kwds
    /// ```
    pub fn topacked(&self) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
        let w_args = pyre_object::w_tuple_new(self.arguments_w.clone());
        // Each `setitem` below hashes a key, can grow the dictionary's storage
        // and can run a subclass `__setitem__`, so every turn is a collection
        // point.  The dictionary moves under one, and the names and values sit
        // in native vectors no root walker updates, so the whole set is
        // published before the first turn and read back on the turn that uses
        // it.  `w_args` is nursery-allocated too, so it is read back at the
        // return rather than carried across the loop in a local.
        let _roots = pyre_object::gc_roots::push_roots();
        let args_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_args);
        let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::dictmultiobject::w_dict_new());
        if let (Some(names), Some(values)) =
            (self.keyword_names_w.as_ref(), self.keywords_w.as_ref())
        {
            // `zip` stops at the shorter side; the pinned pairs match it.
            let npairs = names.len().min(values.len());
            let mut pairs = Vec::with_capacity(npairs * 2);
            for (w_key, w_value) in names.iter().zip(values.iter()).take(npairs) {
                pairs.push(*w_key);
                pairs.push(*w_value);
            }
            let pairs_base = pyre_object::gc_roots::pin_roots(&pairs);
            for i in 0..npairs {
                let w_key = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i);
                let w_value = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i + 1);
                crate::baseobjspace::setitem(
                    pyre_object::gc_roots::shadow_stack_get(kwds_slot),
                    w_key,
                    w_value,
                )?;
            }
        }
        Ok((
            pyre_object::gc_roots::shadow_stack_get(args_slot),
            pyre_object::gc_roots::shadow_stack_get(kwds_slot),
        ))
    }
}

/// pypy/interpreter/argument.py `_match_signature` raises
/// either `ArgErr` (shape mismatch) or `OperationError` (PyError, from
/// `space.setitem` in `_collect_keyword_args`).  Pyre keeps both
/// exception types separate via this enum, so callers
/// (`parse_obj` / `parse_into_scope` in Phase B) can match on the
/// shape arm to reformat as `oefmt(space.w_TypeError, "%s() %8",
/// fnname, e.getmsg())` (argument.py:354) and propagate the Py arm
/// unchanged.
#[derive(Debug, Clone)]
pub enum MatchSignatureError {
    /// `ArgErr*` raised from upstream `_match_signature` /
    /// `_match_keywords`.
    Shape(ArgErr),
    /// `OperationError` raised from `space.setitem` /
    /// `space.finditem_str` etc. (collect_keyword_args fallout).
    Py(crate::PyError),
}

impl From<ArgErr> for MatchSignatureError {
    fn from(e: ArgErr) -> Self {
        MatchSignatureError::Shape(e)
    }
}

impl From<crate::PyError> for MatchSignatureError {
    fn from(e: crate::PyError) -> Self {
        MatchSignatureError::Py(e)
    }
}

/// pypy/interpreter/argument.py `_match_keywords`.
///
/// ```python
/// def _match_keywords(space, signature, blindargs, co_posonlyargcount,
///                     input_argcount, keyword_names_w, kwds_mapping, _):
///     num_kwds = num_remainingkwds = len(keyword_names_w)
///     wrong_posonly = None
///     for i in range(num_kwds):
///         w_name = keyword_names_w[i]
///         assert w_name is not None
///         j = signature.find_w_argname(w_name)
///         if 0 <= j < co_posonlyargcount:
///             if signature.has_kwarg():
///                 j = -1
///             else:
///                 if wrong_posonly is None:
///                     wrong_posonly = []
///                 wrong_posonly.append(signature.argnames[j])
///                 continue
///         if j < input_argcount:
///             if blindargs <= j:
///                 raise ArgErrMultipleValues(space.text_w(w_name))
///         else:
///             kwds_mapping[j - input_argcount] = i
///             num_remainingkwds -= 1
///     if wrong_posonly:
///         raise ArgErrPosonlyAsKwds(wrong_posonly)
///     return num_remainingkwds
/// ```
pub fn match_keywords(
    signature: &crate::gateway::Signature,
    blindargs: usize,
    co_posonlyargcount: usize,
    input_argcount: usize,
    keyword_names_w: &[PyObjectRef],
    kwds_mapping: &mut [isize],
) -> Result<usize, MatchSignatureError> {
    let num_kwds = keyword_names_w.len();
    let mut num_remainingkwds = num_kwds;
    let mut wrong_posonly: Option<Vec<String>> = None;
    for i in 0..num_kwds {
        let w_name = keyword_names_w[i];
        debug_assert!(!w_name.is_null(), "keyword_names_w[i] is None");
        let mut j = signature.find_w_argname(w_name);
        // argument.py:474-484 — posonly conflict.
        if j >= 0 && (j as usize) < co_posonlyargcount {
            if signature.has_kwarg() {
                j = -1;
            } else {
                let argname = signature.argnames[j as usize].to_string();
                if let Some(list) = wrong_posonly.as_mut() {
                    list.push(argname);
                } else {
                    wrong_posonly = Some(vec![argname]);
                }
                continue;
            }
        }
        if j < input_argcount as isize {
            // argument.py:485-495 — multiple-values check.  PyPy raises
            // `ArgErrMultipleValues(space.text_w(w_name))`; if `w_name`
            // is non-string (hand-built `Arguments::with_kw`), `text_w`
            // raises a TypeError that propagates through the
            // `_match_signature` caller as `OperationError`.  Pyre
            // mirrors that propagation via `MatchSignatureError::Py`
            // rather than substituting an empty string (baseobjspace.py:
            // 313-315 `_typed_unwrap_error` message).
            if (blindargs as isize) <= j {
                let name = if !w_name.is_null() && unsafe { pyre_object::is_str(w_name) } {
                    unsafe { crate::baseobjspace::str_utf8_w(w_name)?.to_string() }
                } else {
                    let tp = type_name_of(w_name);
                    return Err(MatchSignatureError::Py(crate::PyError::type_error(
                        format!("expected str, got {tp} object"),
                    )));
                };
                return Err(MatchSignatureError::Shape(ArgErr::MultipleValues {
                    argname: name,
                }));
            }
        } else {
            kwds_mapping[(j - input_argcount as isize) as usize] = i as isize;
            num_remainingkwds -= 1;
        }
    }
    if let Some(wrong) = wrong_posonly {
        return Err(MatchSignatureError::Shape(ArgErr::PosonlyAsKwds {
            posonly_kwds: wrong,
        }));
    }
    Ok(num_remainingkwds)
}

/// pypy/interpreter/argument.py `_collect_keyword_args`.
///
/// ```python
/// def _collect_keyword_args(space, keyword_names_w, keywords_w, w_kwds,
///                           kwds_mapping, _):
///     for i in range(len(keyword_names_w)):
///         for j in kwds_mapping:
///             if i == j:
///                 break
///         else:
///             w_key = keyword_names_w[i]
///             space.setitem(w_kwds, w_key, keywords_w[i])
/// ```
///
/// `kwds_mapping[k] == i` means slot `k` matched a real argname; this
/// helper walks the kwarg names and forwards every name that did NOT
/// match (i.e. did not appear in `kwds_mapping`) into the `**kwargs`
/// dict via `setitem`.
pub fn collect_keyword_args(
    keyword_names_w: &[PyObjectRef],
    keywords_w: &[PyObjectRef],
    w_kwds: PyObjectRef,
    kwds_mapping: &[isize],
) -> Result<(), crate::PyError> {
    // Every `setitem` is a collection point, so the dictionary and the pairs it
    // forwards are published before the first one and read back at each turn.
    // A name the mapping claimed keeps an empty pair of slots so the two loops
    // agree on the index.
    let _roots = pyre_object::gc_roots::push_roots();
    let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_kwds);
    let mut pairs = Vec::with_capacity(keyword_names_w.len() * 2);
    for i in 0..keyword_names_w.len() {
        if mapping_contains(kwds_mapping, i as isize) {
            pairs.push(pyre_object::PY_NULL);
            pairs.push(pyre_object::PY_NULL);
            continue;
        }
        pairs.push(keyword_names_w[i]);
        pairs.push(keywords_w[i]);
    }
    let pairs_base = pyre_object::gc_roots::pin_roots(&pairs);
    for i in 0..keyword_names_w.len() {
        if mapping_contains(kwds_mapping, i as isize) {
            continue;
        }
        let w_key = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i);
        let w_value = pyre_object::gc_roots::shadow_stack_get(pairs_base + 2 * i + 1);
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(kwds_slot),
            w_key,
            w_value,
        )?;
    }
    Ok(())
}

#[inline]
fn mapping_contains(mapping: &[isize], target: isize) -> bool {
    mapping.contains(&target)
}

/// pypy/interpreter/argument.py — `ArgErr` exception hierarchy.
///
/// PyPy declares `class ArgErr(Exception)` with abstract `getmsg()`
/// and 6 concrete subclasses (`ArgErrMissing`, `ArgErrTooMany`,
/// `ArgErrTooManyMethod`, `ArgErrMultipleValues`, `ArgErrUnknownKwds`,
/// `ArgErrPosonlyAsKwds`).  `_match_signature` / `_parse` / `parse_obj`
/// raise these and the surrounding `try/except ArgErr` arms reformat
/// the message via `oefmt(space.w_TypeError, "%s() %8", fnname,
/// e.getmsg())`.
///
/// Pyre folds the class hierarchy into a Rust enum with payload-only
/// variants — the six error shapes carry exactly the upstream
/// constructor parameters and `getmsg()` runs the per-variant
/// formatting line-by-line.
#[derive(Debug, Clone)]
pub enum ArgErr {
    /// argument.py `ArgErrMissing(missing, positional)`.
    Missing {
        missing: Vec<String>,
        positional: bool,
    },
    /// argument.py `ArgErrTooMany(signature, num_defaults, given, kwonly_given)`.
    TooMany {
        signature: crate::gateway::Signature,
        num_defaults: usize,
        given: usize,
        kwonly_given: usize,
    },
    /// argument.py `ArgErrTooManyMethod` — same fields as
    /// `TooMany`, with appended "did you forget self?" hint when the
    /// signature shape suggests a missing self parameter.
    TooManyMethod {
        signature: crate::gateway::Signature,
        num_defaults: usize,
        given: usize,
        kwonly_given: usize,
    },
    /// argument.py:598-605 `ArgErrMultipleValues(argname)`.
    MultipleValues { argname: String },
    /// argument.py:607-627 `ArgErrUnknownKwds(space, num_remainingkwds,
    /// keyword_names_w, kwds_mapping)`.  PyPy's ctor walks the
    /// keyword_names_w list to extract a single offending name when
    /// `num_remainingkwds == 1`; pyre stores the resolved name string
    /// directly so the formatter can stay free of space refs.
    UnknownKwds { num_kwds: usize, kwd_name: Wtf8Buf },
    /// argument.py `ArgErrPosonlyAsKwds(posonly_kwds)`.
    PosonlyAsKwds { posonly_kwds: Vec<String> },
}

impl ArgErr {
    /// pypy/interpreter/argument.py `ArgErrMissing.getmsg`,
    /// :562-580 `ArgErrTooMany.getmsg`, :589-595
    /// `ArgErrTooManyMethod.getmsg`, :603-605
    /// `ArgErrMultipleValues.getmsg`, :620-627
    /// `ArgErrUnknownKwds.getmsg`, :635-640
    /// `ArgErrPosonlyAsKwds.getmsg` — line-by-line ports.
    /// Same text as [`Self::getmsg`], keeping a lone surrogate in the
    /// unexpected-keyword name (`text_w` returns the raw buffer).
    pub fn getmsg_wtf8(&self) -> Wtf8Buf {
        match self {
            ArgErr::UnknownKwds {
                num_kwds: 1,
                kwd_name,
            } => {
                let mut msg = Wtf8Buf::from("got an unexpected keyword argument '");
                msg.push_wtf8(kwd_name);
                msg.push_str("'");
                msg
            }
            other => Wtf8Buf::from(other.getmsg()),
        }
    }

    pub fn getmsg(&self) -> String {
        match self {
            ArgErr::Missing {
                missing,
                positional,
            } => {
                // argument.py:535-546 — comma-and-or-and join.
                let mut arguments_str = String::new();
                let n = missing.len();
                for (i, arg) in missing.iter().enumerate() {
                    if i == 0 {
                        // empty separator
                    } else if i == n - 1 {
                        if n == 2 {
                            arguments_str.push_str(" and ");
                        } else {
                            arguments_str.push_str(", and ");
                        }
                    } else {
                        arguments_str.push_str(", ");
                    }
                    arguments_str.push('\'');
                    arguments_str.push_str(arg);
                    arguments_str.push('\'');
                }
                let kind_str = if *positional {
                    "positional"
                } else {
                    "keyword-only"
                };
                let plural = if n != 1 { "s" } else { "" };
                format!("missing {n} required {kind_str} argument{plural}: {arguments_str}",)
            }
            ArgErr::TooMany {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => format_too_many(signature, *num_defaults, *given, *kwonly_given),
            ArgErr::TooManyMethod {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => {
                let mut msg = format_too_many(signature, *num_defaults, *given, *kwonly_given);
                let n = signature.num_argnames();
                // argument.py:592-594 — "did you forget self?" hint.
                if *given == n + 1 && (n == 0 || signature.argnames[0] != "self") {
                    msg.push_str(". Did you forget 'self' in the function definition?");
                }
                msg
            }
            ArgErr::MultipleValues { argname } => {
                format!("got multiple values for argument '{argname}'")
            }
            ArgErr::UnknownKwds { num_kwds, kwd_name } => {
                if *num_kwds == 1 {
                    format!("got an unexpected keyword argument '{kwd_name}'")
                } else {
                    format!("got {num_kwds} unexpected keyword arguments")
                }
            }
            ArgErr::PosonlyAsKwds { posonly_kwds } => {
                // argument.py:635-637 — always the plural wording, even for
                // a single name; CPython (`positional_only_passed_as_keyword`
                // in Python/ceval.c) matches this, there is no singular form.
                format!(
                    "got some positional-only arguments passed as keyword arguments: '{}'",
                    posonly_kwds.join(", "),
                )
            }
        }
    }
}

/// pypy/interpreter/argument.py `ArgErrTooMany.getmsg` body.
/// Extracted because both `TooMany` and `TooManyMethod` arms share it.
fn format_too_many(
    signature: &crate::gateway::Signature,
    num_defaults: usize,
    given: usize,
    kwonly_given: usize,
) -> String {
    let num_args = signature.num_argnames();
    let takes_str = if num_defaults > 0 {
        format!(
            "from {} to {} positional arguments",
            num_args as isize - num_defaults as isize,
            num_args,
        )
    } else {
        let plural = if num_args != 1 { "s" } else { "" };
        format!("{num_args} positional argument{plural}")
    };
    let given_str = if kwonly_given > 0 {
        let pos_plural = if given != 1 { "s" } else { "" };
        let kw_plural = if kwonly_given != 1 { "s" } else { "" };
        format!(
            "{given} positional argument{pos_plural} (and {kwonly_given} keyword-only argument{kw_plural}) were",
        )
    } else {
        let verb = if given != 1 { "were" } else { "was" };
        format!("{given} {verb}")
    };
    format!("takes {takes_str} but {given_str} given")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pypy/interpreter/argument.py — `w_function is None` arm.
    #[test]
    fn raise_type_error_no_function() {
        let err = raise_type_error(pyre_object::PY_NULL, "boom".to_string());
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(err.message_text(), "boom");
    }

    /// pypy/interpreter/argument.py:16-17 — function-prefixed arm.
    /// With an int as `w_function` the prefix falls back to `str(int) ()`
    /// per `object_functionstr`'s scalar fallback path.  The fallback
    /// invokes `space.str(w_int)` which dispatches to `int.__str__`
    /// via type-MRO `lookup`; that lookup needs `init_typeobjects()`
    /// to have populated the W_TypeObject MRO.
    #[test]
    fn raise_type_error_with_function() {
        crate::typedef::init_typeobjects();
        let func = pyre_object::w_int_new(7);
        let err = raise_type_error(func, "needs an iterable".to_string());
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        // object_functionstr scalar fallback returns the str() of the
        // value, here "7"; raise_type_error joins it with the message.
        assert_eq!(err.message_text(), "7 needs an iterable");
    }

    /// pypy/interpreter/argument.py getmsg single-missing positional case.
    #[test]
    fn arg_err_missing_one_positional() {
        let err = ArgErr::Missing {
            missing: vec!["x".to_string()],
            positional: true,
        };
        assert_eq!(err.getmsg(), "missing 1 required positional argument: 'x'",);
    }

    /// pypy/interpreter/argument.py:540-541 — 2-missing uses ` and `.
    #[test]
    fn arg_err_missing_two_keyword_only() {
        let err = ArgErr::Missing {
            missing: vec!["a".to_string(), "b".to_string()],
            positional: false,
        };
        assert_eq!(
            err.getmsg(),
            "missing 2 required keyword-only arguments: 'a' and 'b'",
        );
    }

    /// pypy/interpreter/argument.py — 3-or-more uses `, and `.
    #[test]
    fn arg_err_missing_three_with_oxford_comma() {
        let err = ArgErr::Missing {
            missing: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            positional: true,
        };
        assert_eq!(
            err.getmsg(),
            "missing 3 required positional arguments: 'a', 'b', and 'c'",
        );
    }

    /// pypy/interpreter/argument.py `ArgErrMultipleValues`.
    #[test]
    fn arg_err_multiple_values() {
        let err = ArgErr::MultipleValues {
            argname: "kwarg".to_string(),
        };
        assert_eq!(err.getmsg(), "got multiple values for argument 'kwarg'");
    }

    /// pypy/interpreter/argument.py `ArgErrUnknownKwds`
    /// — single-name vs multi-count branches.
    #[test]
    fn arg_err_unknown_kwds_branches() {
        let one = ArgErr::UnknownKwds {
            num_kwds: 1,
            kwd_name: Wtf8Buf::from("wibble"),
        };
        assert_eq!(one.getmsg(), "got an unexpected keyword argument 'wibble'");
        let many = ArgErr::UnknownKwds {
            num_kwds: 3,
            kwd_name: Wtf8Buf::new(),
        };
        assert_eq!(many.getmsg(), "got 3 unexpected keyword arguments");
    }

    /// pypy/interpreter/argument.py `ArgErrPosonlyAsKwds.getmsg` —
    /// always the plural wording, regardless of `posonly_kwds` length.
    /// CPython's `positional_only_passed_as_keyword` (Python/ceval.c) has
    /// no singular form either.
    #[test]
    fn arg_err_posonly_as_kwds_always_plural() {
        let one = ArgErr::PosonlyAsKwds {
            posonly_kwds: vec!["a".to_string()],
        };
        assert_eq!(
            one.getmsg(),
            "got some positional-only arguments passed as keyword arguments: 'a'",
        );
        let many = ArgErr::PosonlyAsKwds {
            posonly_kwds: vec!["x".to_string(), "y".to_string()],
        };
        assert_eq!(
            many.getmsg(),
            "got some positional-only arguments passed as keyword arguments: 'x, y'",
        );
    }

    /// pypy/interpreter/argument.py `ArgErrTooMany.getmsg` —
    /// no defaults, single-arg singular path.
    #[test]
    fn arg_err_too_many_no_defaults_singular() {
        let sig = crate::gateway::Signature::new(vec!["x"], None, None, 0, 0);
        let err = ArgErr::TooMany {
            signature: sig,
            num_defaults: 0,
            given: 2,
            kwonly_given: 0,
        };
        assert_eq!(err.getmsg(), "takes 1 positional argument but 2 were given");
    }

    /// pypy/interpreter/argument.py:565-567 — defaults branch produces
    /// "from X to Y positional arguments".
    #[test]
    fn arg_err_too_many_with_defaults() {
        let sig = crate::gateway::Signature::new(vec!["a", "b", "c"], None, None, 0, 0);
        let err = ArgErr::TooMany {
            signature: sig,
            num_defaults: 1,
            given: 4,
            kwonly_given: 0,
        };
        assert_eq!(
            err.getmsg(),
            "takes from 2 to 3 positional arguments but 4 were given",
        );
    }

    /// pypy/interpreter/argument.py `ArgErrTooManyMethod`
    /// appends self-hint when shape matches and argnames[0] != "self".
    #[test]
    fn arg_err_too_many_method_self_hint_appended() {
        let sig = crate::gateway::Signature::new(vec!["x"], None, None, 0, 0);
        let err = ArgErr::TooManyMethod {
            signature: sig,
            num_defaults: 0,
            given: 2,
            kwonly_given: 0,
        };
        let msg = err.getmsg();
        assert!(msg.starts_with("takes 1 positional argument but 2 were given"));
        assert!(msg.contains("Did you forget 'self'"));
    }

    /// pypy/interpreter/argument.py:592-593 — when argnames[0] == "self"
    /// the hint is suppressed.
    #[test]
    fn arg_err_too_many_method_self_hint_suppressed() {
        let sig = crate::gateway::Signature::new(vec!["self", "x"], None, None, 0, 0);
        let err = ArgErr::TooManyMethod {
            signature: sig,
            num_defaults: 0,
            given: 3,
            kwonly_given: 0,
        };
        let msg = err.getmsg();
        assert!(!msg.contains("Did you forget"));
    }

    /// pypy/interpreter/argument.py `contains_w_names` —
    /// string equality match.
    #[test]
    fn contains_w_names_string_match() {
        let names = vec![pyre_object::w_str_new("a"), pyre_object::w_str_new("b")];
        assert!(contains_w_names(pyre_object::w_str_new("a"), &names));
        assert!(contains_w_names(pyre_object::w_str_new("b"), &names));
        assert!(!contains_w_names(pyre_object::w_str_new("c"), &names));
    }

    /// `contains_w_names` compares keyword names without requiring a `&str`
    /// view.
    #[test]
    fn surrogate_keyword_name_matches() {
        let mut name = rustpython_wtf8::Wtf8Buf::new();
        name.push(rustpython_wtf8::CodePoint::from_u32(0xD800).unwrap());
        let w_name = pyre_object::w_str_from_wtf8(name.clone());
        let existing = vec![pyre_object::w_str_from_wtf8(name.clone())];
        let new = vec![w_name];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];

        assert!(contains_w_names(w_name, &existing));
        let err = check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect_err("should raise TypeError on duplicate");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message_text().contains("got multiple values"));
    }

    /// pypy/interpreter/argument.py `_check_not_duplicate_kwargs` —
    /// raises TypeError on overlap.
    #[test]
    fn check_not_duplicate_kwargs_raises_on_overlap() {
        let existing = vec![pyre_object::w_str_new("a")];
        let new = vec![pyre_object::w_str_new("a")];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        let err = check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect_err("should raise TypeError on duplicate");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message_text().contains("got multiple values"));
        assert!(err.message_text().contains("'a'"));
    }

    /// `check_not_duplicate_kwargs` accepts disjoint name sets.
    #[test]
    fn check_not_duplicate_kwargs_disjoint_ok() {
        let existing = vec![pyre_object::w_str_new("a")];
        let new = vec![pyre_object::w_str_new("b")];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect("disjoint names should pass");
    }

    /// pypy/interpreter/argument.py `_combine_starargs_wrapped`:
    /// list extension via fixedview.
    #[test]
    fn combine_starargs_wrapped_extends_from_list() {
        let mut args: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        let stararg =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)]);
        combine_starargs_wrapped(&mut args, stararg, pyre_object::PY_NULL)
            .expect("list star arg should expand");
        assert_eq!(args.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(args[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(args[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(args[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py:97-103 — non-iterable star arg
    /// surfaces "argument after * must be an iterable".
    ///
    /// Rejecting the operand runs `space.iter`, whose `__iter__` probe is a
    /// type-MRO `lookup`; that lookup needs `init_typeobjects()` to have
    /// populated the W_TypeObject MRO, and its hash-hook install is what
    /// enters the runtime thread this test's method-cache probe needs a GIL
    /// from.
    #[test]
    fn combine_starargs_wrapped_non_iterable_raises() {
        crate::typedef::init_typeobjects();
        let mut args: Vec<PyObjectRef> = vec![];
        let stararg = pyre_object::w_int_new(42); // ints aren't iterable
        let err = combine_starargs_wrapped(&mut args, stararg, pyre_object::PY_NULL)
            .expect_err("int star arg should raise TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(
            err.message_text()
                .contains("argument after * must be an iterable")
        );
    }

    /// pypy/interpreter/argument.py:172-338 `match_signature` happy
    /// path: 2 positional args fill `def f(a, b): pass`.
    #[test]
    fn match_signature_positional_pass_through() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let args = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("happy path should pass");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(scope[1]), 2);
        }
    }

    /// pypy/interpreter/argument.py:289-300 `match_signature` raises
    /// `ArgErrTooMany` when avail > co_argcount.
    #[test]
    fn match_signature_too_many_args_raises() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let args = [
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("too many args should ArgErr");
        match err {
            MatchSignatureError::Shape(ArgErr::TooMany { given, .. }) => {
                assert_eq!(given, 3);
            }
            other => panic!("expected TooMany, got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py:335-336 `match_signature` raises
    /// `ArgErrMissing` for missing positional arguments.
    #[test]
    fn match_signature_missing_positional_raises() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let args = [pyre_object::w_int_new(1)];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("missing positional should ArgErr");
        match err {
            MatchSignatureError::Shape(ArgErr::Missing {
                missing,
                positional,
            }) => {
                assert_eq!(missing, vec!["b"]);
                assert!(positional);
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py:223-234 `match_signature` collects
    /// extra positional args into the *vararg slot.
    #[test]
    fn match_signature_collects_vararg() {
        let sig = crate::gateway::Signature::new(vec!["a"], Some("rest"), None, 0, 0);
        let args = [
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("vararg path should pass");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            // vararg lives at index 1 (after the named args).
            assert_eq!(pyre_object::w_tuple_len(scope[1]), 2);
        }
    }

    /// `_match_signature` with `kwargname` and one exact-str keyword that
    /// names no parameter: `newdict(kwargs=True)` stays KwargsDictStrategy.
    #[test]
    fn match_signature_exact_str_kwarg_uses_kwargs_strategy() {
        crate::test_hooks::install_hash_hook();
        let sig = crate::gateway::Signature::new(vec![], None, Some("kw"), 0, 0);
        let names = [pyre_object::w_str_new("a")];
        let values = [pyre_object::w_int_new(1)];
        let arguments = Arguments::with_kw(&[], &names, &values);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("unmatched exact str belongs in **kwargs");
        unsafe {
            assert_eq!(
                pyre_object::dictmultiobject::w_dict_get_strategy(scope[0]).strategy_kind(),
                pyre_object::dictmultiobject::StrategyKind::Kwargs
            );
        }
    }

    /// Same shape with a str-subclass keyword: `_collect_keyword_args`
    /// `setitem`s that object, and the kwargs dict promotes to object strategy.
    #[test]
    fn match_signature_str_subclass_kwarg_keeps_key_on_object_strategy() {
        crate::test_hooks::install_hash_hook();
        let _roots = pyre_object::gc_roots::push_roots();
        let w_class = pyre_object::w_type_new("StrSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let w_class = pyre_object::gc_roots::pin_root(w_class);
        let subclass =
            pyre_object::w_str_subclass_from_wtf8(rustpython_wtf8::Wtf8Buf::from("a"), w_class);
        let subclass_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(subclass);
        let sig = crate::gateway::Signature::new(vec![], None, Some("kw"), 0, 0);
        let values = [pyre_object::w_int_new(1)];
        let names = [pyre_object::gc_roots::shadow_stack_get(subclass_slot)];
        let arguments = Arguments::with_kw(&[], &names, &values);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("str subclass keyword belongs in **kwargs");
        let subclass = pyre_object::gc_roots::shadow_stack_get(subclass_slot);
        unsafe {
            assert_eq!(
                pyre_object::dictmultiobject::w_dict_get_strategy(scope[0]).strategy_kind(),
                pyre_object::dictmultiobject::StrategyKind::Object
            );
            let items = pyre_object::dictmultiobject::w_dict_items(scope[0]);
            assert_eq!(items.len(), 1);
            assert!(std::ptr::eq(items[0].0, subclass));
            assert_eq!(pyre_object::w_int_get_value(items[0].1), 1);
        }
    }

    /// pypy/interpreter/argument.py _match_keywords `match_keywords` matches
    /// keyword names against argnames + writes mapping.
    #[test]
    fn match_keywords_assigns_index() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let names = [pyre_object::w_str_new("b")];
        let mut mapping = vec![-1isize; 2];
        let remaining = match_keywords(&sig, 0, 0, 0, &names, &mut mapping)
            .expect("kwarg name match should succeed");
        assert_eq!(remaining, 0);
        assert_eq!(mapping, vec![-1, 0]); // 'b' is signature index 1, slot 1 (= 1 - input_argcount=0)
    }

    /// pypy/interpreter/argument.py:474-484 `match_keywords` flags
    /// posonly conflict via `ArgErrPosonlyAsKwds`.
    #[test]
    fn match_keywords_posonly_conflict() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 1);
        let names = [pyre_object::w_str_new("a")];
        let mut mapping = vec![-1isize; 2];
        let err = match_keywords(&sig, 0, 1, 0, &names, &mut mapping)
            .expect_err("posonly kwarg should raise");
        match err {
            MatchSignatureError::Shape(ArgErr::PosonlyAsKwds { posonly_kwds }) => {
                assert_eq!(posonly_kwds, vec!["a"]);
            }
            other => panic!("expected Shape(PosonlyAsKwds), got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py `_combine_starstarargs_wrapped`:
    /// dict expansion fills parallel name/value buffers.
    #[test]
    fn combine_starstarargs_wrapped_dict_expansion() {
        crate::test_hooks::install_hash_hook();
        let mut names: Vec<PyObjectRef> = vec![];
        let mut values: Vec<PyObjectRef> = vec![];
        let dict = pyre_object::dictmultiobject::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(dict, "x", pyre_object::w_int_new(10));
            pyre_object::w_dict_setitem_str(dict, "y", pyre_object::w_int_new(20));
        }
        combine_starstarargs_wrapped(&mut names, &mut values, dict, pyre_object::PY_NULL)
            .expect("dict expansion should succeed");
        assert_eq!(names.len(), 2);
        assert_eq!(values.len(), 2);
    }

    /// `_combine_starstarargs_wrapped` on a dict whose only key is a str
    /// subclass returns that same object in the names buffer, with its value.
    #[test]
    fn combine_starstarargs_wrapped_keeps_str_subclass_key() {
        crate::test_hooks::install_hash_hook();
        let _roots = pyre_object::gc_roots::push_roots();
        let w_class = pyre_object::w_type_new("StrSub", pyre_object::PY_NULL, std::ptr::null_mut());
        let w_class = pyre_object::gc_roots::pin_root(w_class);
        let subclass =
            pyre_object::w_str_subclass_from_wtf8(rustpython_wtf8::Wtf8Buf::from("a"), w_class);
        let subclass_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(subclass);
        let value = pyre_object::w_int_new(1);
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        let dict = pyre_object::dictmultiobject::w_dict_new();
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(dict);
        unsafe {
            pyre_object::dictmultiobject::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(subclass_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
        let mut names: Vec<PyObjectRef> = vec![];
        let mut values: Vec<PyObjectRef> = vec![];
        combine_starstarargs_wrapped(
            &mut names,
            &mut values,
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::PY_NULL,
        )
        .expect("str subclass key should expand");
        assert_eq!(names.len(), 1);
        assert_eq!(values.len(), 1);
        assert!(std::ptr::eq(
            names[0],
            pyre_object::gc_roots::shadow_stack_get(subclass_slot)
        ));
        assert!(std::ptr::eq(
            values[0],
            pyre_object::gc_roots::shadow_stack_get(value_slot)
        ));
    }

    /// pypy/interpreter/argument.py `__init__` with `w_stararg`:
    /// star-arg list extends `arguments_w` in-place (Phase A.3 routes
    /// through `combine_starargs_wrapped`).
    #[test]
    fn new_with_w_stararg_extends_arguments_w() {
        let pos = [pyre_object::w_int_new(1)];
        let stararg =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)]);
        let arguments = Arguments::new(&pos, None, None, Some(stararg), None, false, None)
            .expect("star expansion should succeed");
        assert_eq!(arguments.arguments_w.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py `__init__` with `w_starstararg`:
    /// star-star dict expands into the kwargs Vecs (Phase A.3 routes
    /// through `combine_starstarargs_wrapped`).
    #[test]
    fn new_with_w_starstararg_fills_kwargs() {
        crate::test_hooks::install_hash_hook();
        let pos = [pyre_object::w_int_new(1)];
        let dict = pyre_object::dictmultiobject::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(dict, "k", pyre_object::w_int_new(99));
        }
        let arguments = Arguments::new(&pos, None, None, None, Some(dict), false, None)
            .expect("starstar expansion should succeed");
        assert_eq!(arguments.arguments_w.len(), 1);
        let names = arguments.keyword_names_w.expect("kwargs should be Some");
        let values = arguments.keywords_w.expect("kwargs should be Some");
        assert_eq!(names.len(), 1);
        assert_eq!(values.len(), 1);
        unsafe {
            assert_eq!(pyre_object::w_str_get_wtf8(names[0]), "k");
            assert_eq!(pyre_object::w_int_get_value(values[0]), 99);
        }
    }

    /// pypy/interpreter/argument.py `replace_arguments` —
    /// returns a new Arguments with replaced positional list,
    /// keyword names/values shared (cloned in pyre).
    #[test]
    fn replace_arguments_preserves_kwargs() {
        let pos = [pyre_object::w_int_new(1)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let new_pos = vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)];
        let replaced = arguments.replace_arguments(new_pos);
        assert_eq!(replaced.arguments_w.len(), 2);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(replaced.arguments_w[0]), 2);
            assert_eq!(pyre_object::w_int_get_value(replaced.arguments_w[1]), 3);
        }
        let names = replaced.keyword_names_w.expect("kwargs preserved");
        assert_eq!(names.len(), 1);
    }

    /// pypy/interpreter/argument.py `prepend` — `[w_firstarg] + args_w`.
    #[test]
    fn prepend_inserts_at_front() {
        let pos = [pyre_object::w_int_new(2), pyre_object::w_int_new(3)];
        let arguments = Arguments::positional_only(&pos);
        let prepended = arguments.prepend(pyre_object::w_int_new(1));
        assert_eq!(prepended.arguments_w.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py `unpack` — returns
    /// (positional list, kwds dict).
    #[test]
    fn unpack_returns_args_and_kwds() {
        let pos = [pyre_object::w_int_new(10)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (args_w, kwds_w) = arguments.unpack().expect("string keys are valid");
        assert_eq!(args_w.len(), 1);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(args_w[0]), 10);
        }
        assert_eq!(kwds_w.len(), 1);
        let v = kwds_w.get(&Wtf8Buf::from("k")).expect("key 'k' present");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(*v), 99);
        }
    }

    /// pypy/interpreter/argument.py:71-74 — `kwds_w[key] = value`
    /// overwrites on duplicate (RPython interp-level dict semantics).
    /// Hand-built duplicate keyword names collapse to the last value,
    /// matching PyPy's behaviour on the same hand-built input.
    #[test]
    fn unpack_overwrites_duplicate_keys() {
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_str_new("dup"), pyre_object::w_str_new("dup")];
        let values = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (_, kwds_w) = arguments.unpack().expect("string keys are valid");
        assert_eq!(kwds_w.len(), 1, "duplicate keys must collapse");
        let v = kwds_w
            .get(&Wtf8Buf::from("dup"))
            .expect("key 'dup' present");
        unsafe {
            // last-write-wins → expected value 2 (not 1).
            assert_eq!(pyre_object::w_int_get_value(*v), 2);
        }
    }

    /// pypy/interpreter/argument.py:74 `space.text_w(w_name)` raises
    /// TypeError when the keyword name is not a string; `unpack`
    /// propagates.
    #[test]
    fn unpack_non_string_key_returns_err() {
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_int_new(7)];
        let values = [pyre_object::w_int_new(42)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        match arguments.unpack() {
            Ok(_) => panic!("non-string keyword name should TypeError"),
            Err(err) => {
                assert_eq!(err.kind, crate::PyErrorKind::TypeError);
                // baseobjspace.py `_typed_unwrap_error`:
                // `expected str, got <T> object`.
                assert!(err.message_text().contains("expected str"));
            }
        }
    }

    /// `unpack` keeps a lone-surrogate keyword name as WTF-8
    /// (`W_UnicodeObject.text_w` returns `self._utf8`).
    #[test]
    fn unpack_keeps_surrogate_keyword_name() {
        let mut name = Wtf8Buf::new();
        name.push(rustpython_wtf8::CodePoint::from_u32(0xD800).unwrap());
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_str_from_wtf8(name.clone())];
        let values = [pyre_object::w_int_new(2)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (_, kwds_w) = arguments.unpack().expect("surrogate names stay WTF-8");
        assert_eq!(kwds_w.len(), 1);
        let v = kwds_w.get(&name).expect("surrogate key present");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(*v), 2);
        }
    }

    /// argument.py `__init__`: `_jit_few_keywords` is true when
    /// `keyword_names_w is None`.
    #[test]
    fn jit_few_keywords_true_without_names() {
        let pos = [pyre_object::w_int_new(1)];
        let arguments = Arguments::positional_only(&pos);
        assert!(arguments.jit_few_keywords);
        let replaced = arguments.replace_arguments(vec![pyre_object::w_int_new(2)]);
        assert!(replaced.jit_few_keywords);
    }

    /// argument.py `__init__`: `_jit_few_keywords` is
    /// `jit.isconstant(len(keyword_names_w))` when the list is present.
    /// Residual `isconstant` is false, so a present names list is not
    /// "few" until a trace sees a constant length.
    #[test]
    fn jit_few_keywords_false_for_present_names_at_interp() {
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(1)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        assert!(!arguments.jit_few_keywords);
        let replaced = arguments.replace_arguments(Vec::new());
        assert!(!replaced.jit_few_keywords);
    }

    /// pypy/interpreter/argument.py `parse_into_scope` happy
    /// path: matches signature, returns scope_length.
    #[test]
    fn parse_into_scope_happy_path() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&pos);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let scopelen = arguments
            .parse_into_scope(
                pyre_object::PY_NULL,
                &mut scope,
                "f",
                &sig,
                None,
                pyre_object::PY_NULL,
            )
            .expect("happy path should succeed");
        assert_eq!(scopelen, 2);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(scope[1]), 2);
        }
    }

    /// pypy/interpreter/argument.py `parse_into_scope` shape
    /// mismatch surfaces as TypeError with `<fnname>() ` prefix.
    #[test]
    fn parse_into_scope_shape_error_formats_fnname_prefix() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&pos);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .parse_into_scope(
                pyre_object::PY_NULL,
                &mut scope,
                "myfn",
                &sig,
                None,
                pyre_object::PY_NULL,
            )
            .expect_err("too many args should TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message_text().starts_with("myfn() "));
    }

    /// pypy/interpreter/argument.py `parse_obj` special-cases
    /// "takes no keyword arguments" when no **kwarg + no kwonly args.
    #[test]
    fn parse_obj_unknown_kwds_no_kwarg_message() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1)];
        let names = [pyre_object::w_str_new("wibble")];
        let values = [pyre_object::w_int_new(2)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let err = arguments
            .parse_obj(
                pyre_object::PY_NULL,
                "myfn",
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("unknown kwarg should TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(err.message_text(), "myfn() takes no keyword arguments");
    }

    /// pypy/interpreter/argument.py `frompacked` builds
    /// Arguments from packed star args.
    #[test]
    fn frompacked_expands_w_args() {
        let w_args =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(1), pyre_object::w_int_new(2)]);
        let arguments = Arguments::frompacked(Some(w_args), None).expect("frompacked");
        assert_eq!(arguments.arguments_w.len(), 2);
    }

    /// pypy/interpreter/argument.py `topacked` round-trips
    /// arguments_w through newtuple + newdict.
    #[test]
    fn topacked_round_trips() {
        crate::test_hooks::install_hash_hook();
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (w_args, w_kwds) = arguments.topacked().expect("topacked");
        unsafe {
            assert_eq!(pyre_object::w_tuple_len(w_args), 2);
            assert!(pyre_object::is_dict(w_kwds));
        }
    }
}
