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
use rustpython_wtf8::{Wtf8, Wtf8Buf};

/// The `%T` conversion `oefmt` applies, for the `*args` / `**kwargs` parity
/// messages ("Value after * must be an iterable, not %T" /
/// "Value after ** must be a mapping, not %T", no function-name prefix;
/// `_Py_Check_ArgsIterable` / `_PyEval_FormatKwargsError`) and for `text_w`'s
/// non-string TypeError ("expected str, got %T object"). Duplicate-keyword
/// and non-string-keyword text still uses `raise_type_error` /
/// `object_functionstr`.
use crate::error::{FmtArg, type_name_of};

/// `raise_type_error!(w_function, "msg", args)`.
///
/// `@specialize.arg(2)` makes `msg` a per-call-site constant, so
/// `"%s " + msg` folds per specialization. Both format strings are
/// `&'static str` known at the call site: `valuefmt` and
/// `concat!("%s ", valuefmt)`.
#[macro_export]
macro_rules! raise_type_error {
    ($w_function:expr, $valuefmt:literal, $args:expr $(,)?) => {
        $crate::argument::raise_type_error($w_function, $valuefmt, concat!("%s ", $valuefmt), $args)
    };
}

/// `argument.py` `raise_type_error`.
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
/// A null `w_function` builds `oefmt(TypeError, valuefmt, args)`.
/// Otherwise the prefixed format is the per-call-site constant
/// (`@specialize.arg(2)`), and the first argument is
/// `object_functionstr(w_function)`. An error from that lookup is the
/// raised error.
///
/// The body is traced, as `argument.py` `raise_type_error` is (only
/// `@specialize.arg(2)`). `object_functionstr` runs eagerly, as the
/// argument of `oefmt` in `argument.py`. `crate::error::oefmt` still
/// formats at construction, where `OpErrFmt._compute_value` formats at
/// display (pre-existing `oefmt` difference).
pub fn raise_type_error(
    w_function: PyObjectRef,
    valuefmt: &'static str,
    prefixed_valuefmt: &'static str,
    args: &[FmtArg<'_>],
) -> crate::PyError {
    if w_function.is_null() {
        return type_error_oefmt(valuefmt, args);
    }
    // `object_functionstr` runs Python and can collect. The callable and
    // every `%R`/`%S`/`%T`/`%N` operand are caller-owned words; publish the
    // whole set before the first normalize and hand `oefmt` the slots' words.
    let mut objs = Vec::new();
    for arg in args {
        if let FmtArg::Obj(obj) = arg {
            objs.push(*obj);
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    let fn_slot = roots.publish(&[w_function]);
    let objs_base = roots.publish(&objs);
    roots.normalize(fn_slot, 1 + objs.len());
    match crate::baseobjspace::object_functionstr(roots.get(fn_slot)) {
        Ok(prefix) => {
            let mut fmt_args = Vec::with_capacity(1 + args.len());
            fmt_args.push(FmtArg::Bytes(prefix.as_bytes()));
            let mut obj_i = 0;
            for arg in args {
                match *arg {
                    FmtArg::Obj(_) => {
                        fmt_args.push(FmtArg::Obj(roots.get(objs_base + obj_i)));
                        obj_i += 1;
                    }
                    other => fmt_args.push(other),
                }
            }
            type_error_oefmt(prefixed_valuefmt, &fmt_args)
        }
        Err(err) => err,
    }
}

/// `oefmt` of the TypeError class. Before `init_typeobjects` the class is
/// null and `oefmt` reports `RuntimeError`; the carrier stays `TypeError`.
pub(crate) fn type_error_oefmt(valuefmt: &'static str, args: &[FmtArg<'_>]) -> crate::PyError {
    let w_type =
        crate::typedef::gettypeobject(&pyre_object::interp_exceptions::EXC_TYPE_ERROR_TYPE);
    let mut err = crate::error::oefmt(w_type, valuefmt, args);
    if w_type.is_null() {
        err.set_kind(crate::PyErrorKind::TypeError);
    }
    err
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
///
/// `@jit.look_inside_iff(lambda ...:
///     jit.isconstant(len(keyword_names_w) and
///     jit.isconstant(existingkeywords_w)))`.
fn check_not_duplicate_kwargs_iff(
    existingkeywords_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    _keywords_w: &[PyObjectRef],
    _w_function: PyObjectRef,
) -> bool {
    let names_and_existing =
        keyword_names_w.len() != 0 && majit_rlib::jit::isconstant(existingkeywords_w);
    majit_rlib::jit::isconstant(&names_and_existing)
}

#[majit_macros::look_inside_iff(check_not_duplicate_kwargs_iff)]
pub fn check_not_duplicate_kwargs(
    existingkeywords_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    _keywords_w: &[PyObjectRef],
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // argument.py `_check_not_duplicate_kwargs`: the `for w_key in
    // keyword_names_w` / `contains_w_names` loop lives on the
    // `unroll_safe` original (`look_inside_iff`). `contains_w_names` can
    // collect, so both lists and the callable are read from their slots.
    let existing_len = existingkeywords_w.len();
    let names_len = keyword_names_w.len();
    let roots = pyre_object::gc_roots::push_roots();
    let function_slot = roots.pin_roots(&[w_function]);
    let existing_base = roots.pin_roots(existingkeywords_w);
    let names_base = roots.pin_roots(keyword_names_w);
    let mut existing_now = Vec::with_capacity(existing_len);
    for _ in 0..existing_len {
        existing_now.push(pyre_object::PY_NULL);
    }
    for i in 0..names_len {
        pyre_object::gc_roots::shadow_stack_copy_range(existing_base, &mut existing_now);
        if contains_w_names(roots.get(names_base + i), &existing_now) {
            // `_check_not_duplicate_kwargs` calls `raise_type_error` in the
            // loop (`'%S'` of `w_key`). `contains_w_names` can collect, so
            // both words are read back from the slots this bracket published.
            let w_key = roots.get(names_base + i);
            let w_function = roots.get(function_slot);
            return Err(raise_type_error!(
                w_function,
                "got multiple values for keyword argument '%S'",
                &[FmtArg::Obj(w_key)],
            ));
        }
    }
    Ok(())
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
/// `_do_combine_starstarargs_wrapped` — when `values_w` is `Some` (an exact
/// `dict`, or a `dict` subclass whose `__iter__` is still `dict.__iter__`),
/// the value for key `i` is `values_w[i]` from the caller's one walk of the
/// entries. The seen-set still records `text_w` (`key in seen`). When
/// `values_w` is `None`, the value is `space.getitem`, so a mapping's
/// `__getitem__` override runs.
///
/// `keyword_names_w` / `keywords_w` are the caller's two output lists
/// (`[None] * len(keys_w)` in `_combine_starstarargs_wrapped`), passed as
/// shadow-stack base indices of GC-visible ranges the caller reserved.
pub fn do_combine_starstarargs_wrapped(
    keys_w: &[PyObjectRef],
    w_starstararg: PyObjectRef,
    keyword_names_w: usize,
    keywords_w: usize,
    existingkeywords_w: Option<&[PyObjectRef]>,
    values_w: Option<&[PyObjectRef]>,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // argument.py `seen = {}` — membership of already-emitted keys.  A
    // small list, not `HashMap`: the std HashMap graph does not annotate.
    let mut seen: Vec<rustpython_wtf8::Wtf8Buf> = Vec::new();
    // The per-key value lookup below runs `__getitem__` when `values_w` is
    // None, so every turn is a collection point.
    // `gctransform/framework.py push_roots` stores every live GCREF before
    // that call; `pop_roots` / `expand_pop_roots` writes each one back.
    // `argument.py _do_combine_starstarargs_wrapped` writes
    // `keyword_names_w[i] = w_key` / `keywords_w[i] = w_value` into GC lists
    // the caller allocated and owns. Those lists are the two shadow-stack
    // ranges whose bases are `keyword_names_w` / `keywords_w`; this helper
    // fills slot `i` of each during the iteration and returns None.
    let _roots = pyre_object::gc_roots::push_roots();
    let existing_len = existingkeywords_w.map_or(0, |existing| existing.len());
    let n = keys_w.len();
    let values_slice = values_w.unwrap_or(&[]);
    debug_assert!(values_w.is_none() || values_slice.len() == n);
    let root_base = pyre_object::gc_roots::publish_roots(&[w_starstararg, w_function]);
    let existing_base = pyre_object::gc_roots::publish_roots(existingkeywords_w.unwrap_or(&[]));
    let keys_base = pyre_object::gc_roots::publish_roots(keys_w);
    let values_base = pyre_object::gc_roots::publish_roots(values_slice);
    pyre_object::gc_roots::normalize_roots(root_base, 2 + existing_len + n + values_slice.len());
    let w_starstararg = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let w_function = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let mut existing_now = vec![pyre_object::PY_NULL; existing_len];
    for i in 0..n {
        let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
        // argument.py — `key = space.text_w(w_key)`; raise TypeError
        // if w_key is not a string.  `_PyStack_UnpackDict` reports this
        // without the callable's qualname or the offending key's type, where
        // argument.py formats `"keywords must be strings, not '%T'"`
        // through `raise_type_error`.
        if unsafe { !pyre_object::is_str(w_key) } {
            return Err(crate::PyError::type_error("keywords must be strings"));
        }
        // `space.text_w` preserves lone surrogates as WTF-8 bytes; keep the
        // key in WTF-8 so a surrogate keyword name survives the seen-set and
        // the dict value read.
        let key = unsafe { pyre_object::w_str_get_wtf8(w_key).to_owned() };
        // argument.py:439-445 — duplicate check against existing kwargs +
        // already-seen names in this iteration.
        let already_in_existing = existing_len != 0 && {
            pyre_object::gc_roots::shadow_stack_copy_range(existing_base, &mut existing_now);
            contains_w_names(w_key, &existing_now)
        };
        let mut already_seen = false;
        for j in 0..seen.len() {
            if seen.as_slice()[j] == key {
                already_seen = true;
                break;
            }
        }
        if already_in_existing || already_seen {
            // `contains_w_names` can collect; `%S` needs the key object.
            let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
            return Err(raise_type_error!(
                w_function(),
                "got multiple values for keyword argument '%S'",
                &[FmtArg::Obj(w_key)],
            ));
        }
        // `contains_w_names` can collect; the key is read back from its slot.
        let w_key = pyre_object::gc_roots::shadow_stack_get(keys_base + i);
        // argument.py — `keyword_names_w[i] = w_key`.
        pyre_object::gc_roots::shadow_stack_set(keyword_names_w + i, w_key);
        // argument.py — `space.getitem` when `values_w` is None. A str subclass
        // is probed as `w_key`; the WTF-8 `seen` key is the wrong probe.
        let w_value = if values_w.is_some() {
            pyre_object::gc_roots::shadow_stack_get(values_base + i)
        } else {
            crate::baseobjspace::getitem(w_starstararg(), w_key)?
        };
        // argument.py — `keywords_w[i] = w_value`.
        pyre_object::gc_roots::shadow_stack_set(keywords_w + i, w_value);
        // argument.py `seen[key] = None` — record key, value slot
        // is unread.
        seen.push(key);
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
    match crate::baseobjspace::fixedview(pyre_object::gc_roots::shadow_stack_get(star_slot), -1) {
        Ok(args_w) => {
            pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut arguments_w[..]);
            // argument.py — `self.arguments_w = self.arguments_w + args_w`.
            arguments_w.extend(args_w);
            Ok(())
        }
        Err(e) => {
            // [3.15-spec] `_Py_Check_ArgsIterable` formats
            // "Value after * must be an iterable, not %.200s" with no
            // `object_functionstr` prefix. `raise_type_error` in
            // `argument.py` prefixes `"%s " + msg`. Evidence:
            // `lib-python/3/test/pickletester.py` `test_bad_newobj_ex_args`.
            // `raise_type_error` has `@specialize.arg(2)` and no `@jit.*`.
            if e.kind == crate::PyErrorKind::TypeError
                && !crate::baseobjspace::is_iterable(pyre_object::gc_roots::shadow_stack_get(
                    star_slot,
                ))
            {
                Err(type_error_oefmt(
                    "Value after * must be an iterable, not %T",
                    &[FmtArg::Obj(pyre_object::gc_roots::shadow_stack_get(
                        star_slot,
                    ))],
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
    // argument.py:109 — fast path via view_as_kwargs.  Both halves of
    // the tuple are `Option`; PyPy's base default is `(None, None)`
    // while the kwargsdict-aware override returns `(Some, Some)`.
    let (fast_names, fast_values) =
        crate::baseobjspace::view_as_kwargs(pyre_object::gc_roots::shadow_stack_get(root_base));
    if let Some(names) = fast_names {
        let values = fast_values
            .expect("baseobjspace.py:1159 view_as_kwargs returns matching Some/Some or None/None");
        let mut names = names;
        let mut values = values;
        pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut keyword_names_out[..]);
        pyre_object::gc_roots::shadow_stack_copy_range(values_base, &mut keywords_out[..]);
        // argument.py:111-119 — merge with optional duplicate check.
        if !keyword_names_out.is_empty() {
            // The duplicate check compares through `is_w`, which can collect.
            let fast_names_base = pyre_object::gc_roots::pin_roots(&names);
            let fast_values_base = pyre_object::gc_roots::pin_roots(&values);
            check_not_duplicate_kwargs(
                keyword_names_out,
                &names,
                &values,
                pyre_object::gc_roots::shadow_stack_get(root_base + 1),
            )?;
            pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut keyword_names_out[..]);
            pyre_object::gc_roots::shadow_stack_copy_range(values_base, &mut keywords_out[..]);
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
            pyre_object::is_dict(pyre_object::gc_roots::shadow_stack_get(root_base))
        } else {
            crate::baseobjspace::isinstance_w(
                pyre_object::gc_roots::shadow_stack_get(root_base),
                w_dict_type,
            )
        };
        if !isinst {
            false
        } else {
            // `argument.py` — `space.findattr(space.type(...),
            // w_st_iter)`: the `getattr` is issued on the type object
            // itself, going through the metaclass / descriptor lookup
            // chain, so a raising `__getattribute__` on the metaclass
            // surfaces here instead of reading as "no `__iter__`".
            let w_obj_type =
                crate::typedef::r#type(pyre_object::gc_roots::shadow_stack_get(root_base))
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
    let mut dict_values_w: Vec<PyObjectRef> = Vec::new();
    let keys_w: Vec<PyObjectRef> = if is_dict {
        // [3.15-spec] An exact dict, or a dict subclass that keeps `dict.__iter__`,
        // hands over its entries in one walk and no key is hashed again;
        // `_combine_starstarargs_wrapped` (`argument.py`) unpacks the keys and reads
        // each value back through `dict_getitem`, which runs a str subclass's
        // `__hash__` at the call site.  `_PyStack_UnpackDict` walks the entries.
        // Evidence: `test_functools.TestPartial.test_call_safety_against_reentrant_mutation`.
        // Neither `_combine_starstarargs_wrapped` nor `_do_combine_starstarargs_wrapped`
        // carries a `@jit.*` hint.
        let dict = crate::type_methods::resolve_dict_backing(
            pyre_object::gc_roots::shadow_stack_get(root_base),
        );
        let dict = pyre_object::gc_roots::pin_root(if dict.is_null() {
            pyre_object::gc_roots::shadow_stack_get(root_base)
        } else {
            dict
        });
        let items = unsafe { pyre_object::w_dict_items(dict) };
        let mut keys_w = Vec::with_capacity(items.len());
        dict_values_w.reserve(items.len());
        for (k, v) in items {
            keys_w.push(k);
            dict_values_w.push(v);
        }
        keys_w
    } else {
        // argument.py:131-138 — the `try` covers the whole
        // `space.call_method(w_starstararg, "keys")`, so an
        // AttributeError raised either while finding `keys` OR inside
        // the keys() call is converted to the mapping TypeError below.
        let w_keys_meth = match crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(root_base),
            "keys",
        ) {
            Ok(m) => m,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                // [3.15-spec] `_PyEval_FormatKwargsError` formats
                // "Value after ** must be a mapping, not %T" with no
                // function-name prefix. `raise_type_error` in
                // `argument.py` prefixes `"%s " + msg`. Evidence:
                // `lib-python/3/test/pickletester.py` `test_bad_newobj_ex_args`.
                return Err(type_error_oefmt(
                    "Value after ** must be a mapping, not %T",
                    &[FmtArg::Obj(pyre_object::gc_roots::shadow_stack_get(
                        root_base,
                    ))],
                ));
            }
            Err(e) => return Err(e),
        };
        let w_keys = match crate::call::call_function_impl_result(w_keys_meth, &[]) {
            Ok(w) => w,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                return Err(type_error_oefmt(
                    "Value after ** must be a mapping, not %T",
                    &[FmtArg::Obj(pyre_object::gc_roots::shadow_stack_get(
                        root_base,
                    ))],
                ));
            }
            Err(e) => return Err(e),
        };
        crate::baseobjspace::unpackiterable(w_keys, -1)?
    };
    // argument.py:140-144 — the caller allocates both output lists
    // (`keywords_w = [None] * len(keys_w)`, `keyword_names_w = [None] * len(keys_w)`)
    // as GC-visible storage it owns, then `_do_combine_starstarargs_wrapped`
    // fills them in place and returns None. Interpreter-level argument lists
    // are `Vec<PyObjectRef>` plus shadow-stack slots, so the two buffers are
    // contiguous ranges of nulls reserved inside this `push_roots` scope
    // (`gctransform/framework.py push_roots` / `pop_roots`) and passed as
    // base indices.
    pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut keyword_names_out[..]);
    pyre_object::gc_roots::shadow_stack_copy_range(values_base, &mut keywords_out[..]);
    let existing: Option<&[PyObjectRef]> = if keyword_names_out.is_empty() {
        None
    } else {
        Some(keyword_names_out.as_slice())
    };
    let n = keys_w.len();
    let nulls = vec![pyre_object::PY_NULL; n];
    let keywords_w = pyre_object::gc_roots::publish_roots(&nulls);
    let keyword_names_w = pyre_object::gc_roots::publish_roots(&nulls);
    do_combine_starstarargs_wrapped(
        &keys_w,
        pyre_object::gc_roots::shadow_stack_get(root_base),
        keyword_names_w,
        keywords_w,
        existing,
        if is_dict { Some(&dict_values_w) } else { None },
        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
    )?;
    // argument.py:145-150 — merge into the running output buffers.
    pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut keyword_names_out[..]);
    pyre_object::gc_roots::shadow_stack_copy_range(values_base, &mut keywords_out[..]);
    let mut names_add = vec![pyre_object::PY_NULL; n];
    let mut values_add = vec![pyre_object::PY_NULL; n];
    pyre_object::gc_roots::shadow_stack_copy_range(keyword_names_w, &mut names_add);
    pyre_object::gc_roots::shadow_stack_copy_range(keywords_w, &mut values_add);
    keyword_names_out.extend(names_add);
    keywords_out.extend(values_add);
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
    /// `_match_signature` indexes the `make_sure_not_resized` GcArray twins,
    /// not this Option, so a present-but-empty list stays distinct from None
    /// (`_jit_few_keywords` / `_combine_starstarargs_wrapped`).
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
    /// for the `positional_only` / `with_kw` shortcuts;
    /// `pyframe.py make_arguments` stores the CALL methodcall flag here
    /// so `_match_signature` can raise `ArgErrTooManyMethod`.
    pub methodcall: bool,
    /// `make_sure_not_resized(self.arguments_w)`: `_match_signature` indexes
    /// a `Ptr(GcArray(PyObjectRef))` (`GETFIELD` + `getarrayitem_gc`).
    pub arguments_w_gc: *mut pyre_object::FixedObjectArray,
    /// `make_sure_not_resized(self.keyword_names_w)` twin of `arguments_w_gc`.
    pub keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    /// `make_sure_not_resized(self.keywords_w)` twin of `arguments_w_gc`.
    pub keywords_w_gc: *mut pyre_object::FixedObjectArray,
}

/// Pin a `make_sure_not_resized` GcArray without a List→PyObject cast at
/// the caller. `pin_root` takes `PyObjectRef`; `*mut FixedObjectArray` is
/// SomeList, and `__cast_instance_intrinsic` of that List Skips the
/// look-inside body (`match_signature`, `parse_into_scope`, `call_args`).
/// The helper is residual (`dont_look_inside`); `erase_shadow_stack`
/// treats the call as `pin_root` (`Leaf::Pin`) so the jitcode still
/// aliases the slot to the GETFIELD local.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn pin_root_gcarray(
    root: *mut pyre_object::FixedObjectArray,
) -> *mut pyre_object::FixedObjectArray {
    pyre_object::gc_roots::pin_root(root as PyObjectRef) as *mut pyre_object::FixedObjectArray
}

/// Reload a pinned `make_sure_not_resized` twin from its shadow-stack slot.
/// `pin_root_gcarray` returns the pointer at pin time; a collecting call
/// forwards the slot in place (`framework.py pop_roots`).
/// Always inlined so the caller is not a param_slots callee of this get
/// (`copy_shadow_range_to_vec`).
#[inline(always)]
fn gcarray_from_slot(slot: usize) -> *mut pyre_object::FixedObjectArray {
    pyre_object::gc_roots::shadow_stack_get(slot) as *mut pyre_object::FixedObjectArray
}

/// Expanded at each `_match_signature` site so the Gets stay in that
/// body. A helper that takes slot indices is ParamSlots and refuses
/// erase (`copy_shadow_range_to_vec`).
macro_rules! reload_gc_twins {
    ($args_slot:expr, $names_slot:expr, $keywords_slot:expr) => {
        (
            pyre_object::gc_roots::shadow_stack_get($args_slot)
                as *mut pyre_object::FixedObjectArray,
            pyre_object::gc_roots::shadow_stack_get($names_slot)
                as *mut pyre_object::FixedObjectArray,
            pyre_object::gc_roots::shadow_stack_get($keywords_slot)
                as *mut pyre_object::FixedObjectArray,
        )
    };
}

/// `make_sure_not_resized` list stored as a raw pointer on this non-GC
/// `Arguments` struct. Empty is PyPy's `None` / `_ll_prebuilt_empty_array`.
/// Non-empty uses the stable (non-moving) path so the pointer stays valid
/// across later collections; a young collecting twin would leave a
/// forwarded stub in the field.
fn freeze_one_gc_array(values: &[PyObjectRef]) -> *mut pyre_object::FixedObjectArray {
    if values.is_empty() {
        pyre_object::object_array::prebuilt_empty_object_array()
    } else {
        unsafe { pyre_object::object_array::alloc_mro_block_gc(values) }
    }
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
            arguments_w: copy_object_slice(args_w),
            keyword_names_w: keyword_names_w.map(copy_object_slice),
            keywords_w: keywords_w.map(copy_object_slice),
            // argument.py `__init__` writes `_jit_few_keywords` after
            // `_combine_wrapped`; seed the same formula so the field is
            // defined before that helper runs.
            jit_few_keywords: match keyword_names_w {
                None => true,
                Some(names) => majit_rlib::jit::isconstant(&names.len()),
            },
            methodcall,
            arguments_w_gc: std::ptr::null_mut(),
            keyword_names_w_gc: std::ptr::null_mut(),
            keywords_w_gc: std::ptr::null_mut(),
        };
        let w_function = w_function.unwrap_or(pyre_object::PY_NULL);
        arguments._combine_wrapped(w_stararg, w_starstararg, w_function)?;
        // argument.py — recompute after `_combine_wrapped`, since
        // that helper may have grown `keyword_names_w`.
        arguments.jit_few_keywords = match arguments.keyword_names_w.as_deref() {
            None => true,
            Some(names) => majit_rlib::jit::isconstant(&names.len()),
        };
        arguments.freeze_gc_arrays();
        Ok(arguments)
    }

    /// Rebuild the `make_sure_not_resized` GcArray twins from the owned Vecs.
    /// The Vec words are pinned across the three mallocs (`framework.py
    /// push_roots`); each twin is pinned before the next so a collecting
    /// allocation cannot drop a header this struct holds only as a raw
    /// pointer.
    #[majit_macros::dont_look_inside]
    pub fn freeze_gc_arrays(&mut self) {
        let _roots = pyre_object::gc_roots::push_roots();
        let names_is_some = self.keyword_names_w.is_some();
        let values_is_some = self.keywords_w.is_some();
        let names = self.keyword_names_w.as_deref().unwrap_or(&[]);
        let values = self.keywords_w.as_deref().unwrap_or(&[]);
        let args_base = pyre_object::gc_roots::pin_roots(&self.arguments_w);
        let names_base = pyre_object::gc_roots::pin_roots(names);
        let values_base = pyre_object::gc_roots::pin_roots(values);
        let n_args = self.arguments_w.len();
        let n_names = names.len();
        let n_values = values.len();
        let arguments_w: Vec<PyObjectRef> = (0..n_args)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(args_base + i))
            .collect();
        let args_live = pyre_object::gc_roots::pin_roots(&arguments_w);
        let keyword_names_w: Vec<PyObjectRef> = (0..n_names)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(names_base + i))
            .collect();
        let names_live = pyre_object::gc_roots::pin_roots(&keyword_names_w);
        let keywords_w: Vec<PyObjectRef> = (0..n_values)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(values_base + i))
            .collect();
        let values_live = pyre_object::gc_roots::pin_roots(&keywords_w);
        let arguments_w: Vec<PyObjectRef> = (0..n_args)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(args_live + i))
            .collect();
        self.arguments_w_gc = freeze_one_gc_array(&arguments_w);
        let args_gc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pin_root_gcarray(self.arguments_w_gc);
        let keyword_names_w: Vec<PyObjectRef> = (0..n_names)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(names_live + i))
            .collect();
        self.keyword_names_w_gc = freeze_one_gc_array(&keyword_names_w);
        let names_gc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pin_root_gcarray(self.keyword_names_w_gc);
        let keywords_w: Vec<PyObjectRef> = (0..n_values)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(values_live + i))
            .collect();
        self.keywords_w_gc = freeze_one_gc_array(&keywords_w);
        let keywords_gc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pin_root_gcarray(self.keywords_w_gc);
        self.arguments_w_gc = gcarray_from_slot(args_gc_slot);
        self.keyword_names_w_gc = gcarray_from_slot(names_gc_slot);
        self.keywords_w_gc = gcarray_from_slot(keywords_gc_slot);
        self.arguments_w = (0..n_args)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(args_live + i))
            .collect();
        self.keyword_names_w = if names_is_some {
            Some(
                (0..n_names)
                    .map(|i| pyre_object::gc_roots::shadow_stack_get(names_live + i))
                    .collect(),
            )
        } else {
            None
        };
        self.keywords_w = if values_is_some {
            Some(
                (0..n_values)
                    .map(|i| pyre_object::gc_roots::shadow_stack_get(values_live + i))
                    .collect(),
            )
        } else {
            None
        };
    }

    /// Owned-Vec constructor used by CALL_KW packing (`arguments_from_call_kw`)
    /// and the match_signature walker. `__init__` without star expansion.
    pub fn from_owned_vecs(
        arguments_w: Vec<PyObjectRef>,
        keyword_names_w: Option<Vec<PyObjectRef>>,
        keywords_w: Option<Vec<PyObjectRef>>,
        jit_few_keywords: bool,
        methodcall: bool,
    ) -> Self {
        let mut arguments = Self {
            arguments_w,
            keyword_names_w,
            keywords_w,
            jit_few_keywords,
            methodcall,
            arguments_w_gc: std::ptr::null_mut(),
            keyword_names_w_gc: std::ptr::null_mut(),
            keywords_w_gc: std::ptr::null_mut(),
        };
        arguments.freeze_gc_arrays();
        arguments
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
    /// Residual: star expansion walks Python and mutates the owned
    /// keyword Vecs in place.
    #[majit_macros::dont_look_inside]
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
        // (mirrors argument.py `if self.keyword_names_w is None:
        // self.keyword_names_w = keyword_names_w`).  Reassign Some after
        // the helper runs whether or not it produced any keys, mirroring
        // PyPy's unconditional `self.keyword_names_w = ...` after `**{}`.
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
                        return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                            "expected str, got ",
                            tp,
                            " object",
                        )));
                    }
                };
                // argument.py `unpack` — `kwds_w[key] = ...` overwrites on duplicate.
                kwds_w.insert(key, w_value);
            }
        }
        let arguments_w = copy_shadow_range_to_vec(args_base, args_n);
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
        // argument.py `replace_arguments` builds a new `Arguments`
        // through `__init__`, which recomputes `_jit_few_keywords`.
        let jit_few_keywords = match keyword_names_w.as_deref() {
            None => true,
            Some(names) => majit_rlib::jit::isconstant(&names.len()),
        };
        Self::from_owned_vecs(args_w, keyword_names_w, keywords_w, jit_few_keywords, false)
    }

    /// pypy/interpreter/argument.py `prepend`.
    ///
    /// ```python
    /// def prepend(self, w_firstarg):
    ///     "Return a new Arguments with a new argument inserted first."
    ///     return self.replace_arguments([w_firstarg] + self.arguments_w)
    /// ```
    /// Residual: rtyping the copied `GCREF` words has no primitive repr.
    #[majit_macros::dont_look_inside]
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
    /// Returns [`MatchOutcome::Shape`] on shape mismatch. PyPy raises ArgErr
    /// from the call site and converts to `oefmt(space.w_TypeError, ...)`
    /// at `parse_obj` / `parse_into_scope`.
    ///
    /// Inputs (matching the upstream parameter names):
    /// - `w_firstarg`: optional implicit first arg (e.g. `self` for
    ///   bound methods); `PY_NULL` means absent.
    /// - `scope_w`: output buffer, must be `signature.scope_length()`
    ///   long, pre-filled with `PY_NULL`.
    /// - `signature`: callee signature.
    /// - `defaults_w`: positional default values; empty when PyPy passes `None`.
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
        defaults_w: &[PyObjectRef],
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> MatchOutcome {
        let co_posonlyargcount = signature.posonlyargcount;
        let co_argcount = signature.num_argnames();
        let co_kwonlyargcount = signature.num_kwonlyargnames();
        let mut too_many_args = false;

        // argument.py:191-201 — put `w_firstarg` into the scope.
        // `[w_firstarg] + args_w` for `def meth(*args)` is a flag plus
        // an extra stararg element: a sibling `Vec<PyObjectRef>` local
        // unions with `upfront` (`UnionError: RustVec ∪ Integer`) and
        // `Option<Vec>::as_deref` has no ClassDef.
        let mut upfront: usize = 0;
        let prepend_first = !w_firstarg.is_null() && co_argcount == 0;
        if !w_firstarg.is_null() && co_argcount > 0 {
            scope_w[0] = w_firstarg;
            upfront = 1;
        }
        // `make_sure_not_resized` twins (`GETFIELD` + `arraylen` /
        // `getarrayitem_gc`). Reading the owned Vecs first residualizes
        // `ll_vec_*` and clears the heapcache that a CALL_KW walker plants
        // on these fields. `_jit_few_keywords` / `methodcall` are write-once
        // (`Arguments.__init__`); load them with the twins. `pin_root_gcarray`
        // is residual (`dont_look_inside_cannot_raise`) and
        // `invalidate_caches_varargs` drops that seed, so a later GETFIELD
        // of the recording-time native box would stay in the trace.
        // Upstream `argument.py Arguments` is GC-managed and can keep
        // those GETFIELDs; the native box cannot.
        let arguments_w_gc = self.arguments_w_gc;
        let keyword_names_w_gc = self.keyword_names_w_gc;
        let keywords_w_gc = self.keywords_w_gc;
        let jit_few_keywords = self.jit_few_keywords;
        let methodcall = self.methodcall;
        let args_len = unsafe { (&*arguments_w_gc).len() };
        let names_len = unsafe { (&*keyword_names_w_gc).len() };
        let num_args = args_len + usize::from(prepend_first);
        let avail = num_args + upfront;
        // argument.py:206-209 — kwds count. Empty is PyPy's `None`.
        let num_kwds = names_len;

        // `w_kw_defs`, the keyword values, the leftover positionals that
        // become `*args`, the positional defaults and the words already
        // copied into `scope_w` are all read after the allocations below —
        // the vararg tuple, the dict itself, the keyword collection, the
        // kwonly-default lookups — and a `list` or `dict` among them moves
        // under any of those. `_match_signature` assigns `scope_w[i]` in
        // place. Scalars are pinned first (`ShadowStackFrameworkGCTransformer.push_roots`
        // of those locals). `erase_shadow_stack` aliases those pins
        // so the jitcode sees ordinary array ops; Close is `RootScope` Drop.
        let _roots = pyre_object::gc_roots::push_roots();
        let args_gc_slot = pyre_object::gc_roots::shadow_stack_len();
        let arguments_w_gc = pin_root_gcarray(arguments_w_gc);
        let names_gc_slot = args_gc_slot + 1;
        let mut keyword_names_w_gc = pin_root_gcarray(keyword_names_w_gc);
        let keywords_gc_slot = args_gc_slot + 2;
        let mut keywords_w_gc = pin_root_gcarray(keywords_w_gc);

        // argument.py — positional copy. Index the `make_sure_not_resized`
        // GcArray (`GETFIELD` + `getarrayitem_gc`) so a CALL_KW walker's
        // seeded `NEW_ARRAY` of red operands is what the compiled loop reads.
        let mut input_argcount = upfront;
        if input_argcount < co_argcount {
            let take = num_args.min(co_argcount - upfront);
            for i in 0..take {
                scope_w[i + input_argcount] = unsafe { (&*arguments_w_gc)[i] };
            }
            input_argcount += take;
        }

        let has_kwarg = signature.has_kwarg();
        let defaults_len = defaults_w.len();
        let scope_len = scope_w.len();
        let kw_defs_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_kw_defs);
        let firstarg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_firstarg);
        let defaults_base = pyre_object::gc_roots::pin_roots(defaults_w);
        let scope_base = pyre_object::gc_roots::publish_roots(scope_w);
        // Always a scalar pin: joining `scope_base` (`SliceBase`) into this
        // local on the no-kwarg arm rebinds the special and refuses erase.
        let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::PY_NULL);
        pyre_object::gc_roots::normalize_roots(args_gc_slot, 6 + defaults_len + scope_len);
        let w_kw_defs = pyre_object::gc_roots::shadow_stack_get(kw_defs_slot);

        // argument.py:222-236 — *vararg collection.
        let outcome = 'done: {
            if signature.has_vararg() {
                let args_left = co_argcount - upfront;
                // argument.py — `assert args_left >= 0` always holds in
                // pyre (usize subtraction would have panicked above).
                let nstar = if num_args > args_left {
                    num_args - args_left
                } else {
                    0
                };
                let loc = co_argcount + co_kwonlyargcount;
                // argument.py:222-234 — `starargs_w` is `[]`, `args_w`, or
                // `args_w[args_left:]`, then `space.newtuple(starargs_w)`
                // (`tupleobject.py wraptuple` / `wraptuple2`).
                let w_stararg = starargs_newtuple(
                    arguments_w_gc,
                    args_left,
                    nstar,
                    prepend_first,
                    pyre_object::gc_roots::shadow_stack_get(firstarg_slot),
                );
                pyre_object::gc_roots::shadow_stack_set(scope_base + loc, w_stararg);
                scope_w[loc] = w_stararg;
            } else if avail > co_argcount {
                too_many_args = true;
            }
            (_, keyword_names_w_gc, keywords_w_gc) =
                reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);

            // argument.py:238-242 — **kwargs dict allocation.
            let kwarg_loc = co_argcount + co_kwonlyargcount + (signature.has_vararg() as usize);
            if has_kwarg {
                // `_match_signature` — `space.newdict(kwargs=True)`
                // (EmptyKwargsDictStrategy). An exact str key stays on
                // KwargsDictStrategy; a str subclass is not an exact str, so
                // `switch_to_correct_strategy` promotes to the object strategy
                // and `_collect_keyword_args` `setitem`s that same key.
                let w_new = pyre_object::dictmultiobject::w_dict_new_kwargs();
                pyre_object::gc_roots::shadow_stack_set(kwds_slot, w_new);
                pyre_object::gc_roots::shadow_stack_set(scope_base + kwarg_loc, w_new);
                scope_w[kwarg_loc] = w_new;
                (_, keyword_names_w_gc, keywords_w_gc) =
                    reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);
            }

            // argument.py:244-271 — keyword arg matching.
            // `kwds_mapping = [0] * n` then fill -1. A `Vec<isize>` is rvec
            // raw malloc and does not virtualize; a fixed `[i64; N]` is
            // `do_fixed_newlist_clear` → `NewArrayClear` the optimizer
            // virtualizes (`[i64]` is the length-prefixed reader;
            // `[isize]` is headerless and Repeat stays residual).
            // `_match_keywords` indexes `keyword_names_w` itself
            // (`make_sure_not_resized` list), so the GcArray is passed
            // through — a copy would residual-malloc every CALL_KW.
            if num_kwds > 0 {
                let mapping_len = co_argcount + co_kwonlyargcount - input_argcount;
                // A callee that publishes `scope_w` or `shadow_stack_set`s a
                // passed `scope_base` is ParamSlots; erase then refuses this
                // body (`calls-stack-sensitive-fn`) and the walker hits the
                // leftover `shadow_stack_copy_range`. Keep both arms inline
                // so the Open/Close stay this function's. The cap-sized
                // `[i64; N]` is `NewArrayClear` and virtualizes; the
                // longer `Vec` residual-mallocs and is for signatures the
                // JIT constant-folds off.
                if mapping_len > KWDS_MAPPING_CAP {
                    let mut mapping = vec![-1isize; mapping_len];
                    let num_remainingkwds = match match_keywords_vec(
                        signature,
                        blindargs,
                        co_posonlyargcount,
                        input_argcount,
                        keyword_names_w_gc,
                        &mut mapping,
                    ) {
                        MatchKeywordsOutcome::Remaining(n) => n,
                        MatchKeywordsOutcome::Shape(e) => break 'done MatchOutcome::Shape(e),
                        MatchKeywordsOutcome::Py(e) => break 'done MatchOutcome::Py(e),
                    };
                    if num_remainingkwds > 0 {
                        if has_kwarg {
                            let w_kwds = pyre_object::gc_roots::shadow_stack_get(kwds_slot);
                            let names = unsafe { (&*keyword_names_w_gc).as_slice() };
                            let keywords = unsafe { (&*keywords_w_gc).as_slice() };
                            match collect_keyword_args_vec(names, keywords, w_kwds, &mapping) {
                                CollectKeywordArgsOutcome::Ok => {}
                                CollectKeywordArgsOutcome::Py(e) => {
                                    break 'done MatchOutcome::Py(e);
                                }
                            }
                            (_, keyword_names_w_gc, keywords_w_gc) =
                                reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);
                        } else {
                            break 'done unknown_kwds_err_vec(
                                keyword_names_w_gc,
                                &mapping,
                                num_remainingkwds,
                            );
                        }
                    }
                    (_, keyword_names_w_gc, keywords_w_gc) =
                        reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);
                    let mut j = 0usize;
                    for i in input_argcount..(input_argcount + mapping_len) {
                        let kwds_index = mapping[j];
                        j += 1;
                        if kwds_index >= 0 {
                            let w_value = unsafe { (&*keywords_w_gc)[kwds_index as usize] };
                            pyre_object::gc_roots::shadow_stack_set(scope_base + i, w_value);
                            scope_w[i] = w_value;
                        }
                    }
                } else {
                    // `[i64; N]` Repeat is `NewArrayClear` only when it is
                    // never Unsize'd to a pair slice. `&mut mapping[..len]`
                    // (and rustc rewriting a counted fill into `<[T]>::fill`)
                    // allocates `ll_slice_buffer_*`, residualizes, and
                    // invalidates the heapcache the `_gc` GETFIELDs need.
                    let mut mapping = [0i64; KWDS_MAPPING_CAP];
                    {
                        let mut i = 0usize;
                        while i < mapping_len {
                            mapping[i] = -1;
                            i += 1;
                        }
                    }
                    let num_remainingkwds = match match_keywords(
                        signature,
                        blindargs,
                        co_posonlyargcount,
                        input_argcount,
                        keyword_names_w_gc,
                        &mut mapping,
                        mapping_len,
                        jit_few_keywords,
                    ) {
                        MatchKeywordsOutcome::Remaining(n) => n,
                        MatchKeywordsOutcome::Shape(e) => break 'done MatchOutcome::Shape(e),
                        MatchKeywordsOutcome::Py(e) => break 'done MatchOutcome::Py(e),
                    };
                    if num_remainingkwds > 0 {
                        if has_kwarg {
                            // argument.py:266-268 — collect overflow into **kwarg.
                            let w_kwds = pyre_object::gc_roots::shadow_stack_get(kwds_slot);
                            match collect_keyword_args(
                                keyword_names_w_gc,
                                keywords_w_gc,
                                w_kwds,
                                &mapping,
                                mapping_len,
                                jit_few_keywords,
                            ) {
                                CollectKeywordArgsOutcome::Ok => {}
                                CollectKeywordArgsOutcome::Py(e) => {
                                    break 'done MatchOutcome::Py(e);
                                }
                            }
                            (_, keyword_names_w_gc, keywords_w_gc) =
                                reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);
                        } else {
                            // argument.py:270-271 — ArgErrUnknownKwds.
                            break 'done unknown_kwds_err(
                                keyword_names_w_gc,
                                &mapping,
                                mapping_len,
                                num_remainingkwds,
                            );
                        }
                    }
                    // argument.py:273-287 — fill missing args from kwds.
                    if input_argcount < co_argcount + co_kwonlyargcount {
                        (_, keyword_names_w_gc, keywords_w_gc) =
                            reload_gc_twins!(args_gc_slot, names_gc_slot, keywords_gc_slot);
                        let mut j = 0usize;
                        for i in input_argcount..(co_argcount + co_kwonlyargcount) {
                            let kwds_index = mapping[j];
                            j += 1;
                            if kwds_index >= 0 {
                                let w_value = unsafe { (&*keywords_w_gc)[kwds_index as usize] };
                                pyre_object::gc_roots::shadow_stack_set(scope_base + i, w_value);
                                scope_w[i] = w_value;
                            }
                        }
                    }
                }
            }

            let more_filling = input_argcount < co_argcount + co_kwonlyargcount;
            // `argument.py:274` `def_first = co_argcount - len(defaults_w)` is
            // signed: `defaults_w` longer than the positional parameters makes it
            // negative and `defnum = i - def_first` then names the tuple's tail.
            let mut def_first: isize = 0;
            if more_filling {
                def_first = co_argcount as isize - defaults_len as isize;
            }

            // argument.py:289-300 — too-many-args ArgErr.
            if too_many_args {
                break 'done too_many_args_err(
                    signature,
                    scope_w,
                    co_argcount,
                    co_kwonlyargcount,
                    defaults_len,
                    avail,
                    methodcall,
                );
            }

            // argument.py:302-338 — fill defaults + collect missing names.
            if more_filling {
                let mut missing_pos = false;
                // argument.py:306-315 — posonly defaults.
                for i in input_argcount..co_argcount {
                    if !pyre_object::gc_roots::shadow_stack_get(scope_base + i).is_null() {
                        continue;
                    }
                    let defnum = (i as isize) - def_first;
                    if defnum >= 0 {
                        let w_default = pyre_object::gc_roots::shadow_stack_get(
                            defaults_base + defnum as usize,
                        );
                        pyre_object::gc_roots::shadow_stack_set(scope_base + i, w_default);
                        scope_w[i] = w_default;
                    } else {
                        missing_pos = true;
                    }
                }
                // `_match_signature` always runs the kwonly loop after the
                // positional-defaults pass, then raises
                // `ArgErrMissing(missing_positional, True)` if any, else
                // `ArgErrMissing(missing_kwonly, False)`.
                //
                // [3.15-spec] `PyDict_GetItemRef(func->func_kwdefaults, varname, &def)`
                // (Python/ceval.c `initialize_locals`, read at v3.15.0 and v3.15.2) ↔ pypy3
                // `space.finditem_str`.  Storage-level
                // `getitem_str` on `resolve_dict_backing(w_kw_defs)`: a dict subclass
                // `__getitem__` / `__missing__` is never dispatched, while an error
                // from a colliding key's `__eq__` propagates.
                let mut missing_kw = false;
                for i in co_argcount..(co_argcount + co_kwonlyargcount) {
                    if !pyre_object::gc_roots::shadow_stack_get(scope_base + i).is_null() {
                        continue;
                    }
                    if w_kw_defs.is_null() {
                        missing_kw = true;
                        continue;
                    }
                    match kwdefaults_storage_lookup_str_at(
                        pyre_object::gc_roots::shadow_stack_get(kw_defs_slot),
                        signature,
                        i,
                    ) {
                        FinditemStrOutcome::Hit(w_def) => {
                            pyre_object::gc_roots::shadow_stack_set(scope_base + i, w_def);
                            scope_w[i] = w_def;
                        }
                        FinditemStrOutcome::Miss => {
                            missing_kw = true;
                        }
                        FinditemStrOutcome::Py(e) => break 'done MatchOutcome::Py(e),
                    }
                }
                if missing_pos {
                    break 'done match_signature_missing_err(
                        signature,
                        scope_w,
                        input_argcount,
                        co_argcount,
                        true,
                    );
                }
                if missing_kw {
                    break 'done match_signature_missing_err(
                        signature,
                        scope_w,
                        co_argcount,
                        co_argcount + co_kwonlyargcount,
                        false,
                    );
                }
            }

            // `pop_roots` restores `scope_w` (`shadow_stack_copy_range`).
            // `erase_shadow_stack` aliases `scope_base` to `scope_w`, so the
            // copy is a no-op in the jitcode.
            pyre_object::gc_roots::shadow_stack_copy_range(scope_base, scope_w);
            MatchOutcome::Ok
        };
        outcome
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
        defaults_w: &[PyObjectRef],
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<Vec<PyObjectRef>, MatchSignatureError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pin_root_gcarray(self.arguments_w_gc);
        let _ = pin_root_gcarray(self.keyword_names_w_gc);
        let _ = pin_root_gcarray(self.keywords_w_gc);
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; scopelen];
        match self.match_signature(
            w_firstarg,
            &mut scope_w,
            signature,
            defaults_w,
            w_kw_defs,
            blindargs,
        ) {
            MatchOutcome::Ok => Ok(scope_w),
            MatchOutcome::Shape(e) => Err(MatchSignatureError::Shape(e)),
            MatchOutcome::Py(e) => Err(MatchSignatureError::Py(e)),
        }
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
    /// `argument.py parse_into_scope` is look-inside: it calls
    /// `_match_signature` then formats `ArgErr` as TypeError. A `Result<(),
    /// MatchSignatureError>` return on `_match_signature` shares
    /// `result::Result::Err` with `Result<_, PyError>` (`PyErrorObject ∪
    /// MatchSignatureError`); the outcome enum is a distinct ClassDef.
    pub fn parse_into_scope(
        &self,
        w_firstarg: PyObjectRef,
        scope_w: &mut [PyObjectRef],
        fnname: &Wtf8,
        signature: &crate::gateway::Signature,
        defaults_w: &[PyObjectRef],
        w_kw_defs: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        match self.match_signature(w_firstarg, scope_w, signature, defaults_w, w_kw_defs, 0) {
            MatchOutcome::Ok => Ok(signature.scope_length()),
            MatchOutcome::Shape(e) => Err(parse_into_scope_type_error(fnname, e)),
            MatchOutcome::Py(e) => Err(e),
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
        defaults_w: &[PyObjectRef],
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
                    return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                        fnname,
                        "() takes no keyword arguments",
                    )));
                }
                let msg = ArgErr::UnknownKwds { num_kwds, kwd_name }.getmsg_wtf8();
                Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    fnname, "() ", msg
                )))
            }
            Err(MatchSignatureError::Shape(e)) => {
                let msg = e.getmsg_wtf8();
                Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    fnname, "() ", msg
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
        let names: &[PyObjectRef] = self.keyword_names_w.as_deref().unwrap_or(&[]);
        let values: &[PyObjectRef] = self.keywords_w.as_deref().unwrap_or(&[]);
        // argument.py `topacked`: `if self.keyword_names_w is not None`.
        if self.keyword_names_w.is_some() {
            let npairs = names.len().min(values.len());
            let mut pairs = Vec::with_capacity(npairs * 2);
            for i in 0..npairs {
                pairs.push(names[i]);
                pairs.push(values[i]);
            }
            let pairs_base = pyre_object::gc_roots::shadow_stack_len();
            for i in 0..pairs.len() {
                let _ = pyre_object::gc_roots::pin_root(pairs[i]);
            }
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

/// `argument.py _match_signature` raises `ArgErr` or `OperationError`.
/// This is not `result::Result`: that ClassDef is shared with
/// `Result<_, PyError>` and unions `Err.__pos_0`.
#[derive(Debug, Clone)]
pub enum MatchOutcome {
    Ok,
    Shape(ArgErr),
    Py(crate::PyError),
}

/// `_match_keywords` outcome. Not `result::Result`, for the same
/// ClassDef reason as [`MatchOutcome`].
#[derive(Debug, Clone)]
pub enum MatchKeywordsOutcome {
    Remaining(usize),
    Shape(ArgErr),
    Py(crate::PyError),
}

/// `_collect_keyword_args` outcome. Not `result::Result`: that ClassDef
/// is shared with `Result<_, PyError>` and unions `Err.__pos_0`.
#[derive(Debug, Clone)]
pub enum CollectKeywordArgsOutcome {
    Ok,
    Py(crate::PyError),
}

/// pypy/interpreter/argument.py `_match_signature` raises
/// either `ArgErr` (shape mismatch) or `OperationError` (PyError, from
/// `space.setitem` in `_collect_keyword_args`).  Pyre keeps both
/// exception types separate via this enum, so callers
/// (`parse_obj` / `parse_into_scope` in Phase B) can match on the
/// shape arm to reformat as `oefmt(space.w_TypeError, "%s() %8",
/// fnname, e.getmsg())` and propagate the Py arm
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

/// Capacity of the `_match_signature` stack `kwds_mapping`. A longer
/// mapping is a `Vec<isize>` in the same function so erase does not
/// see a ParamSlots callee.
const KWDS_MAPPING_CAP: usize = 64;

/// `argument.py` `_match_signature` `space.newtuple(starargs_w)`.
///
/// `starargs_w` is `[]` when there are no extra positionals, `args_w`
/// when `args_left == 0` (`don't need to copy if its the whole list`),
/// or `args_w[args_left:]`. Bound-method `def meth(*args)` prepends
/// `w_firstarg`. Length 1/2 use the word-ABI tuple residuals; every
/// other length is `wraptuple` of a `newlist_clear` items block
/// (`tupleobject.py wraptuple`).
///
/// No root bracket (`tuple_ll_newlist`): `_match_signature` already
/// pins `arguments_w` and `w_firstarg`. A nested `push_roots` is a
/// residual `root_scope_close` the CALL_KW walker would execute with a
/// virtualized guard.
#[inline(never)]
#[majit_macros::unroll_safe]
fn starargs_newtuple(
    arguments_w_gc: *mut pyre_object::FixedObjectArray,
    args_left: usize,
    nstar: usize,
    prepend_first: bool,
    w_firstarg: PyObjectRef,
) -> PyObjectRef {
    if nstar == 0 {
        let block = unsafe { crate::baseobjspace::tuple_ll_newlist(0) };
        return unsafe {
            pyre_object::wraptuple(block as *mut pyre_object::object_array::ItemsBlock)
        };
    }
    if nstar == 1 {
        let item = if prepend_first {
            w_firstarg
        } else {
            unsafe { (&*arguments_w_gc)[args_left] }
        };
        return pyre_object::jit_w_tuple1(item);
    }
    if nstar == 2 {
        let a = if prepend_first {
            w_firstarg
        } else {
            unsafe { (&*arguments_w_gc)[args_left] }
        };
        // Bound-method `*args` prepends `w_firstarg`; `args_left` is 0
        // then, so the one extra positional is `arguments_w[0]`.
        let b = if prepend_first {
            unsafe { (&*arguments_w_gc)[args_left] }
        } else {
            unsafe { (&*arguments_w_gc)[args_left + 1] }
        };
        return pyre_object::wraptuple2(a, b);
    }
    if !prepend_first && args_left == 0 {
        return unsafe {
            pyre_object::wraptuple(arguments_w_gc as *mut pyre_object::object_array::ItemsBlock)
        };
    }
    starargs_newtuple_copy(arguments_w_gc, args_left, nstar, prepend_first, w_firstarg)
}

/// `args_w[args_left:]` (plus an optional prepend) into a fresh
/// `make_sure_not_resized` array, then `wraptuple`. Residual: the copy
/// is a runtime-length loop the CALL_KW walker constant-folds off for
/// the small arities [`starargs_newtuple`] handles.
#[majit_macros::dont_look_inside]
fn starargs_newtuple_copy(
    arguments_w_gc: *mut pyre_object::FixedObjectArray,
    args_left: usize,
    nstar: usize,
    prepend_first: bool,
    w_firstarg: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pin_root_gcarray(arguments_w_gc);
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_firstarg);
    let block = unsafe { crate::baseobjspace::tuple_ll_newlist(nstar as i64) };
    let block_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(block as PyObjectRef);
    let mut i = 0usize;
    if prepend_first {
        unsafe {
            pyre_object::object_array::items_block_set_ref(
                pyre_object::gc_roots::shadow_stack_get(block_slot)
                    as *mut pyre_object::object_array::ItemsBlock,
                0,
                pyre_object::gc_roots::shadow_stack_get(first_slot),
            );
        }
        i = 1;
    }
    let skip = usize::from(prepend_first);
    while i < nstar {
        let src = gcarray_from_slot(args_slot);
        let item = unsafe { (&*src)[args_left + (i - skip)] };
        unsafe {
            pyre_object::object_array::items_block_set_ref(
                pyre_object::gc_roots::shadow_stack_get(block_slot)
                    as *mut pyre_object::object_array::ItemsBlock,
                i,
                item,
            );
        }
        i += 1;
    }
    unsafe {
        pyre_object::wraptuple(pyre_object::gc_roots::shadow_stack_get(block_slot)
            as *mut pyre_object::object_array::ItemsBlock)
    }
}

/// `argument.py` `_match_keywords` is
/// `@jit.look_inside_iff(lambda ... jiton: jiton)`.
fn match_keywords_jiton(
    _signature: &crate::gateway::Signature,
    _blindargs: usize,
    _co_posonlyargcount: usize,
    _input_argcount: usize,
    _keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    _kwds_mapping: &mut [i64; KWDS_MAPPING_CAP],
    _mapping_len: usize,
    jiton: bool,
) -> bool {
    jiton
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
/// `argument.py` `_match_keywords` is `@jit.look_inside_iff(..., jiton)`.
/// The happy path is `Signature.find_w_argname` (`@jit.elidable`) plus
/// the mapping stores; `str_utf8_w` lives only on the multiple-values
/// error arm, residualized below so the unrolled body records.
/// The jiton argument is `Arguments._jit_few_keywords`; CALL_KW passes
/// a bytecode-constant `nkw` so the iff looks inside.
#[majit_macros::look_inside_iff(match_keywords_jiton)]
pub fn match_keywords(
    signature: &crate::gateway::Signature,
    blindargs: usize,
    co_posonlyargcount: usize,
    input_argcount: usize,
    keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    kwds_mapping: &mut [i64; KWDS_MAPPING_CAP],
    _mapping_len: usize,
    _jiton: bool,
) -> MatchKeywordsOutcome {
    let names = unsafe { &*keyword_names_w_gc };
    let num_kwds = names.len();
    let mut num_remainingkwds = num_kwds;
    let mut wrong_posonly = false;
    let mut i = 0usize;
    while i < num_kwds {
        let w_name = names[i];
        debug_assert!(!w_name.is_null(), "keyword_names_w[i] is None");
        let mut j = signature.find_w_argname(w_name);
        // argument.py:474-484 — posonly conflict.
        if j >= 0 && (j as usize) < co_posonlyargcount {
            if signature.has_kwarg() {
                j = -1;
            } else {
                wrong_posonly = true;
                i += 1;
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
                return match_keywords_multiple_values_err(w_name);
            }
        } else {
            kwds_mapping[(j - input_argcount as isize) as usize] = i as i64;
            num_remainingkwds -= 1;
        }
        i += 1;
    }
    if wrong_posonly {
        return match_keywords_posonly_err(signature, names.as_slice(), co_posonlyargcount);
    }
    MatchKeywordsOutcome::Remaining(num_remainingkwds)
}

/// Residual `Vec<isize>` mapping for signatures longer than
/// [`KWDS_MAPPING_CAP`]. The JIT constant-folds that arm off.
#[majit_macros::dont_look_inside]
fn match_keywords_vec(
    signature: &crate::gateway::Signature,
    blindargs: usize,
    co_posonlyargcount: usize,
    input_argcount: usize,
    keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    kwds_mapping: &mut Vec<isize>,
) -> MatchKeywordsOutcome {
    let names = unsafe { &*keyword_names_w_gc };
    let num_kwds = names.len();
    let mut num_remainingkwds = num_kwds;
    let mut wrong_posonly = false;
    // `find_w_argname` is unresolved from this residual. Names stay in the
    // slots this bracket published and are read back after each call.
    let _roots = pyre_object::gc_roots::push_roots();
    let names_base = pyre_object::gc_roots::pin_roots(names.as_slice());
    for i in 0..num_kwds {
        debug_assert!(
            !pyre_object::gc_roots::shadow_stack_get(names_base + i).is_null(),
            "keyword_names_w[i] is None"
        );
        let mut j =
            signature.find_w_argname(pyre_object::gc_roots::shadow_stack_get(names_base + i));
        let w_name = pyre_object::gc_roots::shadow_stack_get(names_base + i);
        if j >= 0 && (j as usize) < co_posonlyargcount {
            if signature.has_kwarg() {
                j = -1;
            } else {
                wrong_posonly = true;
                continue;
            }
        }
        if j < input_argcount as isize {
            if (blindargs as isize) <= j {
                return match_keywords_multiple_values_err(w_name);
            }
        } else {
            kwds_mapping[(j - input_argcount as isize) as usize] = i as isize;
            num_remainingkwds -= 1;
        }
    }
    if wrong_posonly {
        let mut names_now = vec![pyre_object::PY_NULL; num_kwds];
        pyre_object::gc_roots::shadow_stack_copy_range(names_base, &mut names_now);
        return match_keywords_posonly_err(signature, &names_now, co_posonlyargcount);
    }
    MatchKeywordsOutcome::Remaining(num_remainingkwds)
}

/// Residual: `str_utf8_w` / `w_str_from_wtf8_managed` hit unlowered
/// `is_ascii`. The happy path of `_match_keywords` does not take this arm.
#[majit_macros::dont_look_inside]
fn match_keywords_posonly_err(
    signature: &crate::gateway::Signature,
    keyword_names_w: &[PyObjectRef],
    co_posonlyargcount: usize,
) -> MatchKeywordsOutcome {
    let mut posonly_kwds: Vec<String> = Vec::new();
    for &w_name in keyword_names_w {
        let j = signature.find_w_argname(w_name);
        if j >= 0 && (j as usize) < co_posonlyargcount && !signature.has_kwarg() {
            posonly_kwds.push(signature.argnames[j as usize].as_ref().to_string());
        }
    }
    MatchKeywordsOutcome::Shape(ArgErr::PosonlyAsKwds { posonly_kwds })
}

/// Residual: `str_utf8_w` / `w_str_from_wtf8_managed` hit unlowered
/// `is_ascii`. The happy path of `_match_keywords` does not take this arm.
#[majit_macros::dont_look_inside]
fn match_keywords_multiple_values_err(w_name: PyObjectRef) -> MatchKeywordsOutcome {
    if !w_name.is_null() && unsafe { pyre_object::is_str(w_name) } {
        match unsafe { crate::baseobjspace::str_utf8_w(w_name) } {
            Ok(name) => MatchKeywordsOutcome::Shape(ArgErr::MultipleValues {
                argname: name.to_string(),
            }),
            Err(err) => MatchKeywordsOutcome::Py(err),
        }
    } else {
        let tp = type_name_of(w_name);
        MatchKeywordsOutcome::Py(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected str, got ",
            tp,
            " object",
        )))
    }
}

/// pypy/interpreter/argument.py `_collect_keyword_args`.
///
/// `argument.py` `_collect_keyword_args` is
/// `@jit.look_inside_iff(lambda ... jiton: jiton)`.
fn collect_keyword_args_jiton(
    _keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    _keywords_w_gc: *mut pyre_object::FixedObjectArray,
    _w_kwds: PyObjectRef,
    _kwds_mapping: &[i64; KWDS_MAPPING_CAP],
    _mapping_len: usize,
    jiton: bool,
) -> bool {
    jiton
}

/// `argument.py` `_collect_keyword_args(space, keyword_names_w, keywords_w, w_kwds, kwds_mapping, _)`
/// walks each index `i` of `keyword_names_w`. For each `i` it scans `kwds_mapping`;
/// if some slot `j` equals `i` the name is already bound and the scan stops.
/// Otherwise (`for`/`else`) it forwards the pair with
/// `space.setitem(w_kwds, keyword_names_w[i], keywords_w[i])`.
///
/// `kwds_mapping[k] == i` means slot `k` matched a real argname; this
/// helper walks the kwarg names and forwards every name that did NOT
/// match (i.e. did not appear in `kwds_mapping`) into the `**kwargs`
/// dict via `setitem`.
/// `argument.py` `_collect_keyword_args`. The CALL_KW site passes a
/// constant `nkw` so `jiton` is true and the body records `setitem`.
/// The two lists are the `make_sure_not_resized` GcArrays `_match_keywords`
/// already takes; `FixedObjectArray::as_slice` is an interior items
/// pointer, which `pin_roots_jit_abi` would read as a length-prefixed
/// array.
#[majit_macros::look_inside_iff(collect_keyword_args_jiton)]
pub fn collect_keyword_args(
    keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    keywords_w_gc: *mut pyre_object::FixedObjectArray,
    w_kwds: PyObjectRef,
    kwds_mapping: &[i64; KWDS_MAPPING_CAP],
    mapping_len: usize,
    _jiton: bool,
) -> CollectKeywordArgsOutcome {
    // Every `setitem` is a collection point, so the dictionary and the
    // lists it indexes are published before the first one and read back
    // at each turn.
    let _roots = pyre_object::gc_roots::push_roots();
    let names_gc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pin_root_gcarray(keyword_names_w_gc);
    let values_gc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pin_root_gcarray(keywords_w_gc);
    let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_kwds);
    let n = unsafe { (*gcarray_from_slot(names_gc_slot)).len() };
    let mut i = 0usize;
    while i < n {
        if mapping_contains(kwds_mapping, mapping_len, i as i64) {
            i += 1;
            continue;
        }
        let w_key = unsafe { (&*gcarray_from_slot(names_gc_slot))[i] };
        let w_value = unsafe { (&*gcarray_from_slot(values_gc_slot))[i] };
        let pair = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_key);
        let _ = pyre_object::gc_roots::pin_root(w_value);
        match setitem_kwargs_dict(
            pyre_object::gc_roots::shadow_stack_get(kwds_slot),
            pyre_object::gc_roots::shadow_stack_get(pair),
            pyre_object::gc_roots::shadow_stack_get(pair + 1),
        ) {
            CollectKeywordArgsOutcome::Ok => {}
            CollectKeywordArgsOutcome::Py(e) => return CollectKeywordArgsOutcome::Py(e),
        }
        i += 1;
    }
    CollectKeywordArgsOutcome::Ok
}

/// Residual overflow collect for a `Vec` mapping longer than
/// [`KWDS_MAPPING_CAP`].
#[majit_macros::dont_look_inside]
fn collect_keyword_args_vec(
    keyword_names_w: &[PyObjectRef],
    keywords_w: &[PyObjectRef],
    w_kwds: PyObjectRef,
    kwds_mapping: &[isize],
) -> CollectKeywordArgsOutcome {
    let _roots = pyre_object::gc_roots::push_roots();
    let kwds_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_kwds);
    let names_base = pyre_object::gc_roots::pin_roots(keyword_names_w);
    let values_base = pyre_object::gc_roots::pin_roots(keywords_w);
    for i in 0..keyword_names_w.len() {
        if kwds_mapping.contains(&(i as isize)) {
            continue;
        }
        match setitem_kwargs_dict(
            pyre_object::gc_roots::shadow_stack_get(kwds_slot),
            pyre_object::gc_roots::shadow_stack_get(names_base + i),
            pyre_object::gc_roots::shadow_stack_get(values_base + i),
        ) {
            CollectKeywordArgsOutcome::Ok => {}
            CollectKeywordArgsOutcome::Py(e) => return CollectKeywordArgsOutcome::Py(e),
        }
    }
    CollectKeywordArgsOutcome::Ok
}

/// `space.setitem(w_kwds, w_key, w_value)` for overflow keywords.
/// Residual: `W_DictMultiObject.setitem` reads `dstrategy.imp` as a
/// trait object the annotator types classdef-less, so looking inside
/// the store aborts `_collect_keyword_args`. A non-virtual dict
/// setitem is residual the same way. Returns [`CollectKeywordArgsOutcome`]
/// so the look-inside body never getattr `Result::Err.__pos_0`.
#[majit_macros::dont_look_inside]
fn setitem_kwargs_dict(
    w_kwds: PyObjectRef,
    w_key: PyObjectRef,
    w_value: PyObjectRef,
) -> CollectKeywordArgsOutcome {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&[w_kwds, w_key, w_value]);
        match pyre_object::w_dict_store_checked(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        ) {
            Ok(()) => CollectKeywordArgsOutcome::Ok,
            Err(_) => {
                CollectKeywordArgsOutcome::Py(crate::baseobjspace::take_pending_dict_key_error(
                    pyre_object::gc_roots::shadow_stack_get(base + 1),
                ))
            }
        }
    }
}

fn mapping_contains(mapping: &[i64; KWDS_MAPPING_CAP], mapping_len: usize, target: i64) -> bool {
    let mut j = 0usize;
    while j < mapping_len {
        if mapping[j] == target {
            return true;
        }
        j += 1;
    }
    false
}

/// One-word residual of `len(self.keyword_names_w)`.
#[majit_macros::dont_look_inside]
pub(crate) fn option_vec_len(v: &Option<Vec<PyObjectRef>>) -> usize {
    match v {
        Some(inner) => inner.len(),
        None => 0,
    }
}

/// Residual count of CALL_KW names so `call_args` does not getattr the
/// Vec in a look-inside caller.
#[majit_macros::dont_look_inside]
pub(crate) fn arguments_keyword_count(args: &Arguments) -> usize {
    option_vec_len(&args.keyword_names_w)
}

/// `argument.py parse_into_scope` `except ArgErr`: format `fnname() ` +
/// `e.getmsg()`. Residual so the look-inside happy path does not format.
#[majit_macros::dont_look_inside]
fn parse_into_scope_type_error(fnname: &Wtf8, e: ArgErr) -> crate::PyError {
    let msg = e.getmsg_wtf8();
    crate::PyError::type_error(crate::display::wtf8_format!(fnname, "() ", msg))
}

/// Residual of the keyword-only default storage probe so `_match_signature`
/// does not read a `Result` in the look-inside loop.
#[derive(Debug)]
enum FinditemStrOutcome {
    Hit(PyObjectRef),
    Miss,
    Py(crate::PyError),
}

/// [3.15-spec] `PyDict_GetItemRef(func->func_kwdefaults, varname, &def)`
/// (Python/ceval.c `initialize_locals`, read at v3.15.0 and v3.15.2) ↔ pypy3
/// `space.finditem_str`.  Storage-level
/// `getitem_str` on `resolve_dict_backing(w_kw_defs)`: a dict subclass
/// `__getitem__` / `__missing__` is never dispatched, while an error from a
/// colliding key's `__eq__` propagates.
#[majit_macros::dont_look_inside]
fn kwdefaults_storage_lookup_str(w_dict: PyObjectRef, name: &str) -> FinditemStrOutcome {
    let _roots = pyre_object::gc_roots::push_roots();
    let mapping_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    let backing = crate::type_methods::resolve_dict_backing(
        pyre_object::gc_roots::shadow_stack_get(mapping_slot),
    );
    if backing.is_null() {
        return FinditemStrOutcome::Miss;
    }
    let backing_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(backing);
    match unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str_checked(
            pyre_object::gc_roots::shadow_stack_get(backing_slot),
            name,
        )
    } {
        Ok(Some(v)) => FinditemStrOutcome::Hit(v),
        Ok(None) => FinditemStrOutcome::Miss,
        Err(_) => FinditemStrOutcome::Py(crate::baseobjspace::take_pending_hash_error()),
    }
}

/// One-word residual of `kwdefaults_storage_lookup_str(w_dict, argnames[i])`.
/// `&str` is two words; the look-inside kwonly-defaults loop cannot
/// pass a rustc fat pointer.
#[majit_macros::dont_look_inside]
fn kwdefaults_storage_lookup_str_at(
    w_dict: PyObjectRef,
    signature: &crate::gateway::Signature,
    index: usize,
) -> FinditemStrOutcome {
    kwdefaults_storage_lookup_str(w_dict, signature.argnames[index].as_ref())
}

/// Residual `ArgErrUnknownKwds` builder. `Wtf8Buf::new` / `text_w` stay
/// off the look-inside happy path.
#[majit_macros::dont_look_inside]
fn unknown_kwds_err(
    keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    mapping: &[i64; KWDS_MAPPING_CAP],
    mapping_len: usize,
    num_remainingkwds: usize,
) -> MatchOutcome {
    let keyword_names_w = unsafe { &*keyword_names_w_gc };
    let mut name = Wtf8Buf::new();
    if num_remainingkwds == 1 {
        let mut i = 0usize;
        while i < keyword_names_w.len() {
            let w_n = keyword_names_w[i];
            if !mapping_contains(mapping, mapping_len, i as i64) {
                if unsafe { pyre_object::is_str(w_n) } {
                    name = unsafe { pyre_object::w_str_get_wtf8(w_n) }.to_wtf8_buf();
                } else {
                    let tp = type_name_of(w_n);
                    return MatchOutcome::Py(crate::PyError::type_error(
                        crate::display::wtf8_format!("expected str, got ", tp, " object"),
                    ));
                }
                break;
            }
            i += 1;
        }
    }
    MatchOutcome::Shape(ArgErr::UnknownKwds {
        num_kwds: num_remainingkwds,
        kwd_name: name,
    })
}

/// Residual `ArgErrUnknownKwds` for a `Vec` mapping longer than
/// [`KWDS_MAPPING_CAP`].
#[majit_macros::dont_look_inside]
fn unknown_kwds_err_vec(
    keyword_names_w_gc: *mut pyre_object::FixedObjectArray,
    mapping: &[isize],
    num_remainingkwds: usize,
) -> MatchOutcome {
    let keyword_names_w = unsafe { (&*keyword_names_w_gc).as_slice() };
    let mut name = Wtf8Buf::new();
    if num_remainingkwds == 1 {
        for i in 0..keyword_names_w.len() {
            let w_n = keyword_names_w[i];
            if !mapping.contains(&(i as isize)) {
                if unsafe { pyre_object::is_str(w_n) } {
                    name = unsafe { pyre_object::w_str_get_wtf8(w_n) }.to_wtf8_buf();
                } else {
                    let tp = type_name_of(w_n);
                    return MatchOutcome::Py(crate::PyError::type_error(
                        crate::display::wtf8_format!("expected str, got ", tp, " object"),
                    ));
                }
                break;
            }
        }
    }
    MatchOutcome::Shape(ArgErr::UnknownKwds {
        num_kwds: num_remainingkwds,
        kwd_name: name,
    })
}

/// Residual `ArgErrTooMany` / `ArgErrTooManyMethod` builder. Signature
/// field reads stay in rustc layout.
#[majit_macros::dont_look_inside]
fn too_many_args_err(
    signature: &crate::gateway::Signature,
    scope_w: &[PyObjectRef],
    co_argcount: usize,
    co_kwonlyargcount: usize,
    num_defaults: usize,
    given: usize,
    methodcall: bool,
) -> MatchOutcome {
    let mut kwonly_given: usize = 0;
    for i in co_argcount..(co_argcount + co_kwonlyargcount) {
        if !scope_w[i].is_null() {
            kwonly_given += 1;
        }
    }
    if methodcall {
        MatchOutcome::Shape(ArgErr::TooManyMethod {
            signature: signature.clone(),
            num_defaults,
            given,
            kwonly_given,
        })
    } else {
        MatchOutcome::Shape(ArgErr::TooMany {
            signature: signature.clone(),
            num_defaults,
            given,
            kwonly_given,
        })
    }
}

/// Residual `ArgErrMissing` builder. Collects still-null scope slots.
#[majit_macros::dont_look_inside]
fn match_signature_missing_err(
    signature: &crate::gateway::Signature,
    scope_w: &[PyObjectRef],
    start: usize,
    end: usize,
    positional: bool,
) -> MatchOutcome {
    let mut missing: Vec<String> = Vec::new();
    for i in start..end {
        if scope_w[i].is_null() {
            missing.push(signature.argnames[i].as_ref().to_string());
        }
    }
    MatchOutcome::Shape(ArgErr::Missing {
        missing,
        positional,
    })
}

/// Copy `n` shadow-stack words starting at `base`. A for-loop, not an
/// iterator adapter: RPython has no closures, and `_match_signature`
/// must stay a graph the two-phase prepass can annotate.
fn copy_object_slice(src: &[PyObjectRef]) -> Vec<PyObjectRef> {
    let n = src.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(src[i]);
    }
    out
}

/// Copy `n` shadow-stack words starting at `base`. A for-loop, not an
/// iterator adapter: RPython has no closures. Always inlined so the
/// caller is not a param_slots callee of this get-loop.
#[inline(always)]
fn copy_shadow_range_to_vec(base: usize, n: usize) -> Vec<PyObjectRef> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(pyre_object::gc_roots::shadow_stack_get(base + i));
    }
    out
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
    /// Names are copied from the signature's `Cow` argnames; getmsg reads them.
    Missing {
        missing: Vec<String>,
        positional: bool,
    },
    /// argument.py `ArgErrTooMany(signature, num_defaults, given, kwonly_given)`.
    /// The variant stores the upstream `signature`.
    TooMany {
        signature: crate::gateway::Signature,
        num_defaults: usize,
        given: usize,
        kwonly_given: usize,
    },
    /// argument.py `ArgErrTooManyMethod` — same fields as
    /// `TooMany`, with appended "did you forget self?" hint when the
    /// signature shape suggests a missing self parameter.
    /// The variant stores the upstream `signature`.
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
    /// Residual: `StringBuilder.append` has no arm for a `String`.
    #[majit_macros::dont_look_inside]
    pub fn getmsg_wtf8(&self) -> Wtf8Buf {
        match self {
            ArgErr::UnknownKwds {
                num_kwds: 1,
                kwd_name,
            } => {
                let mut msg = Wtf8Buf::new();
                msg.push_str("got an unexpected keyword argument '");
                msg.push_wtf8(kwd_name);
                msg.push_str("'");
                msg
            }
            other => {
                let mut msg = Wtf8Buf::new();
                msg.push_str(&other.getmsg());
                msg
            }
        }
    }

    #[majit_macros::dont_look_inside]
    pub fn getmsg(&self) -> String {
        match self {
            ArgErr::Missing {
                missing,
                positional,
            } => {
                // argument.py ArgErrMissing.getmsg — comma-and-or-and join.
                let mut arguments_str = String::new();
                let n = missing.len();
                for i in 0..n {
                    let arg = missing.as_slice()[i].as_str();
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
                let mut msg = String::new();
                msg.push_str("missing ");
                msg.push_str(&n.to_string());
                msg.push_str(" required ");
                msg.push_str(kind_str);
                msg.push_str(" argument");
                if n != 1 {
                    msg.push('s');
                }
                msg.push_str(": ");
                msg.push_str(&arguments_str);
                msg
            }
            ArgErr::TooMany {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => format_too_many(
                signature.num_argnames(),
                *num_defaults,
                *given,
                *kwonly_given,
            ),
            ArgErr::TooManyMethod {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => format_too_many_method(
                signature.num_argnames(),
                signature.argnames.first().map(|name| name.as_ref()),
                *num_defaults,
                *given,
                *kwonly_given,
            ),
            ArgErr::MultipleValues { argname } => {
                let mut msg = String::new();
                msg.push_str("got multiple values for argument '");
                msg.push_str(argname);
                msg.push('\'');
                msg
            }
            ArgErr::UnknownKwds { num_kwds, kwd_name } => {
                if *num_kwds == 1 {
                    let mut msg = String::new();
                    msg.push_str("got an unexpected keyword argument '");
                    msg.push_str(&kwd_name.to_string());
                    msg.push('\'');
                    msg
                } else {
                    let mut msg = String::new();
                    msg.push_str("got ");
                    msg.push_str(&num_kwds.to_string());
                    msg.push_str(" unexpected keyword arguments");
                    msg
                }
            }
            ArgErr::PosonlyAsKwds { posonly_kwds } => {
                // argument.py ArgErrPosonlyAsKwds.getmsg — always the plural
                // wording, even for a single name.
                let mut names = String::new();
                for i in 0..posonly_kwds.len() {
                    if i > 0 {
                        names.push_str(", ");
                    }
                    names.push_str(posonly_kwds.as_slice()[i].as_str());
                }
                let mut msg = String::new();
                msg.push_str("got some positional-only arguments passed as keyword arguments: '");
                msg.push_str(&names);
                msg.push('\'');
                msg
            }
        }
    }
}

/// pypy/interpreter/argument.py `ArgErrTooMany.getmsg`.
pub(crate) fn format_too_many(
    num_args: usize,
    num_defaults: usize,
    given: usize,
    kwonly_given: usize,
) -> String {
    let mut takes_str = String::new();
    if num_defaults > 0 {
        takes_str.push_str("from ");
        takes_str.push_str(&(num_args as isize - num_defaults as isize).to_string());
        takes_str.push_str(" to ");
        takes_str.push_str(&num_args.to_string());
        takes_str.push_str(" positional arguments");
    } else {
        takes_str.push_str(&num_args.to_string());
        takes_str.push_str(" positional argument");
        if num_args != 1 {
            takes_str.push('s');
        }
    }
    let mut given_str = String::new();
    if kwonly_given > 0 {
        given_str.push_str(&given.to_string());
        given_str.push_str(" positional argument");
        if given != 1 {
            given_str.push('s');
        }
        given_str.push_str(" (and ");
        given_str.push_str(&kwonly_given.to_string());
        given_str.push_str(" keyword-only argument");
        if kwonly_given != 1 {
            given_str.push('s');
        }
        given_str.push_str(") were");
    } else {
        given_str.push_str(&given.to_string());
        given_str.push(' ');
        if given != 1 {
            given_str.push_str("were");
        } else {
            given_str.push_str("was");
        }
    }
    let mut msg = String::new();
    msg.push_str("takes ");
    msg.push_str(&takes_str);
    msg.push_str(" but ");
    msg.push_str(&given_str);
    msg.push_str(" given");
    msg
}

/// pypy/interpreter/argument.py `ArgErrTooManyMethod.getmsg`: `ArgErrTooMany.getmsg` plus the
/// forgotten-self hint when exactly one positional too many was given and the first parameter is not `self`.
pub(crate) fn format_too_many_method(
    num_args: usize,
    first_argname: Option<&str>,
    num_defaults: usize,
    given: usize,
    kwonly_given: usize,
) -> String {
    let mut msg = format_too_many(num_args, num_defaults, given, kwonly_given);
    let first_is_self = match first_argname {
        Some(name) => name == "self",
        None => false,
    };
    if given == num_args + 1 && (num_args == 0 || !first_is_self) {
        msg.push_str(". Did you forget 'self' in the function definition?");
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pypy/interpreter/argument.py — `w_function is None` arm.
    #[test]
    fn raise_type_error_no_function() {
        let err = raise_type_error!(pyre_object::PY_NULL, "boom", &[]);
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
        let err = raise_type_error!(func, "needs an iterable", &[]);
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

    /// `_check_not_duplicate_kwargs` `'%S'` is `space.str(w_key)`, not `%T`.
    #[test]
    fn check_not_duplicate_kwargs_non_str_key_uses_str() {
        let w_key = pyre_object::w_int_new(7);
        let existing = vec![w_key];
        let new = vec![w_key];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        let err = check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect_err("should raise TypeError on duplicate");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        let msg = err.message_text();
        assert!(msg.contains("got multiple values"));
        assert!(msg.contains("'7'"), "got {msg:?}");
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

    /// `_Py_Check_ArgsIterable` — non-iterable star arg surfaces
    /// "Value after * must be an iterable".
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
                .contains("Value after * must be an iterable")
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
        assert!(matches!(
            arguments.match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                &[],
                pyre_object::PY_NULL,
                0,
            ),
            MatchOutcome::Ok
        ));
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
        match arguments.match_signature(
            pyre_object::PY_NULL,
            &mut scope,
            &sig,
            &[],
            pyre_object::PY_NULL,
            0,
        ) {
            MatchOutcome::Shape(ArgErr::TooMany { given, .. }) => {
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
        match arguments.match_signature(
            pyre_object::PY_NULL,
            &mut scope,
            &sig,
            &[],
            pyre_object::PY_NULL,
            0,
        ) {
            MatchOutcome::Shape(ArgErr::Missing {
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
        assert!(matches!(
            arguments.match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                &[],
                pyre_object::PY_NULL,
                0,
            ),
            MatchOutcome::Ok
        ));
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
        assert!(matches!(
            arguments.match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                &[],
                pyre_object::PY_NULL,
                0,
            ),
            MatchOutcome::Ok
        ));
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
        assert!(matches!(
            arguments.match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                &[],
                pyre_object::PY_NULL,
                0,
            ),
            MatchOutcome::Ok
        ));
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
        let names = unsafe {
            pyre_object::object_array::alloc_mro_block_gc_young(&[pyre_object::w_str_new("b")])
        };
        let mut mapping = [-1i64; KWDS_MAPPING_CAP];
        let remaining = match match_keywords(&sig, 0, 0, 0, names, &mut mapping, 2, true) {
            MatchKeywordsOutcome::Remaining(n) => n,
            other => panic!("kwarg name match should succeed, got {other:?}"),
        };
        assert_eq!(remaining, 0);
        // 'b' is signature index 1, slot 1 (= 1 - input_argcount=0)
        assert_eq!(mapping[0], -1);
        assert_eq!(mapping[1], 0);
    }

    /// pypy/interpreter/argument.py:474-484 `match_keywords` flags
    /// posonly conflict via `ArgErrPosonlyAsKwds`.
    #[test]
    fn match_keywords_posonly_conflict() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 1);
        let names = unsafe {
            pyre_object::object_array::alloc_mro_block_gc_young(&[pyre_object::w_str_new("a")])
        };
        let mut mapping = [-1i64; KWDS_MAPPING_CAP];
        match match_keywords(&sig, 0, 1, 0, names, &mut mapping, 2, true) {
            MatchKeywordsOutcome::Shape(ArgErr::PosonlyAsKwds { posonly_kwds }) => {
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
        let names = arguments
            .keyword_names_w
            .as_ref()
            .expect("kwargs should be Some");
        let values = arguments
            .keywords_w
            .as_ref()
            .expect("kwargs should be Some");
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
        let names = replaced.keyword_names_w.as_ref().expect("kwargs preserved");
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

    /// argument.py `__init__`: a present-but-empty `keyword_names_w`
    /// (`keywords=[]` / after `**{}`) is not `None`, so residual
    /// `isconstant(len(...))` is false.
    #[test]
    fn jit_few_keywords_false_for_present_empty_names_at_interp() {
        let arguments = Arguments::new(&[], Some(&[]), Some(&[]), None, None, false, None)
            .expect("empty keyword lists need no star expansion");
        assert!(arguments.keyword_names_w.is_some());
        assert!(!arguments.jit_few_keywords);
        let replaced = arguments.replace_arguments(Vec::new());
        assert!(replaced.keyword_names_w.is_some());
        assert!(!replaced.jit_few_keywords);
    }

    /// argument.py `_combine_starstarargs_wrapped`: `**{}` assigns a
    /// present empty list, not None.
    #[test]
    fn empty_starstararg_leaves_present_empty_names() {
        crate::test_hooks::install_hash_hook();
        let dict = pyre_object::dictmultiobject::w_dict_new();
        let arguments = Arguments::new(&[], None, None, None, Some(dict), false, None)
            .expect("empty dict starstar should succeed");
        assert_eq!(arguments.keyword_names_w.as_ref().map(|s| s.len()), Some(0));
        assert!(!arguments.jit_few_keywords);
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
                Wtf8::new("f"),
                &sig,
                &[],
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
                Wtf8::new("myfn"),
                &sig,
                &[],
                pyre_object::PY_NULL,
            )
            .expect_err("too many args should TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message_text().starts_with("myfn() "));
    }

    /// `PyCode.funcrun` passes `func.qualname`. A lone surrogate in that
    /// text is part of the TypeError, not a replacement character.
    #[test]
    fn parse_into_scope_keeps_a_surrogate_in_fnname() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&pos);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let mut fnname = Wtf8Buf::new();
        fnname.push_str("q");
        fnname.push(rustpython_wtf8::CodePoint::from_u32(0xDCFF).unwrap());
        let err = arguments
            .parse_into_scope(
                pyre_object::PY_NULL,
                &mut scope,
                &fnname,
                &sig,
                &[],
                pyre_object::PY_NULL,
            )
            .expect_err("too many args should TypeError");
        let mut prefix = fnname.clone();
        prefix.push_str("() ");
        assert!(err.message_wtf8().starts_with(&prefix));
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
                &[],
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
