//! structseq factory — the interpreter-level half of `lib_pypy/_structseq.py`.
//!
//! The app-level module itself (`structseqfield`, `structseqtype`,
//! `structseq_new`, `structseq_repr`, ...) is the bundled `_structseq` module
//! (`module/_structseq`).  [`make_struct_seq`] evaluates what an app-level
//! `class X(metaclass=structseqtype)` statement does: one `structseqfield(i)`
//! per field in the class namespace, then `structseqtype(name, (), ns)`.  Each
//! field descriptor carries its own `index` / `is_positional`, and the class
//! dict carries `n_fields`, `n_sequence_fields`, `_extra_fields` and `_name`,
//! so every reader below goes through the class.
//!
//! What stays here is interpreter-level:
//!
//! * [`new_instance`] / [`new_instance_with_extra`] — `build_stat_result`
//!   (`interp_posix.py`): build the tuple and store the named-only fields
//!   with `setdictvalue`, without going through `structseq_new`.
//! * [`structseq_descr_new`] — `structseq_new` with the CPython 3.14
//!   constructor rules, installed as the class's `__new__`
//!   (`structseqtype.__new__` keeps a `__new__` the namespace supplies).
//! * `structseq_replace` — the 3.13 `__replace__`.

use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8Buf;

use crate::PyError;

/// `_structseq.structseqfield` / `_structseq.structseqtype`, from the bundled
/// app-level module.
fn structseq_app_attr(name: &str) -> PyObjectRef {
    let module = crate::importing::get_builtin_module("_structseq")
        .expect("the _structseq builtin module is always installed");
    crate::baseobjspace::getattr_str(module, name)
        .unwrap_or_else(|e| panic!("_structseq.{name}: {e:?}"))
}

/// Whether `obj` is a class whose metaclass is `structseqtype`.  Structseq
/// types are unacceptable as bases but, unlike most types with that
/// restriction, their constructor accepts the `sequence=` and `dict=`
/// keywords.
pub(crate) fn is_structseq_type(obj: PyObjectRef) -> bool {
    if obj.is_null() || !unsafe { pyre_object::is_type(obj) } {
        return false;
    }
    std::ptr::eq(
        unsafe { (*obj).w_class },
        structseq_app_attr("structseqtype"),
    )
}

/// `cls._name` — `structseqtype.__new__` sets it to the class's `name`
/// attribute, which [`make_struct_seq_impl`] supplies as the dotted name.
fn class_name(cls: PyObjectRef) -> Option<String> {
    let w_name = crate::baseobjspace::getattr_str(cls, "_name").ok()?;
    unsafe { pyre_object::w_str_get_value_opt(w_name) }.map(str::to_owned)
}

/// `[field.__name__ for field in cls._extra_fields]`.
fn extra_field_names(cls: PyObjectRef) -> Result<Vec<String>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let w_fields = crate::baseobjspace::getattr_str(cls, "_extra_fields")?;
    let fields_slot = roots.pin_roots(&[w_fields]);
    let n = unsafe { pyre_object::w_tuple_len(roots.get(fields_slot)) };
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        let field_slot = roots.pin_roots(&[unsafe {
            pyre_object::w_tuple_getitem(roots.get(fields_slot), i as i64)
        }
        .expect("_extra_fields index in range")]);
        let w_name = crate::baseobjspace::getattr_str(roots.get(field_slot), "__name__")?;
        names.push(crate::baseobjspace::str_utf8_w(w_name)?.to_string());
    }
    Ok(names)
}

/// `cls.__match_args__` — the positional names without the unnamed
/// (leading-`_`) ones, in index order.
fn match_args_names(cls: PyObjectRef) -> Result<Vec<String>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let w_names = crate::baseobjspace::getattr_str(cls, "__match_args__")?;
    let names_slot = roots.pin_roots(&[w_names]);
    let n = unsafe { pyre_object::w_tuple_len(roots.get(names_slot)) };
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        let name_slot = roots.pin_roots(&[unsafe {
            pyre_object::w_tuple_getitem(roots.get(names_slot), i as i64)
        }
        .expect("__match_args__ index in range")]);
        names.push(crate::baseobjspace::str_utf8_w(roots.get(name_slot))?.to_string());
    }
    Ok(names)
}

/// CPython 3.14 `structseq___replace__` — copy the positional body and
/// named-only fields, overlay keyword changes, and return the same structseq
/// type.  Types with unnamed positional fields cannot map every tuple slot
/// back to a keyword and therefore reject replacement altogether.
fn structseq_replace_args(
    inst: PyObjectRef,
    args: &crate::argument::Arguments,
) -> Result<PyObjectRef, PyError> {
    if !args.arguments_w.is_empty() {
        return Err(PyError::type_error(
            "__replace__() takes no positional arguments",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let inst_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(inst);
    structseq_replace_from(
        pyre_object::gc_roots::shadow_stack_get(inst_slot),
        crate::builtins::arguments_as_kwargs_dict(args)?,
    )
}

fn structseq_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&inst) = positional.first() else {
        return Err(PyError::type_error(
            "__replace__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() != 1 {
        return Err(PyError::type_error(
            "__replace__() takes no positional arguments",
        ));
    }
    structseq_replace_from(inst, kwargs)
}

fn structseq_replace_from(
    inst: PyObjectRef,
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, PyError> {
    // The class reads below can allocate; the instance is reloaded from its
    // root afterwards, and the class is re-read through it.
    let has_kwargs = kwargs.is_some();
    let roots = pyre_object::gc_roots::push_roots();
    let inst_slot = roots.pin_roots(&[inst]);
    let kwargs_slot = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
    let cls = || unsafe { (*roots.get(inst_slot)).w_class };
    if !is_structseq_type(cls()) {
        return Err(PyError::type_error(
            "__replace__() requires a structseq instance",
        ));
    }
    let name = class_name(cls()).unwrap_or_default();
    if read_class_int(cls(), "n_unnamed_fields").unwrap_or(0) > 0 {
        return Err(PyError::type_error(format!(
            "__replace__() is not supported for {name} because it has unnamed field(s)"
        )));
    }
    // With no unnamed field, `__match_args__` names every positional field.
    let fields = match_args_names(cls())?;
    let extra_fields = extra_field_names(cls())?;
    let kwargs = has_kwargs.then(|| roots.get(kwargs_slot));

    // Key stays the str object. A lone surrogate is not a field name and
    // must survive into the unexpected-field repr (`structseq___replace__`
    // formats the key list with `%R`) instead of a UTF-8 encode.
    let mut changes: Vec<(usize, usize)> = Vec::new();
    for (key, value) in kwargs
        .map(|dict| unsafe { pyre_object::w_dict_items(dict) })
        .unwrap_or_default()
    {
        if unsafe { pyre_object::is_str(key) }
            && unsafe { pyre_object::w_str_get_wtf8(key) } == "__pyre_kw__"
        {
            continue;
        }
        let key_obj = if unsafe { pyre_object::is_str(key) } {
            key
        } else {
            pyre_object::PY_NULL
        };
        let key_slot = roots.pin_roots(&[key_obj]);
        let value_slot = roots.pin_roots(&[value]);
        changes.push((key_slot, value_slot));
    }
    let unexpected: Vec<usize> = changes
        .iter()
        .copied()
        .filter(|(key, _)| !structseq_field_named(roots.get(*key), &fields, &extra_fields))
        .map(|(key, _)| key)
        .collect();
    if !unexpected.is_empty() {
        // `repr` of a str subclass runs user code, which can move every key
        // still waiting its turn; each key is already on the root stack.
        let mut msg = Wtf8Buf::new();
        msg.push_str("Got unexpected field name(s): [");
        for (index, key_slot) in unexpected.iter().copied().enumerate() {
            if index > 0 {
                msg.push_str(", ");
            }
            let key = roots.get(key_slot);
            if key.is_null() {
                msg.push_str("'<non-string>'");
            } else {
                msg.push_wtf8(&unsafe { crate::display::py_repr_wtf8(key)? });
            }
        }
        msg.push_str("]");
        return Err(PyError::type_error(msg));
    }

    let mut body_slots = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let value = changes
            .iter()
            .find(|(key, _)| structseq_key_eq(roots.get(*key), field))
            .map(|(_, value)| roots.get(*value))
            .or_else(|| unsafe { pyre_object::w_tuple_getitem(roots.get(inst_slot), index as i64) })
            .unwrap_or_else(pyre_object::w_none);
        body_slots.push(roots.pin_roots(&[value]));
    }
    let source_dict_slot =
        roots.pin_roots(&[crate::baseobjspace::getdict_native(roots.get(inst_slot))]);
    let mut extra_slots = Vec::with_capacity(extra_fields.len());
    for field in &extra_fields {
        let value = changes
            .iter()
            .find(|(key, _)| structseq_key_eq(roots.get(*key), field))
            .map(|(_, value)| roots.get(*value))
            .or_else(|| {
                let source_dict = roots.get(source_dict_slot);
                (!source_dict.is_null())
                    .then(|| unsafe { pyre_object::w_dict_getitem_str(source_dict, field) })
                    .flatten()
            })
            .unwrap_or_else(pyre_object::w_none);
        extra_slots.push(roots.pin_roots(&[value]));
    }
    let cls_slot = roots.pin_roots(&[cls()]);
    let body: Vec<PyObjectRef> = body_slots.iter().map(|slot| roots.get(*slot)).collect();
    let extras: Vec<(&str, PyObjectRef)> = extra_fields
        .iter()
        .zip(extra_slots)
        .map(|(field, slot)| (field.as_str(), roots.get(slot)))
        .collect();
    Ok(new_instance_with_extra(roots.get(cls_slot), body, extras))
}

/// A keyword matches a structseq field when its WTF-8 view is that UTF-8 name.
/// A lone surrogate never matches (`structseq___replace__` then reports it).
fn structseq_key_eq(key: PyObjectRef, field: &str) -> bool {
    if key.is_null() || unsafe { !pyre_object::is_str(key) } {
        return false;
    }
    match unsafe { pyre_object::w_str_get_wtf8(key) }.as_str() {
        Ok(name) => name == field,
        Err(_) => false,
    }
}

fn structseq_field_named(key: PyObjectRef, fields: &[String], extra_fields: &[String]) -> bool {
    fields.iter().any(|field| structseq_key_eq(key, field))
        || extra_fields
            .iter()
            .any(|field| structseq_key_eq(key, field))
}

/// `lib_pypy/_structseq.py structseq_new` — the `cls(sequence[,
/// dict])` constructor.  The first `n_sequence_fields` items fill the
/// tuple body; any surplus positional items, then the optional dict, then
/// `None` defaults, fill the named-only extra fields.
pub(crate) fn structseq_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() < 2 || args[1].is_null() {
        return Err(PyError::type_error("structseq() requires class + sequence"));
    }
    // `structseq_new(cls, sequence, dict)` keeps its three arguments live
    // across the class reads and the iteration below, each of which can
    // collect.
    let roots = pyre_object::gc_roots::push_roots();
    let arg_base = roots.pin_roots(&[
        args[0],
        args[1],
        args.get(2).copied().unwrap_or(pyre_object::PY_NULL),
    ]);
    let cls = || roots.get(arg_base);
    let sequence = || roots.get(arg_base + 1);
    // A type built by [`disallow_instantiation`] has no `tp_new` upstream, so
    // its `__new__` is `object.__new__`, which refuses it.  The descriptor is
    // reached directly as `type(sys.flags).__new__(...)`, past the check
    // `type.__call__` makes.
    crate::call::check_type_instantiable(cls())?;
    let n_seq = read_class_int(cls(), "n_sequence_fields").unwrap_or(0) as usize;
    let n_fields = read_class_int(cls(), "n_fields").unwrap_or(n_seq as i64) as usize;
    let name = class_name(cls()).unwrap_or_else(|| "structseq".to_string());
    let extra_names = extra_field_names(cls())?;

    // `_structseq.py structseq_new` — the optional second arg is a dict supplying
    // values for the named-only extra fields.
    if args.len() > 3 {
        return Err(PyError::type_error(format!(
            "{name}() takes at most 2 arguments ({} given)",
            args.len() - 1
        )));
    }
    // Signature binding leaves an omitted optional argument as PY_NULL.  An
    // explicit None is different and is rejected by both PyPy's
    // `isinstance(dict, builtin_dict)` and CPython 3.14's `PyDict_Check`.
    let dict_arg = || Some(roots.get(arg_base + 2)).filter(|d| !d.is_null());
    if let Some(d) = dict_arg()
        && !unsafe { pyre_object::is_dict(d) }
    {
        return Err(PyError::type_error(format!(
            "{name} takes a dict as second arg, if any"
        )));
    }

    // `_structseq.py:102-107` — a 1-field structseq wraps its scalar arg;
    // otherwise the arg is iterated into the field values.
    let mut items = if n_seq == 1 {
        vec![sequence()]
    } else {
        crate::builtins::collect_iterable(sequence())?
    };
    if items.len() < n_seq {
        return Err(PyError::type_error(format!(
            "expected a sequence with {} {} items. has {}",
            if n_seq < n_fields {
                "at least"
            } else {
                "exactly"
            },
            n_seq,
            items.len()
        )));
    }
    if items.len() > n_fields {
        return Err(PyError::type_error(format!(
            "expected a sequence with {} {} items. has {}",
            if n_seq < n_fields {
                "at most"
            } else {
                "exactly"
            },
            n_fields,
            items.len()
        )));
    }

    // `_structseq.py:115-143` — first `n_seq` items form the tuple body;
    // surplus items fill leading extras, then the dict, then `None`.
    let surplus = items.len() - n_seq;
    let surplus_vals: Vec<PyObjectRef> = items.split_off(n_seq);
    let body = items;

    // CPython 3.14 consumes only named-only fields that have not already
    // been supplied by surplus sequence items.  Any remaining key is either
    // a duplicate positional value or an unknown field; both use the shared
    // structseq diagnostic.  PyPy's older app-level constructor only noticed
    // duplicates among extra fields, so the 3.14 rule wins here.
    if let Some(d) = dict_arg() {
        let allowed = &extra_names[surplus..];
        let has_unexpected = unsafe { pyre_object::w_dict_items(d) }
            .into_iter()
            .any(|(key, _)| {
                if !unsafe { pyre_object::is_str(key) } {
                    return true;
                }
                let Some(key) = (unsafe { pyre_object::w_str_get_value_opt(key) }) else {
                    return true;
                };
                !allowed.iter().any(|name| name == key)
            });
        if has_unexpected {
            return Err(PyError::type_error(
                "got duplicate or unexpected field name(s)",
            ));
        }
    }

    let mut extras: Vec<(&str, PyObjectRef)> = Vec::with_capacity(extra_names.len());
    for (i, ename) in extra_names.iter().enumerate() {
        let in_dict = dict_arg()
            .is_some_and(|d| unsafe { pyre_object::w_dict_getitem_str(d, ename).is_some() });
        let value = if i < surplus {
            if in_dict {
                return Err(PyError::type_error(
                    "got duplicate or unexpected field name(s)",
                ));
            }
            surplus_vals[i]
        } else if let Some(d) = dict_arg() {
            unsafe { pyre_object::w_dict_getitem_str(d, ename) }.unwrap_or_else(pyre_object::w_none)
        } else {
            pyre_object::w_none()
        };
        extras.push((ename.as_str(), value));
    }

    // `app_posix.py stat_result.__init__` — a tuple-constructed
    // stat_result leaves the float `st_atime`/`st_mtime`/`st_ctime` extras
    // as None; fall back to the integer timestamps at body slots 7..9.
    if name == "os.stat_result" && body.len() > 9 {
        for (slot, ename) in [(7usize, "st_atime"), (8, "st_mtime"), (9, "st_ctime")] {
            if let Some(entry) = extras.iter_mut().find(|(n, _)| *n == ename)
                && unsafe { pyre_object::is_none(entry.1) }
            {
                entry.1 = body[slot];
            }
        }
    }

    Ok(new_instance_with_extra(cls(), body, extras))
}

fn read_class_int(cls: PyObjectRef, attr: &str) -> Option<i64> {
    let v = crate::baseobjspace::getattr_str(cls, attr).ok()?;
    if unsafe { pyre_object::is_int(v) } {
        Some(unsafe { pyre_object::w_int_get_value(v) })
    } else {
        None
    }
}

/// Allocate a structseq instance directly from a Rust-side value
/// vector — host modules use this when they already have all the
/// positional fields materialised and do not need the iteration /
/// arity-check work `structseq_descr_new` does for app-level callers.
pub fn new_instance(cls: PyObjectRef, items: Vec<PyObjectRef>) -> PyObjectRef {
    new_instance_with_extra(cls, items, Vec::new())
}

/// Allocate a structseq instance carrying both the positional tuple body
/// (`items`) and named-only extras (`extras`).  Each extra is stored under
/// its own name so the per-field getter can resolve it (`_structseq.py`
/// extra-field arm); the owning type must have been built with a matching
/// `extra_fields` list via [`make_struct_seq_with_extra`] (which sets
/// `hasdict`).  `os.stat_result` uses this for the float time fields and the
/// `st_*_ns` extras.
pub fn new_instance_with_extra(
    cls: PyObjectRef,
    items: Vec<PyObjectRef>,
    extras: Vec<(&str, PyObjectRef)>,
) -> PyObjectRef {
    // RPython keeps constructor arguments live as GC references.  Mirror that
    // shape explicitly across the tuple/dict allocations instead of relying
    // on raw Rust Vec entries surviving a moving collection.
    let _roots = pyre_object::gc_roots::push_roots();
    // Publish the class, every item and every extra as one batch and read each
    // back from its slot: a slot is what a promotion rewrites, the copies left
    // in these Rust vectors are not.
    let mut roots = Vec::with_capacity(1 + items.len() + extras.len());
    roots.push(cls);
    roots.extend_from_slice(&items);
    roots.extend(extras.iter().map(|&(_, value)| value));
    let cls_slot = pyre_object::gc_roots::pin_roots(&roots);
    let items_slot = cls_slot + 1;
    let extras_slot = items_slot + items.len();
    let rooted_items = (0..items.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(items_slot + index))
        .collect();
    let rooted_cls = pyre_object::gc_roots::shadow_stack_get(cls_slot);
    // `get_unique_interplevel_subclass` gives every tuple subclass the one
    // generated `_getusercls(W_TupleObject)` class, `__dict__` or not
    // (`typedef.py`). Structseq extras therefore live on their owner
    // rather than in the native address-keyed fallback table.
    let obj = pyre_object::w_tuple_subclass_new_array_backed(rooted_items, rooted_cls);
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // `build_stat_result` stores each named-only field with
    // `w_result.setdictvalue(space, name, w_value)` and never builds a dict:
    // its comment names that as the point -- "circumvent the huge mess of
    // structseq_new and a dict argument and just build the object ourselves.
    // then it stays nicely virtual". On a mapdict carrier each store is a map
    // transition, and every later instance of the same structseq reuses the
    // shape those transitions built. Assembling a dict and installing it
    // instead materialises the `("dict", SPECIAL)` wrapper on the instance's
    // map, which moves the whole instance to dict storage -- one hashed insert
    // per extra on every allocation, for `os.stat_result` thirteen of them.
    for (index, (key, _)) in extras.iter().enumerate() {
        let stored = crate::baseobjspace::setdictvalue(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(extras_slot + index),
        )
        .expect("structseq extras: a name store on a fresh tuple subclass cannot raise");
        assert!(
            stored,
            "structseq extras: the owning type must carry a dict (make_struct_seq_with_extra)"
        );
    }
    pyre_object::gc_roots::shadow_stack_get(obj_slot)
}

/// `lib_pypy/_structseq.py structseqtype.__new__` —
/// build a new tuple subclass with the supplied positional field names.
/// The returned type is the value module callers stash so future
/// allocations route through [`new_instance`].
pub fn make_struct_seq(name: &'static str, field_names: &[&'static str]) -> PyObjectRef {
    make_struct_seq_impl(name, field_names, &[])
}

/// Like [`make_struct_seq`] but adds named-only fields beyond the tuple
/// sequence (`_structseq.py __get__` extra-field arm).  `extra_field_names`
/// resolve through the instance `__dict__`, shadowing any same-named
/// positional slot, and the type is marked `hasdict` so [`new_instance_with_extra`]
/// can store them.  `os.stat_result` is the canonical user.
pub fn make_struct_seq_with_extra(
    name: &'static str,
    field_names: &[&'static str],
    extra_field_names: &[&'static str],
) -> PyObjectRef {
    make_struct_seq_impl(name, field_names, extra_field_names)
}

/// Mark a structseq type `Py_TPFLAGS_DISALLOW_INSTANTIATION`, which is how
/// `sys.flags`, `sys.version_info` and `sys.getwindowsversion` are built —
/// their single instance is the module's own answer and there is nothing a
/// second one could describe.  Interpreter-side construction goes through
/// [`new_instance`], which allocates directly, so only `Type(...)` from Python
/// is refused.
pub fn disallow_instantiation(cls: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_type_set_disallow_instantiation(cls) };
    cls
}
fn make_struct_seq_impl(
    name: &'static str,
    field_names: &[&'static str],
    extra_field_names: &[&'static str],
) -> PyObjectRef {
    let (module, short_name) = name
        .rsplit_once('.')
        .map_or((None, name), |(module, short)| (Some(module), short));
    let roots = pyre_object::gc_roots::push_roots();
    let field_type_slot = roots.pin_roots(&[structseq_app_attr("structseqfield")]);
    let meta_slot = roots.pin_roots(&[structseq_app_attr("structseqtype")]);
    let ns_slot = roots.pin_roots(&[pyre_object::w_dict_new()]);
    let store = |key: &str, value: PyObjectRef| {
        let value_slot = roots.pin_roots(&[value]);
        unsafe { pyre_object::w_dict_setitem_str(roots.get(ns_slot), key, roots.get(value_slot)) };
    };

    // `app_posix.py stat_result` numbers the named-only fields past a gap in
    // the indices, which is what makes `structseqtype.__new__` classify them
    // as extra fields rather than positional ones.
    let n_sequence_fields = field_names.len();
    let indexed = field_names.iter().enumerate().chain(
        extra_field_names
            .iter()
            .enumerate()
            .map(|(i, field)| (n_sequence_fields + 1 + i, field)),
    );
    for (index, field) in indexed {
        let w_field = crate::call::call_function_impl_result(
            roots.get(field_type_slot),
            &[pyre_object::w_int_new(index as i64)],
        )
        .unwrap_or_else(|e| panic!("structseqfield({index}) for {name}: {e:?}"));
        store(field, w_field);
    }
    // `structseqtype.__new__` takes `_name` from a `name` class attribute,
    // which the app-level classes (`app_posix.py stat_result`) spell
    // `name = "os.stat_result"`.  A type with a field called `name`
    // (`system.py thread_info`) cannot, and upstream then answers `_name`
    // with that field; its repr would print the field value where the type
    // name goes.  Such a type gets `_name` stored after creation instead.
    let name_is_field = field_names
        .iter()
        .chain(extra_field_names)
        .any(|field| *field == "name");
    if !name_is_field {
        store("name", pyre_object::w_str_new(name));
    }
    if let Some(module) = module {
        store("__module__", pyre_object::w_str_new(module));
    }
    store(
        "__new__",
        crate::typedef::make_new_descr_with_signature(
            structseq_descr_new,
            crate::gateway::Signature::new(vec!["cls", "sequence", "dict"], None, None, 0, 1),
        ),
    );
    store(
        "__replace__",
        crate::gateway::make_builtin_function_passthrough1(
            "__replace__",
            structseq_replace,
            structseq_replace_args,
        ),
    );

    let bases_slot = roots.pin_roots(&[pyre_object::w_tuple_new(Vec::new())]);
    let name_slot = roots.pin_roots(&[pyre_object::w_str_new(short_name)]);
    let cls = crate::call::call_function_impl_result(
        roots.get(meta_slot),
        &[
            roots.get(name_slot),
            roots.get(bases_slot),
            roots.get(ns_slot),
        ],
    )
    .unwrap_or_else(|e| panic!("structseqtype for {name}: {e:?}"));
    // CPython's PyStructSequence types publish no BASETYPE flag; tuple's
    // shared TypeDef stays an acceptable base (`check_and_find_best_base`).
    unsafe { pyre_object::w_type_suppress_cpython_basetype(cls) };
    if name_is_field {
        let cls_slot = roots.pin_roots(&[cls]);
        crate::baseobjspace::setattr_str(
            roots.get(cls_slot),
            "_name",
            pyre_object::w_str_new(name),
        )
        .unwrap_or_else(|e| panic!("{name}._name: {e:?}"));
        return roots.get(cls_slot);
    }
    cls
}

#[cfg(test)]
mod tests {
    #[test]
    fn make_struct_seq_uses_applevel_structseqtype() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let ec = std::rc::Rc::new(crate::PyExecutionContext::default());
        crate::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&ec));
        crate::importing::install_builtin_modules();
        let cls = super::make_struct_seq("os.stat_result", &["st_mode"]);
        let module =
            crate::importing::get_builtin_module("_structseq").expect("_structseq is installed");
        let meta = crate::baseobjspace::getattr_str(module, "structseqtype")
            .expect("_structseq.structseqtype");
        assert!(
            std::ptr::eq(unsafe { (*cls).w_class }, meta),
            "builtin structseq types use the applevel structseqtype identity"
        );
        let inst = super::new_instance(cls, vec![pyre_object::w_int_new(7)]);
        let mode = crate::baseobjspace::getattr_str(inst, "st_mode").expect("st_mode descriptor");
        assert!(unsafe { pyre_object::pyobject::is_int(mode) });
        assert_eq!(unsafe { pyre_object::intobject::w_int_get_value(mode) }, 7);
    }
}
