//! Pickle reduce protocol for `object` — PyPy: `pypy/objspace/std/objectobject.py`.
//!
//! The app-level helpers `reduce_1` / `reduce_2` / `get_slotvalues` /
//! `slotnames` (objectobject.py) are bundled in
//! `reduce_protocol_app.py` and resolved lazily through
//! `appleveldef_install` into a rooted GC module dict.  The three
//! handles the interp-level code calls (`reduce_1`, `reduce_2`,
//! `get_slotvalues`) keep that namespace as their `__globals__`, so
//! `get_slotvalues` can still reach its sibling `slotnames`.
//!
//! The interp-level descriptors `descr_reduce` / `descr_reduce_ex` /
//! `object_getstate_default` / `getnewargs` mirror objectobject.py's
//! `descr__reduce__` / `descr__reduce_ex__` / `object_getstate_default`
//! / `_getnewargs`.

use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8;

use crate::PyResult;
use crate::error::PyError;

/// Resolved app-level handles, indexed `[reduce_1, reduce_2,
/// get_slotvalues]`.  PyPy resolves these once via `app.interphook`
/// (objectobject.py); pyre resolves them on first use into a
/// thread-local cache.
const REDUCE_1: usize = 0;
const REDUCE_2: usize = 1;
const GET_SLOTVALUES: usize = 2;

static HANDLES: parking_lot::Mutex<[usize; 3]> = parking_lot::Mutex::new([0; 3]);

/// Resolve (and cache) the three app-level handles.
///
/// Executes `reduce_protocol_app.py` into its own fresh module globals and
/// stores the selected handles in process-global GC root slots.  The functions
/// retain the execution namespace as their `__globals__`, which keeps
/// `slotnames` reachable from `get_slotvalues` even though only three names
/// are surfaced.
fn handle(which: usize) -> PyResult {
    let cached = HANDLES.lock()[which];
    if cached != 0 {
        return Ok(cached as PyObjectRef);
    }

    // Do not hold HANDLES while executing Python: applevel installation can
    // collect, and the root walker below must be able to lock the slots.  The
    // shadow root keeps the namespace/functions alive until the finished
    // array is published.  A racing initializer may duplicate this work, but
    // publication remains atomic under the mutex and either complete set is
    // equivalent.
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("reduce_protocol: no execution context");
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
    let _ = pyre_object::gc_roots::pin_root(w_app_globals);
    // The source reads `sys.modules['copyreg']`, so it needs the same `sys`
    // the program sees -- but binding the name through an import would read
    // whatever the program left in `sys.modules['sys']`, and a `None` there
    // is the sentinel that makes an import raise.  The interpreter's own
    // module carries the same `modules` dict, so seeding it keeps what the
    // source reads while taking the name out of the program's reach.
    let Some(w_sys) = crate::importing::get_interpreter_sys_module() else {
        panic!("reduce_protocol: no sys module to read `sys.modules` through");
    };
    // Running the source is running Python -- a fresh frame, the program's
    // recursion limit, the program's `builtins` -- so a failure belongs to the
    // `__reduce__` that forced the install.
    crate::importing::appleveldef_install_seeded(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        include_str!("reduce_protocol_app.py"),
        "reduce_protocol_app.py",
        // `objectobject.py` builds these with a bare `applevel(...)`, so they
        // take the `ApplevelClass.__init__` default rather than a module's
        // name.  Nothing binds them into a module, so this reaches app level
        // only as their `__globals__['__name__']`.
        "__builtin__",
        &["reduce_1", "reduce_2", "get_slotvalues"],
        &[("sys", w_sys)],
    )?;
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
    let get = |name: &str| {
        unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) }
            .unwrap_or_else(|| panic!("reduce_protocol: `{name}` not bound"))
    };
    let initialized = [
        get("reduce_1") as usize,
        get("reduce_2") as usize,
        get("get_slotvalues") as usize,
    ];
    let mut handles = HANDLES.lock();
    if handles[which] == 0 {
        *handles = initialized;
    }
    Ok(handles[which] as PyObjectRef)
}

/// RPython's GC transform treats the app-level interphook cache as a normal
/// object graph.  Pyre stores the equivalent shared handles outside the GC,
/// so expose each slot to the global root walker and write relocated pointers
/// back in place.
pub(crate) fn walk_handle_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut handles = HANDLES.lock();
    for slot in handles.iter_mut().filter(|slot| **slot != 0) {
        unsafe { visitor(&mut *(slot as *mut usize as *mut majit_ir::GcRef)) };
    }
}

/// `%T`-style class name of `w_obj` for error messages.
fn typename(w_obj: PyObjectRef) -> String {
    match crate::typedef::r#type(w_obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string(),
        None => "object".to_string(),
    }
}

/// objectobject.py `get_slotvalues(obj)` — app-level handle.
pub fn get_slotvalues(mut w_obj: PyObjectRef) -> PyResult {
    let w_func = pyre_object::with_roots!(w_obj => handle(GET_SLOTVALUES))?;
    crate::call::call_function_impl_result(w_func, &[w_obj])
}

/// objectobject.py `object_getstate_default(space, w_obj, required)`.
///
/// `required` is always 0 from both callers (`descr__getstate__` and the
/// proto>=2 path), so the variable-sized raise is unreachable and pyre
/// has no `variable_sized` layout notion to gate it on — the structure
/// is ported without that dead arm.
pub fn object_getstate_default(w_obj: PyObjectRef) -> PyResult {
    // Allocating the copy, storing into it and `get_slotvalues` all collect,
    // so the receiver, its `__dict__`, the items being copied and the copy
    // itself are read back from the shadow stack rather than kept in Rust
    // locals across those points.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_objdict = crate::baseobjspace::findattr(current_obj(), "__dict__")?;
    let mut state_slot = None;
    if let Some(d) = w_objdict {
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(d);
        if crate::baseobjspace::len_w(pyre_object::gc_roots::shadow_stack_get(dict_slot))? > 0 {
            let copy_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            let items = unsafe {
                pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(dict_slot))
            };
            let mut live = Vec::with_capacity(items.len() * 2);
            for (k, v) in &items {
                live.push(*k);
                live.push(*v);
            }
            let items_slot = pyre_object::gc_roots::pin_roots(&live);
            let mut count = 0usize;
            for index in 0..items.len() {
                let k = pyre_object::gc_roots::shadow_stack_get(items_slot + 2 * index);
                let v = pyre_object::gc_roots::shadow_stack_get(items_slot + 2 * index + 1);
                unsafe {
                    pyre_object::w_dict_store(
                        pyre_object::gc_roots::shadow_stack_get(copy_slot),
                        k,
                        v,
                    )
                };
                count += 1;
            }
            if count > 0 {
                state_slot = Some(copy_slot);
            }
        }
    }
    let w_slots = get_slotvalues(current_obj())?;
    let w_ret = match state_slot {
        Some(slot) => pyre_object::gc_roots::shadow_stack_get(slot),
        None => pyre_object::w_none(),
    };
    if unsafe { pyre_object::is_none(w_slots) } {
        return Ok(w_ret);
    }
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_ret);
    fields.push(w_slots);
    Ok(pyre_object::w_tuple_new(fields.take()))
}

/// objectobject.py `_getnewargs(space, w_obj)` — returns
/// `(hasargs, w_args, w_kwargs)`.
pub fn getnewargs(w_obj: PyObjectRef) -> Result<(bool, PyObjectRef, PyObjectRef), PyError> {
    // `__getnewargs_ex__`/`__getnewargs__` run arbitrary Python and the
    // no-args branch allocates the empty tuple, so `w_obj` moves underneath
    // this frame.  Read it back from the shadow stack for every later use.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_type = crate::typedef::r#type(current_obj())
        .ok_or_else(|| PyError::type_error("cannot determine type for __getnewargs__"))?;
    let w_descr = unsafe { crate::baseobjspace::lookup(current_obj(), "__getnewargs_ex__") };
    let hasargs;
    let w_args;
    let w_kwargs;
    if let Some(w_descr) = w_descr {
        let w_result = unsafe {
            crate::baseobjspace::get_and_call_function(w_descr, current_obj(), w_type.as_ptr(), &[])
        }?;
        if !unsafe { pyre_object::is_tuple(w_result) } {
            return Err(PyError::type_error(format!(
                "__getnewargs_ex__ should return a tuple, not '{}'",
                typename(w_result)
            )));
        }
        let n = unsafe { pyre_object::w_tuple_len(w_result) };
        if n != 2 {
            return Err(PyError::value_error(format!(
                "__getnewargs_ex__ should return a tuple of length 2, not {n}"
            )));
        }
        let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(w_result) };
        let wa = items[0];
        let wk = items[1];
        if !unsafe { pyre_object::is_tuple(wa) } {
            return Err(PyError::type_error(format!(
                "first item of the tuple returned by __getnewargs_ex__ must be a tuple, not '{}'",
                typename(wa)
            )));
        }
        if !unsafe { pyre_object::is_dict(wk) } {
            return Err(PyError::type_error(format!(
                "second item of the tuple returned by __getnewargs_ex__ must be a dict, not '{}'",
                typename(wk)
            )));
        }
        hasargs = true;
        w_args = wa;
        w_kwargs = wk;
    } else {
        let w_descr = unsafe { crate::baseobjspace::lookup(current_obj(), "__getnewargs__") };
        if let Some(w_descr) = w_descr {
            let wa = unsafe {
                crate::baseobjspace::get_and_call_function(
                    w_descr,
                    current_obj(),
                    w_type.as_ptr(),
                    &[],
                )
            }?;
            if !unsafe { pyre_object::is_tuple(wa) } {
                return Err(PyError::type_error(format!(
                    "__getnewargs__ should return a tuple, not '{}'",
                    typename(wa)
                )));
            }
            hasargs = true;
            w_args = wa;
        } else {
            hasargs = false;
            w_args = pyre_object::w_tuple_new(vec![]);
        }
        w_kwargs = pyre_object::w_none();
    }
    Ok((hasargs, w_args, w_kwargs))
}

/// objectobject.py `descr__reduce__(space, w_obj)` — `reduce_1(obj, 0)`.
pub fn descr_reduce(w_obj: PyObjectRef) -> PyResult {
    reduce_1(w_obj, 0)
}

/// setobject.c `set_reduce` — `(type(self), (list(self),), state)` where
/// `state` is `_PyObject_GetState(self)` (`object_getstate_default`).  Both
/// `set` and `frozenset` expose this as `__reduce__`, so a subclass with a
/// payload round-trips through `copy`/`pickle` instead of collapsing to the
/// base type.
pub fn set_reduce(w_obj: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_type = crate::typedef::r#type(w_obj)
        .ok_or_else(|| PyError::type_error("cannot determine type for __reduce__"))?;
    let _ = pyre_object::gc_roots::pin_root(w_type.as_ptr());
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // `w_list_new` collects, so publish every already-live member first
    // and rebuild the list from the forwarded slots.
    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    };
    let items_base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(items_base, items.len());
    let reloaded: Vec<_> = (0..items.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(items_base + i))
        .collect();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::listobject::w_list_new(reloaded));
    let w_args = pyre_object::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(list_slot)]);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_args);

    let w_state = object_getstate_default(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
    let state_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_state);

    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        pyre_object::gc_roots::shadow_stack_get(args_slot),
        pyre_object::gc_roots::shadow_stack_get(state_slot),
    ]))
}

/// objectobject.py `reduce_1(obj, proto)` — app-level handle.
fn reduce_1(w_obj: PyObjectRef, proto: i64) -> PyResult {
    // Boxing `proto` allocates, so the receiver cannot be copied into the
    // argument array before it.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let w_proto = pyre_object::w_int_new(proto);
    let proto_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_proto);
    crate::call::call_function_impl_result(
        handle(REDUCE_1)?,
        &[
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(proto_slot),
        ],
    )
}

/// objectobject.py `reduce_2(obj, proto, args, kwargs)` — app-level handle.
fn reduce_2(
    w_obj: PyObjectRef,
    proto: i64,
    w_args: PyObjectRef,
    w_kwargs: PyObjectRef,
) -> PyResult {
    // Same as `reduce_1`: boxing `proto` allocates, and here it would move
    // three already-copied references.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[w_obj, w_args, w_kwargs]);
    let w_proto = pyre_object::w_int_new(proto);
    let proto_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_proto);
    crate::call::call_function_impl_result(
        handle(REDUCE_2)?,
        &[
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(proto_slot),
            pyre_object::gc_roots::shadow_stack_get(obj_slot + 1),
            pyre_object::gc_roots::shadow_stack_get(obj_slot + 2),
        ],
    )
}

/// objectobject.py `descr__reduce_ex__(space, w_obj, proto)`.
pub fn descr_reduce_ex(w_obj: PyObjectRef, proto: i64) -> PyResult {
    // Every attribute lookup below can run Python, and `getnewargs` allocates
    // even when the type defines neither hook, so the receiver and the
    // arguments it produced have to come back off the shadow stack rather than
    // out of a Rust local.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    // Honour a user `__reduce__` override:
    // `type(obj).__reduce__ is not object.__reduce__`.
    let w_reduce = crate::baseobjspace::findattr(current_obj(), "__reduce__")?;
    if let Some(w_reduce) = w_reduce {
        let reduce_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_reduce);
        let w_type = crate::typedef::r#type(current_obj())
            .ok_or_else(|| PyError::type_error("cannot determine type for __reduce_ex__"))?;
        let w_cls_reduce = crate::baseobjspace::getattr_str(w_type.as_ptr(), "__reduce__")?;
        // Both sides of the identity test have to survive the second lookup,
        // or a collection between them reports a spurious override.
        let cls_reduce_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_cls_reduce);
        let w_obj_reduce =
            crate::baseobjspace::getattr_str(crate::typedef::w_object(), "__reduce__")?;
        let obj_reduce_slot = _roots.pin_roots(&[w_obj_reduce]);
        let w_cls_reduce = pyre_object::gc_roots::shadow_stack_get(cls_reduce_slot);
        let w_reduce = pyre_object::gc_roots::shadow_stack_get(reduce_slot);
        let mut override_ = !crate::baseobjspace::is_w(w_cls_reduce, _roots.get(obj_reduce_slot));
        // Built-in types (range, the iterators) expose `__reduce__`
        // through instance dispatch rather than the type MRO, so the
        // type-level comparison above sees `object.__reduce__` and
        // misses the override.  Compare the bound instance method's
        // `__func__` against `object.__reduce__` to catch it: a genuine
        // override binds a different function.
        if !override_ {
            let w_inst_func = if unsafe { pyre_object::function::is_method(w_reduce) } {
                unsafe { pyre_object::function::w_method_get_func(w_reduce) }
            } else {
                w_reduce
            };
            override_ = !crate::baseobjspace::is_w(w_inst_func, _roots.get(obj_reduce_slot));
        }
        if override_ {
            return crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(reduce_slot),
                &[],
            );
        }
    }
    if proto >= 2 {
        // CPython 3.14 `typeobject.c:reduce_newobj`: `tp_new == NULL`
        // means there is no constructor with which `__newobj__` can rebuild
        // the instance.  Pyre represents that slot state with
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION`; check it before collecting new
        // arguments, exactly as `reduce_newobj` does.  Coroutine objects and
        // their `__await__` wrappers both have this shape.
        let w_type = crate::typedef::r#type(current_obj())
            .ok_or_else(|| PyError::type_error("cannot determine type for __reduce_ex__"))?;
        if unsafe { pyre_object::w_type_disallows_instantiation(w_type.as_ptr()) } {
            return Err(PyError::type_error(format!(
                "cannot pickle '{}' object",
                typename(current_obj())
            )));
        }
        let (hasargs, w_args, w_kwargs) = getnewargs(current_obj())?;
        let args_slot = pyre_object::gc_roots::pin_roots(&[w_args, w_kwargs]);
        // CPython 3.14 `object_getstate_default(required)`: when no
        // `__getnewargs__` supplied constructor state, a native layout whose
        // basicsize exceeds the plain object + managed dict/weakref/slots
        // baseline cannot be rebuilt through an empty `__newobj__` call.
        // PyPy reaches the same safe shape by giving picklable native TypeDefs
        // their own reducer instead of teaching object.__reduce_ex__ their
        // fields.  `ob_type` is the RPython-vtable/layout tag: `W_ObjectObject`
        // and `typedef.py` `_getusercls`'s `W_ObjectObjectUserDictWeakrefable`
        // child have no hidden native payload.  Lists and dicts are the two
        // upstream exceptions because reduce_2 serializes their contents
        // through listitems/dictitems below.  This one layout test replaces
        // the incomplete per-type census (module, memoryview, staticmethod,
        // classmethod) and also covers property and every itertools TypeDef
        // whose type supplies no reducer.
        let native_layout = unsafe { !pyre_object::is_instance(current_obj()) };
        // `object_getstate` hands the call to an overriding `__getstate__`
        // and only falls through to `object_getstate_default(required)` when
        // the type still uses `object.__getstate__`; `descr__reduce_ex__`
        // spells the same gate as a `space.lookup` of the hook.  A native
        // layout that publishes its own hook is rebuilt from the state that
        // hook returns, so the refusal below must not see it: `_io.BytesIO`
        // and `_io.StringIO` hold a native buffer and pickle for exactly this
        // reason.  `getnewargs` above can collect, so the type is read back
        // rather than reused.
        let w_type = crate::typedef::r#type(current_obj())
            .ok_or_else(|| PyError::type_error("cannot determine type for __reduce_ex__"))?;
        let supplies_getstate = unsafe {
            let w_cls_getstate = crate::baseobjspace::lookup_in_type(
                w_type.as_ptr(),
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__getstate__")),
            );
            let w_obj_getstate = crate::baseobjspace::lookup_in_type(
                crate::typedef::w_object(),
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__getstate__")),
            );
            match (w_cls_getstate, w_obj_getstate) {
                (Some(w_cls), Some(w_obj)) => !crate::baseobjspace::is_w(w_cls, w_obj),
                (w_cls, _) => w_cls.is_some(),
            }
        };
        if !hasargs
            && native_layout
            && !supplies_getstate
            && !unsafe { pyre_object::is_list(current_obj()) }
            && !unsafe { pyre_object::is_dict(current_obj()) }
        {
            return Err(PyError::type_error(format!(
                "cannot pickle '{}' object",
                typename(current_obj())
            )));
        }
        return reduce_2(
            current_obj(),
            proto,
            pyre_object::gc_roots::shadow_stack_get(args_slot),
            pyre_object::gc_roots::shadow_stack_get(args_slot + 1),
        );
    }
    reduce_1(current_obj(), proto)
}
