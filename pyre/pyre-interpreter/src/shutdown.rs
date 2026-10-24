//! Process shutdown shared by the native launcher and the wasm guest.
//!
//! `baseobjspace.py` `ObjSpace.finish`, then the module teardown
//! (`finalize_modules`). `app_main.py` `run_toplevel` prints an uncaught
//! exception before this runs.

/// PyPy object-space finalization / `threading._shutdown`: run the app-level
/// callback before module teardown, then join native non-daemon handles.
fn run_threading_shutdown() {
    if let Some(threading) = crate::importing::get_sys_module("threading")
        && let Ok(shutdown) = crate::getattr(threading, pyre_object::w_str_new("_shutdown"))
    {
        let result = crate::call_function(shutdown, &[]);
        if result.is_null()
            && let Some(err) = crate::call::take_call_error()
        {
            crate::eprint_exception(&err, true);
        }
    }
}

/// The app-level atexit callback stack. Any escaping error is unraisable and
/// must not prevent the remaining shutdown phases.
fn run_atexit_callbacks(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
) {
    let result = crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("atexit"),
        canonical,
        pyre_object::PY_NULL,
        0,
        ec_ptr,
    )
    .and_then(|module| crate::getattr(module, pyre_object::w_str_new("_run_exitfuncs")))
    .and_then(|callback| crate::call::call_function_impl_result(callback, &[]));
    if let Err(mut error) = result {
        error.write_unraisable(
            pyre_object::w_none(),
            rustpython_wtf8::Wtf8::new("_run_exitfuncs"),
            pyre_object::w_none(),
        );
    }
}

/// `interp_has_atexit_callbacks`: whether `atexit._ncallbacks()` is nonzero.
/// A failed lookup or call reads as no callbacks, so shutdown proceeds.
fn has_atexit_callbacks(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
) -> bool {
    crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("atexit"),
        canonical,
        pyre_object::PY_NULL,
        0,
        ec_ptr,
    )
    .and_then(|module| crate::getattr(module, pyre_object::w_str_new("_ncallbacks")))
    .and_then(|count| crate::call::call_function_impl_result(count, &[]))
    .and_then(crate::baseobjspace::int_w)
    .is_ok_and(|n| n != 0)
}

/// Sweep the whole heap and run whatever that made finalizable, reporting
/// whether anything was. `false` says the heap held no unreachable finalizer at
/// this point, so a caller that has changed nothing since can skip its next
/// sweep: that one would walk the same heap to the same empty queue.
///
/// The answer comes from the collector rather than from the drain: a sweep runs
/// the death-queue trigger only for a queue it has just put something in, so a
/// bumped `finalizer_trigger_count` is that sweep reporting what it found.
fn collect_and_run_finalizers(ec_ptr: *const crate::executioncontext::PyExecutionContext) -> bool {
    // pypy.module.gc.interp_gc.collect clears both semantic lookup caches
    // before collecting: cached finalizer methods otherwise retain their
    // declaring type and globals after the finalizer returns.
    crate::baseobjspace::clear_method_cache();
    crate::objspace::std::mapdict::clear_map_attr_cache();
    let triggers_before = crate::executioncontext::finalizer_trigger_count();
    pyre_object::gc_hook::try_gc_collect(2);
    let made_finalizable = crate::executioncontext::finalizer_trigger_count() != triggers_before;
    if !ec_ptr.is_null() {
        unsafe {
            (&mut *(ec_ptr as *mut crate::executioncontext::PyExecutionContext))
                ._run_finalizers_now()
        };
    }
    made_finalizable
}

/// Whether releasing this binding can remove anything from the reachable set,
/// and so whether the collection that follows it could find garbage an earlier
/// one did not. Answering `false` skips a full mark-and-sweep of the whole
/// heap, which is what the release loop below otherwise costs per name.
///
/// Two values answer `false` outright:
///
/// * An exact `int`, `float`, `bool`, `str`, `bytes` or `None`. It holds no
///   reference to another object, so nothing but the value itself can lose its
///   last referrer, and no builtin scalar type defines `__del__`, so nothing
///   observes it going. `is_exact_type` is what makes this safe rather than
///   `is_str`/`is_int`, which key off the layout `ob_type` a subclass keeps: a
///   `class MyStr(str)` instance retags `w_class` and is rejected here, so its
///   `__del__` and its own attributes stay on the collecting path.
/// * A module still registered in `sys.modules` under its own `__name__`. It
///   stays reachable from there no matter what `__main__` does, so the release
///   removes no object at all from the reachable set. This is every `import`
///   name. Reading the live `sys.modules` rather than assuming is what keeps a
///   program that replaced or deleted the entry on the collecting path.
///
/// The point is not that these values are cheap to collect — it is that the
/// collection cannot reach a different answer, so every `__del__` still runs at
/// exactly the same place in the loop.
fn release_frees_nothing(mut value: pyre_object::PyObjectRef) -> bool {
    if value.is_null() {
        return true;
    }
    unsafe {
        if pyre_object::is_none(value)
            || pyre_object::is_exact_type(value, &pyre_object::INT_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::BOOL_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::FLOAT_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::STR_TYPE)
            || pyre_object::is_exact_type(value, &pyre_object::BYTES_TYPE)
        {
            return true;
        }
        if pyre_object::is_module(value) {
            // `Module.w_name` is the interpreter's own field, not the
            // program-writable `__name__` attribute. `module.__new__` leaves it
            // null until `__init__` seeds it, and a valid Python name may carry
            // a lone surrogate that cannot key pyre's native UTF-8 registry.
            // Either shape proves nothing about reachability and therefore
            // takes the collecting path.
            let w_name = pyre_object::w_module_get_name(value);
            if w_name.is_null() || !pyre_object::is_str(w_name) {
                return false;
            }
            let Ok(name) = pyre_object::w_str_get_wtf8(w_name).as_str() else {
                return false;
            };
            return pyre_object::with_roots!(value => crate::importing::get_sys_module(name))
                .is_some_and(|m| m == value);
        }
    }
    false
}

/// Whether releasing this binding leaves the collection that would follow with
/// no finalizer to deliver — and so whether that whole-heap mark-and-sweep can
/// be skipped without moving where any `__del__` runs.
///
/// `release_frees_nothing` answers first: a value the release removes nothing
/// from the reachable set can obviously deliver nothing. Past it two questions
/// remain, and both are `O(1)` where the collection is `O(heap)`:
///
/// * Does the collector still owe *any* delivery? `deal_with_objects_with_
///   finalizers` is the only pass that hands control back to the program, and
///   with its queues empty a sweep can free memory but cannot run a line of
///   Python. Re-asked per name because a `__del__` this loop runs can register
///   one (`test_start_new_thread_at_finalization`'s starts a thread).
/// * Is *this* object one it owes? `GcFlags::FINALIZER_REGISTERED` is set by
///   `allocate_instance` for every instance of a `hasuserdel` class and by
///   `_io`, coroutine and weakref-lifeline construction for objects whose type
///   carries no such flag, so it is the whole of PyPy's `hasuserdel` test and
///   more.
///
/// The second is where this stops being exact, and deliberately: releasing a
/// *container* of a finalizable — `holder = [AtFinalization()]` — frees one
/// that no `O(1)` test on `holder` can see, and its `__del__` then runs at the
/// walk's trailing collection instead of at its own name, reading `None` for
/// the `__main__` globals released after it rather than their values. It still
/// runs. Buying that case back costs a whole-heap mark-and-sweep per name, and
/// a namespace binds one per function, class and import: releasing `inspect`'s
/// 182 names as `__main__` was 1.5s of `pyre -m inspect`'s 1.9s against pypy3's
/// 0.09s total, and `test.test_inspect` pays it four times over in
/// subprocesses. PyPy itself neither clears `__main__` nor collects here
/// (`baseobjspace.py finish`), so nothing upstream prices that case higher.
fn release_delivers_no_finalizer(mut value: pyre_object::PyObjectRef) -> bool {
    pyre_object::with_roots!(value => release_frees_nothing(value))
        || !majit_gc::gc_has_pending_finalizers()
        || !majit_gc::gc_object_finalizer_pending(value as usize)
}

/// `PYRE_GC_DIAG`: how many `__main__` bindings the release walk let go, and
/// how many of them it collected after. `swept` is the count
/// [`release_delivers_no_finalizer`] exists to keep near zero; a run where it
/// tracks `released` is one where the walk is back to a mark-and-sweep per name.
fn teardown_census(released: usize, swept: usize) {
    if std::env::var_os("PYRE_GC_DIAG").is_none() {
        return;
    }
    let registered = majit_gc::gc_registered_finalizer_count();
    eprintln!(
        "[jit-gc-diag] teardown_released={released} teardown_swept={swept} \
         finalizers_registered={registered}"
    );
}

fn shutdown_module_private_name(name: &rustpython_wtf8::Wtf8) -> bool {
    let bytes = name.as_bytes();
    bytes.first() == Some(&b'_') && bytes.get(1) != Some(&b'_')
}

fn clear_shutdown_module_name(dict: pyre_object::PyObjectRef, name: &rustpython_wtf8::Wtf8) {
    let should_clear = unsafe { pyre_object::w_dict_getitem_wtf8(dict, name) }
        .is_some_and(|value| unsafe { !pyre_object::is_none(value) });
    if should_clear {
        unsafe {
            pyre_object::w_dict_setitem_wtf8(dict, name, pyre_object::w_none());
        }
    }
}

/// `_PyModule_ClearDict`: clear string-keyed module globals in two name passes.
fn clear_shutdown_module_dict(dict: pyre_object::PyObjectRef) {
    if dict.is_null() {
        return;
    }
    let keys: Vec<rustpython_wtf8::Wtf8Buf> = unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    for name in &keys {
        if shutdown_module_private_name(name) {
            clear_shutdown_module_name(dict, name);
        }
    }
    for name in &keys {
        if name.as_bytes() != b"__builtins__" {
            clear_shutdown_module_name(dict, name);
        }
    }
}

/// `finalize_modules`: clear detached modules newest-first while their peers
/// remain available to finalizers that run between module dictionaries.
fn clear_shutdown_modules(
    released: crate::importing::ReleasedSysModules,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
) {
    let crate::importing::ReleasedSysModules { modules } = released;
    let _roots = pyre_object::gc_roots::push_roots();
    let roots_start = pyre_object::gc_roots::shadow_stack_len();
    let mut names = Vec::with_capacity(modules.len());
    let mut sys_module_slot = None;
    let mut builtins_module_slot = None;
    for (name, module) in modules {
        let index = names.len();
        let bytes = name.as_bytes();
        if bytes == b"sys" {
            sys_module_slot = Some(index);
        } else if bytes == b"builtins" {
            builtins_module_slot = Some(index);
        }
        names.push(name);
        let _ = pyre_object::gc_roots::pin_root(module);
    }
    // finalize_remove_modules in CPython v3.14.6 retains weak references:
    // holding these modules strongly would keep their cycles alive until
    // their globals are cleared. PyPy Module objects likewise have ordinary
    // GC lifetime; ObjSpace.finish does not clear their dictionaries. Keep
    // only GC-managed weak carriers across the collection so finalizers can
    // inspect the intact graph (test_module_finalization_at_shutdown).
    for index in 0..names.len() {
        let slot = roots_start + index;
        let module = pyre_object::gc_roots::shadow_stack_get(slot);
        let weak = unsafe { pyre_object::weakref::w_weakref_new(module) };
        pyre_object::gc_roots::shadow_stack_set(slot, weak.cast());
    }
    let module_at = |index| unsafe {
        pyre_object::weakref::w_weakref_deref(
            pyre_object::gc_roots::shadow_stack_get(roots_start + index).cast(),
        )
    };
    // CPython v3.14.6 pylifecycle.c finalize_modules collects unconditionally
    // after detaching sys.modules and before clearing surviving module dicts.
    // A previous finalizer can release the next link in a chain even when
    // detaching the cache itself made nothing unreachable. Collect while those
    // finalizers can still read their globals (test_module's
    // test_module_finalization_at_shutdown).
    // PyPy ObjSpace.finish runs module shutdown hooks without this dict-clear
    // phase; this collection preserves the existing CPython shutdown contract.
    // incminimark.IncrementalMiniMarkGC.deal_with_objects_with_finalizers
    // preserves dependency order across successive collections. Unlike
    // CPython's refcount/cyclic-GC finalization, one collection need not finish
    // the unreachable module graph. Allow the required first collection,
    // then at most one pass per finalizer still registered at entry. The first
    // pass also drains objects already delivered to the death queue. Stop early
    // when no finalizer is delivered. Snapshot the bound: a __del__ that
    // creates another finalizable cycle must not extend shutdown indefinitely.
    // PyPy ObjSpace.finish and CPython finalize_modules both have bounded
    // shutdown phases; this bound preserves the GC's dependency ordering
    // without chasing newly created generations forever.
    let collection_limit = majit_gc::gc_registered_finalizer_count().saturating_add(1);
    for _ in 0..collection_limit {
        if !collect_and_run_finalizers(ec_ptr) {
            break;
        }
    }
    for index in (0..names.len()).rev() {
        let module = module_at(index);
        let is_core_module = sys_module_slot.is_some_and(|slot| module == module_at(slot))
            || builtins_module_slot.is_some_and(|slot| module == module_at(slot));
        if is_core_module {
            continue;
        }
        if module.is_null() || !unsafe { pyre_object::is_module(module) } {
            continue;
        }
        // _PyWeakref_GET_REF owns the surviving module through _PyModule_Clear.
        // The weak carrier alone must not be its root while a store allocates.
        let module_root = pyre_object::gc_roots::push_roots();
        let module = module_root.pin_root(module);
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        clear_shutdown_module_dict(dict);
    }
    // One collection for the whole walk, not one per module. `finalize_modules`
    // clears the module dictionaries and lets refcounting release what they
    // held; a sweep per module buys no ordering here, because a finalizer that
    // reads a global reaches its own already-cleared namespace either way, and
    // it costs a full mark-and-sweep for each of the ~100 modules a bare
    // `import unittest` loads.
    collect_and_run_finalizers(ec_ptr);
}

/// The module dict of one of the two names finalization reaches for, or
/// `None` when the runtime never minted it.
///
/// The reads are from the interpreter's own registry, not from `sys.modules`:
/// `finalize_modules_delete_special` clears `interp->sysdict` and
/// `interp->builtins`, and a program can park any value under either name in
/// the mapping. `w_module_get_w_dict` is a raw field read whose contract is
/// "points to a valid `Module`", so a value that is not one yields a wild
/// pointer its caller then stores through -- `sys.modules['sys'] = 42`
/// segfaulted at exit. `clear_shutdown_modules` already tests the same way
/// before the same cast.
fn interpreter_module_dict(
    module: Option<pyre_object::PyObjectRef>,
) -> Option<pyre_object::PyObjectRef> {
    let module = module?;
    if module.is_null() || !unsafe { pyre_object::is_module(module) } {
        return None;
    }
    let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    (!dict.is_null()).then_some(dict)
}

/// `pylifecycle.c finalize_modules_delete_special`, the step `finalize_modules`
/// takes before it releases any module namespace.
///
/// `sys` and `builtins` are torn down last of all, so a user value parked on
/// one of them outlives every destructor that might complain about it; upstream
/// clears the usual hiding places up front for that reason.
///
/// `sys.meta_path` is among them, and clearing it is what makes an import
/// attempted from a `__del__` running this late raise `ImportError` instead of
/// succeeding.  `finalize_modules` reaches that state for every module because
/// it goes on to empty `sys.modules`, which this step does not: a module
/// already imported stays importable from a destructor running here.
///
/// The three standard streams are restored from their `__`-prefixed originals
/// rather than cleared, so a destructor still has somewhere to write.
fn finalize_delete_special() {
    // `path_hooks` and `path_importer_cache` are cleared by
    // `_PyImport_FiniExternal`, past where this runs, so they are not here.
    const SYS_CLEARED: [&str; 10] = [
        "path",
        "argv",
        "ps1",
        "ps2",
        "last_exc",
        "last_type",
        "last_value",
        "last_traceback",
        "__interactivehook__",
        "meta_path",
    ];
    const SYS_STREAMS: [(&str, &str); 3] = [
        ("stdin", "__stdin__"),
        ("stdout", "__stdout__"),
        ("stderr", "__stderr__"),
    ];
    if let Some(dict) = interpreter_module_dict(crate::importing::get_interpreter_builtins_module())
    {
        unsafe { pyre_object::w_dict_setitem_str(dict, "_", pyre_object::w_none()) };
    }
    let Some(dict) = interpreter_module_dict(crate::importing::get_interpreter_sys_module()) else {
        return;
    };
    for name in SYS_CLEARED {
        // `_PySys_ClearAttrString` binds `None` rather than deleting the name,
        // so a late reader finds a value that is not there instead of a name
        // that is not there.
        unsafe { pyre_object::w_dict_setitem_str(dict, name, pyre_object::w_none()) };
    }
    for (name, original) in SYS_STREAMS {
        let value = unsafe { pyre_object::w_dict_getitem_str(dict, original) }
            .unwrap_or_else(pyre_object::w_none);
        unsafe { pyre_object::w_dict_setitem_str(dict, name, value) };
    }
}

/// PyPy `ObjSpace.finish()` / module teardown ordering: join non-daemon
/// threads, collect already-unreachable cycles, then release `__main__`
/// globals from newest to oldest while the older globals their `__del__`
/// methods may reference are still present.
///
/// Newest-to-oldest is what keeps those references working, and it is not
/// interchangeable with the insertion order `_PyModule_ClearDict` uses. A name
/// is bound before every name that could be finalized while reading it — most
/// of all `import sys`, which is usually the very first — so releasing in
/// insertion order strands `sys` at `None` and every finalizer that writes to
/// `sys.stderr` dies with an `AttributeError` instead of running. Refcounting
/// makes the question moot upstream: a finalizer there runs from the decref of
/// the name being released, while every other slot still holds its original
/// value, which is not reproducible without leaving dangling pointers in the
/// dict.
///
/// The value is rebound to `None` rather than deleted, as `_PyModule_ClearDict`
/// does: a `__del__` that reads an already-released name then sees `None`, the
/// way it would upstream, instead of raising `NameError` at a name the program
/// can see is still defined.
///
/// `__main__` is not the whole reachable set: an object stored in another
/// module's namespace stays alive through that module's dict. The
/// `finalize_modules` phase that follows detaches `sys.modules` and clears the
/// remaining module dictionaries newest-first.
pub fn finalize_runtime(
    mut canonical: pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
) {
    // `make_pre_finalization_calls`: join the non-daemon threads, then run the
    // atexit callbacks, and repeat while a callback or a thread it started
    // left either a non-daemon thread (`interp_has_threads`) or another
    // callback (`interp_has_atexit_callbacks`).
    loop {
        pyre_object::with_roots!(canonical => run_threading_shutdown());
        pyre_object::with_roots!(canonical => run_atexit_callbacks(canonical, ec_ptr));
        if !crate::module::thread::has_shutdown_handles()
            && !pyre_object::with_roots!(canonical => has_atexit_callbacks(canonical, ec_ptr))
        {
            break;
        }
    }
    // PyPy sets `sys.finalizing` only after atexit callbacks.  Those callbacks
    // may still start threads; reject new starts only when module/finalizer
    // teardown is actually about to begin.
    crate::module::thread::set_finalizing();
    // Past this point a handler would run against a half-torn-down module
    // graph, so the teardown below reports signals instead of delivering
    // them.  atexit ran above and may legitimately have used signals.
    crate::module::signal::interp_signal::clear_handlers();
    // baseobjspace.py `finish()` runs every started module's shutdown
    // hook; `_io`'s (moduledef.py:37-40) flushes the streams that are still
    // alive.  The per-global teardown below reaches only the ones `__main__`
    // itself holds, so without this a stream owned by any other module loses
    // its buffered writes.
    pyre_object::with_roots!(canonical => crate::module::_io::flush_all_streams());
    // `faulthandler`'s (moduledef.py `shutdown`) cancels the watchdog and
    // restores the signal handlers before any module is torn down.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    crate::module::faulthandler::handler::finish();
    // `finalize_modules` opens with this and every release below is its module
    // teardown, so the clearing has to precede the whole walk rather than sit
    // beside `clear_shutdown_modules`.
    finalize_delete_special();
    let mut swept_something_finalizable =
        pyre_object::with_roots!(canonical => collect_and_run_finalizers(ec_ptr));
    let (mut released, mut swept) = (0usize, 0usize);
    let mut entries = unsafe { pyre_object::w_dict_str_entries(canonical) };
    entries.reverse();
    for (name, _) in entries {
        if name == "__builtins__" {
            continue;
        }
        // Re-read rather than trusting the snapshot: a `__del__` already run by
        // this loop may have rebound the name, and the decision below is only
        // sound about the value actually being released.
        let value = unsafe { pyre_object::w_dict_getitem_str(canonical, &name) };
        unsafe {
            pyre_object::w_dict_setitem_str(canonical, &name, pyre_object::w_none());
        }
        released += 1;
        if value.is_none_or(release_delivers_no_finalizer) {
            continue;
        }
        swept += 1;
        swept_something_finalizable =
            pyre_object::with_roots!(canonical => collect_and_run_finalizers(ec_ptr));
    }
    teardown_census(released, swept);
    // Close the loop on a swept heap and a drained queue. A release the loop
    // skipped removes nothing from the reachable set, and one it did not is
    // followed immediately by the sweep above, so the only way this heap can
    // hold a finalizer the last sweep missed is a `__del__` that sweep ran and
    // that dropped the last reference to something else.
    if swept_something_finalizable {
        collect_and_run_finalizers(ec_ptr);
    }
    let shutdown_modules = crate::importing::release_sys_modules_for_shutdown();
    clear_shutdown_modules(shutdown_modules, ec_ptr);
}
