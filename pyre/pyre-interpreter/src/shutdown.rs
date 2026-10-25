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
///   stays reachable from there no matter what another module's dict does, so
///   the release removes no object at all from the reachable set. This is
///   every `import` name. Reading the live mapping rather than assuming is
///   what keeps a program that replaced or deleted the entry on the collecting
///   path. After [`crate::importing::release_sys_modules_for_shutdown`] has
///   emptied that mapping, the answer is always false: a detached module is
///   no longer proof of reachability.
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

/// `PYRE_GC_DIAG`: how many module-dict bindings the shutdown walk let go, and
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

/// `_PyModule_ClearDict`: one name pass, forward dict order. Re-reads the
/// live value rather than trusting the snapshot, because a `__del__` already
/// run by this walk may have rebound the name.
fn clear_shutdown_module_dict_pass(
    dict: &mut pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
    released: &mut usize,
    swept: &mut usize,
    private_only: bool,
) {
    let mut keys: Vec<rustpython_wtf8::Wtf8Buf> =
        unsafe { pyre_object::w_dict_str_entries_wtf8(*dict) }
            .into_iter()
            .map(|(name, _)| name)
            .collect();
    let mut index = 0;
    while index < keys.len() {
        let name = keys[index].clone();
        index += 1;
        if private_only {
            if !shutdown_module_private_name(&name) {
                continue;
            }
        } else if name.as_bytes() == b"__builtins__" {
            continue;
        }
        let Some(old) = (unsafe { pyre_object::w_dict_getitem_wtf8(*dict, &name) }) else {
            continue;
        };
        if unsafe { pyre_object::is_none(old) } {
            continue;
        }
        let mut d = *dict;
        // Classify `old` before the store because it is not rooted across it.
        // Do not root `old`: a pin would keep a finalizable value reachable
        // and move its `__del__` to a later sweep, after names this pass
        // has already replaced. Root the dict across the store and any
        // collection it triggers.
        let delivers_none = release_delivers_no_finalizer(old);
        let did_sweep = pyre_object::with_roots!(d => {
            unsafe {
                pyre_object::w_dict_setitem_wtf8(d, &name, pyre_object::w_none());
            }
            *released += 1;
            if delivers_none {
                false
            } else {
                *swept += 1;
                collect_and_run_finalizers(ec_ptr);
                true
            }
        });
        *dict = d;
        if did_sweep {
            // Re-read remaining entries from the live dict: a `__del__` this
            // collection ran may have rebound a name still ahead of us, and
            // the dict pointer itself may have moved.
            keys = unsafe { pyre_object::w_dict_str_entries_wtf8(*dict) }
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            index = keys.iter().position(|n| n == &name).map_or(0, |at| at + 1);
        }
    }
}

/// `_PyModule_ClearDict`: clear string-keyed module globals in two name
/// passes (private `_x` names first, then every name except `__builtins__`),
/// both in forward dict order. A store that can deliver a finalizer is
/// followed by a collection while this dict is still rooted.
fn clear_shutdown_module_dict(
    dict: &mut pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
    released: &mut usize,
    swept: &mut usize,
) {
    if dict.is_null() {
        return;
    }
    clear_shutdown_module_dict_pass(dict, ec_ptr, released, swept, true);
    clear_shutdown_module_dict_pass(dict, ec_ptr, released, swept, false);
}

/// `finalize_modules`: collect after detaching `sys.modules`, then clear
/// surviving module dictionaries newest-first with `_PyModule_ClearDict`.
/// `sys` and `builtins` dictionaries are left intact: they outlive every
/// other namespace, the same way `ObjSpace.sys` / `ObjSpace.builtin` remain
/// on the space after `ObjSpace.finish`.
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
    // Holding these modules strongly would keep their cycles alive until
    // their globals are cleared. PyPy `module.py Module` has ordinary GC
    // lifetime; `ObjSpace.finish` does not clear their dictionaries. Keep
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
    // Collect after detaching `sys.modules` and before clearing surviving
    // module dictionaries. A previous finalizer can release the next link in
    // a chain even when detaching the cache itself made nothing unreachable.
    // Collect while those finalizers can still read their globals
    // (test_module_finalization_at_shutdown).
    // PyPy `ObjSpace.finish` runs module shutdown hooks without this
    // dict-clear phase.
    // incminimark.IncrementalMiniMarkGC.deal_with_objects_with_finalizers
    // preserves dependency order across successive collections. One
    // collection need not finish the unreachable module graph. Allow the
    // required first collection, then at most one pass per finalizer still
    // registered at entry. The first pass also drains objects already
    // delivered to the death queue. Stop early when no finalizer is
    // delivered. Snapshot the bound: a `__del__` that creates another
    // finalizable cycle must not extend shutdown indefinitely. This bound
    // preserves the GC's dependency ordering without chasing newly created
    // generations forever.
    let collection_limit = majit_gc::gc_registered_finalizer_count().saturating_add(1);
    for _ in 0..collection_limit {
        if !collect_and_run_finalizers(ec_ptr) {
            break;
        }
    }
    let (mut released_count, mut swept) = (0usize, 0usize);
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
        // A strong hold on the surviving module through `_PyModule_Clear`.
        // The weak carrier alone must not be its root while a store allocates.
        let module_root = pyre_object::gc_roots::push_roots();
        let module = module_root.pin_root(module);
        let mut dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        clear_shutdown_module_dict(&mut dict, ec_ptr, &mut released_count, &mut swept);
    }
    teardown_census(released_count, swept);
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

/// PyPy `ObjSpace.finish` joins non-daemon threads and runs module shutdown
/// hooks; it has no module-dict teardown phase. `finalize_modules` then
/// deletes the special sys/builtins names, detaches every `sys.modules`
/// entry, collects, and clears surviving module dictionaries newest-first
/// with `_PyModule_ClearDict`.
pub fn finalize_runtime(
    mut canonical: pyre_object::PyObjectRef,
    ec_ptr: *const crate::executioncontext::PyExecutionContext,
) {
    let _roots = pyre_object::gc_roots::push_roots();
    let canonical_slot = if canonical.is_null() {
        None
    } else {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        canonical = pyre_object::gc_roots::pin_root(canonical);
        Some(slot)
    };
    let reload_canonical = || {
        canonical_slot.map_or(
            pyre_object::PY_NULL,
            pyre_object::gc_roots::shadow_stack_get,
        )
    };
    // `make_pre_finalization_calls`: join the non-daemon threads, then run the
    // atexit callbacks, and repeat while a callback or a thread it started
    // left either a non-daemon thread (`interp_has_threads`) or another
    // callback (`interp_has_atexit_callbacks`).
    loop {
        let mut canonical = reload_canonical();
        pyre_object::with_roots!(canonical => run_threading_shutdown());
        let mut canonical = reload_canonical();
        pyre_object::with_roots!(canonical => run_atexit_callbacks(canonical, ec_ptr));
        let mut canonical = reload_canonical();
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
    // alive. Without this a stream owned by a module whose dict is cleared
    // below loses its buffered writes.
    let mut canonical = reload_canonical();
    pyre_object::with_roots!(canonical => crate::module::_io::flush_all_streams());
    // `faulthandler`'s (moduledef.py `shutdown`) cancels the watchdog and
    // restores the signal handlers before any module is torn down.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    crate::module::faulthandler::handler::finish();
    // `finalize_modules` opens with this; the detach and module-dict walk
    // follow immediately, so the special names are gone before any `__del__`
    // from that walk can hide in them.
    finalize_delete_special();
    // The program has returned: drop the main frame chain so `w_globals`
    // does not keep `__main__`'s namespace reachable through the EC after
    // `sys.modules` is emptied. PyPy `executioncontext.py leave` has already
    // popped the frame; this catches a launcher that left `topframeref` set.
    if !ec_ptr.is_null() {
        unsafe {
            let ec = &mut *(ec_ptr as *mut crate::executioncontext::PyExecutionContext);
            ec.topframeref = std::ptr::null_mut();
        }
    }
    let shutdown_modules = crate::importing::release_sys_modules_for_shutdown();
    // Drop the `canonical` pin so the post-detach collection can reach a
    // `__main__` global's `__del__`. `sys.modules` no longer holds the module.
    if let Some(slot) = canonical_slot {
        pyre_object::gc_roots::shadow_stack_set(slot, pyre_object::w_none());
    }
    clear_shutdown_modules(shutdown_modules, ec_ptr);
}
