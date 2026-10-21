//! `sorted` builtin — PyPy: pypy/module/__builtin__/app_functional.py
//!
//! `pypy/module/__builtin__/moduledef.py` installs `sorted` as an
//! appleveldef (`'sorted': 'app_functional.sorted'`).  The Python body
//! is `list(iterable)` then `list.sort`, which the JIT traces as
//! bytecode and `CALL_ASSEMBLER`s into `list__do_extend_from_iterable`.
//! The interp-level `builtin_sorted` body cannot be descended: its
//! generated wrapper reaches hundreds of un-lowered helpers after an
//! effect (`PYRE_FBW_INLINE_DIAG` `un-lowered helper call in body`).
//!
//! `mixedmodule.py MixedModule._load_lazily` then wraps that Function as
//! `BuiltinFunction` so the published name has no `__get__`.  `type(sorted)`
//! is `builtin_function_or_method` here (PyPy: `builtin_function`); both
//! are the mixed-module builtin type, not `function`.  The wrapper still
//! carries the app-level PyCode (`sorted.__code__` exists), which
//! `resolve_inlinable_callee` inlines.
//!
//! The builtins dict is filled inside `ExecutionContext::new` before the
//! context is registered, so `appleveldef_install` cannot run then
//! (same constraint `async_operation` documents for `aiter`/`anext`).
//! A trampoline occupies the name until the first EC registration, which
//! publishes the BuiltinFunction MixedModule would have stored.  That
//! publish deletes the trampoline slot first: a second `write_cell` over
//! the trampoline wraps it in `ObjectMutableCell`, and LOAD_GLOBAL then
//! keeps a live getfield so the walker refuses to inline the app-level
//! body (`callable_guard_op.is_constant()`).  MixedModule._load_lazily
//! writes the BuiltinFunction once (`write_cell` `StoreBare`).
//! `HANDLE` is the off-GC stand-in for that module-dict slot until then
//! (`async_operation` `HANDLES`); MixedModule's cache is the dict itself.

use pyre_object::PyObjectRef;

use crate::PyResult;

const APP_FUNCTIONAL_SRC: &str = include_str!("module/__builtin__/app_functional.py");

static HANDLE: parking_lot::Mutex<usize> = parking_lot::Mutex::new(0);

/// `mixedmodule.py MixedModule._load_lazily`: an applevel Function published
/// in a mixed-module dict becomes `BuiltinFunction`.  `function.py
/// BuiltinFunction.__init__` allocates a new object and copies `code`,
/// `w_func_globals`, `defs_w`, `closure`, `name`, `w_doc`, `w_func_dict`,
/// `w_module`, `w_kw_defs`, `w_text_signature`.  `MixedModule._cleanup_`
/// forces every lazy loader at translation time, so the wrapper is a
/// prebuilt constant and `rgc.can_move` is false.  Allocate the sibling
/// with `lltype.malloc_typed_stable` (old-gen, non-moving) so LOAD_GLOBAL
/// of `sorted` folds to ConstPtr the same way.
fn wrap_as_builtin_function(func: PyObjectRef) -> PyObjectRef {
    unsafe {
        if !pyre_object::py_type_check(func, &crate::FUNCTION_TYPE) {
            return func;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let src_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        let src = pyre_object::gc_roots::shadow_stack_get(src_slot) as *const crate::Function;
        let mut payload = std::ptr::read(src);
        payload.ob.ob_type = &crate::BUILTIN_FUNCTION_TYPE as *const _;
        payload.ob.w_class = pyre_object::pyobject::get_instantiate(&crate::BUILTIN_FUNCTION_TYPE);
        payload.can_change_code = false;
        // `ptr::read` copied the source's watchers; overwrite without dropping
        // them so the source keeps its own `mutate_<name>` instances.
        for slot in crate::QuasiImmutSlot::ALL {
            std::ptr::write(
                payload.mutate_field_mut(slot),
                pyre_object::quasiimmut::QuasiImmutField::new(),
            );
        }
        // BuiltinFunction.typedef has no instance dict (`hasdict=false`);
        // a copied Function dict would leak `__code__` / `__defaults__`
        // / `__globals__` as hasattr-true instance attributes.
        payload.w_func_dict = pyre_object::PY_NULL;
        pyre_object::lltype::malloc_typed_stable(payload) as PyObjectRef
    }
}

fn app_sorted() -> PyResult {
    let cached = *HANDLE.lock();
    if cached != 0 {
        return Ok(cached as PyObjectRef);
    }

    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("app_functional: no execution context");
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
    let _ = pyre_object::gc_roots::pin_root(w_app_globals);
    crate::importing::appleveldef_install(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        APP_FUNCTIONAL_SRC,
        "app_functional.py",
        "builtins",
        &["sorted"],
    )?;
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
    let raw = unsafe { pyre_object::w_dict_getitem_str(w_app_globals, "sorted") }
        .unwrap_or_else(|| panic!("app_functional: `sorted` not bound"));
    let raw_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(raw);
    // Stamp the doc on the still-mutable Function; wrap copies `w_doc`.
    // `function.py` keeps PyPy's `sorted(...) --> new sorted list` on the
    // PyCode; the GetSet reads `w_doc` first.
    let doc = pyre_object::w_str_new(
        "Return a new list containing all items from the iterable in ascending order.\n\nA custom key function can be supplied to customize the sort order, and the\nreverse flag can be set to request the result in descending order.",
    );
    let raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
    unsafe {
        crate::function::function_set_doc(raw, doc).expect("app-level Function is mutable");
    }
    let raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
    let func = wrap_as_builtin_function(raw);
    let mut handle = HANDLE.lock();
    if *handle == 0 {
        *handle = func as usize;
    }
    Ok(*handle as PyObjectRef)
}

/// Publish the MixedModule `BuiltinFunction` into the live builtins dict so
/// subsequent `LOAD_GLOBAL` sees it.  Called from the first EC registration,
/// after `getexecutioncontext` is live.
pub(crate) fn install_applevel_builtins() {
    let Ok(func) = app_sorted() else {
        return;
    };
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        return;
    }
    let builtins = unsafe { (*ctx).get_builtin_dict() };
    if builtins.is_null() {
        return;
    }
    // A worker thread's first EC registration reaches this again, and
    // `clone_for_thread` hands it the builtins dict the running threads
    // share.  The delete below would leave `sorted` unbound until the store,
    // so a `LOAD_GLOBAL sorted` crossing that window raises `NameError`.
    // A binding that is already the published function needs neither.
    if crate::module_ns_get(builtins, "sorted").is_some_and(is_published_sorted) {
        return;
    }
    // `typeobject.py write_cell`: overwriting the trampoline wraps the
    // slot in `ObjectMutableCell`.  Delete first so the insert is
    // `StoreBare` — MixedModule._load_lazily writes the applevel
    // BuiltinFunction once, and LOAD_GLOBAL then folds to ConstPtr.
    let _roots = pyre_object::gc_roots::push_roots();
    let func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(func);
    crate::module_ns_delete(builtins, "sorted");
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    crate::module_ns_store(builtins, "sorted", func);
    // `func` is the old-gen BuiltinFunction; `w_str_new` may allocate in
    // the nursery.  Reload `func` after that alloc (it cannot move) and
    // let `fset_func_text_signature` run the old-to-young write barrier.
    //
    // `function.py BuiltinFunction.descr_builtinfunction__self__` is
    // `always_none` unless `w_moduleobj` is set.  The clinic
    // `$module` text signature plus the builtins module as `__self__`
    // is what inspect.signature strips to `(iterable, /, *, key=None,
    // reverse=False)`.
    let sig = pyre_object::w_str_new("($module, iterable, /, *, key=None, reverse=False)");
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    unsafe {
        crate::function::fset_func_text_signature(func, sig);
    }
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    let w_builtin = unsafe { (*ctx).get_builtin() };
    unsafe {
        crate::function::builtin_function_set_module_obj(func, w_builtin);
    }
}

pub(crate) fn walk_handle_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut handle = HANDLE.lock();
    if *handle != 0 {
        unsafe { visitor(&mut *(&mut *handle as *mut usize as *mut majit_ir::GcRef)) };
    }
}

/// Trampoline occupying `builtins.sorted` until
/// [`install_applevel_builtins`] replaces it.  First call also publishes
/// the app-level function, then delegates.
///
/// Named `__majit_wrap_*` because it is a `BuiltinCode.func`: the descriptor
/// below puts it in that PBC family, which admits `__majit_wrap_` leaves only.
pub fn __majit_wrap_builtin_sorted(args: &[PyObjectRef]) -> PyResult {
    // `args` is the gateway's native copy; publishing `sorted` allocates.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    install_applevel_builtins();
    let w_sorted = app_sorted()?;
    let mut live_args = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        live_args.push(_roots.get(base + i));
        i += 1;
    }
    crate::call::call_function_impl_result(w_sorted, &live_args)
}

/// Keyword path: names stay on `Arguments` (`func__args__`).  Applevel
/// `sorted` reports unknown keywords as `sort()`, so this forwards the
/// same `Arguments` object rather than binding a Signature on the trampoline.
pub fn builtin_sorted_args(args: &crate::argument::Arguments) -> PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let pos = &args.arguments_w;
    let names = args.keyword_names_w.as_deref().unwrap_or(&[]);
    let values = args.keywords_w.as_deref().unwrap_or(&[]);
    let pos_base = roots.pin_roots(pos);
    let names_base = roots.pin_roots(names);
    let values_base = roots.pin_roots(values);
    let npos = pos.len();
    let nnames = names.len();
    install_applevel_builtins();
    let w_sorted = app_sorted()?;
    let sorted_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_sorted);
    let pos_now: Vec<PyObjectRef> = (0..npos).map(|i| roots.get(pos_base + i)).collect();
    let names_now: Vec<PyObjectRef> = (0..nnames).map(|i| roots.get(names_base + i)).collect();
    let values_now: Vec<PyObjectRef> = (0..nnames).map(|i| roots.get(values_base + i)).collect();
    let forwarded = crate::argument::Arguments::with_kw(&pos_now, &names_now, &values_now);
    crate::call::call_args(
        pyre_object::gc_roots::shadow_stack_get(sorted_slot),
        &forwarded,
    )
}

/// Identity of the published app-level `sorted` BuiltinFunction.  Compares
/// addresses only, so a stale word never matches: the published object is
/// allocated old and never moves.
pub(crate) fn is_published_sorted(func: PyObjectRef) -> bool {
    let cached = *HANDLE.lock();
    cached != 0 && func as usize == cached
}

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_app_sorted,
    __majit_wrap_builtin_sorted
);
