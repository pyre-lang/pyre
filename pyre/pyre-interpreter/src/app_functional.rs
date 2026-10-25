//! `sorted` / `sum` builtins — PyPy: pypy/module/__builtin__/app_functional.py
//!
//! `pypy/module/__builtin__/moduledef.py` installs both as appleveldefs
//! (`'sorted': 'app_functional.sorted'`, `'sum': 'app_functional.sum'`).
//! The Python bodies are what the JIT traces: `sorted` as `list(iterable)`
//! then `list.sort` (`CALL_ASSEMBLER` into `list__do_extend_from_iterable`);
//! `sum` as `_regular_sum`'s `last = last + x` (ints stay unboxed).  The
//! interp-level `builtin_sorted` body cannot be descended: its generated
//! wrapper reaches hundreds of un-lowered helpers after an effect
//! (`PYRE_FBW_INLINE_DIAG` `un-lowered helper call in body`).
//!
//! `mixedmodule.py MixedModule._load_lazily` then wraps that Function as
//! `BuiltinFunction` so the published name has no `__get__`.  `type(sorted)`
//! / `type(sum)` is `builtin_function_or_method` here (PyPy: `builtin_function`);
//! both are the mixed-module builtin type, not `function`.  The wrapper still
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
//! `HANDLES` is the off-GC stand-in for those module-dict slots until then
//! (`async_operation` `HANDLES`); MixedModule's cache is the dict itself.

use pyre_object::PyObjectRef;

use crate::PyResult;

const APP_FUNCTIONAL_SRC: &str = include_str!("module/__builtin__/app_functional.py");

const SORTED: usize = 0;
const SUM: usize = 1;

const NAMES: [&str; 2] = ["sorted", "sum"];
const DOCS: [&str; 2] = [
    "Return a new list containing all items from the iterable in ascending order.\n\nA custom key function can be supplied to customize the sort order, and the\nreverse flag can be set to request the result in descending order.",
    "Return the sum of a 'start' value (default: 0) plus an iterable of numbers\n\nWhen the iterable is empty, return the start value.\nThis function is intended specifically for use with numeric values and\nmay reject non-numeric types.",
];
const SIGS: [&str; 2] = [
    "($module, iterable, /, *, key=None, reverse=False)",
    "($module, iterable, /, start=0)",
];

static HANDLES: parking_lot::Mutex<[usize; 2]> = parking_lot::Mutex::new([0; 2]);

/// `mixedmodule.py MixedModule._load_lazily`: an applevel Function published
/// in a mixed-module dict becomes `BuiltinFunction`.  `function.py
/// BuiltinFunction.__init__` allocates a new object and copies `code`,
/// `w_func_globals`, `defs_w`, `closure`, `name`, `w_doc`, `w_func_dict`,
/// `w_module`, `w_kw_defs`, `w_text_signature`.  `MixedModule._cleanup_`
/// forces every lazy loader at translation time, so the wrapper is a
/// prebuilt constant and `rgc.can_move` is false.  Allocate the sibling
/// with `lltype.malloc_typed_stable` (old-gen, non-moving) so LOAD_GLOBAL
/// of `sorted` / `sum` folds to ConstPtr the same way.
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

fn app_handle(which: usize) -> PyResult {
    let cached = HANDLES.lock()[which];
    if cached != 0 {
        return Ok(cached as PyObjectRef);
    }
    Ok(install_handles()?[which] as PyObjectRef)
}

fn install_handles() -> Result<[usize; 2], crate::PyError> {
    let cached = *HANDLES.lock();
    if cached[SORTED] != 0 {
        return Ok(cached);
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
        &NAMES,
    )?;
    let get = |name: &str| {
        let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
        unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) }
            .unwrap_or_else(|| panic!("app_functional: `{name}` not bound"))
    };
    let mut initialized = [0usize; 2];
    for i in 0..2 {
        let raw = get(NAMES[i]);
        let raw_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(raw);
        // Stamp the doc on the still-mutable Function; wrap copies `w_doc`.
        // `function.py` keeps PyPy's one-liner on the PyCode; the GetSet
        // reads `w_doc` first.
        let doc = pyre_object::w_str_new(DOCS[i]);
        let raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
        unsafe {
            crate::function::function_set_doc(raw, doc).expect("app-level Function is mutable");
        }
        let raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
        let func = wrap_as_builtin_function(raw);
        initialized[i] = func as usize;
        let _ = pyre_object::gc_roots::pin_root(func);
    }
    let mut handles = HANDLES.lock();
    if handles[SORTED] == 0 {
        *handles = initialized;
    }
    Ok(*handles)
}

/// Publish the MixedModule `BuiltinFunction` into the live builtins dict so
/// subsequent `LOAD_GLOBAL` sees it.  Called from the first EC registration,
/// after `getexecutioncontext` is live.
pub(crate) fn install_applevel_builtins() {
    let Ok(handles) = install_handles() else {
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
    // share.  MixedModule._load_lazily replaces only the lazy placeholder;
    // a published function, a user replacement, or an absent name stay.
    // No remaining placeholder needs neither delete nor store.
    let sorted_placeholder = crate::module_ns_get(builtins, "sorted")
        .is_some_and(|binding| is_placeholder(binding, "sorted"));
    let sum_placeholder =
        crate::module_ns_get(builtins, "sum").is_some_and(|binding| is_placeholder(binding, "sum"));
    if !sorted_placeholder && !sum_placeholder {
        return;
    }
    // `typeobject.py write_cell`: overwriting the trampoline wraps the
    // slot in `ObjectMutableCell`.  Delete first so the insert is
    // `StoreBare` — MixedModule._load_lazily writes the applevel
    // BuiltinFunction once, and LOAD_GLOBAL then folds to ConstPtr.
    let _roots = pyre_object::gc_roots::push_roots();
    let sorted_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(handles[SORTED] as PyObjectRef);
    let sum_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(handles[SUM] as PyObjectRef);
    if sorted_placeholder {
        publish_one(builtins, "sorted", sorted_slot, SIGS[SORTED], ctx);
    }
    if sum_placeholder {
        publish_one(builtins, "sum", sum_slot, SIGS[SUM], ctx);
    }
}

/// True while `binding` is still the trampoline `builtins.rs` installed
/// for `name`.  That object's `BuiltinCode.func` is
/// [`__majit_wrap_builtin_sorted`] / [`__majit_wrap_builtin_sum`].
/// `mixedmodule.py MixedModule._load_lazily` replaces only that placeholder.
fn is_placeholder(binding: PyObjectRef, name: &str) -> bool {
    let expected = match name {
        "sorted" => __majit_wrap_builtin_sorted as crate::gateway::BuiltinCodeFn,
        "sum" => __majit_wrap_builtin_sum as crate::gateway::BuiltinCodeFn,
        _ => return false,
    };
    unsafe {
        if binding.is_null() || !crate::is_function(binding) {
            return false;
        }
        let code = crate::function_get_code(binding) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        crate::gateway::builtin_code_fn_eq(crate::gateway::builtin_code_get(code), expected)
    }
}

fn publish_one(
    builtins: PyObjectRef,
    name: &str,
    func_slot: usize,
    sig_text: &str,
    ctx: *const crate::PyExecutionContext,
) {
    crate::module_ns_delete(builtins, name);
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    crate::module_ns_store(builtins, name, func);
    // `func` is the old-gen BuiltinFunction; `w_str_new` may allocate in
    // the nursery.  Reload `func` after that alloc (it cannot move) and
    // let `fset_func_text_signature` run the old-to-young write barrier.
    //
    // `function.py BuiltinFunction.descr_builtinfunction__self__` is
    // `always_none` unless `w_moduleobj` is set.  The clinic `$module`
    // text signature plus the builtins module as `__self__` is what
    // inspect.signature strips to the public signature.
    let sig = pyre_object::w_str_new(sig_text);
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
    let mut handles = HANDLES.lock();
    for slot in handles.iter_mut().filter(|slot| **slot != 0) {
        unsafe { visitor(&mut *(slot as *mut usize as *mut majit_ir::GcRef)) };
    }
}

fn call_published(which: usize, args: &[PyObjectRef]) -> PyResult {
    // `args` is the gateway's native copy; publishing allocates.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    install_applevel_builtins();
    let w_func = app_handle(which)?;
    let mut live_args = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        live_args.push(_roots.get(base + i));
        i += 1;
    }
    crate::call::call_function_impl_result(w_func, &live_args)
}

fn call_published_args(which: usize, args: &crate::argument::Arguments) -> PyResult {
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
    let w_func = app_handle(which)?;
    let func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_func);
    let pos_now: Vec<PyObjectRef> = (0..npos).map(|i| roots.get(pos_base + i)).collect();
    let names_now: Vec<PyObjectRef> = (0..nnames).map(|i| roots.get(names_base + i)).collect();
    let values_now: Vec<PyObjectRef> = (0..nnames).map(|i| roots.get(values_base + i)).collect();
    let forwarded = crate::argument::Arguments::with_kw(&pos_now, &names_now, &values_now);
    crate::call::call_args(
        pyre_object::gc_roots::shadow_stack_get(func_slot),
        &forwarded,
    )
}

/// Trampoline occupying `builtins.sorted` until
/// [`install_applevel_builtins`] replaces it.  First call also publishes
/// the app-level function, then delegates.
///
/// Named `__majit_wrap_*` because it is a `BuiltinCode.func`: the descriptor
/// below puts it in that PBC family, which admits `__majit_wrap_` leaves only.
pub fn __majit_wrap_builtin_sorted(args: &[PyObjectRef]) -> PyResult {
    call_published(SORTED, args)
}

/// Keyword path: names stay on `Arguments` (`func__args__`).  Applevel
/// `sorted` reports unknown keywords as `sort()`, so this forwards the
/// same `Arguments` object rather than binding a Signature on the trampoline.
pub fn builtin_sorted_args(args: &crate::argument::Arguments) -> PyResult {
    call_published_args(SORTED, args)
}

/// Trampoline occupying `builtins.sum` until
/// [`install_applevel_builtins`] replaces it.
pub fn __majit_wrap_builtin_sum(args: &[PyObjectRef]) -> PyResult {
    call_published(SUM, args)
}

/// Keyword path for the `sum` trampoline.  Forwards the same `Arguments`
/// object so `start=` binds on the app-level `sum(iterable, /, start=0)`.
pub fn builtin_sum_args(args: &crate::argument::Arguments) -> PyResult {
    call_published_args(SUM, args)
}

/// Identity of the published app-level `sorted` BuiltinFunction.  Compares
/// addresses only, so a stale word never matches: the published object is
/// allocated old and never moves.
pub(crate) fn is_published_sorted(func: PyObjectRef) -> bool {
    is_published(SORTED, func)
}

/// Identity of the published app-level `sum` BuiltinFunction.
pub(crate) fn is_published_sum(func: PyObjectRef) -> bool {
    is_published(SUM, func)
}

fn is_published(which: usize, func: PyObjectRef) -> bool {
    let cached = HANDLES.lock()[which];
    cached != 0 && func as usize == cached
}

crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_app_sorted,
    __majit_wrap_builtin_sorted
);
crate::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_app_sum,
    __majit_wrap_builtin_sum
);
