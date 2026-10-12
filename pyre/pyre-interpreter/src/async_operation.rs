//! `aiter` / `anext` builtins — PyPy: pypy/module/__builtin__/app_operation.py
//!
//! The two names are pure app-level functions in PyPy's MixedModule.  The
//! builtins module dict is populated inside `ExecutionContext::new`, before
//! the execution context is registered, so `appleveldef_install` cannot run
//! at builtins-setup time (there is no execution context yet).  Instead the
//! two names are registered as interp-level builtin functions that resolve
//! the app-level implementations lazily on first call, when an execution
//! context exists — the same lazy-resolution shape `reduce_protocol` uses.

use pyre_object::PyObjectRef;

use crate::PyResult;

/// `aiter` / `anext` ported from app_operation.py.  The
/// `_NOT_PROVIDED` sentinel stays in the intermediate namespace that both
/// functions retain as their `__globals__`, so only the two public names
/// need to be surfaced.  `anext` calls `__anext__` before constructing its
/// default wrapper, matching CPython 3.14's `builtin_anext_impl` synchronous
/// exception behavior.
const ASYNC_OP_SRC: &str = r#"
_NOT_PROVIDED = object()

def aiter(obj):
    """aiter(async_iterable) -> async_iterator
    aiter(async_callable, sentinel) -> async_iterator
    Like the iter() builtin but for async iterables and callables.
    """
    typ = type(obj)
    try:
        meth = typ.__aiter__
    except AttributeError:
        raise TypeError(f"'{type(obj).__name__}' object is not an async iterable")
    ait = meth(obj)
    if not hasattr(ait, '__anext__'):
        raise TypeError(f"aiter() returned not an async iterator of type '{type(ait).__name__}'")
    return ait

def anext(iterator, default=_NOT_PROVIDED):
    """anext(async_iterator[, default])
    Return the next item from the async iterator.
    If default is given and the iterator is exhausted,
    it is returned instead of raising StopAsyncIteration.
    """
    typ = type(iterator)

    try:
        __anext__ = typ.__anext__
    except AttributeError:
        raise TypeError(f"'{type(iterator).__name__}' object is not an async iterator")

    awaitable = __anext__(iterator)

    if default is _NOT_PROVIDED:
        return awaitable

    async def anext_impl():
        try:
            return await awaitable
        except StopAsyncIteration:
            return default

    return anext_impl()
"#;

const AITER: usize = 0;
const ANEXT: usize = 1;

static HANDLES: parking_lot::Mutex<[usize; 2]> = parking_lot::Mutex::new([0; 2]);

/// Resolve (and cache) the two app-level handles.  Executes `ASYNC_OP_SRC`
/// into its own fresh module globals and stores the `aiter` / `anext`
/// function objects; both retain that namespace as their `__globals__`,
/// keeping the `_NOT_PROVIDED` sentinel reachable.
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
        panic!("async_operation: no execution context");
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
    let _ = pyre_object::gc_roots::pin_root(w_app_globals);
    // The source imports nothing, but running it is still running Python: it
    // builds two functions in a fresh frame, so it answers a recursion limit
    // the program set and a `builtins` the program emptied. That belongs to
    // the `aiter`/`anext` call that forced the install.
    crate::importing::appleveldef_install(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        ASYNC_OP_SRC,
        "app_operation.py",
        // `pypy/module/__builtin__/moduledef.py Module.applevel_name`, which
        // `get_applevel_name` hands to every appleveldef of that module.
        "builtins",
        &["aiter", "anext"],
    )?;
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
    let get = |name: &str| {
        unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) }
            .unwrap_or_else(|| panic!("async_operation: `{name}` not bound"))
    };
    let initialized = [get("aiter") as usize, get("anext") as usize];
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

/// `aiter(obj)` — delegates to the app-level `aiter` after stripping the
/// trailing `__pyre_kw__` marker and refusing keywords.  Clinic
/// `builtin_aiter` takes no keywords and exactly one positional.
pub fn builtin_aiter(args: &[PyObjectRef]) -> PyResult {
    let positional = args;
    crate::gateway::check_declared_arity("aiter", 1, positional.len())?;
    // `positional` is the gateway's native copy; `handle` may allocate.
    let mut w_obj = positional[0];
    let w_aiter = pyre_object::with_roots!(w_obj => handle(AITER))?;
    crate::call::call_function_impl_result(w_aiter, &[w_obj])
}

/// `anext(iterator[, default])` — delegates to the app-level `anext` after
/// stripping the trailing `__pyre_kw__` marker and refusing keywords.
/// `_PyArg_CheckPositional` on `builtin_anext` is at least 1, at most 2.
pub fn builtin_anext(args: &[PyObjectRef]) -> PyResult {
    // Bound scope from `parse_obj`: `aiterator`, `default` (`PY_NULL` omitted).
    if args.is_empty() || args[0].is_null() {
        return Err(crate::PyError::type_error(
            "anext expected at least 1 argument, got 0",
        ));
    }
    // `args` is the gateway's native copy; `handle` may allocate.
    let mut w_iterator = args[0];
    let mut w_default = args
        .get(1)
        .copied()
        .filter(|w| !w.is_null())
        .unwrap_or(pyre_object::PY_NULL);
    let w_anext = pyre_object::with_roots!(w_iterator, w_default => handle(ANEXT))?;
    let live = [w_iterator, w_default];
    let n = if w_default.is_null() { 1 } else { 2 };
    crate::call::call_function_impl_result(w_anext, &live[..n])
}
