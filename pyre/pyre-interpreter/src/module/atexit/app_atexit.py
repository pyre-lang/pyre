# CPython 3.14 permits distinct atexit module objects while keeping one
# interpreter-owned callback stack.  PyPy's app-level list is the same storage
# shape; anchor it on pyre's interpreter-owned sys module so re-imported module
# instances share it.  `_sys` is seeded by the module initializer rather than
# imported here: the name can be blocked in `sys.modules`, and this body runs
# during finalization for a program that never imported `atexit`.
try:
    atexit_callbacks = _sys._pyre_atexit_callbacks
except AttributeError:
    atexit_callbacks = []
    _sys._pyre_atexit_callbacks = atexit_callbacks


def register(func, *args, **kwargs):
    """Register a function to be executed upon normal program termination.

    func - function to be called at exit
    args - optional arguments to pass to func
    kwargs - optional keyword arguments to pass to func

    func is returned to facilitate usage as a decorator."""

    if not callable(func):
        # CPython 3.14's public diagnostic takes precedence over PyPy 3.11's
        # older ``func must be callable`` spelling.
        raise TypeError("the first argument must be callable")

    atexit_callbacks.append((func, args, kwargs))
    return func


def _run_exitfuncs():
    """Run all registered exit functions."""
    for (func, args, kwargs) in reversed(atexit_callbacks):
        if func is None:
            continue
        try:
            func(*args, **kwargs)
        except BaseException as e:
            import __pypy__
            # CPython 3.14 reports atexit failures without attaching the
            # callback object, but includes its repr in err_msg.
            __pypy__.write_unraisable(
                f"in atexit callback {func!r}", e, None)

    _clear()


def _clear():
    """Clear the list of previously registered exit functions."""
    del atexit_callbacks[:]


def unregister(func):
    """Unregister an exit function previously registered with atexit."""
    for i, (f, _, _) in enumerate(atexit_callbacks):
        if f == func:
            atexit_callbacks[i] = (None, None, None)


def _ncallbacks():
    return len(atexit_callbacks)
