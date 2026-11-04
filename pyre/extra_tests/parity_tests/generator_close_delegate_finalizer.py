# pyre-check: pypy-diverges: prompt finalization at close() is a refcount
# boundary; pypy3's tracing GC reaches the delegate's local later
# CPython-suite gap: test_yield_from closes delegating generators, but never
# with a local whose finalizer is the observable, so nothing there grades when
# the delegate's `__del__` runs relative to the outer `close()` returning.
# parity-tests reason: one `close()` finishes every frame in the `yield from`
# chain, and pyre carries a single pending-finalizer census from the frame
# teardown out to `descr_close`. The delegate is finished first, so a census
# that is assigned rather than accumulated loses its answer to the delegating
# frame's own.

"""`close()` on a delegating generator runs the delegate's finalizer.

The object's only reference is a local of the *inner* frame, so it becomes
garbage when `close_yield_from` finishes that frame -- before the outer frame
is finished. CPython's refcount boundary runs `__del__` there, and `close()`
does not return until it has. The file also covers a private-globals function
released by `close()`.
"""


class Probe:
    ran = False

    def __del__(self):
        type(self).ran = True


def inner():
    probe = Probe()
    assert probe is not None
    yield 1


def outer():
    yield from inner()


gen = outer()
assert next(gen) == 1
assert Probe.ran is False
gen.close()
assert Probe.ran is True, "the delegate's finalizer did not run before close() returned"

# A function private to the generator owns its globals.  close() releases the
# frame's locals, the function and that dict together, so the finalizer held in
# the dict runs at close().
import types

finalized = []


class LoggedOnDel:
    def __del__(self):
        finalized.append("del")


def holds_private_globals():
    fn = types.FunctionType(compile("victim", "<private>", "eval"), {"victim": LoggedOnDel()})
    assert fn() is not None
    yield 1


gen = holds_private_globals()
assert next(gen) == 1
gen.close()
assert finalized == ["del"], finalized

print("OK")
