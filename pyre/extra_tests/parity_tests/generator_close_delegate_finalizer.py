# CPython-suite gap: test_yield_from closes delegating generators, but never
# with a local whose finalizer is the observable, so nothing there grades when
# the delegate's `__del__` runs relative to the outer `close()`.
# parity-tests reason: close() of a delegating generator releases the
# delegate's locals and a private-globals function, so a collection
# finalizes them.

"""close() of a delegating generator releases the delegate's locals.

The object's only reference is a local of the inner frame. After close() and
three gc.collect() calls (`support.gc_collect`), that finalizer has run.
A function private to the generator owns its globals; close() releases that
function too, so a collection finalizes the object held in the dict.
"""

import gc
import types


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
gc.collect()
gc.collect()
gc.collect()
assert Probe.ran is True, "the delegate's finalizer did not run after close() and collect"

# A function private to the generator owns its globals.  close() releases the
# frame's locals, the function and that dict together, so a collection
# finalizes the object held in the dict.

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
gc.collect()
gc.collect()
gc.collect()
assert finalized == ["del"], finalized

print("OK")
