# pyre-check: gate=1
"""A SimpleQueue subclass stamped on an old instance survives a minor collection.

`W_SimpleQueue.__new__` allocates the queue old (`allocate_stable`) and then
stores the caller's class in `w_class`. Heap types are born young, so that
store is an old→young edge and `incminimark.py write_barrier` must record it.
Without the barrier a minor collection reclaims the class while the queue is
live, and `type(q)` reads a swept type.
"""
import gc
import _queue


class C(_queue.SimpleQueue):
    pass


q = C()
del C
gc.collect(0)
assert type(q).__name__ == "C", type(q)
assert isinstance(q, _queue.SimpleQueue)
print("OK")
