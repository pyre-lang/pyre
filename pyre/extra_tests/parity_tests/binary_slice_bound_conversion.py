# pyre-check: pypy-diverges: 3.15 `_PyEval_SliceIndex` dropped "or None"; pypy 3.11 still emits it; pypy3 has no list.__sizeof__ (AttributeError)
# CPython-suite gap: bound conversion (int subclass, bigint clamp, rewritten
# TypeError) and lock-free conversion vs the copy stripe are not in the suite.
# parity-tests reason: `_eval_slice_index` is `space.getindex_w`; the list copy
# lock starts after conversion.

"""BINARY_SLICE bound conversion, then the list-copy lock.

`sliceobject.py` `_eval_slice_index` calls `space.getindex_w`. Exact ints,
int subclasses, and values that overflow a machine word must slice the same
way; a bound with no `__index__` raises the rewritten TypeError.

The list copy takes its stripe only after those conversions. An `__index__`
that waits for another thread to mutate this list must return, because that
worker needs the same lock the copy later takes.
"""

import os
import threading
import time

N = 4000
WATCHDOG_S = 12


class MyInt(int):
    pass


class Idx:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def expect_typeerror(fn):
    try:
        fn()
    except TypeError as exc:
        assert str(exc) == (
            "slice indices must be integers or have an __index__ method"
        )
        return
    raise AssertionError("expected TypeError")


def bound_conversion():
    xs = [0, 1, 2, 3, 4]
    assert xs[1:4] == [1, 2, 3]
    huge = 10**100
    assert xs[huge:] == []
    assert xs[-huge:] == [0, 1, 2, 3, 4]
    assert xs[MyInt(1) : MyInt(4)] == [1, 2, 3]
    expect_typeerror(lambda: xs[1.5:2])
    expect_typeerror(lambda: xs[object():2])


def publish_threads():
    done = threading.Event()

    def mark():
        done.set()

    thread = threading.Thread(target=mark)
    thread.start()
    thread.join()


def warmup_list_slice():
    xs = list(range(8))
    total = 0
    for i in range(N):
        total += len(xs[Idx(1) : Idx(4)])
    return total


def slice_index_waits_for_append():
    started = threading.Event()
    proceed = threading.Event()
    lst = list(range(5))

    class BlockingIdx:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            started.set()
            assert proceed.wait(timeout=5), "mutator did not run"
            return self.value

    def mutator():
        assert started.wait(timeout=5), "slice bound did not start"
        lst.append(99)
        proceed.set()

    worker = threading.Thread(target=mutator, daemon=True)
    worker.start()
    result = lst[BlockingIdx(1) : 4]
    worker.join(timeout=5)
    assert not worker.is_alive(), "mutator still blocked on the list lock"
    assert result == [1, 2, 3]


def watchdog():
    time.sleep(WATCHDOG_S)
    os._exit(2)


bound_conversion()
threading.Thread(target=watchdog, daemon=True).start()
publish_threads()
assert warmup_list_slice() == 3 * N
slice_index_waits_for_append()

# AbstractUnwrappedStrategy.deleteslice: compact then one tail shrink.
l = list(range(20))
del l[::3]
print(l)
assert l == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19]
l = list(range(20))
del l[15:2:-2]
print(l)
assert l == [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19]
l = list(range(20))
del l[::-1]
print(l)
assert l == []


_B = [].__sizeof__()


def _slots(lst):
    return (lst.__sizeof__() - _B) // 8


class EmptyHint:
    def __init__(self, hint):
        self.hint = hint

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __length_hint__(self):
        return self.hint


# list_extend_iter_lock_held cuts an unused reservation back with list_resize:
# an allocation of LIST_SMALL_ALLOCATED (32) or fewer is kept, a larger one is freed.
small = []
small.extend(EmptyHint(32))
small.append(1)
big = []
big.extend(EmptyHint(100))
big.append(1)
print("empty-hint", _slots(small), _slots(big))

l = [1]
l.pop()
print("pop1", _slots(l))
l = [1]
l.remove(1)
print("remove1", _slots(l))
l = [1]
l[:] = []
print("setslice-empty", _slots(l))
l = [1]
del l[0]
print("delitem1", _slots(l))
l = [1, 2]
del l[::-1]
print("del-reversed", _slots(l))
l = [1, 2]
del l[::1]
print("del-step1", _slots(l))
l = []
l.extend(EmptyHint(32))
l.clear()
print("eh32-clear", _slots(l))
l = []
l.extend(EmptyHint(32))
l.sort()
print("eh32-sort", _slots(l))
l = []
l.extend(EmptyHint(32))
l.extend([1.5])
print("eh32-ext-float", _slots(l))

dest = []
before = dest.__sizeof__()
grew = []


class Hinted:
    def __iter__(self):
        def gen():
            grew.append(dest.__sizeof__() > before)
            yield 1

        return gen()

    def __length_hint__(self):
        return 32


dest.extend(Hinted())
print(grew[0])
assert grew[0]
print("OK")
