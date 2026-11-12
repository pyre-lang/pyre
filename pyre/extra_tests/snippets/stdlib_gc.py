# pyre-check: gate=1
import gc

from testutils import assert_raises


class Index:
    def __index__(self) -> int:
        return 0


# The generation argument is bound and integer-unwrapped, and every value is
# accepted -- `interp_gc.py collect` ignores it.  Which generation an
# integer selects is an engineering choice and follows upstream, so no value is
# rejected and `bench/synth/gc_pypy_frontend.py` pins that against the pypy
# oracle.  The return value is a spec axis and is an int.
assert isinstance(gc.collect(), int)
assert isinstance(gc.collect(0), int)
assert isinstance(gc.collect(2), int)
assert isinstance(gc.collect(generation=2), int)
assert isinstance(gc.collect(True), int)
assert isinstance(gc.collect(Index()), int)

for generation in (None, 1.25, "0"):
    with assert_raises(TypeError):
        gc.collect(generation)

with assert_raises(TypeError):
    gc.collect(0, 1)

# A cycle of a few hundred objects, dropped, then collected twice. The return
# is the old-generation objects this collection released (`minimarkpage.py`
# `walk_page`, `incminimark.py` `free_unvisited_rawmalloc_objects_step`); a
# minor does not enumerate dead nursery objects, so the cycle is collected
# once while still live and then dropped.
class Node:
    def __init__(self):
        self.next = None


cycle = [Node() for _ in range(400)]
for i, node in enumerate(cycle):
    node.next = cycle[(i + 1) % len(cycle)]
gc.collect()
del cycle, node
first = gc.collect()
second = gc.collect()
assert isinstance(first, int)
assert isinstance(second, int)
assert first + second > 0

assert isinstance(gc.get_objects(), list)
assert isinstance(gc.get_objects(None), list)
assert isinstance(gc.get_objects(generation=None), list)

marker = []
assert any(obj is marker for obj in gc.get_objects())

with assert_raises(TypeError):
    gc.get_objects(None, None)

# A frozen object in a cycle survives later collections until unfreeze; the
# self-reference keeps refcount from freeing it under 3.15t as well.
import weakref
class Frozen: pass
obj = Frozen(); obj.cycle = obj; ref = weakref.ref(obj)
gc.freeze(); assert gc.get_freeze_count() > 0
del obj; gc.collect(); assert ref() is not None
gc.unfreeze(); assert gc.get_freeze_count() == 0
