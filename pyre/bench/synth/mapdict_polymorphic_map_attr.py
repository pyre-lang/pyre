# pyre-check: requires-modules=gc
# pyre-check: max-pypy-ratio=15
# One LOAD_ATTR site alternates between maps whose insertion order assigns
# different storage slots to `q`; promoting the map must guard the selected
# storage coordinate, while a class-layout guard alone cannot (mapdict.py _get_mapdict_map).
import gc


class A:
    def __init__(self):
        self.p = 1
        self.q = 2


class B:
    def __init__(self):
        self.q = 20
        self.p = 10


def run(n):
    objects = [A(), B()]
    # `must_compile` keys a failing ref GUARD_VALUE on the promoted map by its
    # address (compile.py `lltype.cast_ptr_to_int`). Promote the objects and
    # their maps before the loop warms up, so a minor collection does not move
    # them mid-warm-up and restart their counters: without it, guard_failures
    # reads 409/462/445/401 at a 128K/256K/512K/1M+ nursery, and 512 on the
    # windows runner.
    gc.collect()
    total = 0
    i = 0
    while i < n:
        total += objects[i % 2].q
        i += 1
    return total


print(run(9000000))
