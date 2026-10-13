# CPython-suite gap: test.test_collections ABC mixins JIT-trace
# ValuesView.__contains__ (nlocals=4, stacksize=3, locals array length 7:
# self/value/key/v live, three unused stack NULLs). The old tid-9 array
# then holds a stale nursery address at items[1:4] while item[0] (self)
# is already old.
# parity-tests reason: pyframe.py STORE_FAST is setarrayitem_gc on the
# virtualizable locals GcArray; framework.py transform_generic_set emits
# write_barrier_from_array before that store; rewrite.py
# handle_write_barrier_setarrayitem / gen_write_barrier_array emit
# COND_CALL_GC_WB on the array. pyframe_object_custom_trace forwards the
# locals pointer and does not re-scan an already-old one, so a frame-only
# barrier is not enough. Fresh object() stays young at the store;
# intern + a 4096-byte nursery promotes the frame first.
# parity-env: PYPY_GC_NURSERY=4096

"""Old cap-7 frame locals that later receive young args survive minors."""

import gc
from collections.abc import ValuesView

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


def promote():
    junk = [[object() for _ in range(16)] for _ in range(24)]
    del junk
    gc.collect()


def f4t(a, b, c, d):
    return a, b, c


class Mapping:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


N = 600
kept = []
old_self = object()
promote()
for i in range(N):
    row = f4t(old_self, object(), object(), object())
    kept.append(row)
    if i % 8 == 0:
        promote()
if kept[-1][0] is not old_self:
    raise AssertionError(("f4t", kept[-1][0], old_self))

data = {}
for i in range(8):
    data[object()] = object()
view = ValuesView(Mapping(data))
promote()
for i in range(N):
    needle = object()
    if needle in view:
        raise AssertionError(("contains", i, needle))
    if i % 8 == 0:
        promote()

print(kept[-1][0] is old_self)
print("OK")
