# CPython-suite gap: an empty list can be traced, then grow into an
# Object-strategy items block that a minor has already promoted, and store
# interned names into that block. The suite does not check those slots stay
# live.
# parity-tests reason: record_list_and_items_cond_call_gc_wb must not skip
# the array half from a tracing-time box_value of 0. rewrite.py
# handle_write_barrier_setarrayitem / gen_write_barrier_array emit
# COND_CALL_GC_WB on the SETARRAYITEM array operand from what the trace
# proves (ConstPtr(NULL) via rewrite.rs is_null_constant), not from one
# observed Getfield. Tuples force Object strategy so the store is
# list.obj_setitem → setarrayitem_gc_r (ascii interned strs would pick
# AsciiListStrategy and never touch the Object items block).
# parity-env: PYPY_GC_NURSERY=4096

"""A list traced while empty keeps later young interned strs across minors."""

import gc
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 4000


def promote():
    junk = [[i] for i in range(400)]
    del junk
    gc.collect()


def run():
    xs = []
    promote()
    for i in range(N):
        xs.append((sys.intern("k%d-%d" % (i, i)),))
        if i % 32 == 0:
            promote()
    for i in range(4):
        last = N - 4 + i
        expected = "k%d-%d" % (last, last)
        s = xs[last][0]
        if s != expected:
            raise AssertionError((last, s, expected))
        if sys.intern(expected) is not s:
            raise AssertionError(("identity", last, s, expected))
    return xs[-1][0]


print(run())
print("OK")
