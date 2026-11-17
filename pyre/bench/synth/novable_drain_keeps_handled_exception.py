# A compiled novable drain (list.extend, filter, unpack_into) finishes on
# StopIteration as an interpreter-level OperationError. pyjitpl.py
# finishframe / clear_exception null last_exc_value; the caller's handled
# exception is untouched, as in _do_extend_from_iterable /
# _create_unpack_into.unpack_into / functional.py filter. A portal that
# wrote PY_NULL into sys_exc_info cleared sys.exception() inside this
# except. Expected: handled ValueError, and the three drains compile.
#
# Call shapes: list.extend(range) is _do_extend_from_iterable (jd3);
# filter(None, ...) with leading falsy items is filter_jitdriver (jd5);
# list.extend(generator) is unpack_into_w (jd6). Each drain is long
# enough for that portal to compile a loop on the first visits.
import sys

N = 8
EXTEND_N = 2000
FILTER_SKIP = 2000


def drain_gen(n):
    for i in range(n):
        yield i


def run():
    total = 0
    try:
        raise ValueError("outer")
    except ValueError:
        for _ in range(N):
            l = []
            l.extend(range(EXTEND_N))
            total += len(l)
            total += len(list(filter(None, [0] * FILTER_SKIP + [1, 2, 3, 4, 5, 6])))
            g = []
            g.extend(drain_gen(EXTEND_N))
            total += len(g)
            if not isinstance(sys.exc_info()[1], ValueError):
                print("LOST", sys.exc_info())
                return
        print("handled", type(sys.exc_info()[1]).__name__, total)


run()
