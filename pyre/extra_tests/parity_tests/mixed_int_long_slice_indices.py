# CPython-suite gap: test.test_slice.test_indices mixes machine ints and
# bigints as slice.indices bounds in one product loop.
# parity-tests reason: tupleobject.py `_descr_eq` calls `space.eq_w` per item.
# `W_IntObject` (`intobject.py`) is the machine-int class; `W_LongObject` is
# a sibling. A fold that treats Python `type is int` as `W_IntObject` records
# `GetfieldGcI intval` after `GuardClass(LONG)` and aliases it with
# `GetfieldGcR value` on the same box.

"""slice.indices over mixed int/long bounds in one compiled loop."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30
vals = (None, 2**100, -(2**100), 2**30, -(2**30), 2, -2, 0)
lengths = (0, 50, 2**100)


def indices_ref(start, stop, step, length):
    if step is None:
        step = 1
    if step > 0:
        defstart, defstop = 0, length
        lo, hi = 0, length
    else:
        defstart, defstop = length - 1, -1
        lo, hi = -1, length - 1

    def conv(value, default):
        if value is None:
            return default
        if value < 0:
            value += length
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    return (conv(start, defstart), conv(stop, defstop), step)


def main():
    acc = 0
    n = 0
    while n < N:
        for start in vals:
            for stop in vals:
                for step in vals:
                    if step == 0:
                        continue
                    for length in lengths:
                        a = slice(start, stop, step).indices(length)
                        b = slice(start, stop, step).indices(length)
                        expect = indices_ref(start, stop, step, length)
                        if a != expect or b != expect:
                            raise AssertionError((a, b, expect, start, stop, step, length))
                        acc += int(a == b)
                        acc += a[0] + a[1] + a[2]
        n += 1
    return acc


assert main() == main()
print("OK")
