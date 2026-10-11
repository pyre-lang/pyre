# CPython-suite gap: test_argparse crashed with SIGSEGV after
# TestFileTypeXB.test_failures_no_groups_listargs reported a spurious ERROR.
# parity-tests reason: the failure needs a compiled GUARD_EXCEPTION whose
# bridge is traced while an enclosing `except` block is handling an unrelated
# exception, which only a JIT-compiled run reaches.
#
# `conv` is inlined into `run`'s loop and traced through the raising `int(s)`,
# so the guard sits in the callee of a two-frame resume. The second and third
# `run` calls feed it strings that convert: the guard fails because the call
# did not raise, and its bridge is compiled with `sys.exc_info()` holding the
# outer KeyError. The bridge has to continue with the conversion result;
# taking the handled KeyError for the guard's exception re-raises it out of
# `int(s)`.
N = 20000


def conv(s):
    try:
        return int(s)
    except ValueError:
        return -1


def run(items):
    t = 0
    for s in items:
        t += conv(s)
    return t


def main():
    bad = ["x"] * 3000
    good = ["7"] * N
    try:
        raise KeyError("outer")
    except KeyError:
        a = run(bad)
        b = run(good)
        c = run(good)
    assert (a, b, c) == (-3000, 7 * N, 7 * N), (a, b, c)


main()
print("OK")
