# pyre-check: max-pypy-ratio=19
# `__defaults__` replaced part-way through a compiled loop, in the two shapes
# that break differently from a same-length replacement.
#
# `long_tail` swaps in a tuple LONGER than the parameter list.  Binding then has
# to take the tuple's tail (`argument.py:302-315` keeps `defaults_w[i -
# def_first]` for a `def_first = co_argcount - len(defaults_w)` that has gone
# negative), and only the guard-failure resume path recomputes it — the trace
# and the interpreter's flat-call both derive the tail themselves, so setting
# the same tuple before the loop stays correct while swapping it mid-loop did
# not.
#
# `two_ints` swaps in a two-int tuple, which is a `W_SpecialisedTupleObject_ii`.
# Its `value0` is field 0, exactly where `W_TupleObject.wrappeditems` sits, so a
# trace that reads `wrappeditems` off the live `defs_w` without proving the
# class reads the raw integer instead of the items array.
N = 15000000


def long_tail(n):
    def f(a, b=2):
        return a * 10 + b

    acc = 0
    for i in range(n):
        if i == n // 2:
            f.__defaults__ = (7, 9, 11)
        acc += f(1)
    return acc


def two_ints(n):
    def f(a, b=2):
        return a * 10 + b

    acc = 0
    for i in range(n):
        if i == n // 2:
            f.__defaults__ = (7, 9)
        acc += f(1)
    return acc


print(long_tail(N), two_ints(N))
