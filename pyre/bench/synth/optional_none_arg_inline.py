# pyre-check: max-pypy-ratio=2.7
# A hot call into a callee that branches on `arg is None` -- the shape every
# optional argument in the standard library takes.  The tested local holds the
# default `None`, a Ref; the other parameter is an int.
#
# The gate is `loops_compiled` first: declined, the callee takes a trace of
# its own and the count reads 2; inlined, it folds into the caller's and reads
# 1. That census is host-independent. On this run dynasm reads 1.0x and
# cranelift 1.3x; the ceiling is twice the slower, and a ceiling under 3
# derives no floor.
#
# Its sibling `is_none_unboxed_operand_inline` is this fixture with the
# default changed from `None` to an int; it inlines as well.
#
# `clamp` is called once more off the hot path so the printed total depends on
# the arm the trace guards away as well as the one it keeps: an identity test
# answered the wrong way changes the sum rather than the timing.
N = 175933800


def clamp(v, lo=None):
    if lo is None:
        return v
    return lo


def main():
    total = 0
    for i in range(N):
        total += clamp(i)
    total += clamp(1, 7)
    print(total)


main()
