# pyre-check: requires-modules=gc
# pyre-check: max-pypy-ratio=4
# The measured reading stays under 4, so the ceiling is 4.
# pyre-check: skip-cpython
# cpython 1.89s vs pyre 0.37s (5.1x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
import gc

N = 24729800


class Base:
    def value(self, x):
        return x + 1


class Left(Base):
    def value(self, x):
        return x + 3


class Right(Base):
    def value(self, x):
        return x - 5


def main():
    objs = [Base(), Left(), Right()]
    # `must_compile` keys a failing ref GUARD_VALUE by the object's address
    # (compile.py `lltype.cast_ptr_to_int`). Promote the receivers before the
    # loop warms up, so a minor collection does not move them mid-warm-up and
    # restart their counters at a point set by the nursery size.
    gc.collect()
    i = 0
    acc = 0
    while i < N:
        acc = acc + objs[i % 3].value(i)
        i = i + 1
    print(acc)


main()
