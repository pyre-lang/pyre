# pyre-check: spec-folds=setslice,newlist
# pyre-check: max-pypy-ratio=1.0
# pyre-check: skip-cpython
# pyre-check: trace-shape=main:absent=CallMayForceN,absent=CallMayForceR
# Integer list setslice (per-strategy ops).  Exercises W_ListObject slice
# assignment: lst[a:b] = [...] stays in Integer strategy when the new items are
# plain ints, rather than falling back to Object.  guard_class on
# IntegerListStrategy and a new_array(3, ArrayS 8) are what the census reads
# back as `setslice` and `newlist`.

N = 46999728


def main():
    lst = [0] * 10
    i = 0
    while i < N:
        lst[2:5] = [i, i + 1, i + 2]
        i = i + 1
    print(lst)


main()
