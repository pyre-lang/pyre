# pyre-check: spec-folds=make_function
# An annotated `def` in a hot FOR_ITER body. A return annotation compiles to
# two MAKE_FUNCTIONs, one for the `__annotate__` closure, and a
# SET_FUNCTION_ATTRIBUTE annotate stamp before any defaults stamp.
#
# `spec-folds` gates MAKE_FUNCTION. Low thresholds keep the loop compiled
# without hundreds of millions of arithmetic iterations.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 2500000


def main():
    total = 0
    for i in range(N):

        def add(value) -> int:
            return value + 1

        total += add(i)
    print(total)


main()
# Expected: 3125001250000
