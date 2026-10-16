# pyre-check: spec-folds=make_function
# MAKE_FUNCTION plus SET_FUNCTION_ATTRIBUTE in a hot FOR_ITER body. The default
# value forces the attribute initializer onto the definition path.
#
# `spec-folds` gates MAKE_FUNCTION. Low thresholds keep the loop compiled
# without hundreds of millions of arithmetic iterations.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 3000000


def main():
    total = 0
    for i in range(N):

        def add(value=i):
            return value + 1

        total += add()
    print(total)


main()
# Expected: 4500001500000
