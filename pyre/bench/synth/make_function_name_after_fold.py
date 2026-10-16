# pyre-check: spec-folds=make_function
# pyre-check: requires-modules=gc
# A function built by MAKE_FUNCTION in a hot loop reports its name through
# the `__code__` setter's free-variable error, which reads `Function.name`
# rather than the wrapped `__name__`. The name must be the code object's
# `co_name` on compiled iterations and after a collection.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

import gc

N = 600000
EXPECTED = "add_one() requires a code object with 0 free vars, not 1"


def make_closure_code():
    cell = 1

    def inner(value):
        return value + cell

    return inner.__code__


CLOSURE_CODE = make_closure_code()


def main():
    good = 0
    for i in range(N):

        def add_one(value):
            return value + 1

        if i % 1000 == 0:
            if i % 50000 == 0:
                gc.collect()
            try:
                add_one.__code__ = CLOSURE_CODE
            except ValueError as e:
                if str(e) == EXPECTED:
                    good += 1
        else:
            good += add_one(0)
    print(good)


main()
# Expected: 600000
