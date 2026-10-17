# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,inner
# Generator context-manager `__enter__` is `next(self.gen)`, and the body
# runs `FOR_ITER` over a tuple, both inside a hot inlined callee.
#
# A multi-frame resume that retraced the portal frame from a callee pc
# handed `space.next` a leftover builtin (`TypeError: 'builtin_function_or_method'
# object is not an iterator`) or NULL. `contextlib._GeneratorContextManager.__enter__`
# is the unique `next(non_iterator)` site in `test_mimetypes.test_unknown_flag`.
from contextlib import contextmanager

N = 20000


@contextmanager
def cm():
    yield


def inner(xs):
    s = 0
    with cm():
        for x in xs:
            s += x
    return s


def hot(n):
    t = 0
    xs = (1, 2, 3)
    for i in range(n):
        t += inner(xs)
    return t


print("PASS", hot(N))
