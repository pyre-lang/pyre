# Nested `@contextmanager` `__exit__` does `next(self.gen)`, which raises
# StopIteration into `except StopIteration: return False`. Synthesizing
# `record_context`'s `guard_isnull` on `sys_exc_operror` /
# `current_gen_or_coroutine` fails when the compiled `__exit__` body is
# reused under an outer generator; resume is at the CALL of `next(self.gen)`
# whose valuestack was consumed (`TypeError: call on null callable`).
# Residual-call `resolve_exception_context` instead; it reads
# `get_sys_exception` at run time (`error.py record_context`).
#
# Expected: N
# No `max-pypy-ratio`: this is a shape oracle, not a workload.
from contextlib import contextmanager

N = 20000


@contextmanager
def inner():
    yield


@contextmanager
def outer():
    with inner():
        yield


def main():
    n = 0
    i = 0
    while i < N:
        with outer():
            n = n + 1
        i = i + 1
    print(n)


main()
