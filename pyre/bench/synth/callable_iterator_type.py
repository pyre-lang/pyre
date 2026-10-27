# pyre-check: max-pypy-ratio=205
# iter(callable, sentinel) yields a properly-registered iterator type: type(it)
# is a real type object, the iterator is an object instance, and __iter__
# returns self. The concrete type name differs across implementations, so only
# the structural type-identity invariants are compared. Output verified against
# CPython/PyPy.
N = 1280000


def f():
    return 1


def main():
    n = 0
    for _ in range(N):
        it = iter(f, 0)
        if isinstance(type(it), type) and isinstance(it, object) and iter(it) is it:
            n += 1
    print(n)


main()
