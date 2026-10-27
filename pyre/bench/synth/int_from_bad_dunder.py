# pyre-check: max-pypy-ratio=175
# int() rejects a __int__ result that is not an int with TypeError (not an
# internal RuntimeError). Output verified against CPython/PyPy.
N = 320000


class BadInt:
    def __int__(self):
        return 1.5


def main():
    caught = 0
    for _ in range(N):
        try:
            int(BadInt())
        except TypeError:
            caught += 1
    print(caught)


main()
