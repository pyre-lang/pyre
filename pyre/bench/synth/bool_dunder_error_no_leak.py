# pyre-check: max-pypy-ratio=110
# A __bool__ that raises surfaces the error exactly once; the interpreter does
# not stash a duplicate pending exception that resurfaces on a later unrelated
# statement. Output verified against CPython/PyPy.
N = 5760000


class Boom:
    def __bool__(self):
        raise ValueError("boom")


def check(b):
    try:
        if b:
            return "no-raise"
    except ValueError:
        pass
    # A leaked pending error would resurface on this call boundary.
    return str(len("ok"))


def main():
    b = Boom()
    n = 0
    for _ in range(N):
        if check(b) == "2":
            n += 1
    print(n)


main()
