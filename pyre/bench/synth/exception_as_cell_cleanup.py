# pyre-check: max-pypy-ratio=12
# An `except E as e:` handler whose exception variable `e` is captured by a
# nested function makes `e` a cell variable, so the implicit handler cleanup
# emits DELETE_DEREF (clear the cell contents) rather than DELETE_FAST. The
# traced DELETE_DEREF dereferences the cell to raise if unbound, then clears
# its contents; this bench pins that a captured-exception handler in a hot
# loop compiles and returns byte-identically.
N = 10000000


def run(n):
    acc = 0
    i = 0
    while i < n:
        try:
            if (i % 5) == 0:
                raise ValueError(i)
            acc += 1
        except ValueError as e:
            def grab():
                return e          # captures e -> e becomes a cell
            acc += grab().args[0] & 7
        i += 1
    return acc


def main():
    print(run(N))


main()
