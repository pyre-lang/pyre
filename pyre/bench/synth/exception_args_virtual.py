# pyre-check: max-pypy-ratio=21
N = 12000000


def may_fail(i):
    # Object-strategy args (str + int + float): the args list is built
    # inline in the trace and virtualizes when the exception is caught
    # without reading e.args.
    if (i & 31) == 0:
        raise ValueError("boom", i, 0.5)
    return i & 7


def main():
    i = 0
    acc = 0
    read = 0
    while i < N:
        try:
            acc = acc + may_fail(i)
        except ValueError as e:
            # Every 4th raise reads e.args, forcing the virtualized
            # list to materialize; the rest never touch it.
            if (i & 127) == 0:
                a, b, c = e.args
                acc = acc + len(a) + (b & 15) + int(c * 2)
                read = read + 1
            else:
                acc = acc + 1
        finally:
            acc = acc + 1
        i = i + 1
    print(acc, read)


main()
