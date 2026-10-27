# pyre-check: max-pypy-ratio=460
N = 800000


def main():
    acc = 0
    i = 0
    while i < N:
        import errno
        if errno.EINTR > 3:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
