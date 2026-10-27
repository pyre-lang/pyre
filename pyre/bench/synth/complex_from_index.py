# pyre-check: max-pypy-ratio=9.9
# complex() consults __index__ before __float__, matching
# complexobject.py unpackcomplex. This loop only defines __index__.
N = 11520000


class Idx:
    def __index__(self):
        return 3


def main():
    obj = Idx()
    acc = 0.0
    for _ in range(N):
        c = complex(obj)
        acc += c.real
    print(complex(obj), acc == N * 3.0)


main()
