# pyre-check: max-pypy-ratio=60
# A NaN in an all-float list: the same NaN object is found by index/count/
# containment and compares equal in list == / <= (bit-pattern identity), even
# though `nan == nan` is false. Output verified against CPython/PyPy.
N = 120000


def main():
    x = float("nan")
    hits = 0
    for _ in range(N):
        L = [1.0, x, 2.0]
        ok = (
            L.index(x) == 1
            and L.count(x) == 1
            and (x in L)
            and ([x] == [x])
            and ([1.0, x] == [1.0, x])
            and ([x] <= [x])
            and not ([x] < [x])
        )
        if ok:
            hits += 1
    print(hits)


main()
