# pyre-check: max-pypy-ratio=9
# Current CI reads 2.9-7.4x (slowest: windows dynasm 7.4x, pyre-lang #2616
# / #2619). 9 is ~20% over that reading; 9/4 stays under the fastest 2.9x.
# A fresh empty list promoted by append takes RPython's first 0 -> 4 backing
# grow. The following subscript revisits that block through
# W_ListObject.int_items.block after the grow helper updates the owner field.


def f(n):
    acc = 0
    for i in range(n):
        l = []
        l.append(i & 1)
        acc += l[0]
    return acc


print(f(45000000))
