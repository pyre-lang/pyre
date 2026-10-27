# pyre-check: max-pypy-ratio=18
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


print(f(18000000))
