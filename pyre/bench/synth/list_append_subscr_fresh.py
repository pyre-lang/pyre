# pyre-check: max-pypy-ratio=29
# After `_opimpl_recursive_call` / FBW inline screens dropped (#2600), CI
# cranelift reads 21-25x and linux dynasm 20x. 29 is 15% over the slower
# ubuntu cranelift reading (25.1x, pyre-lang #2604).
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
