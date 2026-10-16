# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# Identity-keyed dict length. `{int, str, list}` keys compare by identity
# (`IdentityDictStrategy`). `len` is `rordereddict.py` `ll_dict_len`:
# `return d.num_live_items`. RandomState hasher storage keeps that word at
# a different offset from the ZST-hasher strategies.
N = 20000


def f():
    d = {int: 1, str: 2, list: 3}
    return len(d)


def main():
    acc = 0
    i = 0
    while i < N:
        acc += f()
        i += 1
    if acc != 60000:
        raise AssertionError(acc)
    print("PASS")


main()
