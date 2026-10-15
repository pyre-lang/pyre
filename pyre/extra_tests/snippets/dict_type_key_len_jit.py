# pyre-check: gate=1
"""len, iteration and lookup of a dict whose keys are type objects.

IdentityDictStrategy stores those keys. AbstractTypedStrategy.length is
len(unerase(dstorage)), which rtypes as getfield num_live_items.
"""

N = 4000


def type_key_len(n):
    acc = 0
    i = 0
    while i < n:
        acc += len({object: 0})
        i += 1
    return acc


def two_type_key_len(n):
    acc = 0
    i = 0
    while i < n:
        acc += len({object: 0, int: 0})
        i += 1
    return acc


def type_key_iter(n):
    acc = 0
    i = 0
    while i < n:
        d = {object: 1, int: 2, str: 3}
        for k in d:
            acc += d[k]
        i += 1
    return acc


def type_key_lookup(n):
    acc = 0
    i = 0
    while i < n:
        d = {object: 7, int: 9}
        acc += d[object] + d[int]
        i += 1
    return acc


one = type_key_len(N)
two = two_type_key_len(N)
walked = type_key_iter(N)
looked = type_key_lookup(N)
print("one", one)
print("two", two)
print("iter", walked)
print("lookup", looked)
assert one == N, one
assert two == 2 * N, two
assert walked == 6 * N, walked
assert looked == 16 * N, looked
