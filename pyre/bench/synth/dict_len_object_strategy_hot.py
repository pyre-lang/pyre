# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,entry-bridge:object_len
# JIT `len(d)` across dict strategies. `type` keys `compares_by_identity`, so
# `EmptyDictStrategy.switch_to_correct_strategy` selects IdentityDictStrategy
# (`r_dict` with the default `RandomState` hasher). Int keys take
# IntDictStrategy, str keys UnicodeDictStrategy, bytes keys BytesDictStrategy.
# `AbstractTypedStrategy.length` is `len(unerase(dstorage))`;
# `rordereddict.py ll_dict_len` is `return d.num_live_items`.
#
# Without `repr(C)` on `RDict`, rustc laid `RandomState` in front of
# `num_live_items` and the compiled getfield (offset of the ZST-hasher
# instantiations) read a hasher word.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def object_len():
    d = {object: 0, type: 1, int: 2}
    return len(d)


def int_len():
    d = {1: 0, 2: 1}
    return len(d)


def str_len():
    d = {"a": 0}
    return len(d)


def bytes_len():
    d = {b"a": 0}
    return len(d)


def main():
    n = 5000
    a = b = c = d = 0
    for _ in range(n):
        a += object_len()
        b += int_len()
        c += str_len()
        d += bytes_len()
    return a, b, c, d, len({object: 1}), len({1: 2}), len({"x": 1}), len({b"x": 1})


EXPECTED = (15000, 10000, 5000, 5000, 1, 1, 1, 1)
got = main()
if got == EXPECTED:
    print("PASS dict_len_object_strategy_hot")
else:
    print("FAIL %r against %r" % (got, EXPECTED))
    raise SystemExit(1)
