# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive
# Nested generator `__iter__` driven from a list comprehension in a hot
# loop, same shape as `generator_nested_items_dictcomp.py`. The inner
# generator ends with `return 99` after its last yield. `_send_ex`
# (`GeneratorOrCoroutine._send_ex`) treats a finished frame as
# StopIteration, so 99 must not appear as a FOR_ITER item.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Inner:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        # Yield every item, then `return`. The last resume is this
        # return as a nested generator SubReturn (`_send_ex`). Enough
        # items that `drive`'s inlined comprehension compiles as a loop
        # (same scale as `generator_nested_items_dictcomp.py`).
        for item in self._data:
            yield item
        return 99


class Outer:
    def __init__(self, inner):
        self._inner = inner

    def __iter__(self):
        for key in self._inner:
            yield key


KEYS = 40
ROUNDS = 5
DATA = ["A%d" % i for i in range(KEYS)]
EXPECTED = list(DATA)
env = Outer(Inner(DATA))


def drive(rounds):
    n = 0
    expected = list(EXPECTED)
    for _ in range(rounds):
        got = [key for key in env]
        if got != expected:
            raise SystemExit("items %s != %s" % (got, expected))
        n += len(got)
    return n


total = drive(ROUNDS)
expected_n = len(EXPECTED) * ROUNDS
if total != expected_n:
    raise SystemExit("checksum %s != %s" % (total, expected_n))
print(total)
print("PASS")
