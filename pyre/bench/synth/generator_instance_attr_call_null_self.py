# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive
# Generator `__iter__` yields `self.decodekey(key)` where `decodekey` is a
# plain function in the instance dict (CALL with a NULL self slot). The
# inline reads `Function.code` through the heapcache (`pyjitpl.py`
# `_opimpl_getfield_gc_any_pureornot`) and pins it with `implement_guard_value`.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

ROUNDS = 300
KEYS = 40


def make_decode():
    marker = 1

    def decode(value):
        _ = marker
        return value

    return decode


class Env:
    def __init__(self, data):
        self._data = data
        self.decodekey = make_decode()

    def __iter__(self):
        for key in self._data:
            yield self.decodekey(key)


def drive(rounds):
    env = Env(list(range(KEYS)))
    n = 0
    for _ in range(rounds):
        for _name in env:
            n += 1
    return n


total = drive(ROUNDS)
if total != ROUNDS * KEYS:
    raise SystemExit(1)
print("PASS")
