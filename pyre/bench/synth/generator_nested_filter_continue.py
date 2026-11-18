# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive
# Nested generator: inner filters with `if x % 3: continue` so several
# loop back-edges pass between yields. Outer iterates inner in a hot loop.
# The inner body's `SubLoopCalleeCallAssembler` must bind at pop
# (`finishframe` / `opimpl_jit_merge_point`) without replaying `_send_ex`.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Inner:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        for x in self._data:
            if x % 3:
                continue
            yield x


class Outer:
    def __init__(self, inner):
        self._inner = inner

    def __iter__(self):
        for key in self._inner:
            yield key


KEYS = 40
ROUNDS = 5
DATA = list(range(KEYS))
EXPECTED = [x for x in DATA if x % 3 == 0]
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
