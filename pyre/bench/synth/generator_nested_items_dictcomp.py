# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive
# Dict comprehension over ItemsView.__iter__, which FOR_ITERs a nested
# mapping `__iter__` generator. `walk_generator_resume` must propagate
# `ChangeFrame` (`SubWalkSuspended`) so the nested body is pushed as an
# inlined frame (`pyjitpl.py` `perform_call` / `_interpret` over
# `framestack`). Swallowing it as a FOR_ITER decline leaves a pending
# callee in the sub-walk driver.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Env:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        for key in self._data:
            yield key

    def __getitem__(self, key):
        return self._data[key]

    def items(self):
        return Items(self)


class Items:
    def __init__(self, mapping):
        self._mapping = mapping

    def __iter__(self):
        for key in self._mapping:
            yield (key, self._mapping[key])


KEYS = 40
SKIP = 8
ROUNDS = 5
DATA = {"A%d" % i: i for i in range(KEYS)}
DATA.update({"PYTHON%d" % i: i for i in range(SKIP)})
env = Env(DATA)


def drive(rounds):
    n = 0
    expected = {
        key: value for key, value in DATA.items() if not key.startswith("PYTHON")
    }
    for _ in range(rounds):
        got = {key: value for key, value in env.items() if not key.startswith("PYTHON")}
        if got != expected:
            raise SystemExit("dict %s != %s" % (got, expected))
        n += len(got)
    return n


total = drive(ROUNDS)
expected = KEYS * ROUNDS
if total != expected:
    raise SystemExit("checksum %s != %s" % (total, expected))
print(total)
print("PASS")
