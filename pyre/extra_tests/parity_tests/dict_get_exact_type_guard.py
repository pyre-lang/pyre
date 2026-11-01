# CPython-suite gap: dict.get tests do not warm a compiled exact-str path before
# changing the key's exact type or the receiver's storage strategy.
# parity-tests reason: traced strategy type checks must guard their assumptions
# and preserve the checked lookup's hash exceptions after a side exit.


def lookup(mapping, key):
    return mapping.get(key, 99)


def hot(mapping, key):
    total = 0
    for _ in range(5000):
        total += lookup(mapping, key)
    return total


mapping = {"a": 7}
assert hot(mapping, "a") == 35000


class HashError(Exception):
    pass


class RaisingString(str):
    def __hash__(self):
        raise HashError("key hash")


try:
    hot(mapping, RaisingString("a"))
except HashError as exc:
    assert str(exc) == "key hash"
else:
    raise AssertionError("dict.get swallowed the subclass hash error")


class PlainString(str):
    pass


assert hot(mapping, PlainString("a")) == 35000
mapping[42] = 13
assert hot(mapping, "a") == 35000
assert hot(mapping, 42) == 65000
assert hot(mapping, "missing") == 495000


class DictSubclass(dict):
    def __getitem__(self, key):
        raise AssertionError("dict.get must not invoke __getitem__")


assert hot(DictSubclass(a=11), "a") == 55000
print("OK")
