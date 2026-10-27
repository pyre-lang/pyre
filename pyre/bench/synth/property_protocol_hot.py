# pyre-check: max-pypy-ratio=365
"""Hot property get/set plus Python 3.14 metadata surface."""


class Counter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        "counter value"
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class OverridingProperty(property):
    def __get__(self, obj, objtype=None):
        return "override-get"

    def __set__(self, obj, value):
        obj.recorded = "override-set"

    def __delete__(self, obj):
        obj.recorded = "override-del"


class PropertySubclassOwner:
    recorded = None
    value = OverridingProperty(lambda self: "wrapped")


def main():
    counter = Counter()
    total = 0
    i = 0
    while i < 8000000:
        counter.value = i
        total += counter.value
        i += 1

    prop = Counter.__dict__["value"]
    print(total, counter.value)
    # PyPy 3.11 has not yet gained property.__name__; keep the synthetic
    # output version-neutral while the dedicated 3.14 parity test covers it.
    print(prop.fget.__name__, prop.__doc__)

    # The property fold is exact-type only.  A subclass keeps the same layout
    # but must still dispatch through its overridden descriptor methods.
    owner = PropertySubclassOwner()
    for _ in range(200000):
        assert owner.value == "override-get"
    owner.value = 1
    assert owner.recorded == "override-set"
    del owner.value
    assert owner.recorded == "override-del"


main()
