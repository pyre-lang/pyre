"""A property whose owner overrides __setattr__/__getattribute__.

The property getter/setter folds must decline when a custom read/write hook
owns the access: `obj.value = i` dispatches through `__setattr__` (raw store,
no `* 100`, and a write counter), and `obj.value` through `__getattribute__`
(raw read, no `+ 1`, and a read counter), never the property `fset`/`fget`.
Inlining the property body would drop the hooks and diverge silently.
"""


class Hooked:
    def __init__(self):
        object.__setattr__(self, "_value", 0)
        object.__setattr__(self, "writes", 0)
        object.__setattr__(self, "reads", 0)

    @property
    def value(self):
        return self._value + 1

    @value.setter
    def value(self, value):
        object.__setattr__(self, "_value", value * 100)

    def __setattr__(self, name, value):
        if name == "value":
            object.__setattr__(self, "_value", value)
            object.__setattr__(self, "writes", object.__getattribute__(self, "writes") + 1)
        else:
            object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if name == "value":
            object.__setattr__(self, "reads", object.__getattribute__(self, "reads") + 1)
            return object.__getattribute__(self, "_value")
        return object.__getattribute__(self, name)


def main():
    obj = Hooked()
    total = 0
    i = 0
    while i < 100000:
        obj.value = i
        total += obj.value
        i += 1
    print(total, obj._value, obj.writes, obj.reads)


main()
