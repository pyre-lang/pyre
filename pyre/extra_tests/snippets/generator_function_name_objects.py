# pyre-check: gate=1
# CPython-suite gap: generator tests do not cover function-name object reuse.
# parity-tests reason: PyPy stores the function's existing names on a new generator.

"""Generator construction retains the function's current immutable names."""


def items():
    yield 1


name = "renamed"
qualname = "qualified.items"
items.__name__ = name
items.__qualname__ = qualname
generator = items()

assert generator.__name__ is name
assert generator.__qualname__ is qualname

items.__name__ = "later"
items.__qualname__ = "later.items"
assert generator.__name__ == "renamed"
assert generator.__qualname__ == "qualified.items"

surrogate = "items\ud800"
items.__name__ = surrogate
items.__qualname__ = surrogate
surrogate_generator = items()
assert surrogate_generator.__name__ is surrogate
assert surrogate_generator.__qualname__ is surrogate

import types

code = (lambda: None).__code__.replace(co_name="\ud800x")
f = types.FunctionType(code, {})
assert f.__name__ == "\ud800x"
repr(f)

abc = "abc"
f.__name__ = abc
assert f.__name__ == "abc"
assert repr(f).startswith("<function ")

lone = "\ud800"
f.__name__ = lone
assert f.__name__ == lone
repr(f)

print("OK")
