# CPython-suite gap: test_call does not pin the keyword-only count when a
# `**kwargs` extra is present alongside a filled keyword-only slot.
# parity-tests reason: `argument.py Arguments._match_signature` counts filled
# keyword-only slots, so a `**kwargs` extra is not a keyword-only argument.

"""A **kwargs extra is not counted as a keyword-only argument."""


def f(a, *, k, **kw):
    pass


try:
    f(1, 2, **{"k": 1, "z": 2})
except TypeError as e:
    r_starstar_extra = str(e)

assert r_starstar_extra == (
    "f() takes 1 positional argument but 2 positional arguments "
    "(and 1 keyword-only argument) were given"
)
print("OK")
