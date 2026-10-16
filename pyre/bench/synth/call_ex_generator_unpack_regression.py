# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code. The jitstats baselines gate it.
# Regression guard: `f(*gen())` runs `unpack_into` (baseobjspace.py
# `unpackiterable` generator fast path), which resumes the suspended frame
# once per yield inside one root bracket and reads its results back from a
# base slot. `_invoke_execute_frame` must not leave its own pins on that
# bracket: with them the read-back interleaves the generator and the resume
# value with the yielded values.
def three():
    yield 1
    yield 2
    yield 3


def five():
    x = yield 10
    y = yield 20 + (x or 0)
    yield 30
    yield 40
    yield 50 + (y or 0)


def collect(*args):
    return args


total = 0
bad = 0
for i in range(5100):
    a = collect(*three())
    b = collect(*five())
    if a != (1, 2, 3):
        bad += 1
    if b != (10, 20, 30, 40, 50):
        bad += 1
    total += len(a) + len(b) + a[-1] + b[-1]
print(total, bad)
