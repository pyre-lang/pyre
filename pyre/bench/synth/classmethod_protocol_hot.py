# pyre-check: max-pypy-ratio=80
# pypy's exec time is pinned to the startup-subtraction floor here at every
# size, so the ratio is not a measurement: pypy traces the loop down to a couple
# of nanoseconds an iteration, and a pypy side big enough to measure would need
# tens of millions of them — a minute of pyre.  What the count does decide is
# whether pyre's own side is a measurement, and at the 1001 it was recorded with
# it was not: 0.01s here, one scheduler tick, which a macOS runner read as 0.04s
# and failed a 6x ceiling with.  The count is the smallest that puts pyre's side
# an order of magnitude above the tick; the ceiling is four times the slowest
# the runners observe (19.4x).
"""Hot classmethod binding and wrapper-dictionary parity."""


class Base:
    @classmethod
    def calculate(cls, value):
        return value + len(cls.__name__)


class Derived(Base):
    pass


wrapper = Base.__dict__["calculate"]
wrapper.offset = 3

total = 0
for i in range(100005):
    total += Derived.calculate(i) + wrapper.offset

print(total)
print(callable(wrapper))
print(wrapper.__func__.__name__, wrapper.__dict__["offset"])
