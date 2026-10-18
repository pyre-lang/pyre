# pyre-check: requires-modules=gc
# pyre-check: max-pypy-ratio=3.5
# Local cranelift 2.4x; the ceiling is 3.5.
# pyre-check: skip-cpython
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S`.  At 120000
# iterations pypy exec is inside the `?` band, so a ceiling of 40 applied
# and the floor did not.  48000000 iterations land pypy near 0.11s.
# cpython cannot run this many inside the reference timeout.
import gc

N = 48000000
M = 4000

# `cls.__name__` read from inside a method the enclosing `for` loop inlines.
# A rename must be seen and the name object's identity must survive. Two
# classes sharing the trace must each report their own name. A class-dict
# entry of the same name loses, because the metatype descriptor is consulted
# first. A metaclass other than `type` must not take that path.


class Base:
    @classmethod
    def tag(cls, i):
        return i + len(cls.__name__)


class Derived(Base):
    pass


class Longer(Base):
    pass


class Shadowed(Base):
    # `type.__name__` is a data descriptor on the metatype, so this entry is
    # what an INSTANCE of the class reads and never what the class does.
    __name__ = 'shadow-entry'


class GetattrMeta(type):
    def __getattribute__(cls, name):
        # 8 characters, which no class here is named, so the total says whether
        # this ran.
        if name == '__name__':
            return 'via-meta'
        return type.__getattribute__(cls, name)


class NameMeta(type):
    # A metaclass may define `__name__` itself, and then it is the one that
    # answers -- also 8 characters.
    @property
    def __name__(cls):
        return 'metaname'


class PlainMeta(type):
    # Answers exactly as `type` would, and still must not be folded past: the
    # fold's precondition is the metaclass, not what it happens to do.
    pass


class ByGetattr(metaclass=GetattrMeta):
    @classmethod
    def tag(cls, i):
        return i + len(cls.__name__)


class ByNameProp(metaclass=NameMeta):
    pass


class ByPlain(metaclass=PlainMeta):
    pass


class Box:
    def __init__(self, v):
        self.v = v

    # The instance-attribute twin: the same inline the type-name fold reaches,
    # over the mapdict fold instead.
    def at(self, i):
        return self.v + i


def folded():
    box = Box(4)
    # `must_compile` keys a failing ref GUARD_VALUE by the object's address
    # (compile.py `lltype.cast_ptr_to_int`). Promote the receivers before the
    # loop warms up, so a minor collection does not move them mid-warm-up and
    # restart their counters at a point set by the nursery size.
    gc.collect()
    total = 0
    for i in range(N):
        # Three classes down one trace; each must answer with its own name.
        total = total + Derived.tag(i) - i
        total = total + Longer.tag(i) - i
        total = total + Shadowed.tag(i) - i
        total = total + box.at(i) - i
        # Renaming mid-loop, after the loop is compiled: the fold reads the
        # name slot rather than a constant, so the shorter name is seen from
        # the next iteration on.
        if i == N // 2:
            Derived.__name__ = 'D'
    return total


def declined():
    # Every receiver here has a metaclass that is not `type`, so the fold must
    # not run.  Long enough to compile the loop and reach that decision, and no
    # longer -- what this pins is the decision, not a speed.
    total = 0
    for i in range(M):
        total = total + ByGetattr.tag(i) - i
        total = total + len(ByNameProp.__name__)
        total = total + len(ByPlain.__name__)
    return total


print(folded())
print(declined())
print(Derived.__name__, Longer.__name__, Shadowed.__name__)
print(Shadowed.__dict__['__name__'], Shadowed().__name__)
print(ByGetattr.__name__, ByNameProp.__name__, ByPlain.__name__)

# A rename between two compiled loops, rather than inside one: the second loop
# reads the slot the first one's trace was built against, so a baked name would
# survive here too.
Box.__name__ = 'Renamed'
seen = None
for _ in range(M):
    seen = Box.__name__
print(seen)
