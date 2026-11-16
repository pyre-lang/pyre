# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=held,rebound,slotted,floated
# pyre-check: trace-shape=floated:absent=ArraylenGc
# `AbstractAttribute.read` answers through the `@jit.elidable`
# `_pure_direct_read` when the attribute and the receiver are both green and
# `ever_mutated` is still false.  Without it the trace keeps the three loads
# `_prim_direct_read` is -- the storage block, the attribute's slot, and the
# item -- and carries all of them across the loop back edge.
#
# Both storage shapes take the fold: a boxed slot through
# `PlainAttribute._pure_direct_read` and an unboxed one through
# `UnboxedPlainAttribute._pure_unboxed_read`, whose boxing stays visible to the
# trace so a numeric consumer still unwraps the constant.
#
# The deopt arms -- a write, a delete, a devolved dict, a sibling instance's
# write, and a slot -- live in
# `extra_tests/parity_tests/mapdict_pure_read_deopt_arms.py`; this fixture only
# proves the fold fires and that the answer survives a mid-loop rebind.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30000


class Held:
    def __init__(self):
        self.value = 11


class Slotted:
    __slots__ = ("value",)

    def __init__(self):
        self.value = 8


class Floated:
    def __init__(self):
        self.value = 1.5


holder = Held()
slot_holder = Slotted()
float_holder = Floated()
rebind_holder = Held()


def held(n):
    total = 0
    i = 0
    while i < n:
        total += holder.value
        i += 1
    return total


def slotted(n):
    total = 0
    i = 0
    while i < n:
        total += slot_holder.value
        i += 1
    return total


def floated(n):
    total = 0.0
    i = 0
    while i < n:
        total += float_holder.value
        i += 1
    return total


def rebound(n):
    # The fold takes the `ever_mutated?` quasi-immutable; the write below sets
    # it, so the read after it must answer with the new value.
    seen = 0
    i = 0
    while i < n:
        seen = rebind_holder.value
        if i == n // 2:
            rebind_holder.value = 12
        i += 1
    return seen


assert held(N) == 11 * N
assert slotted(N) == 8 * N
assert floated(N) == 1.5 * N
assert rebound(N) == 12
print("PASS mapdict pure direct read")
