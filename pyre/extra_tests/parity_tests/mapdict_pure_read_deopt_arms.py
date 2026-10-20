# CPython-suite gap: attribute tests never hot-loop a constant receiver whose
# attribute is written, deleted, or devolved out from under the compiled read.
# parity-tests reason: pins the `_pure_direct_read` fold's deopt arms without
# adding a throughput fixture's guard noise to the synth gate.
#
# getattr_hook_deopt_arms:
# attribute tests do not hot-loop every receiver/type invalidation arm. pins __getattr__
# deopts without adding guard noise to the synth gate.
#
# instance_iter_deopt_arms:
# iterator-protocol tests never hot-loop a `for` over a user instance whose `__iter__` is
# rebound, blanked, or returns a different object on a later iteration than the one the
# trace recorded. pins the arms `instance_iter` must NOT take, and the one it takes only
# until the type changes under it, without adding their guard traffic to the synth gate.

"""`AbstractAttribute.read` answers a green receiver's never-written attribute
from the elidable read; every way of making it written must be observed."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30000


# The fold's own shape: a module-level instance, so the receiver is a constant,
# and an attribute written once during construction and never again.
class Held:
    def __init__(self):
        self.value = 11
        self.tag = "t"


held = Held()
total = 0
for _ in range(N):
    total += held.value
assert total == 11 * N, total
assert held.tag == "t"


# A write mid-loop marks the attribute mutated; the very next iteration must
# read the new value, not the one the trace folded in.
class Rebound:
    def __init__(self):
        self.value = 1


rebound = Rebound()
seen = 0
for i in range(N):
    seen = rebound.value
    if i == N // 2:
        rebound.value = 2
assert seen == 2, seen


# Deleting the attribute marks it mutated too, and the read that follows must
# raise instead of answering from the folded value.
class Dropped:
    def __init__(self):
        self.value = 7


dropped = Dropped()
misses = 0
for i in range(N):
    try:
        assert dropped.value == 7
    except AttributeError:
        misses += 1
    if i == N // 2:
        del dropped.value
assert misses == N - N // 2 - 1, misses


# Devolving the instance dict moves the attribute out of the map chain without
# touching the attribute node, so the map pin is what has to notice.
class Devolved:
    def __init__(self):
        self.value = 3


devolved = Devolved()
seen = 0
for i in range(N):
    seen = devolved.value
    if i == N // 2:
        devolved.__dict__[1] = "devolve"
        devolved.__dict__["value"] = 4
assert seen == 4, seen


# The attribute node is shared by every instance of the class, so a write
# through a sibling instance marks it mutated for the folded receiver too.
class Shared:
    def __init__(self, v):
        self.value = v


pinned = Shared(5)
other = Shared(5)
seen = 0
for i in range(N):
    seen = pinned.value
    if i == N // 2:
        other.value = 6
assert seen == 5, seen


# A slot resolves to a `("slot", SLOTS_STARTING_FROM + index)` attribute node
# on the same map chain, and its write marks the same flag.
class Slotted:
    __slots__ = ("value",)

    def __init__(self):
        self.value = 8


slotted = Slotted()
seen = 0
for i in range(N):
    seen = slotted.value
    if i == N // 2:
        slotted.value = 9
assert seen == 9, seen


# An unboxed float slot takes the `_pure_unboxed_read` half of the same fold.
class Floated:
    def __init__(self):
        self.value = 1.5


floated = Floated()
acc = 0.0
for _ in range(N):
    acc += floated.value
assert acc == 1.5 * N, acc

def getattr_hook_deopt_arms():
    """A compiled __getattr__ fold observes receiver, type, and hook mutations."""

    class Mutable:
        def __getattr__(self, name):
            return "hook"

    obj = Mutable()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            obj.later = "instance"
    assert seen == "instance"

    class Replaced:
        @classmethod
        def __getattr__(cls, name):
            return "first"

    obj = Replaced()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            Replaced.__getattr__ = classmethod(lambda cls, name: "second")
    assert seen == "second"

    obj = Mutable()
    obj.__dict__[1] = "devolve"
    obj.present = "real"
    for _ in range(12000):
        assert obj.present == "real"
        assert obj.missing == "hook"
    for i in range(12000):
        value = obj.later
        if i == 6000:
            obj.later = "assigned"
    assert value == "assigned"

    def type_shadow_case():
        class TypeShadow:
            @staticmethod
            def __getattr__(name):
                return "hook:" + name

        obj = TypeShadow()
        for i in range(12000):
            seen = obj.later
            if i == 6000:
                TypeShadow.later = "class"
        assert seen == "class"

    def receiver_binding_case():
        class BoundByReceiver:
            @classmethod
            def __getattr__(cls, name):
                return cls.__name__ + ":" + name

        class Sub(BoundByReceiver):
            pass

        for _ in range(12000):
            assert BoundByReceiver().missing == "BoundByReceiver:missing"
            assert Sub().missing == "Sub:missing"

    def raising_hook_case():
        class Raising:
            def __getattr__(self, name):
                raise AttributeError("no " + name)

        obj = Raising()
        misses = 0
        for _ in range(12000):
            try:
                obj.missing
            except AttributeError as exc:
                assert str(exc) == "no missing"
                misses += 1
        assert misses == 12000

    def self_installing_hook_case():
        class Installing:
            def __getattr__(self, name):
                self.installed = "real"
                return "hook"

        obj = Installing()
        assert obj.installed == "hook"
        for _ in range(12000):
            seen = obj.installed
        assert seen == "real"

    type_shadow_case()
    receiver_binding_case()
    raising_hook_case()
    self_installing_hook_case()
getattr_hook_deopt_arms()

def instance_iter_deopt_arms():
    """`iter`'s instance arm calls `__iter__` and then requires the result to carry
    `__next__` on its type.  The inline decides that check against the receiver's
    guarded class, so every way of making the answer something else must be
    observed."""

    N = 30000

    class Counter:
        def __init__(self, limit):
            self.limit = limit
            self.j = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.j >= self.limit:
                raise StopIteration
            v = self.j
            self.j += 1
            return v

    # `__iter__` hands back a fresh object, not the receiver: the identity the
    # inline's check rests on does not hold, so the residual runs the real
    # `iter_check_is_iterator`.
    class Fresh:
        def __init__(self, limit):
            self.limit = limit

        def __iter__(self):
            return Counter(self.limit)

    def fresh_result():
        src = Fresh(3)
        total = 0
        for _ in range(N):
            for x in src:
                total += x
        return total

    assert fresh_result() == 3 * N

    # descroperation.py:339-341 — an explicit `__iter__ = None` marks the type
    # non-iterable even though the lookup succeeds.
    class Blanked:
        __iter__ = None

        def __next__(self):
            raise StopIteration

    def blanked():
        obj = Blanked()
        hits = 0
        for _ in range(N):
            try:
                for _x in obj:
                    pass
            except TypeError:
                hits += 1
        return hits

    assert blanked() == N

    # `__iter__` returns the receiver but the receiver has no `__next__`:
    # `iter_check_is_iterator` raises, and the message names the returned object's
    # type rather than the one `for` was written over.
    class NotAnIterator:
        def __iter__(self):
            return self

    def not_an_iterator():
        obj = NotAnIterator()
        hits = 0
        for _ in range(N):
            try:
                for _x in obj:
                    pass
            except TypeError as err:
                if "iterator" in str(err):
                    hits += 1
        return hits

    assert not_an_iterator() == N

    # The fold pins the receiver type's version tag, so replacing `__iter__`
    # mid-loop must switch away from the recorded body on the very next iteration.
    class Rebound:
        def __init__(self):
            self.j = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.j >= 1:
                raise StopIteration
            self.j += 1
            return "recorded"

    def rebound():
        obj = Rebound()
        seen = None
        for i in range(N):
            obj.j = 0
            for x in obj:
                seen = x
            if i == N // 2:
                Rebound.__iter__ = lambda self: iter(["replaced"])
        return seen

    assert rebound() == "replaced"

    # The same rebind in the other direction: a type whose `__iter__` starts out
    # returning a fresh iterator and becomes `return self` mid-loop, so the fold
    # has to start firing on a location that recorded the residual.
    class Widened:
        def __init__(self):
            self.j = 0

        def __iter__(self):
            return iter(("first",))

        def __next__(self):
            if self.j >= 1:
                raise StopIteration
            self.j += 1
            return "second"

    def widened():
        obj = Widened()
        seen = None
        for i in range(N):
            obj.j = 0
            for x in obj:
                seen = x
            if i == N // 2:
                Widened.__iter__ = lambda self: self
        return seen

    assert widened() == "second"

    # A subclass that inherits `__iter__` and overrides `__next__`: the version tag
    # the fold pins is the instance's own type's, and the base's answer must not be
    # reused for it.
    class Doubling(Counter):
        def __next__(self):
            if self.j >= self.limit:
                raise StopIteration
            v = self.j
            self.j += 1
            return v * 2

    def subclass_next():
        total = 0
        for _ in range(N):
            for x in Doubling(3):
                total += x
        return total

    assert subclass_next() == 6 * N
instance_iter_deopt_arms()
print("OK")
