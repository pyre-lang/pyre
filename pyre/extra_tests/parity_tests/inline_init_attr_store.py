# CPython-suite gap: no suite test drives one constructor hot enough to inline.
# parity-tests reason: this guards the replay safety of an inlined `__init__`.
#
# inline_init_nested_call_deopt:
# no suite test deopts in or below a hot inlined `__init__`. this guards the constructor
# continuation both adjacent to the resumed frame and one helper frame deeper.

"""An inlined `__init__` may write only the instance the call just allocated.

A guard inside an inlined constructor resumes at the caller's CALL and re-runs
the instantiation, so a store the first attempt made must not survive.  That
holds for `self.x = v` on the fresh instance and for nothing else, so the three
shapes below have to keep answering the same as the interpreter: the plain
constructor, a method mutating a receiver that outlives the call, and a
constructor that publishes `self` before it stores.
"""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

ROUNDS = 3000


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.total = x + y


total = 0
for i in range(ROUNDS):
    p = Point(i, i + 1)
    assert p.x == i, (i, p.x)
    assert p.y == i + 1, (i, p.y)
    assert p.total == 2 * i + 1, (i, p.total)
    total += p.total
assert total == sum(2 * i + 1 for i in range(ROUNDS)), total


class Counter:
    def __init__(self):
        self.n = 0

    def bump(self, by):
        # The receiver outlives the call, so this write is NOT an
        # initialization: re-running it after a rewind would double it.
        self.n += by
        return self.n


c = Counter()
for i in range(ROUNDS):
    c.bump(1)
assert c.n == ROUNDS, c.n

acc = Counter()
seen = []
for i in range(ROUNDS):
    seen.append(acc.bump(2))
assert acc.n == 2 * ROUNDS, acc.n
assert seen == list(range(2, 2 * ROUNDS + 2, 2)), seen[:4]

REGISTRY = []


class Published:
    def __init__(self, v):
        # `self` escapes before the store, so a rewind no longer discards it.
        REGISTRY.append(self)
        self.v = v


for i in range(ROUNDS):
    Published(i)
assert len(REGISTRY) == ROUNDS, len(REGISTRY)
assert [o.v for o in REGISTRY[:5]] == [0, 1, 2, 3, 4], [o.v for o in REGISTRY[:5]]
assert REGISTRY[-1].v == ROUNDS - 1, REGISTRY[-1].v

def inline_init_nested_call_deopt():
    """A class call returns the instance when the deopt happens one frame deeper.

    `inline_init_deopt_result.py` deopts inside `__init__` itself, so the level
    that discards `__init__`'s result sits directly beneath the frame that
    resumes.  Here `__init__` calls a helper and the guard fails inside THAT, so
    the chain is `caller -> descr_call -> __init__ -> helper`.  The discarding
    level is no longer adjacent to the resumed frame, which is the arrangement
    that catches a continuation recorded at a fixed offset from the innermost
    callee rather than at its own depth.

    A chain that misplaces it answers `None` from the CALL, or delivers the
    helper's return where the instance belongs, so the checks read the call's
    result rather than only its side effects.
    """

    ROUNDS = 4000
    # Rare enough that the loop is long compiled before the second arm is first
    # taken, so that arm is reached through a guard failure rather than by tracing.
    RARE = 997

    class DirectBadOnRarePath:
        def __init__(self, n):
            self.n = n
            if n < 0:
                return n
            return None

    direct_raised = 0
    direct_built = 0
    for i in range(ROUNDS):
        value = -i if i % RARE == RARE - 1 else i
        try:
            obj = DirectBadOnRarePath(value)
        except TypeError:
            direct_raised += 1
        else:
            assert isinstance(obj, DirectBadOnRarePath)
            assert obj.n == value
            direct_built += 1
    assert direct_raised == ROUNDS // RARE
    assert direct_built == ROUNDS - direct_raised

    def classify(n):
        # The arm taken on the rare iterations is not the traced one, so the guard
        # that fails is inside this helper, one frame below `__init__`.
        if n < 0:
            return "neg", -n
        return "pos", n

    class Boxed:
        def __init__(self, n):
            self.tag, self.magnitude = classify(n)
            self.n = n

    total = 0
    rare_seen = 0
    for i in range(ROUNDS):
        v = -i if i % RARE == RARE - 1 else i
        b = Boxed(v)
        # The call's result, not a side effect: a misplaced continuation answers
        # `None` here, and one recorded a frame too low answers the helper's tuple.
        assert isinstance(b, Boxed), (i, b)
        assert b.n == v, (i, b.n, v)
        assert b.tag == ("neg" if v < 0 else "pos"), (i, b.tag, v)
        assert b.magnitude == abs(v), (i, b.magnitude, v)
        if v < 0:
            rare_seen += 1
        total += b.n

    assert rare_seen == ROUNDS // RARE, (rare_seen, ROUNDS // RARE)
    assert total == sum(-i if i % RARE == RARE - 1 else i for i in range(ROUNDS)), total

    def maybe_bad(n):
        # Only ever reached after the loop is compiled, so the `TypeError` has to
        # come out of the resumed chain rather than out of tracing — and it is
        # raised by the level ABOVE the frame that returned the offending value.
        if n < 0:
            return n
        return None

    class BadOnRarePath:
        def __init__(self, n):
            self.n = n
            return maybe_bad(n)

    raised = 0
    built = 0
    for i in range(ROUNDS):
        v = -i if i % RARE == RARE - 1 else i
        try:
            obj = BadOnRarePath(v)
        except TypeError:
            raised += 1
        else:
            assert isinstance(obj, BadOnRarePath), (i, obj)
            assert obj.n == v, (i, obj.n, v)
            built += 1

    # `n == 0` is not negative, so iteration 0 builds; every rare iteration after
    # it raises.
    assert raised == ROUNDS // RARE, (raised, ROUNDS // RARE)
    assert built == ROUNDS - raised, (built, ROUNDS, raised)
inline_init_nested_call_deopt()
print("OK")
