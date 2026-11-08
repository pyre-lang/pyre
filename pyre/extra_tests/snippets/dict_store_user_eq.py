# pyre-check: gate=1
"""User-key dict and module-dict stores: equality, a raising probe, a mutation.

The hot loop is what the JIT traces. The per-key probe stays a residual leaf.
"""

import gc
import types


class Key:
    def __init__(self, k):
        self.k = k

    def __hash__(self):
        return hash(self.k)

    def __eq__(self, other):
        return isinstance(other, Key) and self.k == other.k

    def __repr__(self):
        return f"Key({self.k})"


class Boom:
    def __init__(self, k, explode=False):
        self.k = k
        self.explode = explode

    def __hash__(self):
        return 7

    def __eq__(self, other):
        if self.explode or getattr(other, "explode", False):
            raise ValueError("boom in __eq__")
        return isinstance(other, Boom) and self.k == other.k


class Mut:
    def __init__(self, k, target=None):
        self.k = k
        self.target = target
        self.n = 0

    def __hash__(self):
        return 1

    def __eq__(self, other):
        self.n += 1
        if self.target is not None and self.n == 1:
            self.target.clear()
        return isinstance(other, Mut) and self.k == other.k


def store(mapping, key, value):
    mapping[key] = value


acc = 0
box = {}
for i in range(400):
    store(box, i, i)
    acc += box[i]
print("hot", acc, len(box))

left = {}
store(left, Key(1), "a")
store(left, Key(2), "b")
store(left, Key(1), "c")
print("user", sorted((item.k, value) for item, value in left.items()))

module = types.ModuleType("m")
store(module.__dict__, Key(3), "d")
store(module.__dict__, Key(4), "e")
store(module.__dict__, "name", "kept")
picked = []
for item, value in module.__dict__.items():
    if isinstance(item, Key):
        picked.append((0, item.k, value))
    elif item == "name":
        picked.append((1, item, value))
print("module", sorted(picked))

try:
    store({Boom(1): 1}, Boom(1, explode=True), 2)
    print("raise", "swallowed")
except ValueError as exc:
    print("raise", str(exc))


def mutated():
    box = {}
    first = Mut(1)
    store(box, first, "a")
    second = Mut(2)
    second.target = box
    store(box, Mut(1), "b")
    try:
        store(box, second, "c")
    except RuntimeError as exc:
        return "runtime", str(exc)
    return "ok", sorted(item.k for item in box)


print("mutate", mutated())


class CollectingKey:
    def __init__(self, k):
        self.k = k

    def __hash__(self):
        gc.collect()
        return hash(self.k)

    def __eq__(self, other):
        return isinstance(other, CollectingKey) and self.k == other.k


def collected():
    module = types.ModuleType("c")
    space = vars(module)
    store(space, CollectingKey(0), "switch")
    for i in range(1, 30):
        store(space, CollectingKey(i), [i] * 3)
    total = 0
    for item, value in space.items():
        if isinstance(item, CollectingKey) and item.k:
            total += sum(value)
    return total, space[CollectingKey(0)]


print("collect", collected())

# Hash/eq of a large int against Decimal after a short warmup.
# `blackhole_if_trace_too_long` raises `ABORT_TOO_LONG` rather than
# continuing the inlined helper past `trace_limit`.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

from decimal import Decimal as D


def check_int_decimal(i):
    d = D(i)
    assert hash(i) == hash(d) and i == d


for i in range(80):
    check_int_decimal(i)
check_int_decimal(2**32)
print("tower", 2**32)
