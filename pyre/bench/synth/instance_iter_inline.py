# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=self_iter,nested_call,in_callee
# pyre-check: trace-shape=nested_call:absent=CallMayForceR
# `iter`'s instance arm dispatches a user class's `__iter__` and then runs
# `iter_check_is_iterator` on what came back.  Left residual the method is one
# opaque `CallMayForce` per `for` statement -- a real interpreter frame, plus
# the `ForceToken` / virtualref shell that forces the caller's frame around it
# -- even when the body is the `return self` every iterator class writes.
#
# `instance_iter` inlines the body in its place.  The check it owes is decided
# at record time against the class the receiver guard pins under the type's
# `_version_tag?`, and only for the shape where the body hands the receiver's
# own box back; anything else keeps the residual, which runs the check itself.
#
# The arms that must NOT take the fold -- a `__iter__` returning a fresh
# object, a `__iter__ = None`, a receiver with no `__next__`, and a mid-loop
# rebind -- live in `extra_tests/parity_tests/instance_iter_deopt_arms.py`.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

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


class Restarting:
    """`__iter__` resets the cursor before returning the receiver, so the body
    is more than a bare `return self` and still ends on the same box."""

    def __init__(self, limit):
        self.limit = limit
        self.j = 0

    def __iter__(self):
        self.j = 0
        return self

    def __next__(self):
        if self.j >= self.limit:
            raise StopIteration
        v = self.j
        self.j += 1
        return v


def self_iter():
    total = 0
    for _ in range(N):
        it = Counter(2)
        for x in it:
            total += x
    return total


def nested_call():
    src = Restarting(3)
    total = 0
    for _ in range(N):
        for x in src:
            total += x
    return total


def step(go):
    s = 0
    for x in go:
        s += x
        break
    return s


def in_callee():
    """The `for` lives in an inlined callee, the position `instance_next`
    needed its own green key for."""
    go = Counter(N * 10)
    acc = 0
    for _ in range(N):
        acc += step(go)
    return acc


def main():
    print("self_iter", self_iter())
    print("nested_call", nested_call())
    print("in_callee", in_callee())
    print("PASS instance iter inline")


main()
