# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=argument_shapes,hot,missing_window
# pyre-check: trace-shape=argument_shapes:max=CallMayForceR:12
# Self-checking guard for the pins under an inlined callee's keyword-only
# defaults.
#
# Seeding those locals no longer probes `__kwdefaults__` per call.  The mapping
# a definition builds carries a version, so the walker bakes each entry's cell,
# pins `Function.w_kw_defs` with a `GuardValue`, records a quasi-immutable
# marker on the mapping's strategy version, drains that marker with a
# `GuardNotInvalidated`, and reads the cell's field live.  Three separate
# mechanisms therefore stand between a mutation and the next call, and a census
# can only say the fold fired -- not that the pins are honest.  Each site below
# moves one of them and prints a wrong number if that one is missing.
#
# Sites:
#   A  overwrite an existing entry in place.  A cell that already holds an
#      object absorbs the store and bumps no version, so nothing invalidates:
#      this is answered by the live field read alone, and a fold that baked the
#      VALUE rather than the cell keeps returning the old one.
#   B  delete an entry, then add it back.  Both move the strategy version, and
#      the deleted window is observable on its own -- the call must raise
#      rather than seed a stale default.
#   C  rebind the attribute.  `f.__kwdefaults__ = d` stores `d` itself, so only
#      the `Function.w_kw_defs` guard can catch this; the version marker is
#      still watching the old mapping, which no longer answers anything.
#   D  keep calling after that rebind.  A plain dict has no version to pin, so
#      the resolve declines and the callee seeds through the ordinary path --
#      which must still see a later in-place mutation.
N = 20000


def positional_defaults(a, b=3, c=5):
    return a * 100 + b * 10 + c


def object_defaults(a, b="x", c=None):
    return a, b, c


def array_defaults(a, b=3, c=5, d=7):
    return a * 1000 + b * 100 + c * 10 + d


def float_defaults(a, b=0.5, c=0.25):
    return a + b * 10 + c * 100


def positional_only(a, /, b=3, c=5):
    return a * 100 + b * 10 + c


def keyword_shapes(a, *rest, p=7, q=11):
    return a, rest, p, q


class Holder:
    def method(self, a, *, p=13):
        return a, p


def argument_shapes(n):
    """All positional/keyword default slot shapes at one compiled owner."""
    bad = 0
    holder = Holder()
    for i in range(n):
        bad += positional_defaults(1) != 135
        bad += positional_defaults(1, c=7) != 137
        bad += object_defaults(1, c="z") != (1, "x", "z")
        # Three defaults use the array-backed tuple representation; leaving
        # holes reaches non-zero element indexes in that block.
        bad += array_defaults(1, d=8) != 1358
        bad += array_defaults(1, 4, d=8) != 1458
        # The specialised float pair deliberately stays residual.
        bad += float_defaults(1.0, c=0.5) != 56.0
        bad += positional_only(1, c=7) != 137
        bad += keyword_shapes(i) != (i, (), 7, 11)
        bad += keyword_shapes(i, i + 1, p=9) != (i, (i + 1,), 9, 11)
        bad += holder.method(i) != (i, 13)
    return bad


def g(x, *, step=1, tag="a"):
    return x + step, tag


def missing_window(n):
    """Site B's deleted window, in a loop of its own so it compiles too.

    Left inside `main` this ran interpreted no matter what the header
    declared -- `selfcheck-compiles` names code objects, and `main` is not
    one of them -- so the one site that asks a compiled trace to REFUSE a
    baked default was the one site no compiled trace ever saw.
    """
    missing = 0
    for i in range(n):
        try:
            g(i)
        except TypeError:
            missing += 1
    return missing


def hot(n, want_step, want_tag):
    """Run the loop hot and count answers that disagree with the mapping."""
    bad = 0
    for i in range(n):
        value, tag = g(i)
        if value != i + want_step or tag != want_tag:
            bad += 1
    return bad


def main():
    if argument_shapes(N):
        print("FAIL argument default shapes disagreed")
        return 1

    # Warm first so the sites below mutate a compiled loop rather than an
    # interpreted one.
    if hot(N, 1, "a"):
        print("FAIL warm-up disagreed with the declared defaults")
        return 1

    g.__kwdefaults__["step"] = 5
    bad = hot(N, 5, "a")
    if bad:
        print(f"FAIL site A missed an in-place overwrite on {bad} of {N} calls")
        return 1

    del g.__kwdefaults__["tag"]
    missing = missing_window(N)
    if missing != N:
        print(f"FAIL site B seeded a deleted default on {N - missing} of {N} calls")
        return 1

    g.__kwdefaults__["tag"] = "z"
    bad = hot(N, 5, "z")
    if bad:
        print(f"FAIL site B missed the re-added default on {bad} of {N} calls")
        return 1

    g.__kwdefaults__ = {"step": 9, "tag": "q"}
    bad = hot(N, 9, "q")
    if bad:
        print(f"FAIL site C missed the rebound mapping on {bad} of {N} calls")
        return 1

    g.__kwdefaults__["step"] = 11
    bad = hot(N, 11, "q")
    if bad:
        print(f"FAIL site D missed a mutation after the rebind on {bad} of {N} calls")
        return 1

    print("PASS kwdefaults invalidation")
    return 0


import sys

sys.exit(main())
