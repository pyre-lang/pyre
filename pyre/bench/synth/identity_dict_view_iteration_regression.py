# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,walk
# Self-checking regression guard for the cost of walking an IDENTITY dict's view.
#
# Same trait-default trap as `module_dict_view_iteration_regression.py`, in the
# other strategy that never overrode it.  The dict-view iterator keeps an integer
# cursor and asks the strategy for the `index`-th entry per step
# (`DictStrategy::nth_item`); the default answers by materialising the whole
# `items()` and taking `.nth(index)`.  `IdentityDictStrategy` -- what a dict
# switches to when its keys compare by identity, i.e. any ordinary instance
# (`identitydict.py`, selected at `dictmultiobject.py`) -- had no
# override, so one walk of an `n`-entry dict built and threw away `n` vectors of
# `n` pairs.
#
# Unlike the module-dict case those vectors are ordinary Rust heap, so the defect
# shows as time, not as retained RSS.  Measured before the override, walking a
# freshly built dict of 2000 entries: str keys 1.0x, int 1.3x, bytes 2.0x,
# `__hash__`-defining instances 1.0x, tuples 1.0x -- and identity keys 55.2x.
# The ratio grows with n (140x at n = 4000), which is the quadratic term.
#
# The gate is that ratio rather than a wall-clock budget, so it is stable across
# machines and load: both walks are the same loop over the same number of
# entries, and after the fix both are O(1) per step.
#
# ORDER IS LOAD-BEARING.  The first `len()`, `bool()` or `list()` on an identity
# dict promotes it to `ObjectDictStrategy`, which always had the override -- so
# every later walk is fast whatever this strategy does.  The promotion is O(n)
# and visible: on a 2000-entry dict the first `len()` costs 0.000288s and every
# one after 0.000002s, against 0.000002s for the first `len()` of a str-keyed
# dict.  A fixture that checks correctness before timing therefore measures the
# object strategy and passes on the defect.  The timing runs on a pristine dict;
# the correctness checks get their own.  A plain walk and `d[k]` do not promote.
import sys
import time

N = 4000
REPEATS = 3
# Post-fix the identity walk is an object-strategy walk, and the pre-fix census
# put every non-quadratic key type between 0.9x and 2.0x, so 8x leaves four-fold
# headroom while still failing the defect with the JIT off -- where the slower
# interpreted str-keyed baseline compresses the ratio to about 11x.
RATIO_LIMIT = 8.0


class Key:
    """No `__hash__` / `__eq__` override, which is what makes instances compare
    by identity and selects the strategy under test."""


def walk(mapping):
    seen = 0
    for _ in mapping:
        seen += 1
    return seen


def best_walk_time(mapping):
    """Min of REPEATS after one warm-up walk, so a trace compiled mid-measure
    cannot decide the result.  Deliberately never calls `len()`."""
    walk(mapping)
    best = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        walk(mapping)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def main():
    failures = []

    # timing half, on dicts nothing has touched yet
    identity = {Key(): i for i in range(N)}
    text = {"k%06d" % i: i for i in range(N)}
    t_text = best_walk_time(text)
    t_identity = best_walk_time(identity)
    ratio = t_identity / t_text if t_text > 0 else float("inf")
    if ratio > RATIO_LIMIT:
        failures.append(
            "identity-dict walk is %.1fx a str-keyed walk of the same size "
            "(limit %.1fx, n=%d)" % (ratio, RATIO_LIMIT, N)
        )

    # correctness half, on its own dict
    # An O(1) cursor that indexes the storage directly must still walk every
    # entry exactly once, in insertion order, and agree across the three views.
    fresh = {Key(): i for i in range(N)}
    keys = list(fresh)
    if len(keys) != N or len(set(map(id, keys))) != N:
        failures.append(
            "walk yielded %d keys, %d distinct, expected %d"
            % (len(keys), len(set(map(id, keys))), N)
        )
    elif [fresh[k] for k in keys] != list(range(N)):
        failures.append("walk lost insertion order")
    if list(fresh.values()) != list(range(N)):
        failures.append("values() view disagrees with the key walk")
    if [v for _, v in fresh.items()] != list(fresh.values()):
        failures.append("items() view disagrees with values()")
    if [k for k, _ in fresh.items()] != keys:
        failures.append("items() view disagrees with the key walk")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS identity-dict view iteration")
    return 0


sys.exit(main())
