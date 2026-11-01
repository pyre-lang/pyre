# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# Self-checking regression guard for the cost of walking a MODULE dict's view.
#
# `celldict.py getiterkeys`/`getitervalues` are lazy iterators upstream.
# Pyre's dict-view iterator cannot hold one — the GC-object layout has no slot
# for it — so it keeps an integer cursor and asks the strategy for the `index`-th
# entry per step (`DictStrategy::nth_item`).  The trait default answers that by
# materialising `items()` and taking `.nth(index)`, which the doc calls "fine for
# the tiny empty strategies"; `ModuleDictStrategy` had no override, and its
# `items` wraps EVERY name with `w_str_new`.  One walk of an `n`-name module dict
# therefore performed `n * n` wraps, and `w_str_new` allocates through
# `malloc_raw` — immortal, so nothing reclaims them.
#
# Measured before the override: 350 names walked ten times grew peak RSS by
# 156.8 MB; CPython grows none.  Removing it also took 10.3 MB off interpreter
# startup (63.9 -> 53.6 MB peak RSS, 48.0 -> 37.7 MB physical footprint), which
# is `dir(module)` inside `importlib._bootstrap` paying the same quadratic.  The
# threshold below is deliberately far above the fixed interpreter floor and far
# below the quadratic figure, so it fails on the defect without tracking
# allocator noise.
#
# The equality half is the correctness guard for the O(1) cursor itself: a
# `nth_item` that indexes the storage directly must agree, entry for entry and in
# order, with the materialised `items()` it replaced, and a `values()` view --
# which skips the name wrap entirely -- must agree with the same walk's values.
import sys
import types

NAMES = 350
ROUNDS = 10
GROWTH_LIMIT_MB = 40.0


def peak_rss_mb():
    """`None` when the platform cannot answer, which turns the memory half of
    this fixture off rather than guessing a unit.

    The wasm guest imports `resource` but raises out of `getrusage` -- it has no
    host to ask -- so the call itself has to be inside the guard, not just the
    import."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # ru_maxrss is bytes on darwin and kilobytes everywhere else.
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / scale


def warm_jit_metadata():
    # Initialize process-wide compiler metadata before measuring view growth.
    # A separate function keeps selfcheck-compiles=main meaningful: main must
    # still compile during the measured walk, within the unchanged RSS budget.
    total = 0
    for i in range(6000):
        total += i
    return total


def main():
    module = types.ModuleType("module_dict_view_probe")
    for i in range(NAMES):
        setattr(module, "name_%04d" % i, i)
    namespace = module.__dict__

    failures = []

    materialised = list(namespace.items())
    walked = [(key, value) for key, value in namespace.items()]
    if walked != materialised:
        failures.append("view walk disagrees with items(): %r" % (walked[:4],))
    if [value for value in namespace.values()] != [v for _, v in materialised]:
        failures.append("values() view disagrees with items()")
    if [key for key in namespace] != [k for k, _ in materialised]:
        failures.append("key iteration disagrees with items()")

    warm_jit_metadata()
    before = peak_rss_mb()
    seen = 0
    for _ in range(ROUNDS):
        for key, value in namespace.items():
            seen += 1
    # `namespace` carries the module's own dunders on top of the names set
    # above, so compare against the materialised length rather than NAMES.
    if seen != len(materialised) * ROUNDS:
        failures.append("walked %d entries, expected %d" % (seen, len(materialised) * ROUNDS))

    if before is not None:
        grew = peak_rss_mb() - before
        if grew > GROWTH_LIMIT_MB:
            failures.append(
                "module-dict view iteration grew peak RSS by %.1f MB (limit %.1f)"
                % (grew, GROWTH_LIMIT_MB)
            )

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS module-dict view iteration")
    return 0


sys.exit(main())
