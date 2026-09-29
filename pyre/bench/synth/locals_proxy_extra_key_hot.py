# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=probe_callee_extra,probe_plain
# pyre-check: trace-shape=probe_plain:present=Jump
# A key written through `frame.f_locals` that names no writable fast local must
# survive into the mapping `locals()` / `vars()` / `dir()` hand back, including
# when the trace folds those builtins.
#
# `framelocalsproxy_setitem` sends such a key to `FrameDebugData.w_extra_locals`
# and touches neither a fast slot nor `w_locals`, and
# `frame_locals_proxy_snapshot` appends that dict after the fastlocals.
# Both `locals()` folds rebuild the mapping WITHOUT it -- the callee arm from
# `CalleeLocalsShadow`, the portal arm from the virtualizable's slot array --
# so each needs its own gate on `w_extra_locals` rather than reading a null
# `w_locals` as "this frame carries no mapping".  Neither gate existed:
# compiled runs answered `['f', 'x', 'y']` where `PYRE_JIT=0` answered
# `['extra_key', 'f', 'x', 'y']`.
#
# The two folds are separate code paths reached by separate gates, so both are
# driven here: `probe_callee_extra` calls the builtin one frame in, where the
# tracer answers from the inlined callee's shadow, and `probe_portal_extra`
# calls it in the compiled loop's own frame, where it answers from the standard
# virtualizable.  `dir()` rides the same gate through its sorted-names tail.
#
# `probe_portal_extra` also pins the ORDER the two halves land in: the portal
# fold merged `f_extra_locals` before the fastlocal chain and answered
# `['extra_key', 'n', 'y', 'last', 'i']` where the interpreter -- and CPython --
# answer `['n', 'y', 'last', 'i', 'extra_key']`.
#
# `probe_plain` carries no proxy write at all, so the fold still FIRES there.
# It is what keeps the fixture honest: the two gates above are declines, and a
# fixture built only from declines passes just as well with the whole fold
# deleted.  The `spec-folds` header censuses `builtin_locals` and fails the run
# unless it fired at least once, which only `probe_plain` can supply.
import sys

# Stay above the production trace threshold; `selfcheck-compiles` and the
# `builtin_locals` fold census independently reject a vacuously cold run.
N = 3000
KEY = "extra_key"


def helper_plain(x):
    y = x + 1
    return sorted(locals())


def helper_extra(x):
    y = x + 1
    # Not a fast local of this frame, so it lands in `f_extra_locals`.
    sys._getframe().f_locals[KEY] = x
    return sorted(locals()), sorted(vars()), sorted(dir())


def probe_plain():
    last = None
    for _ in range(N):
        last = helper_plain(1)
    return last


def probe_callee_extra():
    last = None
    for i in range(N):
        last = helper_extra(i)
    return last


def probe_portal_extra(n):
    y = 0
    sys._getframe().f_locals[KEY] = 1
    last = None
    for i in range(n):
        y = i
        # Unsorted: `framelocalsproxy_keys` reports the localsplus slots in
        # `co_localsplusnames` order and only then walks `f_extra_locals`, so
        # the ORDER is part of the answer and a rebuild that merges the extras
        # first hands back a differently-keyed dict.
        last = list(locals())
    return last


def report(label, seen, expected):
    if seen != expected:
        print(f"FAIL {label}: {seen}")
        print(f"  expected: {expected}")
        return 1
    return 0


def main():
    rc = report("plain callee", probe_plain(), ["x", "y"])

    locals_names, vars_names, dir_names = probe_callee_extra()
    expected_callee = [KEY, "x", "y"]
    rc |= report("locals() in callee", locals_names, expected_callee)
    rc |= report("vars() in callee", vars_names, expected_callee)
    rc |= report("dir() in callee", dir_names, expected_callee)

    # `n` is the first localsplus slot, then the body's own three, then the
    # proxy-written key.
    expected_portal = ["n", "y", "last", "i", KEY]
    rc |= report(
        "locals() in the compiled loop's own frame",
        probe_portal_extra(N),
        expected_portal,
    )
    # The same frame walked once, so a fold that reorders the two halves is
    # separated from a snapshot that does.
    rc |= report(
        "locals() in an uncompiled loop's own frame",
        probe_portal_extra(1),
        expected_portal,
    )

    if rc:
        return rc
    print("PASS proxy-written extra locals survive the locals folds")
    return 0


sys.exit(main())
