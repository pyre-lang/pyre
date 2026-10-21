# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=fmt
# Recursive fmt carries a per-frame name across an inlined listcomp loop.
# Two producers of that name: `type(v).__name__`, and a guarded
# `"list" if type(v) is list else "tuple"` Const. After JIT warm-up the
# compiled FOR_ITER loop must keep the name of the *current* fmt frame.
#
#     target/release/pyre-dynasm pyre/bench/synth/listcomp_recursive_fmt_typename_resume.py
#
# Correct: fails == 0.  The pre-fix binary mismatches around round 167.

N = 250


def fmt(v, name):
    if isinstance(v, (list, tuple)):
        return "%s%s" % (name(v), [fmt(x, name) for x in v])
    return repr(v)


def typename(v):
    return type(v).__name__


def tag(v):
    return "list" if type(v) is list else "tuple"


def drive(name):
    return [fmt(x, name) for x in [(1, [1, 1]), (2, [2, 2])]]


EXPECTED = [
    "tuple['1', \"list['1', '1']\"]",
    "tuple['2', \"list['2', '2']\"]",
]


def run():
    fails = 0
    for name in (typename, tag):
        for _ in range(N):
            if drive(name) != EXPECTED:
                fails += 1
    return fails


fails = run()
if fails == 0:
    print("PASS recursive fmt listcomp keeps inner type name")
else:
    print(f"FAIL inner type name: {fails} mismatches")
    raise SystemExit(1)
