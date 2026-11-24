# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=outer,outer_arming
# `inner` does not compile here on PyPy either -- the profile hook reads the
# traced `inner` frame's `f_code`, `_nonstandard_virtualizable` forces it,
# and the trace ends `ABORT_ESCAPE`.
# `c_call` carries the frame that made the call, and for a builtin called from
# inside a Python callee that is the CALLEE's frame -- never the caller whose
# compiled trace would have inlined it.
#
# The hazard is real and structural: a walker-inlined callee has no frame on
# `topframeref`/`f_backref` while compiled code runs, and
# `call_jit.rs residual_call_c_profile_frame` reads exactly that chain.  A
# builtin residual firing inside an inlined callee would therefore report the
# inliner's frame.  Two independent guards make the conjunction unreachable,
# and this fixture pins both:
#
#   * At RECORD time `inline_call.rs` declines on `ec_hook_installed()` for any
#     installed hook, so a trace recorded under a profiler inlines nothing and
#     the callee gets a real frame.  `armed_before` is that arm.
#   * At EXECUTE time `is_being_profiled` is an `interp_jit.py` portal green, so
#     a profiled frame cannot enter a trace compiled under the unprofiled key;
#     and a frame that entered before the hook was armed keeps
#     `is_being_profiled` false, so the residual's own third test declines --
#     which is what `call_valuestack` does upstream, gating on the same
#     per-frame flag.  `armed_mid_loop` is that arm: it compiles `outer` with
#     `inner` inlined and only then arms the profiler, from inside `inner`.
#
# A relaxation of the inline decline breaks the first guard, and this fixture is
# what says so.  Measured on cpython 3.14.6 and pypy3, which agree on every
# line below.
import sys

NAME = 'abcdef'
WARM = 75000
REPEAT = 37500
MID_REPEAT = 4500


def inner(x):
    return len(NAME) + x


def outer(n):
    t = 0
    for _ in range(n):
        t = inner(t) % 1000003
    return t


TRIGGER = [None]


def inner_arming(x, hook):
    if x == TRIGGER[0]:
        sys.setprofile(hook)
    return len(NAME) + x


def outer_arming(n, hook):
    t = 0
    for _ in range(n):
        t = inner_arming(t, hook) % 1000003
    return t


def collect():
    seen = {}

    def hook(frame, event, arg):
        if event in ('c_call', 'c_return'):
            key = (event, getattr(arg, '__name__', repr(arg)), frame.f_code.co_name)
            seen[key] = seen.get(key, 0) + 1

    return seen, hook


def armed_before():
    outer(WARM)
    seen, hook = collect()
    sys.setprofile(hook)
    try:
        outer(REPEAT)
    finally:
        sys.setprofile(None)
    return seen, REPEAT


def armed_mid_loop():
    seen, hook = collect()
    outer_arming(WARM, hook)
    TRIGGER[0] = 0
    try:
        outer_arming(MID_REPEAT, hook)
    finally:
        sys.setprofile(None)
    return seen, MID_REPEAT


def check(arm, seen, want_frame, want_count, failures):
    # `sys.setprofile` is the instrument: both the arming and the disarming
    # call are builtin calls made inside the window they open, from whichever
    # frame happens to make them.
    seen = {key: n for key, n in seen.items() if key[1] != 'setprofile'}
    expected = {
        ('c_call', 'len', want_frame): want_count,
        ('c_return', 'len', want_frame): want_count,
    }
    for key, want in sorted(expected.items()):
        got = seen.get(key, 0)
        if got != want:
            failures.append('%s: %s = %d, expected %d' % (arm, key, got, want))
    for key in sorted(seen):
        if key not in expected:
            failures.append(
                '%s: %s = %d, expected no event — the reported frame is not the '
                'one that made the call' % (arm, key, seen[key])
            )


def main():
    failures = []
    seen, n = armed_before()
    check('armed before', seen, 'inner', n, failures)
    seen, n = armed_mid_loop()
    check('armed mid loop', seen, 'inner_arming', n, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS c_call reports the callee frame, not the inliner')
    return 0


sys.exit(main())
