# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=settrace_direct,settrace_nested,setprofile_direct,setprofile_nested,entry-bridge:callee,entry-bridge:wrapper
# The two `root:` arms are the premise, not a relaxation: the events this fixture
# counts are owed precisely because compiled code enters the callee past
# `execute_frame`'s bracket.  A callee that stopped reaching the JIT would make
# every count below pass without testing anything.
# Self-checking guard for the events a callee owes a hook when the loop that
# calls it is running compiled.
#
# `pyframe.py PyFrame.execute_frame` brackets every Python call with
# `ec.call_trace(self)` and `ec.return_trace(self, w_exitvalue)`, and
# `executioncontext.py ExecutionContext.leave` adds `_trace(frame,
# 'leaveframe', ...)` when a profiler is installed.  A tracing JIT reaches a
# callee through doors that begin at the callee's first bytecode -- an inlined
# body, or a direct jump into the callee's own compiled trace -- and every one
# of them starts past that bracket.  Upstream keeps the events anyway because
# it traces THROUGH `execute_frame`: the `gettrace()` reads become guards in
# the compiled loop, so a hook installed later fails one.  An implementation
# that synthesizes the callee entry instead has to answer for them some other
# way, or the loop runs to completion with the hook installed and silent.
#
# THE TAIL IS THE TEST, AND SO IS RUNNING IT TWICE.  The loss is total rather
# than partial: the hook is installed mid-loop, and what arrives is the handful
# of events from the re-warmup before compiled code takes over again -- a
# number set by the JIT's own threshold, NOT by how long the tail is.  So each
# column is asserted at two tails, and a count that reads the same at both is
# reported as the blind-forever signature it is.  Measured before the fix:
# 1043 at a 2500 tail and 1043 at a 10000 one, against exactly the tail on
# cpython 3.14.6 and on pypy3 7.3.22.
#
# The `nested` column is the control and it is on the other side of the line:
# the loop calls a wrapper, and the counted callee is invoked from THAT frame,
# which is interpreted.  It reports on every implementation, which is what
# makes a short `direct` column attributable to the door compiled code enters a
# callee through and not to the setter, the tail length or the counting rule.
#
# `sys.settrace` and `sys.setprofile` are both exercised because they are
# answered by different machinery: a profiler is part of the portal driver's
# green key (`interp_jit.py greens = ['next_instr', 'is_being_profiled',
# 'pycode']`) so arming one mints a different cell, while a trace function has
# no green and needs the compiled loop to be carrying a guard on the slot
# `gettrace()` reads.  An implementation that answers one and not the other
# passes exactly one pair of columns here.
import sys

WARM = 20000  # past the JIT's compilation threshold many times over
TAILS = (2500, 10000)

calls = 0


def hook(frame, event, arg):
    global calls
    if event == 'call' and frame.f_code.co_name == 'callee':
        calls += 1
    return hook


def callee(x):
    return x + 1


def wrapper(x):
    return callee(x)


# One loop per column, spelled out rather than generated.  Two reasons, both
# load-bearing.  A column that reused another column's loop would inherit
# whatever state the previous arm left on it -- a green key already
# deoptimized, a code object already marked don't-trace-here -- and could then
# report every event for the plainest reason of all: never having compiled; a
# fresh name is a fresh green key (`interp_jit.py greens` carries `pycode`).
# And a loop built by `exec` into a fresh dict is NOT the same subject: its
# globals are a plain dict rather than a module's, which is its own JIT path,
# and measured, every column built that way reads 2 -- including the control
# that must pass.  Four near-identical bodies is the price of four columns
# that are each the shape this file is about.


def settrace_direct(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.settrace(hook)
        acc = callee(acc)
        i += 1
    return acc


def settrace_nested(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.settrace(hook)
        acc = wrapper(acc)
        i += 1
    return acc


def setprofile_direct(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.setprofile(hook)
        acc = callee(acc)
        i += 1
    return acc


def setprofile_nested(n, arm_at):
    acc = 0
    i = 0
    while i < n:
        if i == arm_at:
            sys.setprofile(hook)
        acc = wrapper(acc)
        i += 1
    return acc


def run(loop, unsetter, tail):
    global calls
    calls = 0
    loop(WARM + tail, WARM)
    unsetter(None)
    return calls


# Inlined callee with the hook armed inside so the CALL compiles before
# `ec_hook_installed`. settrace misses the first return (`f_trace` unset
# on that frame); setprofile reports every return. A doubled leave lists
# the first profile return twice.
ADOPT_N = 200
MIDBODY_N = 8
adopt_trace_returns = []
adopt_profile_returns = []
midbody_trace_returns = []
midbody_profile_returns = []


def _return_hook(name, dest, ret_self):
    def hook(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == name:
            dest.append(arg)
        if ret_self:
            return hook
    return hook


adopt_tracer = _return_hook('adopt_leaf', adopt_trace_returns, True)
adopt_profiler = _return_hook('adopt_leaf', adopt_profile_returns, False)
midbody_tracer = _return_hook('midbody_leaf', midbody_trace_returns, True)
midbody_profiler = _return_hook('midbody_leaf', midbody_profile_returns, False)


def adopt_leaf(x):
    sys.settrace(adopt_tracer)
    sys.setprofile(adopt_profiler)
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    i = h + 9
    j = i + 10
    return x + 1 + (j - j)


def adopt_loop(n):
    s = 0
    i = 0
    while i < n:
        s += adopt_leaf(i)
        i += 1
    return s


def midbody_leaf(x):
    # Long body so TraceTooLong latches past the callee's entry merge point.
    # Later calls take the short path once the hook is live.
    if sys.gettrace() is not None:
        return x + 1
    s = x
    s = s + 1
    s = s + 2
    s = s + 3
    s = s + 4
    s = s + 5
    s = s + 6
    s = s + 7
    s = s + 8
    s = s + 9
    s = s + 10
    s = s + 11
    s = s + 12
    s = s + 13
    s = s + 14
    s = s + 15
    s = s + 16
    s = s + 17
    s = s + 18
    s = s + 19
    s = s + 20
    s = s + 21
    s = s + 22
    s = s + 23
    s = s + 24
    s = s + 25
    s = s + 26
    s = s + 27
    s = s + 28
    s = s + 29
    s = s + 30
    i = 0
    t = 0
    while i < 24:
        t = t + i
        t = t + 1
        t = t + 2
        i = i + 1
    sys.settrace(midbody_tracer)
    sys.setprofile(midbody_profiler)
    return x + 1 + (s - s) + (t - t)


def midbody_loop(n):
    acc = 0
    i = 0
    while i < n:
        acc += midbody_leaf(i)
        i += 1
    return acc


def _run_traced(loop, n, trace_limit):
    import atexit

    atexit._ncallbacks()
    try:
        import pypyjit
    except ImportError:
        pypyjit = None
    if pypyjit is not None:
        pypyjit.set_param(
            'trace_limit=%d,threshold=1,function_threshold=1' % trace_limit
        )
    try:
        return loop(n)
    finally:
        sys.settrace(None)
        sys.setprofile(None)
        if pypyjit is not None:
            pypyjit.set_param('default')


def _check_exact_returns(label, total, n, traces, profiles):
    expected_trace = list(range(2, n + 1))
    expected_profile = list(range(1, n + 1))
    failures = []
    if total != n * (n + 1) // 2:
        failures.append(
            '%s loop returned %r, expected %r' % (label, total, n * (n + 1) // 2)
        )
    if traces != expected_trace:
        failures.append(
            '%s settrace return events were %r, expected %r'
            % (label, traces, expected_trace)
        )
    if profiles != expected_profile:
        failures.append(
            '%s setprofile return events were %r, expected %r'
            % (label, profiles, expected_profile)
        )
    return failures


def check_adopted_return_trace():
    del adopt_trace_returns[:]
    del adopt_profile_returns[:]
    total = _run_traced(adopt_loop, ADOPT_N, 70)
    return _check_exact_returns(
        'adopt', total, ADOPT_N, adopt_trace_returns, adopt_profile_returns
    )


def check_midbody_adopted_return_trace():
    del midbody_trace_returns[:]
    del midbody_profile_returns[:]
    total = _run_traced(midbody_loop, MIDBODY_N, 180)
    return _check_exact_returns(
        'midbody', total, MIDBODY_N, midbody_trace_returns, midbody_profile_returns
    )


def main():
    columns = [('settrace/direct', settrace_direct, sys.settrace),
               ('settrace/nested', settrace_nested, sys.settrace)]
    setprofile = getattr(sys, 'setprofile', None)
    if setprofile is not None:
        columns.append(('setprofile/direct', setprofile_direct, setprofile))
        columns.append(('setprofile/nested', setprofile_nested, setprofile))

    failures = []
    for name, loop, unsetter in columns:
        seen = [run(loop, unsetter, tail) for tail in TAILS]
        if seen == list(TAILS):
            continue
        failures.append('%s: got %r for tails %r' % (name, seen, list(TAILS)))
        if seen[0] == seen[1]:
            failures.append(
                '%s: the count does not track the tail, so the loop is blind '
                'from the arming onward' % (name,)
            )
    failures.extend(check_adopted_return_trace())
    failures.extend(check_midbody_adopted_return_trace())
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a callee called from a compiled loop reports to a hook')
    return 0


sys.exit(main())
