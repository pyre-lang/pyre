# CPython-suite gap: test_sys_settrace installs its tracer before the traced
# function has run enough to be compiled, so every case it covers is the
# already-hooked order; nothing in the suite warms a loop first and installs a
# tracer afterwards, which is the only order in which a compiled trace can
# outlive the hook's installation.
# parity-tests reason: `sys.settrace` does not change the portal green — `call_trace`
# sets `is_being_profiled` only for a *profile* function — so an already
# compiled trace keeps running, and the callee it inlined reports no `call`
# event at all until a guard on `ExecutionContext.w_tracefunc` leaves it.
#
# jit_compiled_frame_leaves_on_f_trace_assignment:
# test_sys_settrace assigns `f_trace` only from a tracer the `call` event already
# installed, on a frame that has run a handful of bytecodes; nothing in the suite assigns
# it to a frame that has been looping long enough to be compiled, which is the only order
# in which a compiled trace can outlive the assignment. `bytecode_only_trace` opens on
# `frame.get_w_f_trace() is None`, and compiled code makes that call rather than running

"""A compiled trace has to leave when `sys.settrace` is installed.

The tracer returns `None` from every `call` event, so it claims no frame and
`frame_tracing_active` lets the loop keep compiling — what this measures is the
compiled path's own frame events rather than the interpreter's.

The loop is warmed with no hook installed, so its callee is inlined into a
trace recorded against a null `w_tracefunc`.  Installing a tracer afterwards
does not change the portal green — `call_trace` sets `is_being_profiled` only
when a *profile* function is installed — so nothing but a guard on that field
can stop the recorded trace from running on, and an inlined callee that runs
inside compiled code reports no `call` event at all.
"""
import sys

WARM = 4000
N = 2000

seen = []


def tracer(frame, event, arg):
    if frame.f_code.co_name == "callee":
        seen.append(event)
    return None


def callee(x):
    return x + 1


def hot(n):
    total = 0
    for i in range(n):
        total += callee(i)
    return total


hot(WARM)
sys.settrace(tracer)
try:
    hot(N)
finally:
    sys.settrace(None)

calls = seen.count("call")
print("callee call events:", calls)
assert calls == N, "callee reported %d call events, expected %d" % (calls, N)

def jit_compiled_frame_leaves_on_f_trace_assignment():
    """Assigning `f_trace` to a running compiled frame has to leave the trace.

    The loop is warmed with a global tracer already installed that returns `None`
    from every `call` event, so it claims no frame: every `w_f_trace` stays null
    and the loop compiles, with the callee emitted as a residual rather than
    inlined.  The callee then reaches back with `sys._getframe(1)` and hands the
    looping frame its own trace function, which is what `_trace` reads for every
    event but `'call'` — from that iteration on, `line` events are due.
    """

    import sys

    WARM = 4000
    N = 2000

    seen = []


    def watcher(frame, event, arg):
        # Claims nothing: a `None` return leaves every frame's `f_trace` null, so
        # the loop below keeps compiling while this is installed.
        return None


    def recorder(frame, event, arg):
        seen.append(event)
        return recorder


    def claim(i, at):
        if i == at:
            sys._getframe(1).f_trace = recorder
        return i


    def hot(n, at):
        total = 0
        for i in range(n):
            total += claim(i, at)
        return total


    sys.settrace(watcher)
    try:
        hot(WARM, -1)
        hot(N, N // 2)
    finally:
        sys.settrace(None)

    lines = seen.count("line")
    print("line events after the assignment:", lines >= N // 4)
    assert lines >= N // 4, "hot reported %d line events, expected at least %d" % (
        lines,
        N // 4,
    )
jit_compiled_frame_leaves_on_f_trace_assignment()
print("OK")
