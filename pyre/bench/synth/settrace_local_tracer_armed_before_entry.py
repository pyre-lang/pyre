# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# Self-checking guard for the other half of the local-tracing contract: a
# global tracer that RETURNS ITSELF, so `_trace` (executioncontext.py) installs
# it as `w_f_trace` on every frame it sees a `call` event for, and the loop
# inside that frame then runs with local tracing already armed.  A global hook
# that returns `None` declines local tracing and only ever fires frame-level
# events, so it cannot observe any of this -- which is why the events below
# were not covered by anything before.
#
# SCOPE -- the hook is armed BEFORE the frame is entered, which is what the
# name says. `execute_frame` (`pyframe.py`) runs `call_trace` then `dispatch`
# unconditionally; a global tracer that returns itself installs `w_f_trace`
# in that `call_trace`, and `ll_portal_runner` (`warmspot.py ll_portal_runner`)
# then dispatches the armed frame through the portal like any other. Line
# events come from `dispatch_bytecode`'s jitted arm (`bytecode_only_trace`).
# Measured: it passes identically with and without the `debugdata` guard its
# sibling fixtures were added for, so it discriminates nothing about that fix
# and must not be read as covering it.
#
# In particular it does NOT cover the open GC abort (`GC BUG: invalid type_id
# ... site=marking_regray_root`).  That one needs local tracing armed on a
# frame that is ALREADY compiled: `f_trace` set on the frames up the stack from
# inside an already-hot loop, via a SIGALRM handler, with the tracer returning
# itself.  A tracer returning `None` is clean and so is a global-only one --
# that is the discriminating rung, and none of it is what this file does.
#
# Only the loop is measured.  The frame-entry and frame-exit events are
# deliberately outside the window: pyre instruments the `RESUME` at pc 0 and so
# reports one extra `line` event at the `def` line, with and without the JIT,
# and that belongs to a separate open defect rather than to this one.  What is
# stated here is exact where it counts -- one `line` event on the body
# statement per iteration, no more and no fewer.
import sys

N = 60000

events = []


def tracer(frame, event, arg):
    if frame.f_code.co_name == "hot":
        events.append((event, frame.f_lineno - frame.f_code.co_firstlineno))
    return tracer


def hot(n):                      # +0
    acc = 0                      # +1
    for i in range(n):           # +2
        acc += i                 # +3
    return acc                   # +4


def main():
    sys.settrace(tracer)
    try:
        acc = hot(N)
    finally:
        sys.settrace(None)

    failures = []
    if acc != N * (N - 1) // 2:
        failures.append(f"acc: got {acc!r}, want {N * (N - 1) // 2!r}")

    # Counted with `list.count` rather than a `for` loop on purpose.  A
    # bookkeeping loop here is long enough to become hot and compile, and
    # `run_selfcheck`'s JIT floor is corpus-level (`loops_compiled >= 1`), not
    # "the loop under test compiled" -- so it would be satisfied by this loop
    # even if the loop under test did not compile, which is the vacuous pass
    # the floor exists to catch.
    counts = {
        key: events.count(key)
        for key in (("line", 3), ("line", 2), ("call", 0), ("return", 4))
    }
    body = counts.get(("line", 3), 0)
    if body != N:
        failures.append(f"loop body: got {body!r} line events, want exactly {N!r}")
    # The loop header runs once per iteration plus once for the exhausted
    # iterator.  How many `line` events that is depends on where the back edge
    # lands: the header fires again for a backwards jump onto a lower
    # instruction (`run_trace_func`'s `frame.last_instr <
    # d.instr_prev_plus_one`), and whether the jump and the FOR_ITER share the
    # line is a property of the compiled bytecode -- cpython 3.14.6 reads
    # N + 1 where pypy3 7.3.22 reads more.  Only the lower bound is portable.
    header = counts.get(("line", 2), 0)
    if header < N + 1:
        failures.append(f"loop header: got {header!r} line events, want at least {N + 1!r}")
    for want in (("call", 0), ("return", 4)):
        if counts.get(want, 0) != 1:
            failures.append(f"{want!r}: got {counts.get(want, 0)!r}, want 1")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS local tracer returning itself, armed before entry")
    return 0


sys.exit(main())
