# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# The empty compile set is measured, not a relaxation: `PYRE_LOOP_CENSUS=1`
# emits no `[loop-census]` line for this file. The defect is on residual
# portal entries (`portal_activation_*`) — they consumed the resume payload
# before entering their own bracket, which every frame reaching those entries
# does whether or not anything compiled — so warming a loop would add nothing
# to the guard. The discriminator is the JIT build against `PYRE_JIT=0`, not
# compiled against interpreted: `PYRE_JIT=0` reports all six events and the
# default build reported two.
#
# A delegating generator / coroutine is resumed with a hook installed, and the
# activation of the DELEGATING frame is the one being counted.
#
# `pyframe.py execute_frame` puts the resume INSIDE the bracket:
#
#     executioncontext.enter(self)
#     try:
#         executioncontext.call_trace(self)
#         try:
#             try:
#                 next_instr = self.resume_execute_frame(w_arg_or_err)
#             except pyopcode.Yield:
#                 w_exitvalue = self.popvalue()
#         finally:
#             executioncontext.return_trace(self, w_exitvalue)
#     finally:
#         executioncontext.leave(self, w_exitvalue, got_exception)
#
# Native `execute_frame` puts the resume inside that bracket. Residual
# `portal_activation_*` entries used to consume the resume payload BEFORE
# their own bracket, and a suspended `yield from` / `await` delegate that
# yields again finishes the resumption there and returns. That return jumped
# `call_trace`, `return_trace`, `leaveframe_trace` and the frame-chain leave
# in one step.
#
# TWO DEFECTS, ONE HOIST, and the fixture checks both because they fail
# differently:
#   * LOSS. When the delegate yields again, the delegating frames report
#     nothing at all. Measured on the `await` arm: 2 of the 6 events cpython
#     3.14, pypy3 and `PYRE_JIT=0` all report.
#   * ORDER. When the delegate instead raises StopIteration, the bracket does
#     run — but the delegate's own Python already ran ahead of it, so the
#     callee's whole call/return pair is reported OUTSIDE the caller's. A
#     pairing profiler mis-nests: it sees a `return` for a frame it never got a
#     `call` for. Measured on the `yield from` arm as
#     `['call/inner', 'return/inner', 'call/outer', 'return/outer']`.
#
# THE ORDER IS THE TEST for the second arm, not the multiset: both orders carry
# the same four events, so a fixture that counted them would pass while the
# nesting was inverted.
import sys


def record(events, names):
    def hook(frame, event, arg):
        name = frame.f_code.co_name
        if name in names:
            events.append(event + '/' + name)
        return hook

    return hook


def yield_from_arm():
    def inner():
        yield 1
        return

    def outer():
        yield from inner()
        yield 'after'

    gen = outer()
    gen.send(None)
    events = []
    sys.setprofile(record(events, ('outer', 'inner')))
    try:
        gen.send(None)
    finally:
        sys.setprofile(None)
    return events


def await_arm():
    class Fut:
        def __await__(self):
            yield
            yield
            return 7

    async def inner():
        return await Fut()

    async def outer():
        return await inner()

    coro = outer()
    coro.send(None)
    events = []
    sys.setprofile(record(events, ('outer', 'inner', '__await__')))
    try:
        coro.send(None)
    except StopIteration:
        pass
    finally:
        sys.setprofile(None)
    return events


def check(label, got, expected, failures):
    if got != expected:
        failures.append('%s: %s, expected %s' % (label, got, expected))


def main():
    failures = []
    # The delegate raises StopIteration, so `outer` resumes and completes: the
    # bracket runs, and `inner`'s pair must sit INSIDE `outer`'s.
    check(
        'yield from',
        yield_from_arm(),
        ['call/outer', 'call/inner', 'return/inner', 'return/outer'],
        failures,
    )
    # The delegate yields again, so the whole stack re-suspends: every frame
    # still owes its pair, innermost last out.
    check(
        'await',
        await_arm(),
        [
            'call/outer',
            'call/inner',
            'call/__await__',
            'return/__await__',
            'return/inner',
            'return/outer',
        ],
        failures,
    )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a delegating resume reports its activation bracket, nested')
    return 0


sys.exit(main())
