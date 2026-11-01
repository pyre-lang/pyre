# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# A compiled loop later fails a guard, and a `sys.settrace` hook is owed the
# resumed frame's `return` event.
#
# `pyframe.py execute_frame` is two nested `try`s: `call_trace` in the outer
# `try`, `return_trace` in the inner `finally`, `leave` in the outer `finally`.
# A frame resumed mid-body sits inside the inner `try`, so on the way out it
# still owes `return_trace` — including when the callback raises, with
# `w_exitvalue` still None on the exceptional path.  `continue_entered_frame`
# is that resume door (`portal_blackhole_recursive_result`, the CALL_ASSEMBLER
# force/CRN legs in `call_jit.rs`): enter and `call_trace` already ran, and
# skipping `return_trace` drops the event and any exception the callback
# raised.
#
# THE SHAPE IS THE TEST: warmup compiles `hot` on ints, then a later call
# flips a loop variable to float so the compiled type guard fails and the
# rest of the frame is resumed.  The tracer is armed mid-loop from a callee
# via `sys._getframe(1)` (same door as `settrace_f_trace_armed_mid_loop`) so
# the compiled body is entered first; a tracer installed before entry would
# arm `f_trace` in `call_trace` and never reach the resume path.  It records
# `('return', name, value)`.  A missing event is a silent empty list; a hook
# exception that vanished is a None where the callback's message should be.
import sys

WARM = 20000  # past the loop threshold (1039) many times over
TAIL = 100
FLIP_AT = 20
EXPECTED_VALUE = float(TAIL)

events = []
raise_on_return = False


def tracer(frame, event, arg):
    if event == 'return' and frame.f_code.co_name == 'hot':
        events.append(('return', frame.f_code.co_name, arg))
        if raise_on_return:
            raise RuntimeError('boom at return')
    # Returning the hook arms `f_trace`, which is what makes the `return`
    # arm reachable at all -- `return_trace` fires on `gettrace()`.
    return tracer


def arm():
    frame = sys._getframe(1)
    frame.f_trace = tracer
    sys.settrace(tracer)


def hot(n, flip):
    x = 1
    s = 0
    i = 0
    while i < n:
        s = s + x
        i = i + 1
        if flip and i == FLIP_AT:
            arm()
            x = 1.0
    return s


def run(do_raise):
    global raise_on_return
    del events[:]
    raise_on_return = do_raise
    hot(WARM, False)
    raised = None
    try:
        hot(TAIL, True)
    except RuntimeError as exc:
        raised = str(exc)
    finally:
        sys.settrace(None)
        raise_on_return = False
    return list(events), raised


def main():
    failures = []
    seen, raised = run(False)
    print(seen)
    print(raised)
    if seen != [('return', 'hot', EXPECTED_VALUE)]:
        failures.append(
            'return events were %r, expected the resumed frame to report '
            "('return', 'hot', %r)" % (seen, EXPECTED_VALUE)
        )
    if raised is not None:
        failures.append('unraised path leaked %r' % (raised,))

    seen_r, raised_r = run(True)
    print(seen_r)
    print(raised_r)
    if seen_r != [('return', 'hot', EXPECTED_VALUE)]:
        failures.append(
            'raising return hook events were %r, expected the event to be '
            'recorded before the callback raised' % (seen_r,)
        )
    if raised_r != 'boom at return':
        failures.append(
            'raising return hook escaped %r, expected the callback exception '
            'to replace the body result' % (raised_r,)
        )

    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS settrace return event after jit resume')
    return 0


sys.exit(main())
