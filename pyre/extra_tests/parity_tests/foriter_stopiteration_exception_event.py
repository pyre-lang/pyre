# CPython-suite gap: `test_sys_settrace` traces loops and it traces raising
# code, but never a loop whose iterator ends by raising StopIteration from a
# Python-level `__next__` while a tracer is installed.  A runtime that
# swallowed that StopIteration without reporting it would pass the whole
# module.
#
# parity-tests reason: FOR_ITER's exhaustion arm catches the StopIteration and
# jumps, so the consuming frame never dispatches it and never reaches the
# code that reports an exception to the tracer.  Whether an `exception` event
# is delivered there is therefore a decision the exhaustion arm has to make on
# its own, and it is one a debugger sees: `pdb` stops on `exception` events.
#
# The rule is not "always" and not "never".  3.14 reports the event exactly
# when the StopIteration carries a traceback -- which is to say, when it was
# raised by Python code rather than signalled by an iterator implemented
# natively.  A list ending, a range ending and a dict ending all deliver
# nothing; a class whose `__next__` raises delivers one.  That asymmetry is
# what this pins: reporting always would fire on every `for` over a list, and
# reporting never would lose the one case a debugger cares about.
#
# A generator's ending carries no traceback and is the other half of the rule:
# an event is reported because the iterator is a generator.  3.14 answers a
# cold loop differently -- the first execution runs the unspecialised
# `_FOR_ITER`, whose exhaustion arm jumps past `END_FOR` without reporting,
# and only once the loop specialises to `FOR_ITER_GEN` does the end arrive at
# `INSTRUMENTED_END_FOR` where `monitor_stop_iteration` mints it.  So the
# warm loop is what is pinned here: it is the answer 3.14 settles on, the
# answer pypy3 gives from the first call, and the only one of the two that
# does not depend on which tier is running.
import sys


def exception_events_in(consumer):
    """The 'exception' events a tracer sees while `consumer` runs."""
    seen = []

    def record(frame, event, arg):
        if event == 'exception':
            seen.append((frame.f_code.co_name, arg[0].__name__))
        return record

    sys.settrace(record)
    try:
        consumer()
    finally:
        sys.settrace(None)
    return seen


def a_python_level_next_reports_to_the_consuming_frame():
    class Ending:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    def consume():
        for _ in Ending():
            raise AssertionError('the iterator was not empty')

    seen = exception_events_in(consume)
    # Two events: the raise inside `__next__`, and the report FOR_ITER makes
    # in the frame running the loop.  The second is the one under test.
    assert seen == [('__next__', 'StopIteration'), ('consume', 'StopIteration')], seen


def the_rule_is_the_traceback_and_not_the_iterator():
    """`map`'s `__next__` is native, but the StopIteration is not."""

    def stop(_):
        raise StopIteration

    def consume():
        for _ in map(stop, [1, 2]):
            raise AssertionError('the map was not empty')

    # The exhaustion came out of a native `__next__`, so a rule that keyed off
    # the iterator's kind would report nothing here.  It travelled through
    # `stop`'s frame on the way, which is what the consuming frame reports on.
    assert exception_events_in(consume) == [
        ('stop', 'StopIteration'),
        ('consume', 'StopIteration'),
    ], 'unexpected events'


def a_natively_signalled_end_reports_nothing():
    def consume_list():
        for _ in [1, 2]:
            pass

    def consume_range():
        for _ in range(2):
            pass

    def consume_dict():
        for _ in {'a': 1}:
            pass

    def consume_callable_iter():
        box = [0]

        def step():
            box[0] += 1
            return box[0]

        # `iter(callable, sentinel)` ends by comparing, not by raising through
        # `step`, so this is a native end even though `step` is Python.
        for _ in iter(step, 3):
            pass

    for consumer in (
        consume_list,
        consume_range,
        consume_dict,
        consume_callable_iter,
    ):
        assert exception_events_in(consumer) == [], consumer.__name__


def the_loop_still_ends_normally():
    class Ending:
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            return self

        def __next__(self):
            if self.n == 0:
                raise StopIteration
            self.n -= 1
            return self.n

    collected = []

    def consume():
        for item in Ending(3):
            collected.append(item)

    # The report must not disturb what the loop collects, and must not leave
    # the StopIteration visible to the loop body.
    assert exception_events_in(consume) == [
        ('__next__', 'StopIteration'),
        ('consume', 'StopIteration'),
    ], 'unexpected events'
    assert collected == [2, 1, 0], collected


def a_warm_generator_loop_reports_its_end():
    def empty():
        return
        yield

    def consume():
        for _ in empty():
            raise AssertionError('the generator yields nothing')

    # Repeated so the loop is warm on the last call, which is the one graded:
    # 3.14 reports nothing on the first and reports from the second onwards.
    rows = [exception_events_in(consume) for _ in range(8)]
    assert rows[-1] == [('consume', 'StopIteration')], rows


def events_of(fn, on_event):
    """`(event, lineno-rel)` for `fn` only, plus `fn`'s return value."""
    seen = []
    first = fn.__code__.co_firstlineno

    def tracer(frame, event, arg):
        if frame.f_code is not fn.__code__:
            return tracer
        seen.append((event, frame.f_lineno - first))
        on_event(frame, event, arg)
        return tracer

    sys.settrace(tracer)
    try:
        result = fn()
    finally:
        sys.settrace(None)
    return seen, result


def a_line_callback_arms_opcode_events_for_that_instruction():
    """`run_trace_func` reads `f_trace_opcodes` after the line event."""

    def traced():
        a = 1
        b = 2
        return b

    def on_event(frame, event, arg):
        if event == 'line' and frame.f_lineno == traced.__code__.co_firstlineno + 1:
            frame.f_trace_opcodes = True

    seen, result = events_of(traced, on_event)
    print('opcode-arm events:', seen)
    assert result == 2, result
    assert seen == [
        ('call', 0),
        ('line', 1),
        ('opcode', 1),
        ('opcode', 1),
        ('line', 2),
        ('opcode', 2),
        ('opcode', 2),
        ('line', 3),
        ('opcode', 3),
        ('opcode', 3),
        ('return', 3),
    ], seen


def a_line_callback_jump_backwards_rereads_last_instr():
    """`run_trace_func` writes `instr_prev_plus_one` from live `last_instr`."""

    def traced():
        n = 0
        n += 1
        n += 10
        return n

    first = traced.__code__.co_firstlineno
    jumped = {'done': False}

    def on_event(frame, event, arg):
        if jumped['done']:
            return
        if event == 'line' and frame.f_lineno == first + 3:
            frame.f_lineno = first + 2
            jumped['done'] = True

    seen, result = events_of(traced, on_event)
    print('lineno-jump events:', seen)
    print('lineno-jump result:', result)
    assert result == 12, result
    assert seen == [
        ('call', 0),
        ('line', 1),
        ('line', 2),
        ('line', 3),
        ('line', 3),
        ('line', 4),
        ('return', 4),
    ], seen


a_python_level_next_reports_to_the_consuming_frame()
the_rule_is_the_traceback_and_not_the_iterator()
a_natively_signalled_end_reports_nothing()
the_loop_still_ends_normally()
a_warm_generator_loop_reports_its_end()
a_line_callback_arms_opcode_events_for_that_instruction()
a_line_callback_jump_backwards_rereads_last_instr()
print('OK')
