# CPython-suite gap: `test_sys_settrace` traces loops and it traces raising
# code, but never a loop whose iterator ends by raising StopIteration from a
# Python-level `__next__` while a tracer is installed.  A runtime that
# swallowed that StopIteration without reporting it would pass the whole
# module.
# pyre-check: pypy-diverges: SEND reports StopIteration in the yield-from frame only for a generator or coroutine delegate; pypy3 also reports a Python __next__ iterator and AsyncGenASend
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


def send_yield_from_a_generator_reports_in_that_frame():
    """SEND's StopIteration event is the generator-or-coroutine arm only.

    FOR_ITER reports a generator iterator, a Python `__next__` with a
    traceback, and AsyncGenASend.  SEND does not: `INSTRUMENTED_END_SEND` /
    `monitor_stop_iteration` fires in the `yield from` / `await` frame only
    when the delegate is a generator or an exact coroutine.  Measured python3.15.0rc2:

        yield from inner()                 -> ('g_gen', 'StopIteration')
        await coro_inner()                 -> ('coro_outer', 'StopIteration')
        yield from It() Python __next__    -> only ('__next__', 'StopIteration')
        await a.__anext__()                -> no StopIteration in the consumer

    pypy3 reports the last two in the consumer as well, because
    `next_yield_from` calls `_report_stopiteration_sometimes` unchanged.
    The FOR_ITER rows above stay on that helper; these four rows pin SEND.
    """

    def inner():
        return 7
        yield

    def g_gen():
        r = yield from inner()
        assert r == 7

    def consume():
        for _ in g_gen():
            pass

    seen = exception_events_in(consume)
    assert ('g_gen', 'StopIteration') in seen, seen


def send_await_a_coroutine_reports_in_that_frame():
    async def coro_inner():
        return 3

    async def coro_outer():
        r = await coro_inner()
        return r

    def drive(c):
        try:
            c.send(None)
        except StopIteration as e:
            return e.value

    seen = exception_events_in(lambda: drive(coro_outer()))
    assert ('coro_outer', 'StopIteration') in seen, seen


def send_yield_from_a_python_next_does_not_report_in_the_consumer():
    class It:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration(5)

    def g_it():
        r = yield from It()
        assert r == 5

    def consume():
        for _ in g_it():
            pass

    seen = exception_events_in(consume)
    assert ('g_it', 'StopIteration') not in seen, seen
    assert ('__next__', 'StopIteration') in seen, seen


def send_await_asend_does_not_report_stopiteration_in_the_consumer():
    async def agen():
        yield 1

    async def use_asend():
        a = agen()
        v = await a.__anext__()
        try:
            await a.__anext__()
        except StopAsyncIteration:
            pass
        return v

    def drive(c):
        try:
            c.send(None)
        except StopIteration as e:
            return e.value

    seen = exception_events_in(lambda: drive(use_asend()))
    assert ('use_asend', 'StopIteration') not in seen, seen


a_python_level_next_reports_to_the_consuming_frame()
the_rule_is_the_traceback_and_not_the_iterator()
a_natively_signalled_end_reports_nothing()
the_loop_still_ends_normally()
a_warm_generator_loop_reports_its_end()
a_line_callback_arms_opcode_events_for_that_instruction()
a_line_callback_jump_backwards_rereads_last_instr()
send_yield_from_a_generator_reports_in_that_frame()
send_await_a_coroutine_reports_in_that_frame()
send_yield_from_a_python_next_does_not_report_in_the_consumer()
send_await_asend_does_not_report_stopiteration_in_the_consumer()
print('OK')
