# CPython-suite gap: the suite does not run hot user `__next__` methods through
# changed guards, non-function descriptors, polymorphic classes, or try edges.
# parity-tests reason: this guards pyre's FOR_ITER inline admission and resume
# state without duplicating the same hot-loop fixture across five files.
#
# for_iter_user_next_exhaustion:
# the suite does not exhaust hot user iterators through StopIteration subclasses or nested
# Python `__next__` frames. this guards pyre's inlined FOR_ITER exhaustion mapping across
# explicit, deep, and residual StopIteration paths.

"""User `__next__` effects, dispatch, and exceptions survive FOR_ITER resume."""


# An effect before a changed branch guard is not replayed.
effects = []


class SwitchingIterator:
    def __init__(self, limit, switch_at):
        self.index = 0
        self.limit = limit
        self.switch_at = switch_at

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        value = self.index
        self.index += 1
        effects.append(value)
        return value + 1 if value < self.switch_at else value - 1


def consume_switching(limit, switch_at):
    total = 0
    for value in SwitchingIterator(limit, switch_at):
        total += value
    return total


for _ in range(12):
    consume_switching(1600, 1200)
assert len(effects) == 12 * 1600


# Non-function descriptors decline the user-function inline route.
class ClassMethodIterator:
    remaining = 0

    def __init__(self, count):
        type(self).remaining = count

    def __iter__(self):
        return self

    @classmethod
    def __next__(cls):
        if cls.remaining == 0:
            raise StopIteration
        cls.remaining -= 1
        return cls.remaining


def builtin_iterator(count):
    source = iter(range(count))

    class BuiltinIterator:
        def __iter__(self):
            return self

        __next__ = source.__next__

    return BuiltinIterator()


class NextCallable:
    def __init__(self):
        self.remaining = 0

    def __call__(self):
        if self.remaining == 0:
            raise StopIteration
        self.remaining -= 1
        return self.remaining


class CallableIterator:
    __next__ = NextCallable()

    def __init__(self, count):
        type(self).__next__.remaining = count

    def __iter__(self):
        return self


def count_at_most(iterator, limit):
    count = 0
    for _ in iterator:
        count += 1
        if count == limit:
            break
    return count


for _ in range(12):
    assert count_at_most(ClassMethodIterator(1600), 1600) == 1600
    assert count_at_most(builtin_iterator(1600), 1600) == 1600
    assert count_at_most(CallableIterator(1600), 1600) == 1600


# A residual exception after advancing propagates once without replay.
advances = []


class ResidualRaisingIterator:
    def __init__(self, limit, raise_at):
        self.index = 0
        self.limit = limit
        self.raise_at = raise_at

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        advances.append(self.index)
        if self.index == self.raise_at:
            int("not-an-integer")
        return self.index


def consume_residual(limit, raise_at):
    count = 0
    for _ in ResidualRaisingIterator(limit, raise_at):
        count += 1
    return count


for _ in range(8):
    assert consume_residual(1600, 2000) == 1600
advances.clear()
try:
    consume_residual(1600, 1300)
except ValueError:
    pass
else:
    raise AssertionError("residual ValueError was swallowed")
assert len(advances) == 1300


# One hot FOR_ITER site dispatches by class, not merely physical layout.
class Ascending:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        return self.index


class Descending:
    def __init__(self, limit):
        self.index = limit
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= 0:
            raise StopIteration
        self.index -= 1
        return self.index


class Tagging:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        return "t%d" % self.index


def collect(iterator):
    seen = []
    for value in iterator:
        seen.append(value)
    return seen


limit = 1600
ascending = list(range(1, limit + 1))
descending = list(range(limit - 1, -1, -1))
for _ in range(8):
    assert collect(Ascending(limit)) == ascending
for _ in range(8):
    assert collect(Descending(limit)) == descending
    assert collect(Ascending(limit)) == ascending
    assert collect(Tagging(4)) == ["t1", "t2", "t3", "t4"]


# FOR_ITER's materialized catch reaches the enclosing try/finally correctly.
handler_advances = []


class HandlerRaisingIterator:
    def __init__(self, limit, raise_at, exc):
        self.index = 0
        self.limit = limit
        self.raise_at = raise_at
        self.exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        handler_advances.append(self.index)
        if self.index == self.raise_at:
            raise self.exc("boom")
        return self.index


def consume_caught(limit, raise_at, exc):
    total = 0
    caught = None
    try:
        for value in HandlerRaisingIterator(limit, raise_at, exc):
            total += value
    except ValueError as error:
        caught = ("ValueError", str(error))
    except TypeError as error:
        caught = ("TypeError", str(error))
    return total, caught


for _ in range(12):
    assert consume_caught(2000, 0, ValueError) == (2000 * 2001 // 2, None)
for exc_type in (TypeError, ValueError):
    handler_advances.clear()
    total, caught = consume_caught(2000, 1500, exc_type)
    assert caught == (exc_type.__name__, "boom")
    assert total == 1499 * 1500 // 2
    assert len(handler_advances) == 1500


def consume_finally(limit, raise_at):
    marks = []
    try:
        for _ in HandlerRaisingIterator(limit, raise_at, ValueError):
            pass
    except ValueError:
        marks.append("caught")
    finally:
        marks.append("finally")
    return marks


for _ in range(12):
    assert consume_finally(1200, 0) == ["finally"]
assert consume_finally(1200, 900) == ["caught", "finally"]


def consume_escapes(limit, raise_at):
    try:
        for _ in HandlerRaisingIterator(limit, raise_at, KeyError):
            pass
    except ValueError:
        return "wrong"
    return "no-raise"


for _ in range(12):
    assert consume_escapes(1200, 0) == "no-raise"
try:
    consume_escapes(1200, 700)
except KeyError:
    pass
else:
    raise AssertionError("unhandled iterator exception was swallowed")

def for_iter_user_next_exhaustion():
    """User-defined iterator exhaustion reaches the owning FOR_ITER exactly once."""

    class MyStop(StopIteration):
        pass


    class SubclassStoppingIterator:
        def __init__(self, limit):
            self.index = 0
            self.limit = limit

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= self.limit:
                raise MyStop
            self.index += 1
            return self.index


    class DeepStoppingIterator:
        def __init__(self, limit):
            self.index = 0
            self.limit = limit

        def __iter__(self):
            return self

        def _advance(self):
            if self.index >= self.limit:
                raise StopIteration
            self.index += 1
            return self.index

        def __next__(self):
            return self._advance()


    def count_explicit(iterator):
        count = 0
        for value in iterator:
            assert value == count + 1
            count += 1
        return count


    for _ in range(12):
        assert count_explicit(SubclassStoppingIterator(1600)) == 1600
        assert count_explicit(DeepStoppingIterator(1600)) == 1600


    class Wrap:
        def __init__(self, inner):
            self._it = inner

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)


    def sum_wrapped(n):
        total = 0
        for value in Wrap(iter(range(n))):
            total += value
        return total


    expected = 20000 * 19999 // 2
    for _ in range(12):
        assert sum_wrapped(20000) == expected


    def count_nested(n):
        count = 0
        for _ in Wrap(Wrap(iter(range(n)))):
            count += 1
        return count


    for _ in range(12):
        assert count_nested(4000) == 4000


    class StopEarly:
        def __init__(self, inner, stop_at):
            self._it = inner
            self._stop_at = stop_at
            self._seen = 0

        def __iter__(self):
            return self

        def __next__(self):
            value = next(self._it)
            self._seen += 1
            if value == self._stop_at:
                raise StopIteration
            return value


    for _ in range(12):
        early = StopEarly(iter(range(20000)), 1500)
        assert sum(1 for _ in early) == 1500
        assert early._seen == 1501
for_iter_user_next_exhaustion()

# An outer function's first call inlines a loop-owning callee whose `for`
# pulls from a user `__next__` and from a generator expression.
# The thresholds make the outer call the first to drain the callee.


def _sum_loop(seq, start=0):
    total = start
    for item in seq:
        total = total + item
    return total


def _sum_through_outer(seq):
    return _sum_loop(seq)


class _OddCounter:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        value = 2 * self.i + 1
        self.i += 1
        return value


try:
    import pypyjit
    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass
for _ in range(2):
    assert _sum_through_outer(_OddCounter(3000)) == 9000000
    assert _sum_through_outer(2 * i + 1 for i in range(3000)) == 9000000
try:
    import pypyjit
    pypyjit.set_param("default")
except ImportError:
    pass

print("OK")
