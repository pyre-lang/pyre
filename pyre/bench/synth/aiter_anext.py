# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# aiter()/anext() builtins over an async iterator: aiter returns the async
# iterator, anext drives it item by item, and the two-argument anext yields the
# default once the iterator is exhausted. Also covers the three TypeError paths
# (not an async iterable, not an async iterator, __aiter__ returning a
# non-async-iterator). Coroutines are stepped by hand so no event loop is
# needed.
def drive(coro):
    try:
        while True:
            coro.send(None)
    except StopIteration as e:
        return e.value


class ARange:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= self.n:
            raise StopAsyncIteration
        v = self.i
        self.i += 1
        return v


def main():
    it = ARange(3)
    ait = aiter(it)
    print("aiter returns self:", ait is it)

    out = []
    while True:
        try:
            out.append(drive(anext(ait)))
        except StopAsyncIteration:
            break
    print("collected:", out)

    print("anext default:", drive(anext(ait, "DONE")))
    print("anext default2:", drive(anext(ait, 42)))

    try:
        aiter(object())
    except TypeError as e:
        print("aiter err:", type(e).__name__)

    try:
        anext(object())
    except TypeError as e:
        print("anext err:", type(e).__name__)

    class Bad:
        def __aiter__(self):
            return 123

    try:
        aiter(Bad())
    except TypeError as e:
        print("aiter bad-return err:", type(e).__name__)


main()
