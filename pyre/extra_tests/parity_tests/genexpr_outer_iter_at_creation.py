# CPython-suite gap: test_generators checks the creation-time TypeError only
# once, never from a hot loop or a closure-carrying generator expression.
# parity-tests reason: the generator prefix runs before the generator exists.

"""A generator expression acquires its outermost iterator when created,
with `async for` as well."""


def make(s):
    return (n for n in s)


def make_closure(s, k):
    return (n + k for n in s)


def frames_of(exc):
    names = []
    tb = exc.__traceback__
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return names


for i in range(300):
    try:
        make(i)
    except TypeError as e:
        assert str(e) == "'int' object is not iterable", e
        assert frames_of(e) == ["<module>", "make"], frames_of(e)
    else:
        raise AssertionError("no TypeError at creation")

    g = make_closure([i, i + 1], 2)
    assert g.gi_frame.f_lineno == make_closure.__code__.co_firstlineno + 1
    assert list(g) == [i + 2, i + 3]


calls = []


class Iterable:
    def __iter__(self):
        calls.append("iter")
        return iter([1, 2])


g = make(Iterable())
assert calls == ["iter"], calls
assert next(g) == 1
assert calls == ["iter"], calls
assert list(g) == [2]

class AsyncIterable:
    def __aiter__(self):
        calls.append("aiter")
        return self

    def __init__(self):
        self.left = [1, 2]

    async def __anext__(self):
        if not self.left:
            raise StopAsyncIteration
        return self.left.pop(0)


async def make_async(s):
    return (n async for n in s)


async def drain(ag):
    return [n async for n in ag]


def run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise AssertionError("coroutine suspended")


for i in range(300):
    try:
        run(make_async(i))
    except TypeError as e:
        assert "__aiter__" in str(e), e
    else:
        raise AssertionError("no TypeError at async creation")

calls.clear()
ag = run(make_async(AsyncIterable()))
assert calls == ["aiter"], calls
assert run(drain(ag)) == [1, 2]

print("OK")
