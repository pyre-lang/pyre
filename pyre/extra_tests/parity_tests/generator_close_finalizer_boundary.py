# pyre-check: pypy-diverges: unretained close() runs `__del__` before it
# returns; pypy3's tracing GC reaches that local later. A retained
# `gi_frame` keeps the locals on both sides.
# CPython-suite gap: test_yield_from exercises close() on a generator that
# yields from its finally, but only with locals that hold no finalizer, so the
# prompt-finalization collection this path can take is never entered there.
# parity-tests reason: pyre runs a whole-heap collection at close()'s
# finalization boundary, and that collection may only run once the propagation
# root is consumed; an arm that leaves an exception in flight must skip it.
#
# close_keeps_retained_frame_locals:
# test_close_clears_frame never retains `gi_frame` across `close()`.  A
# retained frame keeps its locals (`arg`, `x`); an unretained frame's
# objects run `__del__` before `close()` returns.  `frame.clear()` is the
# path that drops the slots.

"""close() propagates its error across the finalization boundary.

The generator's frame holds an object with `__del__`, so releasing the frame
releases a graph that can run application code.  Two of close()'s outcomes
raise rather than return: the generator that yields again after GeneratorExit
becomes RuntimeError, and one that raises from its finally propagates that
exception.  Both must survive the boundary with their message intact.
"""


class Finalized:
    def __del__(self):
        pass


def ignores_generator_exit():
    keeper = Finalized()

    def inner():
        assert keeper is not None
        try:
            yield 1
        finally:
            yield 2

    def outer():
        return (yield from inner())

    return outer()


def raises_from_finally():
    keeper = Finalized()

    def inner():
        assert keeper is not None
        try:
            yield 1
        finally:
            raise ValueError("boom")

    def outer():
        return (yield from inner())

    return outer()


gen = ignores_generator_exit()
assert next(gen) == 1
try:
    gen.close()
except RuntimeError as exc:
    assert str(exc) == "generator ignored GeneratorExit", str(exc)
else:
    raise AssertionError("close() must raise RuntimeError")

# The generator is exhausted, and reading it again must not revisit the
# released graph.
try:
    next(gen)
except StopIteration as exc:
    assert exc.value is None
else:
    raise AssertionError("next() must raise StopIteration")

gen = raises_from_finally()
assert next(gen) == 1
try:
    gen.close()
except ValueError as exc:
    assert str(exc) == "boom", str(exc)
else:
    raise AssertionError("close() must propagate ValueError")


finalized = []


class D:
    def __init__(self, label):
        self.label = label

    def __del__(self):
        finalized.append(self.label)


def gen_with_arg(arg):
    x = 123
    yield


def check_close_keeps_retained_frame_locals():
    g = gen_with_arg(D("suspended_retained"))
    next(g)
    retained = g.gi_frame
    g.close()
    names = sorted(retained.f_locals)
    x = retained.f_locals["x"]
    deleted = "suspended_retained" in finalized
    print(f"suspended_retained names={names} x={x} deleted={deleted}")
    assert names == ["arg", "x"], names
    assert x == 123, x
    assert deleted is False
    retained = None
    g = None

    g = gen_with_arg(D("suspended_unretained"))
    next(g)
    g.close()
    deleted = "suspended_unretained" in finalized
    print(f"suspended_unretained deleted={deleted}")
    assert deleted is True
    g = None

    g = gen_with_arg(D("unstarted_retained"))
    retained = g.gi_frame
    g.close()
    names = sorted(retained.f_locals)
    deleted = "unstarted_retained" in finalized
    print(f"unstarted_retained names={names} deleted={deleted}")
    assert names == ["arg"], names
    assert deleted is False
    retained = None
    g = None

    g = gen_with_arg(D("unstarted_unretained"))
    g.close()
    deleted = "unstarted_unretained" in finalized
    print(f"unstarted_unretained deleted={deleted}")
    assert deleted is True


check_close_keeps_retained_frame_locals()
print("OK")
