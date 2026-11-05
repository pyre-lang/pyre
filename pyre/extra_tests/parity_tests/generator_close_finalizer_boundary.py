# CPython-suite gap: test_close_clears_frame asserts `__del__` before close()
# returns (refcount); test_close_releases_frame_locals uses gc_collect after
# close.  Neither grades a retained gi_frame's locals, nor the raising close()
# messages across a yield-from chain.
# parity-tests reason: close() drops the generator's frame so a later
# collection finalizes unretained locals; a retained gi_frame keeps them;
# RuntimeError "generator ignored GeneratorExit" and ValueError "boom" stay
# the close() messages.

"""close() releases the frame's locals so a collection can finalize them.

An unretained local with `__del__` runs after close() and three gc.collect()
calls (`support.gc_collect`).  A retained `gi_frame` keeps those locals.
Raising close() arms keep their message: RuntimeError "generator ignored
GeneratorExit", ValueError "boom".
"""

import gc


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
    gc.collect()
    gc.collect()
    gc.collect()
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
    gc.collect()
    gc.collect()
    gc.collect()
    deleted = "unstarted_unretained" in finalized
    print(f"unstarted_unretained deleted={deleted}")
    assert deleted is True


check_close_keeps_retained_frame_locals()
print("OK")
