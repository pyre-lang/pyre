# CPython-suite gap: generator tests do not hot-loop PEP 479 over mixed-MRO stop subclasses.
# parity-tests reason: pins generator and async-generator subclass translation off the synth gate.

"""PEP 479 converts StopIteration subclasses in every relevant MRO order."""

class Sub(StopIteration):
    pass


class SV(StopIteration, ValueError):
    pass


class VS(ValueError, StopIteration):
    pass


class SubAsync(StopAsyncIteration):
    pass


class VSA(ValueError, StopAsyncIteration):
    pass


class Exhausted:
    def __init__(self, stop_type):
        self.stop_type = stop_type

    def __iter__(self):
        return self

    def __next__(self):
        raise self.stop_type


def g_subclass(stop_type, leak):
    yield 1
    if leak:
        next(Exhausted(stop_type))
    raise stop_type


async def ag_subclass(stop_type):
    yield 1
    raise stop_type


def finish(awaitable):
    try:
        while True:
            awaitable.send(None)
    except StopIteration as stop:
        return stop.value


def subclass_paths():
    for _ in range(2000):
        # Consumer spellings are covered above; this loop owns MRO selection.
        for stop_type in (Sub, SV, VS):
            for leak in (False, True):
                try:
                    list(g_subclass(stop_type, leak))
                except RuntimeError as exc:
                    assert type(exc.__cause__) is stop_type
                    assert str(exc) == "generator raised StopIteration"
                else:
                    raise AssertionError("StopIteration subclass escaped")
        for stop_type, message in (
            (StopAsyncIteration, "async generator raised StopAsyncIteration"),
            (SubAsync, "async generator raised StopAsyncIteration"),
            (VSA, "async generator raised StopAsyncIteration"),
            (Sub, "async generator raised StopIteration"),
            (VS, "async generator raised StopIteration"),
        ):
            iterator = ag_subclass(stop_type)
            assert finish(iterator.__anext__()) == 1
            try:
                finish(iterator.__anext__())
            except RuntimeError as exc:
                assert type(exc.__cause__) is stop_type
                assert str(exc) == message
            else:
                raise AssertionError("async generator stop escaped")


subclass_paths()


def _join(left, right):
    if left:
        return left + "/" + right
    return right


def iterdir_tree(node, encode=False):
    # Two YIELD_VALUE opcodes, like glob._iterdir's fsencode if/else.
    for name, children in node:
        if encode:
            yield name, children
        else:
            yield name, children


def listdir_tree(node, dir_fd, dironly):
    return list(iterdir_tree(node))


def rlistdir_kw(node, dir_fd, dironly, include_hidden=False):
    names = listdir_tree(node, dir_fd, dironly)
    for name, children in names:
        if include_hidden or not name.startswith("."):
            yield name
            for y in rlistdir_kw(
                children, dir_fd, dironly, include_hidden=include_hidden
            ):
                yield _join(name, y)


def rlistdir_pos(node, dir_fd, dironly, include_hidden=False):
    names = listdir_tree(node, dir_fd, dironly)
    for name, children in names:
        if include_hidden or not name.startswith("."):
            yield name
            for y in rlistdir_pos(children, dir_fd, dironly, include_hidden):
                yield _join(name, y)


def recursive_kw_generator_exhaustion_is_not_a_leak():
    """Inner generator StopIteration ends the for-loop; PEP 479 does not fire.

    glob._rlistdir calls itself with include_hidden= after the first yield.
    A compiled or residual next of that inner generator must not leak
    StopIteration into the outer body.
    """
    tree = [
        ("a", [("D", []), ("bcd", [("EF", [])])]),
        ("aaa", [("zzzF", [])]),
        ("EF", []),
        (".aa", [("G", [])]),
        (".bb", []),
    ]
    expect_false = ["a", "a/D", "a/bcd", "a/bcd/EF", "aaa", "aaa/zzzF", "EF"]
    expect_true = expect_false + [".aa", ".aa/G", ".bb"]
    for _ in range(2000):
        got = list(rlistdir_kw(tree, None, False, include_hidden=False))
        assert got == expect_false, got
        got = list(rlistdir_pos(tree, None, False, include_hidden=False))
        assert got == expect_false, got
    got = list(rlistdir_kw(tree, None, False, include_hidden=True))
    assert got == expect_true, got
    got = list(rlistdir_pos(tree, None, False, include_hidden=True))
    assert got == expect_true, got


recursive_kw_generator_exhaustion_is_not_a_leak()
print("OK")
