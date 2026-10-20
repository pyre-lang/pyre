# CPython-suite gap: test_pickle does not deterministically compile the
# non-empty list.pop path before an underflow and traceback frame.clear().
# parity-tests reason: exercise generated pop guards and exception resume
# with both Object and Integer list storage.
#
# list_indexerror_traceback_not_accumulated: the suite does not catch a hot list
# IndexError and inspect that each iteration is a fresh exception whose traceback
# is one frame, not a chain prepended onto a CSE'd instance (residual raise caught
# in-frame, list subscript; KeyError and ValueError siblings).

import pickle
import traceback

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class Reader:
    pass


def pop_twice(items):
    first = items.pop()
    second = items.pop()
    return first + second


def run():
    reader = Reader()
    load_setitem = pickle._Unpickler.load_setitem
    for _ in range(1000):
        reader.stack = [{}, 4, None]
        load_setitem(reader)
        assert reader.stack == [{4: None}]
        assert pop_twice([3, 7]) == 10

    for _ in range(100):
        for stack in ([], [None], [None, None]):
            reader.stack = stack
            try:
                load_setitem(reader)
            except IndexError as exc:
                assert reader.stack == []
                traceback.clear_frames(exc.__traceback__)
            else:
                raise AssertionError("expected unpickler stack underflow")

        items = [7]
        try:
            pop_twice(items)
        except IndexError as exc:
            assert str(exc) == "pop from empty list"
            assert items == []
            traceback.clear_frames(exc.__traceback__)
        else:
            raise AssertionError("expected integer list underflow")


def names(traceback):
    out = []
    while traceback is not None:
        out.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return tuple(out)


def missing_key(d, i):
    try:
        return d[i]
    except KeyError as e:
        return names(e.__traceback__)


def bad_index(seq, i):
    try:
        seq[i]
        raise AssertionError("expected IndexError")
    except IndexError as e:
        return e


def bad_convert(s):
    try:
        return int(s)
    except ValueError as e:
        return names(e.__traceback__)


def run_fresh_tracebacks():
    d = {}
    seq = []
    key_shapes = set()
    index_shapes = set()
    convert_shapes = set()
    index_exc = []
    for i in range(5000):
        key_shapes.add(missing_key(d, i))
        e = bad_index(seq, i)
        index_exc.append(e)
        index_shapes.add(names(e.__traceback__))
        convert_shapes.add(bad_convert("x"))

    assert key_shapes == {("missing_key",)}, key_shapes
    assert index_shapes == {("bad_index",)}, sorted(index_shapes)[:5]
    assert convert_shapes == {("bad_convert",)}, convert_shapes
    assert len({id(e) for e in index_exc}) == 5000, len({id(e) for e in index_exc})


run()
run_fresh_tracebacks()
print("OK")
