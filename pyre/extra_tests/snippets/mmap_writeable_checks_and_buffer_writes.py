# pyre-check: gate=1
# CPython-suite gap: test_mmap never writes a memoryview or array into a
# mapping, never writes to a closed ACCESS_READ mapping, never calls len() or
# enters `with` on a closed mapping, and never calls __exit__ directly.
#
# `interp_mmap.py` checks `check_valid()` before `check_writeable()` in
# `write`, `write_byte`, `move` and `descr_setitem`, acquires `write` data
# with `bufferstr_w`, and its `__len__` / `descr_enter` call `check_valid()`.
# The messages are `rmmap.py MMap.check_writeable` and `MMap.write_byte`.
import array
import mmap
import sys

m = mmap.mmap(-1, 16)
assert m.write(memoryview(b"ab")) == 2
assert m.write(array.array("b", [1, 2])) == 2
m[4:6] = memoryview(b"xy")
assert m[:6] == b"ab\x01\x02xy"
m.seek(16)
try:
    m.write_byte(1)
except ValueError as e:
    assert str(e) == "write byte out of range", e
else:
    raise AssertionError("write_byte past the end accepted")
assert m.__exit__(None, None, None) is None

r = mmap.mmap(-1, 16, access=mmap.ACCESS_READ)
for call in (
    lambda: r.write(b"a"),
    lambda: r.write_byte(1),
    lambda: r.move(0, 1, 1),
    lambda: r.__setitem__(0, 1),
):
    try:
        call()
    except TypeError as e:
        assert str(e) == "mmap can't modify a readonly memory map.", e
    else:
        raise AssertionError("read-only mapping modified")
r.close()
for call in (
    lambda: r.write(b"a"),
    lambda: r.write_byte(1),
    lambda: r.move(0, 1, 1),
    lambda: len(r),
    lambda: r.__enter__(),
):
    try:
        call()
    except ValueError as e:
        assert str(e) == "mmap closed or invalid", e
    else:
        raise AssertionError("closed mapping used")

w = mmap.mmap(-1, 12)
w.write(b"one two ones")
assert w.find(b"", 13) == -1
assert w.rfind(b"", 13) == -1
assert w.find(b"", 12) == 12

events = []
sys.addaudithook(lambda e, a: events.append((e, a)) if e == "mmap.__new__" else None)
mmap.mmap(-1, 8).close()
assert events == [("mmap.__new__", (-1, 8, 0, 0))], events

print("OK")
