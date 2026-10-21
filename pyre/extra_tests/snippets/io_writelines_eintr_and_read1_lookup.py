# pyre-check: gate=1
# CPython-suite gap: no `test_io` case interrupts a `write` inside
# `IOBase.writelines`, and none gives `TextIOWrapper` a buffer whose `read1`
# lookup raises something other than AttributeError.
#
# `IOBase.writelines` retries a `write` that fails with an OSError whose
# errno is EINTR (`interp_iobase.py trap_eintr`, `iobase.c
# _PyIO_trap_eintr`); any other error propagates.  The `read1` probe in
# `TextIOWrapper.__init__` treats only AttributeError as "no read1"
# (`textio.c` `PyObject_HasAttrWithError`); PyPy's `space.findattr` would
# swallow the KeyError and NameError below too.
import errno
import io


class Interrupted(io.IOBase):
    def __init__(self):
        self.calls = 0
        self.out = []

    def write(self, b):
        self.calls += 1
        if self.calls == 2:
            raise InterruptedError(errno.EINTR, "interrupted")
        self.out.append(b)


w = Interrupted()
w.writelines([b"a", b"b", b"c"])
assert w.out == [b"a", b"b", b"c"], w.out
assert w.calls == 4, w.calls


class Failing(io.IOBase):
    def write(self, b):
        raise OSError(errno.EIO, "eio")


try:
    Failing().writelines([b"a"])
except OSError as e:
    assert e.errno == errno.EIO
else:
    raise AssertionError("EIO was retried")


class Read1KeyError(io.BytesIO):
    def __getattribute__(self, name):
        if name == "read1":
            raise KeyError("boom")
        return super().__getattribute__(name)


try:
    io.TextIOWrapper(Read1KeyError(b"x"), encoding="utf-8")
except KeyError:
    pass
else:
    raise AssertionError("read1 lookup error swallowed")


class SeekableFirst(Read1KeyError):
    def seekable(self):
        raise ValueError("seekable")


# `seekable()` runs before the `read1` probe (`descr_init`, `textio.c`).
try:
    io.TextIOWrapper(SeekableFirst(b"x"), encoding="utf-8")
except ValueError:
    pass
else:
    raise AssertionError("read1 probed before seekable()")


class Read1NameError(io.BytesIO):
    def __getattribute__(self, name):
        if name == "read1":
            raise NameError("boom")
        return super().__getattribute__(name)


try:
    io.TextIOWrapper(Read1NameError(b"x"), encoding="utf-8")
except NameError:
    pass
else:
    raise AssertionError("read1 NameError swallowed")


class Missing(AttributeError):
    pass


class NoRead1(io.BytesIO):
    def __getattribute__(self, name):
        if name == "read1":
            raise Missing(name)
        return super().__getattribute__(name)


assert io.TextIOWrapper(NoRead1(b"xy"), encoding="utf-8").read() == "xy"
print("PASS")
