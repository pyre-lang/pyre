# pyre-check: platforms=win32
# pyre-check: requires-modules=msvcrt,_winapi
# pyre-check: pypy-diverges: PyPy's CFFI reports its own integer overflow text
# and accepts __int__; get_osfhandle also truncates the descriptor to C int.
# Its unguarded CRT setmode also aborts on invalid descriptors.
# CPython-suite gap: test_msvcrt.TestFileOperations tests valid descriptors,
# modes and lengths, but not integers outside the Windows C int/long range.
# parity-tests reason: lib_pypy.msvcrt passes these arguments through CFFI's
# checked int/int/long conversions before the CRT can touch the file. A Rust
# narrowing cast must not wrap a large descriptor, mode, flag or lock length.
import errno
import gc
import msvcrt
import os
import struct
import tempfile
import _winapi


def overflows(function, *args, c_type="int"):
    try:
        result = function(*args)
    except OverflowError as error:
        assert str(error) == "Python int too large to convert to C " + c_type, str(error)
        return
    else:
        raise AssertionError((function.__name__, args, result))


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        gc.collect()
        return self.value


class IntOnly:
    def __int__(self):
        raise AssertionError("CRT integer arguments must not call __int__")


def refuses_int(function, *args):
    try:
        function(*args)
    except TypeError:
        return
    raise AssertionError((function.__name__, "accepted __int__ without __index__"))


with tempfile.TemporaryFile() as file:
    fd = file.fileno()
    # The CRT validates the mode before the descriptor. In particular, an
    # early Rust Borrowed-fd check must not replace EINVAL with EBADF.
    for invalid_fd in (-1, 999999):
        for mode, expected_errno in ((0, errno.EINVAL), (os.O_BINARY, errno.EBADF)):
            try:
                msvcrt.setmode(invalid_fd, mode)
            except OSError as error:
                assert error.errno == expected_errno, (invalid_fd, mode, error)
            else:
                raise AssertionError('setmode accepted an invalid descriptor')
    for value in (2**31, -(2**31) - 1, 2**32, -(2**32), 2**100, -(2**100)):
        overflows(msvcrt.setmode, value, os.O_BINARY)
        overflows(msvcrt.setmode, fd, value)
        overflows(msvcrt.locking, value, msvcrt.LK_NBLCK, 1)
        overflows(msvcrt.locking, fd, value, 1)
        overflows(msvcrt.locking, fd, msvcrt.LK_NBLCK, value, c_type="long")
        # Invalid handle: conversion must fail before _open_osfhandle runs.
        overflows(msvcrt.open_osfhandle, -1, value)
        overflows(msvcrt.get_osfhandle, value)

    # Arguments remain rooted when an earlier index conversion collects.
    assert msvcrt.get_osfhandle(Index(fd)) == msvcrt.get_osfhandle(fd)
    previous = msvcrt.setmode(Index(fd), Index(os.O_BINARY))
    msvcrt.setmode(fd, previous)
    msvcrt.locking(Index(fd), Index(msvcrt.LK_NBLCK), Index(1))
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

    refuses_int(msvcrt.get_osfhandle, IntOnly())
    refuses_int(msvcrt.setmode, IntOnly(), os.O_BINARY)
    refuses_int(msvcrt.setmode, fd, IntOnly())
    refuses_int(msvcrt.locking, IntOnly(), msvcrt.LK_NBLCK, 1)
    refuses_int(msvcrt.locking, fd, IntOnly(), 1)
    refuses_int(msvcrt.locking, fd, msvcrt.LK_NBLCK, IntOnly())
    refuses_int(msvcrt.open_osfhandle, -1, IntOnly())

    # The handle is a real integer, not an index argument. Its conversion
    # precedes flags; malformed flags keep these probes away from the CRT.
    for handle in (IntOnly(), Index(-1), None):
        try:
            msvcrt.open_osfhandle(handle, 2**100)
        except TypeError as error:
            assert str(error) == "an integer is required", str(error)
        else:
            raise AssertionError("open_osfhandle accepted a non-integer handle")
    bits = struct.calcsize("P") * 8
    for handle in (2**(bits - 1), 2**bits - 1):
        overflows(msvcrt.open_osfhandle, handle, Index(2**100))
    for handle in (2**bits, -(2**(bits - 1)) - 1, -(2**100)):
        try:
            msvcrt.open_osfhandle(handle, 2**100)
        except OverflowError as error:
            assert str(error) == "int too big to convert", str(error)
        else:
            raise AssertionError("open_osfhandle accepted an oversized handle")

    # A full-width unsigned handle reaches the CRT unchanged. Truncating its
    # upper bits would instead adopt this valid duplicate and succeed.
    process = _winapi.GetCurrentProcess()
    duplicate = _winapi.DuplicateHandle(
        process, msvcrt.get_osfhandle(fd), process, 0, False,
        _winapi.DUPLICATE_SAME_ACCESS,
    )
    try:
        try:
            adopted = msvcrt.open_osfhandle(duplicate | (1 << (bits - 1)), os.O_BINARY)
        except OSError as error:
            assert error.errno == errno.EBADF, error
        else:
            duplicate = None  # The CRT took ownership of the duplicate.
            os.close(adopted)
            raise AssertionError("open_osfhandle truncated the handle")
    finally:
        if duplicate is not None:
            _winapi.CloseHandle(duplicate)

    class OverflowIndex:
        def __index__(self):
            raise OverflowError("from __index__")

    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, OverflowIndex())
    except OverflowError as error:
        assert str(error) == "from __index__", str(error)
    else:
        raise AssertionError("__index__ error lost")

    # Refused calls must leave the descriptor usable and the region unlocked.
    file.write(b"x")
    file.flush()
    file.seek(0)
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    previous = msvcrt.setmode(fd, os.O_BINARY)
    msvcrt.setmode(fd, previous)
    assert file.read() == b"x"

print("OK")
