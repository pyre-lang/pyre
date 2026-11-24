# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# The invariant is interpreter-level, so the run compiles no loop.
# pyre-check: skip-backends=wasm
# The wasm guest filesystem is read-only, so the pid-named directory this
# fixture creates cannot be made (`OSError: [Errno 30] Read-only file system`)
# and the FileExistsError half cannot run. open() of a missing path already
# raises FileNotFoundError(2) on wasm.
# A failed syscall raises the errno-specific OSError subclass with
# .errno/.strerror/.filename set and args=(errno, strerror): open() of a
# missing path -> FileNotFoundError(2), os.mkdir of an existing path ->
# FileExistsError(17).
#
# Both fixtures come from `os` alone: a pid-named directory this script
# creates in the cwd, rather than tempfile.mkdtemp, so the guard stays
# independent of the shutil import chain.  The mkdir target has to be a
# directory of its own because "/" is a drive root on Windows and the cwd is
# in use there, and both report access-denied rather than EEXIST.
#
# The errno numbers are asserted on every platform, Windows included: the
# posix module is the one installed there too, and the OS error kind is
# translated back to a POSIX errno on the way into OSError.
import os

MISSING = "pyre_enoent_probe_missing_file"
EXISTING = "pyre_eexist_probe_dir_%d" % os.getpid()

os.mkdir(EXISTING)


def check():
    try:
        open(MISSING, "r")
    except FileNotFoundError as e:
        assert type(e).__name__ == "FileNotFoundError", type(e).__name__
        assert e.errno == 2, e.errno
        assert isinstance(e.strerror, str), e.strerror
        assert e.filename == MISSING, e.filename
        assert e.args == (2, e.strerror), e.args
    else:
        raise AssertionError("open() of a missing path did not raise")

    try:
        os.mkdir(EXISTING)
    except FileExistsError as e:
        assert type(e).__name__ == "FileExistsError", type(e).__name__
        assert e.errno == 17, e.errno
        assert isinstance(e.strerror, str), e.strerror
        assert e.args == (17, e.strerror), e.args
    else:
        raise AssertionError("os.mkdir of an existing path did not raise")


try:
    for _ in range(6000):
        check()
finally:
    os.rmdir(EXISTING)
print("PASS")
