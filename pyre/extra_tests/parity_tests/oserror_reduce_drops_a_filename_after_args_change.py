# CPython-suite gap: `test_exceptions` pickles an `OSError` built by the
# constructor. It does not replace `args` afterwards, and it does not delete
# `filename`.
#
# parity-tests reason: `__reduce__` is the constructor call pickle will replay.
# Reattaching a filename after `args` has stopped being the trimmed pair stores
# different arguments. Deleting the member clears the slot, so the message and
# the rebuilt arguments both drop it.
#
# pyre-check: pypy-diverges: `W_OSError.descr_reduce` appends `w_filename`
# whenever the slot is set, so a replaced `args` tuple grows.
# `readwrite_attrproperty_w` installs no `fdel`, so `del exc.filename` raises
# AttributeError there. `OSError_reduce` splices only when `args` still has
# length 2, and `PyMember_SetOne` clears the member.
#
# oserror_str_omits_deleted_errno_strerror:
# `test_exceptions` stringifies a constructed `OSError` and never deletes `errno` or
# `strerror`. `OSError_str` tests the `myerrno` and `strerror` pointers. `PyMember_SetOne`
# clears a `_Py_T_OBJECT` member, so after `del` those tests fail and `BaseException_str`
# prints the args tuple. An explicit `None` store stays present. Replacing `args` does not
# restamp the slots. `readwrite_attrproperty_w` installs no `fdel`, so `del e.strerror`
import sys


def rebuilt(exc):
    return exc.__reduce__()[1]


exc = OSError(2, "m", "a")
exc.args = (9, "z", "nope")
assert rebuilt(exc) == (9, "z", "nope"), rebuilt(exc)
assert str(exc) == "[Errno 2] m: 'a'", str(exc)

exc = FileNotFoundError(2, "m", "a")
exc.args = (1,)
assert exc.__reduce__()[0] is FileNotFoundError
assert rebuilt(exc) == (1,), rebuilt(exc)
assert str(exc) == "[Errno 2] m: 'a'", str(exc)

exc = OSError()
exc.filename = "a"
assert str(exc) == "[Errno None] None: 'a'", str(exc)
assert rebuilt(exc) == (), rebuilt(exc)
exc.filename2 = "b"
assert str(exc) == "[Errno None] None: 'a' -> 'b'", str(exc)
assert rebuilt(exc) == (), rebuilt(exc)

exc = OSError(2, "m", "a")
del exc.filename
assert exc.filename is None
assert str(exc) == "[Errno 2] m", str(exc)
assert rebuilt(exc) == (2, "m"), rebuilt(exc)

exc = OSError(2, "m", "a", None, "b")
del exc.filename2
assert exc.filename == "a"
assert exc.filename2 is None
assert rebuilt(exc) == (2, "m", "a"), rebuilt(exc)
if sys.platform != "win32":
    assert str(exc) == "[Errno 2] m: 'a'", str(exc)

def oserror_str_omits_deleted_errno_strerror():
    def deleted_strerror_falls_back_to_args():
        exc = OSError(2, "m")
        del exc.strerror
        assert exc.strerror is None
        assert str(exc) == "(2, 'm')", str(exc)


    def deleted_errno_falls_back_to_args():
        exc = OSError(2, "m")
        del exc.errno
        assert exc.errno is None
        assert str(exc) == "(2, 'm')", str(exc)


    def deleted_both_fall_back_to_args():
        exc = OSError(2, "m")
        del exc.errno
        del exc.strerror
        assert str(exc) == "(2, 'm')", str(exc)


    def explicit_none_strerror_stays_present():
        exc = OSError(2, "m")
        exc.strerror = None
        assert str(exc) == "[Errno 2] None", str(exc)


    def explicit_none_errno_stays_present():
        exc = OSError(2, "m")
        exc.errno = None
        assert str(exc) == "[Errno None] m", str(exc)


    def deleted_strerror_with_filename_renders_none():
        exc = OSError(2, "m", "a")
        del exc.strerror
        assert str(exc) == "[Errno 2] None: 'a'", str(exc)


    def replacing_args_does_not_restamp_slots():
        exc = OSError(2, "m")
        exc.args = (9, "x")
        assert str(exc) == "[Errno 2] m", str(exc)


    deleted_strerror_falls_back_to_args()
    deleted_errno_falls_back_to_args()
    deleted_both_fall_back_to_args()
    explicit_none_strerror_stays_present()
    explicit_none_errno_stays_present()
    deleted_strerror_with_filename_renders_none()
    replacing_args_does_not_restamp_slots()
oserror_str_omits_deleted_errno_strerror()
print("OK")
