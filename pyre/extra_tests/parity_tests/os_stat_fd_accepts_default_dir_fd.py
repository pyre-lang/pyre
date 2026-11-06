# pyre-check: platforms=linux,darwin
# CPython-suite gap: test.test_os / test_posix DirFd tests pass a live directory
# descriptor from os_helper.open_dir_fd, so they never pass dir_fd=AT_FDCWD
# next to an fd path.
# parity-tests reason: os.stat(fd, dir_fd=AT_FDCWD) (or the host libc sentinel
# when the constant is unpublished) is the omitted-dir_fd spelling.  A converter
# that treats any non-None integer as a real descriptor raises ValueError here.

"""dir_fd=AT_FDCWD next to an fd path is the omitted-dir_fd spelling."""

import os


def default_dir_fd_sentinels():
    seen = []
    if hasattr(os, "AT_FDCWD"):
        seen.append(os.AT_FDCWD)
    # Darwin AT_FDCWD is -2; Linux (and PyPy's fallback) is -100.
    for candidate in (-2, -100):
        if candidate not in seen:
            seen.append(candidate)
    return seen


fd = os.open(".", os.O_RDONLY)
try:
    omitted = os.fstat(fd)
    default = None
    for sentinel in default_dir_fd_sentinels():
        try:
            st = os.stat(fd, dir_fd=sentinel)
        except (ValueError, OSError):
            continue
        if st.st_ino == omitted.st_ino and st.st_dev == omitted.st_dev:
            default = sentinel
            break
    assert default is not None, "no DEFAULT_DIR_FD sentinel accepted next to an fd"
    other = -100 if default != -100 else -2
    try:
        os.stat(fd, dir_fd=other)
        raise AssertionError("non-default dir_fd next to fd was accepted")
    except ValueError as exc:
        assert "dir_fd" in str(exc), str(exc)
finally:
    os.close(fd)

print("OK")
