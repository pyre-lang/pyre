# pyre-check: requires-modules=fcntl
# A build without `pyre-module` has no `fcntl`, and neither has the wasm guest.
# pyre-check: skip-cpython
# pyre-check: skip-platforms=win32
# Hot flock in a compiled loop, then a failing call whose errno comes back
# through `ccall_c_flock`'s GIL release and errno save. `flock` retries in a
# `while True`, so the codewriter policy leaves it residual (`look_inside_graph`
# `contains_loop`) and the trace calls it rather than `c_flock` directly.
import errno
import fcntl
import os
import tempfile

fd, path = tempfile.mkstemp()
try:
    n = 0
    acc = 0
    err = 0
    while n < 645168:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError as e:
            err = e.errno
            break
        fcntl.flock(fd, fcntl.LOCK_UN)
        acc = acc + n
        n = n + 1
        if n == 100000:
            os.close(fd)
    if err != errno.EBADF:
        raise SystemExit("errno %s" % err)
    print(acc)
    print(err)
finally:
    try:
        os.close(fd)
    except OSError:
        pass
    os.unlink(path)
