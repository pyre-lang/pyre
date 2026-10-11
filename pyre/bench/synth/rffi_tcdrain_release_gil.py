# pyre-check: requires-modules=termios
# A build without `pyre-module` has no `termios`, and neither has the wasm guest.
# pyre-check: skip-cpython
# pyre-check: skip-platforms=win32
# Hot tcdrain in a compiled loop, then a failing call whose errno comes back
# through `ccall_c_tcdrain`'s GIL release and errno save. The trace records
# `call_release_gil` to `c_tcdrain` (`interp_termios.tcdrain` →
# `rtermios.tcdrain`, loop-free).
import errno
import os
import termios

master, slave = os.openpty()
try:
    n = 0
    acc = 0
    err = 0
    while n < 20001:
        try:
            termios.tcdrain(slave)
        except termios.error as e:
            # `termios.error` derives from Exception, so the errno is args[0].
            err = e.args[0]
            break
        acc = acc + n
        n = n + 1
        if n == 20000:
            os.close(slave)
    if err != errno.EBADF:
        raise SystemExit("errno %s" % err)
    print(acc)
    print(err)
finally:
    try:
        os.close(master)
    except OSError:
        pass
    try:
        os.close(slave)
    except OSError:
        pass
