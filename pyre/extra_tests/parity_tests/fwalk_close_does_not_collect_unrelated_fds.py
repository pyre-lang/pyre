# pyre-check: platforms=linux,darwin
# CPython-suite gap: test_os.FwalkTests.test_fd_finalization now collects
# before the snapshot (same as test_fd_leak), so leftover FileIO from earlier
# tests do not move getfd(). It still does not pin that `os.fwalk().close()`
# itself must not collect those FileIO: a collection taken from close() would
# run `__del__` of the leaked handles and `getfd()` would drop.
# parity-tests reason: `os.fwalk` holds an exhausted `scandir` iterator at the
# topdown yield.  `W_ScandirIterator.fail` / `_close` (interp_scandir.py) is
# what makes that iterator's finalizer a no-op.

"""fwalk().close() must not finalize FileIO that is not in its frame."""

import os
import shutil
import tempfile


def getfd():
    fd = os.dup(1)
    os.close(fd)
    return fd


root = tempfile.mkdtemp()
try:
    os.makedirs(os.path.join(root, "sub"))
    leaked = [open(os.path.join(root, "keep"), "w") for _ in range(3)]
    del leaked
    old = getfd()
    it = os.fwalk(root, topdown=True)
    next(it)
    it.close()
    now = getfd()
    assert now == old, (now, old)
finally:
    shutil.rmtree(root, ignore_errors=True)

print("OK")
