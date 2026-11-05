# pyre-check: platforms=linux,darwin
# CPython-suite gap: test_os.FwalkTests.test_fd_finalization only grades the
# fwalk generator's own descriptors.  On a tracing GC, a collection taken
# from close() would also run `__del__` of unrelated FileIO left by earlier
# tests, so `getfd()` drops below the snapshot and the assertion reads
# `3 != 6`.
# parity-tests reason: `os.fwalk` holds an exhausted `scandir` iterator at the
# topdown yield.  `W_ScandirIterator.fail` / `_close` (interp_scandir.py) is
# what makes that iterator's finalizer a no-op; `may_ignore_finalizer` is the
# same hint so `fwalk().close()` does not keep the exhausted scandir on the
# finalizer queue.

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
