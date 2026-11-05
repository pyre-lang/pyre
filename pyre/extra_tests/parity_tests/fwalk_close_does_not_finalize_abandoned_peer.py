# CPython-suite gap: test_os.FwalkTests.test_fd_finalization only fails
# after test_walk_symlink abandons a live fwalk; the suite does not pin
# that close() of one fwalk must not run a peer's finally.
# parity-tests reason: close() of one fwalk must not collect the heap and
# close another generator's dirfds.  An exhausted scandir's leftover
# finalizer registration is a no-op (`may_ignore_finalizer` / `_close`).
import os
import shutil
import tempfile

if not hasattr(os, "fwalk"):
    print("OK")
    raise SystemExit(0)


def getfd():
    fd = os.dup(1)
    os.close(fd)
    return fd


root = tempfile.mkdtemp()
try:
    os.makedirs(os.path.join(root, "a", "b"))
    held = os.fwalk(root, follow_symlinks=True)
    next(held)
    del held

    old = getfd()
    it = os.fwalk(root, topdown=True)
    next(it)
    it.close()
    after = getfd()
    assert after == old, (after, old)
finally:
    shutil.rmtree(root, ignore_errors=True)

print("OK")
