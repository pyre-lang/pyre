# CPython-suite gap: test_linecache keeps closed FileIO objects across
# minors and later traces w_name; the suite does not assert that slot.
# parity-tests reason: interp_fileio.py descr_init space.setattr(self, "name",
# w_name) stores a young str on an old rawmalloced FileIO; the original
# w_name is a livevar, not a getattr of opened after a collecting call.
# parity-env: PYPY_GC_NURSERY=4096

"""Closed FileIO.name stays a live str across minor collections."""

import gc
import io
import os
import sys
import tempfile

N = 40
CHURN = 80

tmpdir = tempfile.mkdtemp()
files = []
paths = []
try:
    for i in range(N):
        path = os.path.join(tmpdir, "f%d.txt" % i)
        with open(path, "w") as fh:
            fh.write("x")
        path = sys.intern(path)
        f = io.FileIO(path, "rb")
        f.close()
        files.append(f)
        paths.append(path)
    junk = []
    for i in range(CHURN):
        junk.append([object() for _ in range(64)])
        if i % 8 == 0:
            gc.collect()
    for f, path in zip(files, paths):
        name = f.name
        if name != path:
            raise AssertionError((name, path))
finally:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
print("OK")
