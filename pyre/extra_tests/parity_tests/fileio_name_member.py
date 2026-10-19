# CPython-suite gap: test_fileio.testRepr covers del name, not assignment.
# test_linecache keeps closed FileIO objects across minors and later traces
# w_name; the suite does not assert that slot.
# parity-tests reason: interp_fileio.py interp_member_w w_name drives repr.
# interp_fileio.py descr_init space.setattr(self, "name", w_name) stores a
# young str on an old rawmalloced FileIO; the original w_name is a livevar,
# not a getattr of opened after a collecting call.
# parity-env: PYPY_GC_NURSERY=4096

"""FileIO.name is the interp_member_w w_name field.

`del f.name` and `f.name = value` update the typed field, so `repr(f)`
follows the live name or switches to the fd form.
"""

import gc
import io
import os
import sys
import tempfile


path = tempfile.mktemp()
try:
    with open(path, "wb") as fh:
        fh.write(b"x")
    f = io.FileIO(path, "rb")
    print("open-repr", "name=" in repr(f), f.name == path)
    f.name = "renamed"
    print("set-repr", "name='renamed'" in repr(f) or 'name="renamed"' in repr(f), f.name)
    del f.name
    print("del-repr", "fd=" in repr(f), "name=" not in repr(f))
    try:
        f.name
    except AttributeError:
        print("del-get", "AttributeError")
    else:
        print("del-get", "ok")
    f.close()
    print("closed-repr", "[closed]" in repr(f))
finally:
    try:
        os.unlink(path)
    except OSError:
        pass


def closed_fileio_name_stays_a_live_str_across_minors():
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


closed_fileio_name_stays_a_live_str_across_minors()
print("OK")
