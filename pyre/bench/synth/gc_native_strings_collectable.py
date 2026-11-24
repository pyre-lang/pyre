# pyre-check: no-cpython
# pyre-check: requires-modules=mmap

import _io
import gc
import os
import time


results = []


def managed(label, value):
    results.append((label, value))
    return value


# Native filesystem objects have state-dependent repr branches.
entry = next(os.scandir("."))
managed("DirEntry repr", os.DirEntry.__repr__(entry))

file = _io.FileIO(__file__, "r")
managed("FileIO name repr", _io.FileIO.__repr__(file))
del file.name
managed("FileIO fd repr", _io.FileIO.__repr__(file))
file.close()
managed("FileIO closed repr", _io.FileIO.__repr__(file))


# Unix and Windows have separate libc-backed strftime implementations, while
# asctime and ctime share the upstream-style formatter.
calendar = (2020, 2, 3, 4, 5, 6, 0, 34, -1)
strftime_value = time.strftime("%Y-%m-%d %H:%M:%S", calendar)
assert strftime_value == "2020-02-03 04:05:06"
managed("strftime", strftime_value)

asctime_value = time.asctime(calendar)
assert asctime_value == "Mon Feb  3 04:05:06 2020"
managed("asctime", asctime_value)
managed("ctime", time.ctime(0))


# pyre's mmap implementation is currently Unix-only. The module imports on
# Windows without exposing mmap.mmap, so keep the platform branch in this one
# native fixture instead of maintaining another process/baseline pair.
import mmap


if hasattr(mmap, "mmap"):
    mapping = mmap.mmap(-1, 1)
    managed("mmap live repr", mmap.mmap.__repr__(mapping))
    mapping.close()
    managed("mmap closed repr", mmap.mmap.__repr__(mapping))


# The census names only tracked types and a string is never one, so it can no
# longer report a result directly.  It does name `results`, and the collector's
# own edge walk goes the rest of the way: one hop reaches the entry tuples and
# a second reaches the strings.  A value the native path handed back without
# giving it a managed identity would be missing from that walk.
assert any(obj is results for obj in gc.get_objects())
reached = []
for entry in gc.get_referents(results):
    reached.extend(gc.get_referents(entry))
for label, value in results:
    assert any(ref is value for ref in reached), label

print("native runtime string results are collectable")
