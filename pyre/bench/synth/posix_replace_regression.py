# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# The invariant is interpreter-level, so the run compiles no loop.
# pyre-check: skip-backends=wasm
# The wasm guest filesystem is read-only, so the pid-named directory this
# fixture creates cannot be made (`OSError: [Errno 30] Read-only file system`).
# os.replace(src, dst) renames over an existing destination in one step and
# leaves no source behind, where os.rename is the raw platform call.  It is
# what importlib._bootstrap_external._write_atomic uses to publish a .pyc, so
# the name has to exist before the write-atomic path can run at all.
#
# The fixtures live in a pid-named directory this script creates in the cwd
# rather than under tempfile, so the guard stays independent of the shutil
# import chain.
import os

BASE = "pyre_replace_probe_%d" % os.getpid()
SRC = os.path.join(BASE, "src")
DST = os.path.join(BASE, "dst")

os.mkdir(BASE)


def write(path, text):
    f = open(path, "w")
    try:
        f.write(text)
    finally:
        f.close()


def read(path):
    f = open(path, "r")
    try:
        return f.read()
    finally:
        f.close()


def check():
    # A destination that does not exist yet: plain move.
    write(SRC, "first")
    os.replace(SRC, DST)
    assert not os.path.exists(SRC), "source survived the replace"
    assert read(DST) == "first", read(DST)

    # A destination that does exist: overwritten, no error.
    write(SRC, "second")
    os.replace(SRC, DST)
    assert not os.path.exists(SRC), "source survived the overwrite"
    assert read(DST) == "second", read(DST)

    # A missing source reports ENOENT through the errno-specific subclass.
    try:
        os.replace(os.path.join(BASE, "absent"), DST)
    except FileNotFoundError as e:
        assert e.errno == 2, e.errno
    else:
        raise AssertionError("os.replace of a missing source did not raise")


try:
    for _ in range(200):
        check()
finally:
    if os.path.exists(DST):
        os.unlink(DST)
    if os.path.exists(SRC):
        os.unlink(SRC)
    os.rmdir(BASE)
print("PASS")
