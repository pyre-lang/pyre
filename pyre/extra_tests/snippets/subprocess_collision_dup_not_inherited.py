# pyre-check: platforms=darwin,linux
"""A collision-avoidance dup of stdout is not inherited with close_fds=False."""
import os
import subprocess
import sys
import tempfile

script = r"""
import os, sys
listing = sys.argv[1]
st1 = os.fstat(1)
leaked = []
for fd in range(3, 64):
    try:
        st = os.fstat(fd)
    except OSError:
        continue
    if st.st_dev == st1.st_dev and st.st_ino == st1.st_ino:
        leaked.append(fd)
with open(listing, "w") as out:
    out.write(",".join(str(fd) for fd in leaked))
"""

saved0 = os.dup(0)
listing = tempfile.NamedTemporaryFile(delete=False)
listing.close()
marker = tempfile.NamedTemporaryFile()
try:
    os.dup2(marker.fileno(), 0)
    os.set_inheritable(marker.fileno(), False)
    with open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen(
            [sys.executable, "-c", script, listing.name],
            stdin=devnull,
            stdout=0,
            close_fds=False,
        )
        rc = proc.wait()
    assert rc == 0, rc
    with open(listing.name) as stream:
        leaked = stream.read().strip()
    assert leaked == "", leaked
finally:
    os.dup2(saved0, 0)
    os.close(saved0)
    marker.close()
    try:
        os.unlink(listing.name)
    except OSError:
        pass
