# pyre-check: platforms=darwin,linux
"""sendmsg rejects ancillary data whose CMSG_SPACE exceeds SOCKLEN_T_LIMIT."""
import mmap
import os
import socket
import tempfile

if not hasattr(socket, "CMSG_SPACE"):
    raise SystemExit(0)

toobig = 0x7FFFFFFF - socket.CMSG_SPACE(1) + 1
pair = socket.socketpair()
try:
    try:
        fd, path = tempfile.mkstemp()
        os.unlink(path)
        os.ftruncate(fd, toobig)
        payload = mmap.mmap(fd, toobig)
        os.close(fd)
    except (OSError, MemoryError, OverflowError, ValueError):
        raise SystemExit(0)
    try:
        try:
            pair[0].sendmsg([b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, payload)])
        except MemoryError:
            raise SystemExit(0)
        except OSError as exc:
            assert "ancillary data item too large" in str(exc) or "too much ancillary data" in str(
                exc
            ) or getattr(exc, "errno", None) is not None, exc
        else:
            raise SystemExit("sendmsg accepted an oversized ancillary item")
    finally:
        payload.close()
finally:
    pair[0].close()
    pair[1].close()
