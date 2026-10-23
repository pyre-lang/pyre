# pyre-check: pypy-diverges: pypy3's `settimeout` / `setdefaulttimeout` store
# `space.float_w(w_timeout)` as is, so `inf`, `nan` and `1e300` are accepted and
# `gettimeout()` returns them unchanged, and an `__index__`-only object raises
# TypeError; its `getservbyname` / `getservbyport` look up the prefix before an
# embedded NUL instead of raising ValueError.
#
# CPython-suite gap: `test_timeout` checks only negative and non-numeric
# timeouts, and `test_socket` never passes a non-finite or out-of-range one, a
# protocol with an embedded NUL, or a third argument to a service lookup.
#
# parity-tests reason: a timeout is converted to an int64 nanosecond count
# (`_PyTime_FromSecondsObject`) before the socket keeps it.  So NaN and an
# out-of-range value raise instead of being stored, and `gettimeout()` returns
# the rounded count.  The service lookups parse `"s|s"` / `"i|s"`, so an
# embedded NUL raises ValueError while converting the argument, before the port
# range check and the audit hook.
import socket
import sys

events = []
sys.addaudithook(lambda e, a: events.append(e) if e.startswith("socket.getserv") else None)


def raises(exc, fn, *args, match=None):
    try:
        fn(*args)
    except exc as e:
        if match is not None:
            assert str(e) == match, (fn, args, str(e))
    else:
        raise AssertionError((fn, args, "did not raise", exc))


class Index:
    def __index__(self):
        return 2


class IntWithFloat(int):
    def __float__(self):
        return 9.0


s = socket.socket()
try:
    for setter in (socket.setdefaulttimeout, s.settimeout):
        raises(ValueError, setter, float("nan"), match="Invalid value NaN (not a number)")
        for v in (float("inf"), float("-inf"), 1e300, -1e300, 2**70, -(2**70), 9223372037):
            raises(OverflowError, setter, v, match="timestamp out of range for C PyTime_t")
        for v in (-1, -0.5, -1e-10):
            raises(ValueError, setter, v, match="Timeout value out of range")
        if sys.platform == "win32":
            raises(OverflowError, setter, 2147483.648, match="timeout doesn't fit into C timeval")
    s.settimeout(1e-10)
    assert s.gettimeout() == 1e-09, s.gettimeout()
    s.settimeout(Index())
    assert s.gettimeout() == 2.0, s.gettimeout()
    s.settimeout(IntWithFloat(3))
    assert s.gettimeout() == 3.0, s.gettimeout()
    s.settimeout(-0.0)
    assert s.gettimeout() == 0.0, s.gettimeout()
    socket.setdefaulttimeout(1e-10)
    assert socket.getdefaulttimeout() == 1e-09, socket.getdefaulttimeout()
finally:
    socket.setdefaulttimeout(None)
    s.close()

raises(ValueError, socket.getservbyname, "http\0x", match="embedded null character")
raises(ValueError, socket.getservbyname, "http", "tcp\0x", match="embedded null character")
raises(ValueError, socket.getservbyport, 80, "tcp\0x", match="embedded null character")
raises(ValueError, socket.getservbyport, -1, "tcp\0x", match="embedded null character")
assert events == [], events
raises(TypeError, socket.getservbyname, "http", "tcp", 1)
raises(TypeError, socket.getservbyport, 80, "tcp", 1)

print("OK")
