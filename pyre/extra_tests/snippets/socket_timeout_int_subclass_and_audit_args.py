# pyre-check: gate=1
# CPython-suite gap: `test_socket` never passes an `int` subclass that
# overrides `__float__` as a timeout, never raises KeyboardInterrupt from a
# `setsockopt` value's `__index__`, and never audits `getservbyname` with a
# `str` subclass.
#
# `settimeout_w` / `setdefaulttimeout` read the timeout with `space.float_w`,
# which an `int` answers from its own value (`W_IntObject.float_w`);
# `pytime_from_object` reads an index with `PyLong_AsLongLong`.  Neither
# consults `__float__`.  `setsockopt_w` re-raises an async exception from the
# `c_int_w` attempt instead of retrying the buffer.  `getservbyname` audits
# `space.newtext(name)`, a plain `str`.
import socket
import sys


class I(int):
    def __float__(self):
        return 2.0


s = socket.socket()
try:
    s.settimeout(I(1))
    assert s.gettimeout() == 1.0, s.gettimeout()
    socket.setdefaulttimeout(I(1))
    assert socket.getdefaulttimeout() == 1.0, socket.getdefaulttimeout()
    socket.setdefaulttimeout(None)

    class K:
        def __index__(self):
            raise KeyboardInterrupt

    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, K())
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt swallowed")
finally:
    s.close()


class S(str):
    pass


seen = []


def hook(event, args):
    if event == "socket.getservbyname":
        seen.append(type(args[0]))


sys.addaudithook(hook)
try:
    socket.getservbyname(S("http"), "tcp")
except OSError:
    pass
assert seen == [str], seen

print("OK")
