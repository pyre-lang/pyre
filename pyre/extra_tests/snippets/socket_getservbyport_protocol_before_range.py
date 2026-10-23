# pyre-check: gate=1
# CPython-suite gap: `test_socket` never passes an out-of-range port together
# with a protocol that is not a string.
#
# `interp_func.py getservbyport` converts the protocol (`space.text_w`) before
# it checks the port range, as `getservbyport`'s `"i|s"` parse does, so the
# protocol's TypeError wins over the range's OverflowError.
import socket

for port in (-1, 70000):
    try:
        socket.getservbyport(port, object())
    except TypeError:
        pass
    else:
        raise AssertionError("range checked before the protocol")
    try:
        socket.getservbyport(port)
    except OverflowError:
        pass
    else:
        raise AssertionError("out-of-range port accepted")

print("OK")
