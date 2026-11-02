# pyre-check: platforms=win32
# pyre-check: requires-modules=_overlapped,_winapi
# pyre-check: pypy-diverges: lib_pypy._overlapped masks CFFI pointer casts
# and its constructor expects a cdata event rather than an integer handle.
# CPython-suite gap: Windows events tests use ordinary handles and keys;
# they do not cover pointer overflow or integer-coercion objects.
# parity-tests reason: reject out-of-range pointer words before Win32 calls,
# while retaining full unsigned completion keys and OVERLAPPED addresses.
import _overlapped as ov
import _winapi
import struct


class Index:
    def __index__(self):
        raise AssertionError('pointer arguments must not invoke __index__')


class IntOnly:
    def __int__(self):
        raise AssertionError('pointer arguments must not invoke __int__')


operation = ov.Overlapped(0)
cases = [
    (ov.Overlapped, ()),
    (ov.CreateIoCompletionPort, (0, 0, 0)),
    (ov.GetQueuedCompletionStatus, (0,)),
    (ov.PostQueuedCompletionStatus, (0, 0, 0)),
    (ov.RegisterWaitWithQueue, (0, 0, 0)),
    (ov.UnregisterWait, ()),
    (ov.UnregisterWaitEx, (0,)),
    (ov.BindLocal, (2,)),
    (ov.WSAConnect, (('127.0.0.1', 1),)),
    (ov.SetEvent, ()),
    (ov.ResetEvent, ()),
    (operation.ReadFile, (0,)),
    (operation.ReadFileInto, (bytearray(1),)),
    (operation.WriteFile, (b'',)),
    (operation.WSARecv, (0, 0)),
    (operation.WSARecvInto, (bytearray(1), 0)),
    (operation.WSASend, (b'', 0)),
    (operation.AcceptEx, (0,)),
    (operation.ConnectEx, (('127.0.0.1', 1),)),
    (operation.DisconnectEx, (0,)),
    (operation.TransmitFile, (0, 0, 0, 0, 0, 0)),
    (operation.ConnectNamedPipe, ()),
    (operation.WSARecvFrom, (0, 0)),
    (operation.WSARecvFromInto, (bytearray(1), 0)),
    (operation.WSASendTo, (b'', 0, ('127.0.0.1', 1))),
]
bits = struct.calcsize('P') * 8
for function, rest in cases:
    for value in (1 << bits, -(1 << (bits - 1)) - 1, 1 << 100, -(1 << 100)):
        try:
            function(value, *rest)
        except OverflowError as error:
            assert str(error) == 'int too big to convert', (function, str(error))
        else:
            raise AssertionError((function, 'pointer overflow accepted'))
    for value in (Index(), IntOnly(), None):
        try:
            function(value, *rest)
        except TypeError as error:
            assert str(error) == 'an integer is required', (function, str(error))
        else:
            raise AssertionError((function, 'non-integer pointer accepted'))

# The second handle and key/pointer positions must share the checked boundary.
for function, args in (
    (ov.CreateIoCompletionPort, (0, 1 << bits, 0, 0)),
    (ov.CreateIoCompletionPort, (0, 0, 1 << bits, 0)),
    (ov.PostQueuedCompletionStatus, (0, 0, 1 << bits, 0)),
    (ov.PostQueuedCompletionStatus, (0, 0, 0, 1 << bits)),
    (ov.RegisterWaitWithQueue, (0, 1 << bits, 0, 0)),
    (ov.RegisterWaitWithQueue, (0, 0, 1 << bits, 0)),
    (ov.UnregisterWaitEx, (0, 1 << bits)),
    (operation.AcceptEx, (0, 1 << bits)),
    (operation.TransmitFile, (0, 1 << bits, 0, 0, 0, 0, 0)),
):
    try:
        function(*args)
    except OverflowError as error:
        assert str(error) == 'int too big to convert', str(error)
    else:
        raise AssertionError('later pointer overflow accepted')

port = ov.CreateIoCompletionPort(ov.INVALID_HANDLE_VALUE, 0, 0, 0)
try:
    for key in (1 << (bits - 1), (1 << bits) - 1, -1):
        ov.PostQueuedCompletionStatus(port, 7, key, (1 << bits) - 1)
        assert ov.GetQueuedCompletionStatus(port, 0) == (
            0, 7, key % (1 << bits), (1 << bits) - 1,
        )
finally:
    _winapi.CloseHandle(port)

print('OK')
