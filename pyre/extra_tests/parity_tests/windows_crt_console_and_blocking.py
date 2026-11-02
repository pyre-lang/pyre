# pyre-check: platforms=win32
# pyre-check: requires-modules=msvcrt
# pyre-check: pypy-diverges: PyPy has no msvcrt audit events and different
# console argument errors; its CFFI rejects supplementary wchar_t input.
# CPython-suite gap: test_msvcrt omits console conversion, audit rejection
# and a contended CRT lock while another Python thread collects/unlocks.
# parity-tests reason: preserve lib_pypy.msvcrt's code-unit boundary and
# external-call release, plus the pinned spec's audit and argument contract.
import gc
import msvcrt
import os
import sys
import tempfile
import threading
import time


for function in (msvcrt.putwch, msvcrt.ungetwch):
    for value, kind in ((None, 'None'), (1, 'int'), (b'A', 'bytes')):
        try:
            function(value)
        except TypeError as error:
            assert str(error) == (
                function.__name__ + '() argument must be a unicode character, not ' + kind
            ), str(error)
        else:
            raise AssertionError('accepted a non-string')
    for value in ('', 'ab'):
        try:
            function(value)
        except TypeError as error:
            assert str(error) == (
                function.__name__ + '(): argument must be a unicode character, '
                'not a string of length ' + str(len(value))
            ), str(error)
        else:
            raise AssertionError('accepted multiple code points')

for char in ('A', '\uac00', '\ud800', '\udfff', '\U0001f600'):
    msvcrt.ungetwch(char)
    # The CRT's non-console fallback truncates pushback to a byte; a real
    # console preserves the wchar_t. Neither path may reject a surrogate.
    assert len(msvcrt.getwch()) == 1

for value in (b'A', bytearray(b'A')):
    msvcrt.ungetch(value)
    assert msvcrt.getch() == b'A'
for function in (msvcrt.putch, msvcrt.ungetch):
    for value, kind in ((None, 'None'), (1, 'int'), ('A', 'str')):
        try:
            function(value)
        except TypeError as error:
            assert str(error) == (
                function.__name__ + '() argument must be a byte string of length 1, not ' + kind
            ), str(error)
        else:
            raise AssertionError('accepted a non-bytes argument')
    for value in (b'', b'ab', bytearray(), bytearray(b'ab')):
        try:
            function(value)
        except TypeError as error:
            assert str(error) == (
                function.__name__ + '(): argument must be a byte string of length 1, '
                'not a ' + type(value).__name__ + ' object of length ' + str(len(value))
            ), str(error)
        else:
            raise AssertionError('accepted wrong byte length')
    try:
        function(memoryview(b'A'))
    except TypeError:
        pass
    else:
        raise AssertionError('accepted a general buffer')

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, 'lock')
    first = os.open(path, os.O_CREAT | os.O_RDWR | os.O_BINARY)
    second = os.open(path, os.O_RDWR | os.O_BINARY)
    ready = threading.Event()
    errors = []

    def worker():
        ready.set()
        try:
            msvcrt.locking(second, msvcrt.LK_LOCK, 1)
            msvcrt.locking(second, msvcrt.LK_UNLCK, 1)
        except Exception as error:
            errors.append(type(error).__name__)

    try:
        msvcrt.locking(first, msvcrt.LK_NBLCK, 1)
        thread = threading.Thread(target=worker)
        thread.start()
        assert ready.wait(5)
        time.sleep(0.2)
        gc.collect()
        msvcrt.locking(first, msvcrt.LK_UNLCK, 1)
        thread.join(15)
        assert not thread.is_alive()
        assert not errors, errors
    finally:
        os.close(second)
        os.close(first)

events = []


def hook(event, args):
    if event.startswith('msvcrt.'):
        gc.collect()
        events.append((event, args))
        raise RuntimeError('deny CRT operation')


# A rejecting hook must run after conversion and before native side effects.
# The handle event publishes an unsigned pointer word even for -1.
sys.addaudithook(hook)
with tempfile.TemporaryFile() as file:
    fd = file.fileno()
    for function, args in (
        (msvcrt.get_osfhandle, (fd,)),
        (msvcrt.locking, (fd, msvcrt.LK_NBLCK, 1)),
        (msvcrt.open_osfhandle, (-1, 0)),
    ):
        try:
            function(*args)
        except RuntimeError as error:
            assert str(error) == 'deny CRT operation'
        else:
            raise AssertionError('audit hook ignored')
    import struct
    assert events == [
        ('msvcrt.get_osfhandle', (fd,)),
        ('msvcrt.locking', (fd, msvcrt.LK_NBLCK, 1)),
        ('msvcrt.open_osfhandle', ((1 << (struct.calcsize('P') * 8)) - 1, 0)),
    ], events
    for function, args in (
        (msvcrt.get_osfhandle, (1 << 100,)),
        (msvcrt.locking, (fd, 0, 1 << 100)),
        (msvcrt.open_osfhandle, (1 << 100, 0)),
    ):
        try:
            function(*args)
        except OverflowError:
            pass
        else:
            raise AssertionError('audit ran before conversion')
    assert len(events) == 3
    file.write(b'usable')

print('OK')
