# CPython-suite gap: test_io.PyTextIOWrapperTest.test_seek_and_tell
# parity-tests reason: `_pyio.FileIO.readall` resizes a bytearray after
# `os.readinto(fd, memoryview(result)[n:])`; an unreachable memoryview must
# not keep the bytearray export and raise BufferError.

"""A dead memoryview export must not block FileIO.readall's trailing resize.

`_pyio.FileIO.readall` builds a temporary `memoryview(result)[bytes_read:]`
for each `os.readinto`, then `result.resize(bytes_read)`.  The view and
its owning parent are unreachable before the resize.  A GC that still
traces the parent from a popped operand slot on an inlined or
CALL_ASSEMBLER frame keeps `bytearray._exports` raised and the resize
raises BufferError.
"""

import codecs
import os
import tempfile

from test.test_io import (
    StatefulIncrementalDecoder,
    StatefulIncrementalDecoderTest,
    lookupTestDecoder,
)
import _pyio as pyio

CHUNK_SIZE = 128


def run_one(path, data, min_pos):
    with open(path, "wb") as f:
        f.write(data)
    f = pyio.open(path, encoding="test_decoder")
    f._CHUNK_SIZE = CHUNK_SIZE
    decoded = f.read()
    f.close()
    for i in range(min_pos, len(decoded) + 1):
        for j in (1, 5, len(decoded) - i):
            f = pyio.open(path, encoding="test_decoder")
            assert f.read(i) == decoded[:i]
            cookie = f.tell()
            assert f.read(j) == decoded[i : i + j]
            f.seek(cookie)
            assert f.read() == decoded[i:]
            f.close()


codecs.register(lookupTestDecoder)
StatefulIncrementalDecoder.codecEnabled = 1
fd, path = tempfile.mkstemp()
os.close(fd)
try:
    for inp, _, _ in StatefulIncrementalDecoderTest.test_cases:
        run_one(path, inp, 0)
    for inp, _, _ in StatefulIncrementalDecoderTest.test_cases:
        offset = CHUNK_SIZE - len(inp) // 2
        prefix = b"." * offset
        run_one(path, prefix + inp, offset * 2)
finally:
    StatefulIncrementalDecoder.codecEnabled = 0
    os.unlink(path)

print("OK")
