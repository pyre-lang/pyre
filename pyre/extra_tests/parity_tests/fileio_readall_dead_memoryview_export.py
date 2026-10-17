# CPython-suite gap: test_io.PyTextIOWrapperTest.test_seek_and_tell
# parity-tests reason: `_pyio.FileIO.readall` resizes a bytearray after
# `os.readinto` into a slice of it; the temporary memoryview is released
# before that resize so the export cannot raise BufferError.

"""FileIO.readall must release its temporary memoryview before resize.

`_pyio.FileIO.readall` reads into a slice of `result`, then
`result.resize(bytes_read)`.  The view is released before the resize so
`bytearray._exports` is not still raised.
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
