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

import _pyio as pyio


# 3.15 moved this codec out of `test.test_io` into `test.test_io.test_textio`.
# Importing that module pulls unittest and the rest of the text-io suite and
# pushes this script past the 30s parity timeout, so the codec used by
# `test_seek_and_tell` is inlined here.
class StatefulIncrementalDecoder(codecs.IncrementalDecoder):
    def __init__(self, errors="strict"):
        codecs.IncrementalDecoder.__init__(self, errors)
        self.reset()

    def reset(self):
        self.i = 1
        self.o = 1
        self.buffer = bytearray()

    def getstate(self):
        i, o = self.i ^ 1, self.o ^ 1
        return bytes(self.buffer), i * 100 + o

    def setstate(self, state):
        buffer, io = state
        self.buffer = bytearray(buffer)
        i, o = divmod(io, 100)
        self.i, self.o = i ^ 1, o ^ 1

    def decode(self, input, final=False):
        output = ""
        for b in input:
            if self.i == 0:
                if b == ord("."):
                    if self.buffer:
                        output += self.process_word()
                else:
                    self.buffer.append(b)
            else:
                self.buffer.append(b)
                if len(self.buffer) == self.i:
                    output += self.process_word()
        if final and self.buffer:
            output += self.process_word()
        return output

    def process_word(self):
        output = ""
        if self.buffer[0] == ord("i"):
            self.i = min(99, int(self.buffer[1:] or 0))
        elif self.buffer[0] == ord("o"):
            self.o = min(99, int(self.buffer[1:] or 0))
        else:
            output = self.buffer.decode("ascii")
            if len(output) < self.o:
                output += "-" * self.o
            if self.o:
                output = output[: self.o]
            output += "."
        self.buffer = bytearray()
        return output

    codecEnabled = False


def lookupTestDecoder(name):
    if StatefulIncrementalDecoder.codecEnabled and name == "test_decoder":
        latin1 = codecs.lookup("latin-1")
        return codecs.CodecInfo(
            name="test_decoder",
            encode=latin1.encode,
            decode=None,
            incrementalencoder=None,
            streamreader=None,
            streamwriter=None,
            incrementaldecoder=StatefulIncrementalDecoder,
        )


TEST_CASES = [
    (b"abcd", False, "a.b.c.d."),
    (b"oiabcd", True, "abcd."),
    (b"oi...abcd...", True, "abcd."),
    (b"i.o6.x.xyz.toolongtofit.", False, "x-----.xyz---.toolon."),
    (b"i.i2.o6xyz", True, "xy----.z-----."),
    (b"i.o3.i6.abcdefghijklmnop", True, "abc.ghi.mno."),
    (
        b"i.o29.a.b.cde.o15.abcdefghijabcdefghij.i3.a.b.c.d.ei00k.l.m",
        True,
        "a----------------------------."
        + "b----------------------------."
        + "cde--------------------------."
        + "abcdefghijabcde."
        + "a.b------------."
        + ".c.------------."
        + "d.e------------."
        + "k--------------."
        + "l--------------."
        + "m--------------.",
    ),
]

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
    for inp, _, _ in TEST_CASES:
        run_one(path, inp, 0)
    for inp, _, _ in TEST_CASES:
        offset = CHUNK_SIZE - len(inp) // 2
        prefix = b"." * offset
        run_one(path, prefix + inp, offset * 2)
finally:
    StatefulIncrementalDecoder.codecEnabled = 0
    os.unlink(path)

print("OK")
