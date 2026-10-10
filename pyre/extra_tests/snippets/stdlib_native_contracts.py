# pyre-check: gate=1
"""Observable contracts at native/stdlib boundaries not owned by lib-python."""

import _json
import array
import codecs
import errno
import functools
import io
import json
import marshal
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import types
from ctypes import (
    CFUNCTYPE,
    Structure,
    c_byte,
    c_double,
    c_int,
    c_longlong,
    c_short,
    c_ubyte,
    c_void_p,
    cast,
)


def raised(call, exc_type):
    try:
        call()
    except exc_type as exc:
        return exc
    raise AssertionError("%s was not raised" % exc_type.__name__)


def refusal(call, exc_type=TypeError):
    return str(raised(call, exc_type))


# array: distinct converters must name the operand and refusal they reached.
def ints():
    return array.array("i", [1, 2, 3])


def doubles():
    return array.array("d", [1.0, 2.0])


BAD_ARGUMENT = "bad argument type for built-in operation"
assert refusal(lambda: ints() + object()) == 'can only append array (not "object") to array'
assert refusal(lambda: ints() + [1]) == 'can only append array (not "list") to array'
assert refusal(lambda: ints() + doubles()) == BAD_ARGUMENT
assert refusal(lambda: ints().__iadd__(object())) == 'can only extend array with array (not "object")'
assert refusal(lambda: ints().__iadd__(doubles())) == "can only extend with array of same kind"
assert refusal(lambda: ints().extend(doubles())) == "can only extend with array of same kind"
assert refusal(lambda: ints().__setitem__(slice(0, 2), object())) == (
    'can only assign array (not "object") to array slice'
)
assert refusal(lambda: ints().__setitem__(slice(0, 2), [1, 2])) == (
    'can only assign array (not "list") to array slice'
)
assert refusal(lambda: ints().__setitem__(slice(0, 2), doubles())) == BAD_ARGUMENT
assert refusal(lambda: ints().fromlist(1)) == "arg must be list"
assert refusal(lambda: ints().frombytes("ab")) == "a bytes-like object is required, not 'str'"
assert refusal(lambda: ints() * "x") == "can't multiply sequence by non-int of type 'str'"
for code in "bBhHiIlLqQ":
    assert refusal(lambda c=code: array.array(c, [1.5])) == (
        "'float' object cannot be interpreted as an integer"
    )
    assert refusal(lambda c=code: array.array(c, [None])) == (
        "'NoneType' object cannot be interpreted as an integer"
    )
reconstruct = array._array_reconstructor
assert refusal(lambda: reconstruct(array.array, "i", "x", b"")) == (
    "'str' object cannot be interpreted as an integer"
)
assert refusal(lambda: reconstruct(array.array, "i", 0, 1)) == (
    "fourth argument should be bytes, not int"
)
assert refusal(lambda: reconstruct(array.array, b"i", 0, b"")) == (
    "_array_reconstructor() argument 2 must be a unicode character, not bytes"
)
assert refusal(lambda: reconstruct(int, "i", 0, b"")) == "int is not a subtype of array.array"
assert reconstruct(array.array, "b", 1, b"\x01\x02").tolist() == [1, 2]


# Codec registries distinguish a search function from an error handler.
assert refusal(lambda: codecs.register(1)) == "argument must be callable"
assert refusal(lambda: codecs.register_error("x", 1)) == "handler must be callable"
assert "str" in refusal(lambda: codecs.register_error(1, len))


def codec_handler(exc):
    return "?", exc.end


codecs.register_error("pyre-native-contract", codec_handler)
assert codecs.lookup_error("pyre-native-contract") is codec_handler
assert "\xe9".encode("ascii", "pyre-native-contract") == b"?"


# Wrapper method names are part of tracebacks, help and argument errors.
partial = functools.partial
assert partial.__new__.__qualname__ == "partial.__new__"
assert partial.__repr__.__qualname__ == "partial.__repr__"
assert partial(pow, 2)(3) == 8

stat_value = os.stat_result((0,) * 10)
assert stat_value.st_atime_ns is None
assert stat_value.st_mtime_ns is None


# The native template compiler validates the app-level parser's result.
saved_parse_template = re._parser.parse_template
re._parser.parse_template = lambda template, pattern: ()
try:
    assert "must be list" in refusal(lambda: re.sub("a", r"\g<0>", "a"))
finally:
    re._parser.parse_template = saved_parse_template


# Formatting identifies the integer/string specifier that rejected `z`.
INTEGER_Z = "Negative zero coercion (z) not allowed in integer format specifier"
STRING_Z = "Negative zero coercion (z) not allowed in string format specifier"
for value, spec, want in (
    (1, "z", INTEGER_Z),
    (1, "zd", INTEGER_Z),
    (True, "z", INTEGER_Z),
    ("a", "z", STRING_Z),
    ("a", "zs", STRING_Z),
):
    assert refusal(lambda v=value, s=spec: format(v, s), ValueError) == want
assert format(-0.0, "z") == "0.0"
assert format(-0.0, "zf") == "0.000000"


# A shared builtin formatter is usable only for receivers accepted by the
# descriptor's owner.  The implementation cannot recognise the shared body
# alone: subclasses may install another builtin type's descriptor.
class StrWithIntFormat(str):
    __format__ = int.__format__


class IntWithStrFormat(int):
    __format__ = str.__format__


for value, spec, owner, receiver in (
    (StrWithIntFormat("x"), ">5", "int", "StrWithIntFormat"),
    (IntWithStrFormat(3), ">5", "str", "IntWithStrFormat"),
):
    message = str(raised(lambda v=value, s=spec: format(v, s), TypeError))
    assert owner in message and receiver in message, message


# Closed I/O layers intentionally carry different spellings.
PERIOD = "I/O operation on closed file."
NO_PERIOD = "I/O operation on closed file"
text = io.TextIOWrapper(io.BufferedWriter(io.FileIO(os.open(os.devnull, os.O_WRONLY), "w")))
raw = io.FileIO(os.open(os.devnull, os.O_WRONLY), "w")
string = io.StringIO()
byte = io.BytesIO()
base = io.IOBase()
for stream in (text, raw, string, byte, base):
    stream.close()
for call in (lambda: text.write("x"), base.flush, base.isatty, base._checkClosed,
             lambda: base.writelines([]), lambda: byte.write(b"x")):
    assert refusal(call, ValueError) == PERIOD
for call in (lambda: raw.write(b"x"), lambda: string.write("x"),
             lambda: string.line_buffering, lambda: string.newlines):
    assert refusal(call, ValueError) == NO_PERIOD
reader = io.BufferedReader(io.FileIO(os.open(os.devnull, os.O_RDONLY), "r"))
writer = io.BufferedWriter(io.FileIO(os.open(os.devnull, os.O_WRONLY), "w"))
reader.close()
writer.close()
assert refusal(reader.flush, ValueError) == PERIOD
assert refusal(reader.detach, ValueError) == PERIOD
assert refusal(writer.flush, ValueError) == "flush of closed file"
assert refusal(lambda: reader.read(), ValueError) == "read of closed file"


# Standard streams must not bypass their closed objects via raw descriptors.
CLOSE_PROGRAM = r'''
import sys

report = sys.stderr
def rejects(call):
    try:
        call()
    except ValueError as exc:
        assert "closed file" in str(exc)
        return
    raise AssertionError("closed stream answered")

sys.stdout.close()
# Unbuffered stdout (`-u` / PYTHONUNBUFFERED) has no BufferedWriter layer:
# create_stdio keeps the FileIO as TextIOWrapper.buffer, so there is no `.raw`.
buf = sys.stdout.buffer
raw = getattr(buf, "raw", buf)
assert buf.closed and raw.closed
for call in (lambda: print("lost"), lambda: sys.stdout.write("lost"),
             sys.stdout.flush, sys.stdout.fileno, sys.stdout.writable):
    rejects(call)
assert sys.stdout.readable() is False
sys.stdin.close()
for call in (sys.stdin.read, sys.stdin.flush, sys.stdin.fileno, sys.stdin.readable):
    rejects(call)
assert sys.stdin.writable() is False
print("REFUSED", file=report)
'''
done = subprocess.run(
    [sys.executable, "-c", CLOSE_PROGRAM],
    stdin=subprocess.DEVNULL,
    capture_output=True,
    text=True,
)
assert done.returncode == 0 and done.stdout == "" and "REFUSED" in done.stderr, done


# Native displayhook/json machinery does not import builtins by name.
def displayhook_with(entry, value):
    saved = sys.modules.get("builtins", ...)
    if entry is ...:
        sys.modules.pop("builtins", None)
    else:
        sys.modules["builtins"] = entry
    try:
        try:
            sys.displayhook(value)
        except BaseException as exc:
            return type(exc).__name__, str(exc)
        return None
    finally:
        if saved is ...:
            sys.modules.pop("builtins", None)
        else:
            sys.modules["builtins"] = saved


LOST = ("RuntimeError", "lost builtins module")
assert displayhook_with(..., 3) == LOST
assert displayhook_with(..., None) == LOST
assert displayhook_with(None, None) is None
assert displayhook_with(42, None) is None
assert displayhook_with(None, 3)[0] == "AttributeError"
saved_builtins = sys.modules["builtins"]
sys.modules["builtins"] = None
try:
    dumped = json.dumps({"b": 1, "a": 2}, sort_keys=True)
finally:
    sys.modules["builtins"] = saved_builtins
assert dumped == '{"a": 2, "b": 1}'
raised(lambda: json.dumps({"a": 1, 2: 3}, sort_keys=True), TypeError)


# Module/type name objects survive surrogate and heap-type assignment paths.
module_name = "mod-\udcff"
module = types.ModuleType(module_name)
assert module.__name__ is module_name
reseeded = types.ModuleType("anonymous")
reseeded.__init__(module_name)
assert reseeded.__name__ is module_name
assert repr(module_name)[1:-1] in repr(module)


class TaggedName(str):
    pass


class Renamed:
    pass


tagged = TaggedName("Tagged")
Renamed.__name__ = tagged
assert Renamed.__name__ is tagged and type(Renamed.__name__) is TaggedName


# A function exposes the real dict used for keyword-only defaults.  Switching
# it to general-key storage must not detach the compiled/default reader.
def kwdefault(*, value=5):
    return value


kwdefaults = kwdefault.__kwdefaults__
assert type(kwdefaults) is dict and kwdefault.__kwdefaults__ is kwdefaults
kwdefaults[1] = "one"
kwdefaults[(2, 3)] = "tuple"
assert list(kwdefaults) == ["value", 1, (2, 3)]
assert dict(kwdefaults) == {"value": 5, 1: "one", (2, 3): "tuple"}
kwdefaults["value"] = 8
assert kwdefault() == 8


# CPython's math helpers preserve operation order and reader-specific errors.
for x, base_value in ((1e300, 10), (8, 2), (1 << 60, 2), (3, 7), (1e-300, 10)):
    assert math.log(x, base_value) == math.log(x) / math.log(base_value)
for base_value in (1, 1.0, True):
    assert refusal(lambda b=base_value: math.log(2, b), ZeroDivisionError) == "division by zero"
assert refusal(lambda: math.log(0, -1.0), ValueError) == "expected a positive input"
assert refusal(lambda: math.log(0.0, -1.0), ValueError) == "expected a positive input, got 0.0"
assert refusal(lambda: math.log(float("nan"), 0), ValueError) == "expected a positive input"
assert refusal(lambda: math.log(2, 0.0), ValueError) == "expected a positive input, got 0.0"
assert refusal(lambda: math.log(2, 0), ValueError) == "expected a positive input"
assert refusal(lambda: math.log1p(-1.0), ValueError) == (
    "expected argument value > -1, got -1.0"
)
assert refusal(lambda: math.log1p(-2), ValueError) == (
    "expected argument value > -1, got -2.0"
)
assert refusal(lambda: math.acosh(0.5), ValueError) == (
    "expected argument value not less than 1, got 0.5"
)
assert refusal(lambda: math.acosh(0), ValueError) == (
    "expected argument value not less than 1, got 0.0"
)
assert refusal(lambda: math.lgamma(-1.0), ValueError) == (
    "expected a noninteger or positive integer, got -1.0"
)
assert refusal(lambda: math.lgamma(0.0), ValueError) == (
    "expected a noninteger or positive integer, got 0.0"
)
assert refusal(lambda: math.gamma(-1.0), ValueError) == (
    "expected a noninteger or positive integer, got -1.0"
)
for n in [10**k for k in range(1, 300)] + [1 << k for k in range(1, 1000)]:
    assert math.log(n) == math.log(float(n))
    assert math.log2(n) == math.log2(float(n))
    assert math.log10(n) == math.log10(float(n))


# sys.intern has separate converter and exact-string refusals.
class StringSubclass(str):
    pass


assert refusal(lambda: sys.intern(b"x")) == "intern() argument must be str, not bytes"
assert refusal(lambda: sys.intern(3)) == "intern() argument must be str, not int"
assert refusal(lambda: sys.intern(StringSubclass("x"))) == "can't intern StringSubclass"


# POSIX path/string/tokenizer readers distinguish their embedded-NUL errors.
if sys.platform in ("linux", "darwin"):
    assert refusal(lambda: open("a\0b"), ValueError) == "embedded null byte"
    assert refusal(lambda: open(b"a\0b"), ValueError) == "embedded null byte"
    assert refusal(lambda: os.putenv("a\0b", "c"), ValueError) == "embedded null byte"
    assert refusal(lambda: os.system("a\0b"), ValueError) == "embedded null byte"
    assert refusal(lambda: os.stat("a\0b"), ValueError) == (
        "stat: embedded null character in path"
    )
    assert refusal(lambda: os.listdir("a\0b"), ValueError) == (
        "listdir: embedded null character in path"
    )
    assert refusal(lambda: compile("a\0b", "<s>", "exec"), SyntaxError) == (
        "source code string cannot contain null bytes"
    )
    # A symlink whose destination already exists reports EEXIST and carries
    # both operands: `filename` is the source, `filename2` the destination.
    with tempfile.TemporaryDirectory() as symlink_dir:
        occupied = os.path.join(symlink_dir, "occupied")
        with open(occupied, "w"):
            pass
        try:
            os.symlink(occupied, occupied)
        except OSError as exc:
            assert exc.errno == errno.EEXIST
            assert (exc.filename, exc.filename2) == (occupied, occupied)
        else:
            raise AssertionError("symlink over an existing path succeeded")


# Untrusted UTF-8 is checked before it becomes the runtime's string storage.
def loads_both(payload):
    size = len(payload)
    return (
        lambda: marshal.loads(b"u" + size.to_bytes(4, "little") + payload),
        lambda: pickle.loads(b"\x80\x04\x8c" + bytes([size]) + payload + b"."),
    )


for payload, reason in (
    (b"\xed\xc0\x80", "byte 0xed in position 0: invalid continuation byte"),
    (b"\xed\xa0\x80\xff", "byte 0xff in position 3: invalid start byte"),
    (b"\x41\xff", "byte 0xff in position 1: invalid start byte"),
    (b"\xed\xa0\x80\xc3", "byte 0xc3 in position 3: unexpected end of data"),
):
    want = "'utf-8' codec can't decode " + reason
    for load in loads_both(payload):
        assert refusal(load, UnicodeDecodeError) == want
assert marshal.loads(b"u\x03\x00\x00\x00\xed\xa0\x80") == "\ud800"
assert pickle.loads(b"\x80\x04\x8c\x03\xed\xa0\x80.") == "\ud800"
subject = "中" * 100
assert refusal(lambda: json.JSONDecoder().scan_once(subject, 200), StopIteration) == "200"
assert refusal(lambda: _json.scanstring(subject, 200), ValueError) == "end is out of bounds"
assert refusal(lambda: b"ab".hex(chr(0xDC80)), ValueError) == "sep must be ASCII."
for text_arg, position in ((chr(0xDC80), 0), ("41" + chr(0xDC80) + "42", 2)):
    want = "non-hexadecimal number found in fromhex() arg at position %d" % position
    assert refusal(lambda a=text_arg: bytes.fromhex(a), ValueError) == want
    assert refusal(lambda a=text_arg: bytearray.fromhex(a), ValueError) == want


# ctypes callback argument/result classification, including nested pointers.
class Pair(Structure):
    _fields_ = [("x", c_int), ("y", c_double)]


pair_callback = CFUNCTYPE(c_double, Pair)(lambda pair: pair.x + pair.y)
assert pair_callback(Pair(7, 0.5)) == 7.5
assert CFUNCTYPE(c_longlong, c_longlong)(lambda value: value + 0x100000000)(9) == 0x100000009
for ctype, values in (
    (c_byte, (-1, 127)),
    (c_ubyte, (0, 255)),
    (c_short, (-2, 0x7FFF)),
    (c_int, (-3, 0x7FFFFFFF)),
):
    callback = CFUNCTYPE(ctype, ctype)(lambda value: value)
    for value in values:
        assert callback(value) == value
INNER = CFUNCTYPE(c_int, c_int)
APPLY = CFUNCTYPE(c_int, INNER, c_int)
inner = INNER(lambda value: value + 1)
inner_address = cast(inner, c_void_p).value


def apply_body(function, value):
    assert cast(function, c_void_p).value == inner_address
    return function(value) * 2


assert APPLY(apply_body)(inner, 20) == 42
assert cast(pair_callback, c_void_p).value

print("stdlib native contracts ok")
