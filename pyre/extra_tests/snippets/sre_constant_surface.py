# pyre-check: gate=1
# CPython-suite gap: test_re reads `MAXGROUPS` out of the module under test, so
# both sides of every comparison move together and a wrong value is invisible;
# `MAGIC`, `CODESIZE` and `MAXREPEAT` are never checked against a number at all.
# parity-tests reason: `_sre.MAXGROUPS` carried `INT32_MAX`, because the engine
# crate halves its own unsigned `MAXREPEAT` where `sre.h` halves `INT32_MAX`.

# pyre-check: pypy-diverges: `rsre_char.py` sets `MAXGROUPS = 2**31 - 1` on a
# 64-bit host, which is the wide number pyre used to publish, and pypy3 is 3.11
# so it carries the older `MAGIC` and has no `re.PatternError`.

import re
import re._constants
import _sre

# `Modules/_sre/sre.h`, 64-bit: `SRE_MAXREPEAT` is the whole `Py_UCS4`, while
# `SRE_MAXGROUPS` halves `INT32_MAX` — half the *signed* maximum, which is one
# bit narrower than half of `MAXREPEAT`.
assert _sre.MAGIC == 20230612, _sre.MAGIC
assert _sre.CODESIZE == 4, _sre.CODESIZE
assert _sre.MAXREPEAT == 2**32 - 1, _sre.MAXREPEAT
assert _sre.MAXGROUPS == 2**30 - 1, _sre.MAXGROUPS
assert _sre.MAXGROUPS == 0x7FFFFFFF // 2, _sre.MAXGROUPS

# `re/_constants.py` re-exports the module's numbers rather than restating them.
assert re._constants.MAXGROUPS == _sre.MAXGROUPS
assert re._constants.MAXREPEAT == _sre.MAXREPEAT
# `_compiler.py` asserts this pairing at import time, so a MAGIC that drifts
# from the bundled stdlib breaks `import re` rather than one call.
assert re._constants.MAGIC == _sre.MAGIC

# The number bounds the group references `_parser.py` accepts, but it does not
# bracket them: a reference below the bound reaches a later check that raises
# the same message at the same position.  Only the constant shows the value.
for n in (2**30 - 2, 2**30 - 1, 2**31 - 1):
    try:
        re.compile(r"(?P<a>)(?(%d))" % n)
    except re.PatternError as e:
        assert str(e) == "invalid group reference %d at position 10" % n, (n, e)
    else:
        raise AssertionError("(?(%d)) compiled" % n)

# `AppReCache.get_re` imports through builtin `__import__` and stores `w_re`
# once. A first template compile while `sys.modules["re"]` is None, or with
# `__import__` replaced, observes that import; after a successful import,
# replacing `sys.modules["re"]` does not change which `re` `_sre` calls.
import builtins
import sys

saved_re = sys.modules["re"]
sys.modules["re"] = None
try:
    re.compile(b"a").sub(b"\\g<0>y", b"a")
except ModuleNotFoundError as e:
    assert "import of re halted" in str(e), e
else:
    raise AssertionError("expected ModuleNotFoundError while sys.modules['re'] is None")
finally:
    sys.modules["re"] = saved_re

saved_import = builtins.__import__


def _hijack(*a, **k):
    raise RuntimeError("hijacked")


builtins.__import__ = _hijack
try:
    re.compile(b"a").sub(b"\\g<0>y", b"a")
except RuntimeError as e:
    assert str(e) == "hijacked", e
else:
    raise AssertionError("expected RuntimeError from replaced builtins.__import__")
finally:
    builtins.__import__ = saved_import

assert re.compile(b"a").sub(b"\\g<0>x", b"a") == b"ax"
saved_re = sys.modules["re"]
sys.modules["re"] = None
try:
    assert re.compile(b"a").sub(b"\\g<0>y", b"a") == b"ay"
finally:
    sys.modules["re"] = saved_re

# `W_SRE_Pattern.subx` / `pattern_subx`: a non-callable replacement is not
# type-checked against the subject up front. A literal whose type does not
# match the subject is collected into `sublist_w` and the error comes from
# the subject's `join`; a no-match shortcut returns the subject before join.
# A `getstring` / buffer-acquire failure is not an error here — it makes the
# replacement non-literal, so `re._compile_template` raises.
#
# Spec is 3.15 `pattern_subx`. PyPy 3.11 `W_SRE_Pattern.subx` join messages
# omit "instance" / "a bytes-like object" (`expected str, bytes found` /
# `expected bytes, str found`), raise `a bytes-like object is required, not
# int` from `acquire_py_buffer` instead of the template compiler, have no
# `_sre.template`, and accept a bytearray literal on a str subject
# (`re.sub('a', bytearray(b'x'), 'a')` returns `'x'`). 3.15 raises
# `sequence item 0: expected str instance, bytearray found`.


def _typeerror(fn, msg):
    try:
        fn()
    except TypeError as e:
        assert str(e) == msg, (str(e), msg)
    else:
        raise AssertionError("expected TypeError: %s" % msg)


assert re.sub("z", b"x", "a") == "a"
_typeerror(lambda: re.sub("b", b"x", "ab"), "sequence item 1: expected str instance, bytes found")
_typeerror(
    lambda: re.sub("b", 1, "ab"),
    "decoding to str: need a bytes-like object, int found",
)
_typeerror(
    lambda: re.sub("z", 1, "ab"),
    "decoding to str: need a bytes-like object, int found",
)
_typeerror(
    lambda: re.sub(b"b", "x", b"ab"),
    "sequence item 1: expected a bytes-like object, str found",
)
_typeerror(
    lambda: re.sub(b"b", "x\\0", b"ab"),
    "sequence item 1: expected a bytes-like object, str found",
)
_typeerror(
    lambda: re.sub("b", b"x\\0", "ab"),
    "sequence item 1: expected str instance, bytes found",
)
_typeerror(
    lambda: re.sub("b", bytearray(b"x\\0"), "ab"),
    "sequence item 1: expected str instance, bytes found",
)
_typeerror(
    lambda: re.sub("a", bytearray(b"x"), "a"),
    "sequence item 0: expected str instance, bytearray found",
)
_typeerror(
    lambda: re.compile("(a)").sub(_sre.template(re.compile("a"), ["x", 1, "y"]), "a"),
    "decoding to str: need a bytes-like object, _sre.SRE_Template found",
)

assert re.sub("b", "x", "ab") == "ax"
assert re.sub("(b)", r"\1X", "ab") == "abX"
assert re.sub(b"b", b"x", b"ab") == b"ax"
assert re.sub(b"(b)", br"\1X", b"ab") == b"abX"
_typeerror(
    lambda: re.sub("b", b"x", "abb", count=1),
    "sequence item 1: expected str instance, bytes found",
)
_typeerror(lambda: re.subn("b", b"x", "ab"), "sequence item 1: expected str instance, bytes found")
assert re.subn("z", b"x", "ab") == ("ab", 0)
assert re.sub("b", lambda m: "X", "ab") == "aX"
assert re.sub("z", b"x", "ab") == "ab"

# Callable filter: `use_builder == '\x00'`. Each non-None piece is appended
# to `sublist_w` and the result is the subject's empty-string `join`; type
# errors come from `join`. None is skipped. A no-match shortcut returns the
# subject before the filter is called.
_typeerror(
    lambda: re.sub("b", lambda m: 1, "ab"),
    "sequence item 1: expected str instance, int found",
)
_typeerror(
    lambda: re.sub(b"b", lambda m: "x", b"ab"),
    "sequence item 1: expected a bytes-like object, str found",
)
_typeerror(
    lambda: re.sub("b", lambda m: b"x", "ab"),
    "sequence item 1: expected str instance, bytes found",
)
assert re.sub(b"b", lambda m: bytearray(b"x"), b"ab") == b"ax"
assert re.sub("b", lambda m: None, "ab") == "a"
assert re.sub("z", lambda m: 1, "ab") == "ab"
assert re.subn("b", lambda m: "X", "abab") == ("aXaX", 2)

# PyPy `ObjSpace.acquire_py_buffer` reads a native `W_BufferExporter` first.
# A bytes/bytearray/array subclass that defines `__buffer__` still contributes
# its native payload; the override is not called. CPython honours the slot.
if sys.implementation.name != "cpython":
    import array

    _buf_calls = []

    class BA(bytearray):
        def __buffer__(self, flags):
            _buf_calls.append("ba")
            return memoryview(b"x")

    assert re.sub(b"a", BA(b"z"), b"a") == b"z"
    assert _buf_calls == [], _buf_calls
    _buf_calls.clear()

    class AR(array.array):
        def __buffer__(self, flags):
            _buf_calls.append("ar")
            return memoryview(b"x")

    assert re.sub(b"a", AR("B", b"z"), b"a") == b"z"
    assert _buf_calls == [], _buf_calls


# PEP 688 Python exporter: `getstring` → `PyObject_GetBuffer` accepts
# `P.__buffer__`. CPython may call twice through `bytes.join`; assert the
# value only.
class P:
    def __buffer__(self, flags):
        return memoryview(b"x")


assert re.sub(b"a", P(), b"a") == b"x"

# `W_SRE_Pattern.subx`: no-match shortcut returns the subject before
# allocating `sublist_w`.
assert re.sub(b"q", b"r", b"abc") == b"abc"
assert re.subn("q", "r", "abc") == ("abc", 0)

print("OK")
