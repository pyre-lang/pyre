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

print("OK")
