# pyre-check: gate=1
# Unbound scanner `__next__` must refuse a foreign receiver with one TypeError
# both before and after the call is compiled.
import re

pat = re.compile("a")
scanner = getattr(pat, "scanner", None)
it = scanner("a") if scanner is not None else pat.finditer("a")
T = type(it)

first = None
for i in range(3000):
    try:
        T.__next__(object())
    except BaseException as e:
        m = "%s: %s" % (type(e).__name__, e)
    else:
        raise AssertionError("expected an exception")
    if first is None:
        first = m
    elif m != first:
        raise AssertionError("FLIP at %s | %s | %s" % (i, first, m))
