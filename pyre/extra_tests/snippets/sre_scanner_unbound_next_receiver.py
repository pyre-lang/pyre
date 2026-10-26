# pyre-check: gate=1
# Unbound scanner `__next__` must refuse a foreign receiver with one TypeError
# both before and after the call is compiled.
import re

pat = re.compile("a")
scanner = getattr(pat, "scanner", None)
it = scanner("a") if scanner is not None else pat.finditer("a")
T = type(it)

expected_next = (
    "descriptor '__next__' requires a '_sre.SRE_Scanner' object but received a 'object'"
)
if hasattr(T, "__next__"):
    for i in range(3000):
        try:
            T.__next__(object())
        except TypeError as e:
            if str(e) != expected_next:
                raise AssertionError("at %s: %r" % (i, str(e)))
        else:
            raise AssertionError("expected an exception")


def expect_typeerror(fn, message):
    try:
        fn()
    except TypeError as e:
        if str(e) != message:
            raise AssertionError("got %r want %r" % (str(e), message))
    else:
        raise AssertionError("expected TypeError: %s" % message)


# `_sre_SRE_Scanner_prefixmatch` / `_sre_SRE_Scanner_search`; `__iter__` /
# `__next__` use the slot-wrapper zero-argument refusal.
if hasattr(T, "match"):
    expect_typeerror(lambda: T.match(it, 1), "prefixmatch() takes no arguments")
    expect_typeerror(lambda: T.match(it, x=1), "prefixmatch() takes no arguments")
if hasattr(T, "prefixmatch"):
    expect_typeerror(lambda: T.prefixmatch(it, 1), "prefixmatch() takes no arguments")
if hasattr(T, "search"):
    expect_typeerror(lambda: T.search(it, 1), "search() takes no arguments")
    expect_typeerror(lambda: T.search(it, x=1), "search() takes no arguments")
if hasattr(T, "__iter__"):
    expect_typeerror(lambda: T.__iter__(it, 1), "expected 0 arguments, got 1")
if hasattr(T, "__next__"):
    expect_typeerror(lambda: T.__next__(it, 1), "expected 0 arguments, got 1")
