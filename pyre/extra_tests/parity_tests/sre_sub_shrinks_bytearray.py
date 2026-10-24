# CPython-suite gap: test_re never resizes a bytearray from inside re.sub.
# parity-tests reason: subx and split_w hold one buffer export for the whole scan, so a callback that resizes the bytearray is refused and the export is released when the scan returns.
# pyre-check: pypy-diverges: make_ctx releases the export immediately and BufMatchContext reads the live buffer with ctx.end fixed at entry, so pypy3 reads past the shrunk buffer (the oracle prints b'aaYY') instead of raising BufferError.
#
# sre_template_object:
# test_re does not call _sre.template on a list subclass, a wide group index, or inspect
# the template object's type. _sre.template accepts list subclasses and ssize_t indexes,
# and returns an _sre.SRE_Template rather than a tuple. PyPy 3.11 re has no
# _compile_template and no _sre.SRE_Template; pypy3 raises AttributeError on
# re._compile_template.

"""sub holds the subject's buffer export across the replacement callback."""

import gc
import re


def bytearray_subclass_payload_survives_collect():
    """A `bytearray` subclass keeps its payload across collect, then drops it."""

    class B(bytearray):
        pass

    payload = b"x" * 4096
    items = []
    i = 0
    while i < 300:
        items.append(B(payload))
        i += 1
    i = 0
    while i < 300:
        if i % 2 == 0:
            items[i].extend(b"yz")
        i += 1
    gc.collect()
    contents_ok = True
    i = 0
    while i < 300:
        item = items[i]
        if i % 2 == 0:
            if len(item) != 4098 or bytes(item) != payload + b"yz":
                contents_ok = False
        elif len(item) != 4096 or bytes(item) != payload:
            contents_ok = False
        i += 1
    print(
        "subclass_payload n=%d even_len=%d odd_len=%d contents_ok=%s"
        % (len(items), len(items[0]), len(items[1]), contents_ok)
    )
    assert contents_ok
    del items
    gc.collect()


bytearray_subclass_payload_survives_collect()

subject = bytearray(b"aaXaaXaa")


def repl(match):
    del subject[match.end() :]
    return b"Y"


raised = False
try:
    re.sub(br"X", repl, subject)
except BufferError:
    raised = True
assert raised, "re.sub callback resize must raise BufferError"
assert bytes(subject) == b"aaXaaXaa", bytes(subject)
# state_fini released the export: a later resize is allowed.
subject.append(ord(b"Z"))
assert bytes(subject) == b"aaXaaXaaZ", bytes(subject)

# split_w has no callback. The export is still acquired for the scan and
# released before split returns, so the bytearray is resizable afterwards.
split_subject = bytearray(b"aaXaaXaa")
parts = re.split(br"X", split_subject)
assert parts == [b"aa", b"aa", b"aa"], parts
del split_subject[:]
assert bytes(split_subject) == b"", bytes(split_subject)

# A bytes pattern on memoryview(bytearray): the view exports the bytearray,
# and subx holds that export across the callback, so the resize is refused.
view_subject = bytearray(b"aaXaaXaa")
view = memoryview(view_subject)


def repl_view(match):
    del view_subject[match.end() :]
    return b"Y"


view_raised = False
try:
    re.sub(br"X", repl_view, view)
except BufferError:
    view_raised = True
assert view_raised, "re.sub on memoryview(bytearray) must refuse the resize"
assert bytes(view_subject) == b"aaXaaXaa", bytes(view_subject)

def sre_template_object():
    """_sre.template builds an SRE_Template and accepts wide indexes."""

    import re
    import _sre

    pattern = re.compile("(a)")
    template = re._compile_template(pattern, r"\1")
    assert type(template).__name__ == "SRE_Template", type(template)
    assert type(template).__module__ == "_sre", type(template).__module__
    assert re.sub("(a)", r"\1\1", "aa") == "aaaa"


    class MyList(list):
        pass


    built = _sre.template(pattern, MyList(["", 1, ""]))
    assert type(built).__name__ == "SRE_Template", type(built)


    class I(int):
        pass


    wide = _sre.template(pattern, ["", 2**40, ""])
    assert type(wide).__name__ == "SRE_Template"
    subclass_index = _sre.template(pattern, ["", I(1), ""])
    assert type(subclass_index).__name__ == "SRE_Template"
    bool_index = _sre.template(pattern, ["", True, ""])
    assert type(bool_index).__name__ == "SRE_Template"

    try:
        _sre.template(pattern, ["", 2**70, ""])
    except OverflowError as error:
        assert str(error) == "Python int too large to convert to C ssize_t", error
    else:
        raise AssertionError("2**70 must raise OverflowError")

    try:
        _sre.template(pattern, ["", -(2**70), ""])
    except OverflowError as error:
        assert str(error) == "Python int too large to convert to C ssize_t", error
    else:
        raise AssertionError("-(2**70) must raise OverflowError")

    try:
        _sre.template(pattern, ["", -1, ""])
    except TypeError as error:
        assert str(error) == "invalid template", error
    else:
        raise AssertionError("a negative index must be an invalid template")

    try:
        _sre.template(pattern, ["", "x", ""])
    except TypeError as error:
        assert str(error) == "an integer is required", error
    else:
        raise AssertionError("a non-int index must require an integer")

    try:
        _sre.template(pattern, ("", 1, ""))
    except TypeError as error:
        assert "must be list" in str(error), error
    else:
        raise AssertionError("a tuple must be refused")
sre_template_object()
print("OK")
