# pyre-check: gate=1
# pyre-check: pypy-diverges: pins staticmethod.__text_signature__ (3.15); pypy3's staticmethod has no such attribute
# CPython-suite gap: exact builtin and descriptor text signatures are not asserted.
# parity-tests reason: guard pyre's CPython 3.15 signature metadata surface.

"""CPython 3.15 text signatures for builtins and builtin types.

These cases share one process and one descriptor-checking harness so adding a
type does not duplicate the test machinery or interpreter startup.
"""

import array
import builtins
import inspect
import itertools
import sys
import types


def check_descriptors(owner, expected):
    for name, signature in expected.items():
        descriptor = owner.__dict__[name]
        actual = descriptor.__text_signature__
        assert actual == signature, (owner, name, actual)


# bool_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__invert__": "($self, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
}

check_descriptors(bool, EXPECTED)

assert str(inspect.signature(bool.__repr__)) == "(self, /)"
assert str(inspect.signature(bool.__and__)) == "(self, value, /)"


# builtin_text_signatures_python315

EXPECTED = {
    "__import__": "($module, /, name, globals=None, locals=None, fromlist=(),\n           level=0)",
    "abs": "($module, number, /)",
    "aiter": "($module, async_iterable, /)",
    "all": "($module, iterable, /)",
    "anext": "($module, async_iterator, default=<unrepresentable>, /)",
    "any": "($module, iterable, /)",
    "ascii": "($module, obj, /)",
    "bin": "($module, integer, /)",
    "breakpoint": "($module, /, *args, **kws)",
    "callable": "($module, obj, /)",
    "chr": "($module, i, /)",
    "compile": "($module, /, source, filename, mode, flags=0,\n        dont_inherit=False, optimize=-1, *, module=None,\n        _feature_version=-1)",
    "delattr": "($module, obj, name, /)",
    "divmod": "($module, x, y, /)",
    "eval": "($module, source, /, globals=None, locals=None)",
    "exec": "($module, source, /, globals=None, locals=None, *, closure=None)",
    "format": "($module, value, format_spec='', /)",
    "globals": "($module, /)",
    "hasattr": "($module, obj, name, /)",
    "hash": "($module, obj, /)",
    "hex": "($module, integer, /)",
    "id": "($module, obj, /)",
    "input": "($module, prompt='', /)",
    "isinstance": "($module, obj, class_or_tuple, /)",
    "issubclass": "($module, cls, class_or_tuple, /)",
    "len": "($module, obj, /)",
    "locals": "($module, /)",
    "oct": "($module, integer, /)",
    "open": "($module, /, file, mode='r', buffering=-1, encoding=None,\n     errors=None, newline=None, closefd=True, opener=None)",
    "ord": "($module, character, /)",
    "pow": "($module, /, base, exp, mod=None)",
    "print": "($module, /, *objects, sep=' ', end='\\n', file=None, flush=False)",
    "repr": "($module, obj, /)",
    "round": "($module, /, number, ndigits=None)",
    "setattr": "($module, obj, name, value, /)",
    "sorted": "($module, iterable, /, *, key=None, reverse=False)",
    "sum": "($module, iterable, /, start=0)",
}

for name, signature in EXPECTED.items():
    assert getattr(builtins, name).__text_signature__ == signature, name

for name in ("__build_class__", "dir", "getattr", "iter", "max", "min", "next", "vars"):
    assert getattr(builtins, name).__text_signature__ is None, name

assert str(inspect.signature(len)) == "(obj, /)"
assert str(inspect.signature(sorted)) == (
    "(iterable, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(open)) == (
    "(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, "
    "closefd=True, opener=None)"
)
assert str(inspect.signature(print)) == (
    "(*objects, sep=' ', end='\\n', file=None, flush=False)"
)

# Constructor signatures are metadata on the type object, separate from the
# signatures on each type's ``__new__`` descriptor.
TYPE_EXPECTED = {
    bool: "(object=False, /)",
    float: "(x=0, /)",
    complex: "(real=0, imag=0)",
    memoryview: "(object)",
    list: "(iterable=(), /)",
    tuple: "(iterable=(), /)",
    set: "(iterable=(), /)",
    frozenset: "(iterable=(), /)",
    property: "(fget=None, fset=None, fdel=None, doc=None)",
    enumerate: "(iterable, start=0)",
    reversed: "(object, /)",
    map: "(function, iterable, /, *iterables, strict=False)",
    filter: "(function, iterable, /)",
    zip: "(*iterables, strict=False)",
    itertools.count: "(start=0, step=1)",
}

for cls, signature in TYPE_EXPECTED.items():
    assert cls.__text_signature__ == signature, cls

for cls in (
    int,
    str,
    bytes,
    bytearray,
    dict,
    range,
    slice,
    super,
    type,
    itertools.repeat,
):
    assert cls.__text_signature__ is None, cls


# itertools_count_metadata_python315

COUNT_EXPECTED = {
    "__new__": (
        "Create and return a new object.  See help(type) for accurate signature.",
        "($type, *args, **kwargs)",
    ),
    "__repr__": ("Return repr(self).", "($self, /)"),
    "__getattribute__": ("Return getattr(self, name).", "($self, name, /)"),
    "__iter__": ("Implement iter(self).", "($self, /)"),
    "__next__": ("Implement next(self).", "($self, /)"),
}

for name, (doc, signature) in COUNT_EXPECTED.items():
    descriptor = itertools.count.__dict__[name]
    assert descriptor.__doc__ == doc, name
    assert descriptor.__text_signature__ == signature, name

assert str(inspect.signature(itertools.count)) == "(start=0, step=1)"
assert "__reduce__" not in itertools.count.__dict__

for name, args in {
    "__repr__": ({},),
    "__getattribute__": ({}, "x"),
    "__iter__": ({},),
    "__next__": ({},),
}.items():
    try:
        itertools.count.__dict__[name](*args)
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'itertools.count' object "
            "but received a 'dict'"
        )
    else:
        raise AssertionError(name)


# itertools_repeat_metadata_python315

REPEAT_EXPECTED = {
    "__new__": (
        "builtin_function_or_method",
        "Create and return a new object.  See help(type) for accurate signature.",
        "($type, *args, **kwargs)",
    ),
    "__repr__": ("wrapper_descriptor", "Return repr(self).", "($self, /)"),
    "__getattribute__": (
        "wrapper_descriptor",
        "Return getattr(self, name).",
        "($self, name, /)",
    ),
    "__iter__": ("wrapper_descriptor", "Implement iter(self).", "($self, /)"),
    "__next__": ("wrapper_descriptor", "Implement next(self).", "($self, /)"),
    "__length_hint__": (
        "method_descriptor",
        "Private method returning an estimate of len(list(it)).",
        "($self, /)",
    ),
}

for name, (carrier, doc, signature) in REPEAT_EXPECTED.items():
    descriptor = itertools.repeat.__dict__[name]
    assert type(descriptor).__name__ == carrier, name
    assert descriptor.__doc__ == doc, name
    assert descriptor.__text_signature__ == signature, name

assert "__reduce__" not in itertools.repeat.__dict__

for name, args in {
    "__repr__": ({},),
    "__getattribute__": ({}, "x"),
    "__iter__": ({},),
    "__next__": ({},),
}.items():
    try:
        itertools.repeat.__dict__[name](*args)
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' requires a 'itertools.repeat' object "
            "but received a 'dict'"
        )
    else:
        raise AssertionError(name)

try:
    itertools.repeat.__dict__["__length_hint__"]({})
except TypeError as exc:
    assert str(exc) == (
        "descriptor '__length_hint__' for 'itertools.repeat' objects "
        "doesn't apply to a 'dict' object"
    )
else:
    raise AssertionError("__length_hint__")


# bytearray_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__buffer__": "($self, flags, /)",
    "__release_buffer__": "($self, buffer, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__iadd__": "($self, value, /)",
    "__imul__": "($self, value, /)",
    "__alloc__": "($self, /)",
    "__reduce__": "($self, /)",
    "__reduce_ex__": "($self, proto=0, /)",
    "__sizeof__": "($self, /)",
    "append": "($self, item, /)",
    "capitalize": "($self, /)",
    "center": "($self, width, fillchar=b' ', /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "count": "($self, sub[, start[, end]], /)",
    "decode": "($self, /, encoding='utf-8', errors='strict')",
    "endswith": "($self, suffix[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "extend": "($self, iterable_of_ints, /)",
    "find": "($self, sub[, start[, end]], /)",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "index": "($self, sub[, start[, end]], /)",
    "insert": "($self, index, item, /)",
    "isalnum": "($self, /)",
    "isalpha": "($self, /)",
    "isascii": "($self, /)",
    "isdigit": "($self, /)",
    "islower": "($self, /)",
    "isspace": "($self, /)",
    "istitle": "($self, /)",
    "isupper": "($self, /)",
    "join": "($self, iterable_of_bytes, /)",
    "ljust": "($self, width, fillchar=b' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, bytes=None, /)",
    "partition": "($self, sep, /)",
    "pop": "($self, index=-1, /)",
    "remove": "($self, value, /)",
    "replace": "($self, old, new, /, count=-1)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "resize": "($self, size, /)",
    "reverse": "($self, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=b' ', /)",
    "rpartition": "($self, sep, /)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "rstrip": "($self, bytes=None, /)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "splitlines": "($self, /, keepends=False)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "strip": "($self, bytes=None, /)",
    "swapcase": "($self, /)",
    "title": "($self, /)",
    "translate": "($self, table, /, delete=b'')",
    "upper": "($self, /)",
    "zfill": "($self, width, /)",
}

check_descriptors(bytearray, EXPECTED)

raw_maketrans = bytearray.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert bytearray.maketrans.__text_signature__ == "(frm, to, /)"
assert bytearray.__dict__["fromhex"].__text_signature__ == "($type, string, /)"

assert str(inspect.signature(bytearray.decode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(bytearray.resize)) == "(self, size, /)"
assert str(inspect.signature(bytearray.fromhex)) == "(string, /)"


# bytes_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__buffer__": "($self, flags, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__getnewargs__": "($self, /)",
    "__bytes__": "($self, /)",
    "capitalize": "($self, /)",
    "center": "($self, width, fillchar=b' ', /)",
    "count": "($self, sub[, start[, end]], /)",
    "decode": "($self, /, encoding='utf-8', errors='strict')",
    "endswith": "($self, suffix[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "find": "($self, sub[, start[, end]], /)",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "index": "($self, sub[, start[, end]], /)",
    "isalnum": "($self, /)",
    "isalpha": "($self, /)",
    "isascii": "($self, /)",
    "isdigit": "($self, /)",
    "islower": "($self, /)",
    "isspace": "($self, /)",
    "istitle": "($self, /)",
    "isupper": "($self, /)",
    "join": "($self, iterable_of_bytes, /)",
    "ljust": "($self, width, fillchar=b' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, bytes=None, /)",
    "partition": "($self, sep, /)",
    "replace": "($self, old, new, /, count=-1)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=b' ', /)",
    "rpartition": "($self, sep, /)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "rstrip": "($self, bytes=None, /)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "splitlines": "($self, /, keepends=False)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "strip": "($self, bytes=None, /)",
    "swapcase": "($self, /)",
    "title": "($self, /)",
    "translate": "($self, table, /, delete=b'')",
    "upper": "($self, /)",
    "zfill": "($self, width, /)",
}

check_descriptors(bytes, EXPECTED)

raw_maketrans = bytes.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert bytes.maketrans.__text_signature__ == "(frm, to, /)"
assert bytes.__dict__["fromhex"].__text_signature__ == "($type, string, /)"

assert str(inspect.signature(bytes.decode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(bytes.replace)) == "(self, old, new, /, count=-1)"
assert str(inspect.signature(bytes.fromhex)) == "(string, /)"


# complex_text_signatures_python315

VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "conjugate", "__complex__", "__getnewargs__",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "from_number": "($type, number, /)",
    "__format__": "($self, format_spec, /)",
}

check_descriptors(complex, EXPECTED)

assert str(inspect.signature(complex.from_number)) == "(number, /)"
assert str(inspect.signature(complex.__pow__)) == "(self, value, mod=None, /)"
assert str(inspect.signature(complex.conjugate)) == "(self, /)"


# dict_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__sizeof__": "($self, /)",
    "get": "($self, key, default=None, /)",
    "setdefault": "($self, key, default=None, /)",
    "pop": "($self, key, default=<unrepresentable>, /)",
    "popitem": "($self, /)",
    "keys": "($self, /)",
    "items": "($self, /)",
    "values": "($self, /)",
    "update": None,
    "fromkeys": "($type, iterable, value=None, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "__reversed__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(dict, EXPECTED)

assert str(inspect.signature(dict.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(dict.get)) == "(self, key, default=None, /)"
assert str(inspect.signature(dict.fromkeys)) == "(iterable, value=None, /)"
assert str(inspect.signature(dict.__class_getitem__)) == "(object, /)"


# float_text_signatures_python315

VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__mod__", "__rmod__", "__divmod__", "__rdivmod__", "__floordiv__",
        "__rfloordiv__", "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "__int__", "__float__", "conjugate", "__trunc__", "__floor__", "__ceil__",
        "as_integer_ratio", "hex", "is_integer", "__getnewargs__",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "from_number": "($type, number, /)",
    "__round__": "($self, ndigits=None, /)",
    "fromhex": "($type, string, /)",
    "__getformat__": "($type, typestr, /)",
    "__format__": "($self, format_spec, /)",
}

check_descriptors(float, EXPECTED)

assert str(inspect.signature(float.from_number)) == "(number, /)"
assert str(inspect.signature(float.fromhex)) == "(string, /)"
assert str(inspect.signature(float.__pow__)) == "(self, value, mod=None, /)"


# functional_iterator_text_signatures_python315

METHODS = {
    enumerate: ("__iter__", "__next__", "__reduce__", "__class_getitem__"),
    reversed: (
        "__iter__",
        "__next__",
        "__length_hint__",
        "__reduce__",
        "__setstate__",
    ),
    map: ("__iter__", "__next__", "__reduce__", "__setstate__"),
    filter: ("__iter__", "__next__", "__reduce__"),
    zip: ("__iter__", "__next__", "__reduce__", "__setstate__"),
}

for typ, names in METHODS.items():
    assert typ.__dict__["__new__"].__text_signature__ == "($type, *args, **kwargs)"
    for name in names:
        expected = (
            "($type, object, /)"
            if name == "__class_getitem__"
            else "($self, object, /)"
            if name == "__setstate__"
            else "($self, /)"
        )
        assert typ.__dict__[name].__text_signature__ == expected, (typ, name)

assert str(inspect.signature(enumerate.__next__)) == "(self, /)"
assert str(inspect.signature(reversed.__setstate__)) == "(self, object, /)"
assert str(inspect.signature(enumerate.__class_getitem__)) == "(object, /)"


class Sequence:
    def __getitem__(self, index):
        raise IndexError


ITERATOR_METHODS = [
    (
        type(iter(Sequence())),
        ("__iter__", "__next__", "__reduce__", "__length_hint__", "__setstate__"),
    ),
    (
        type(iter([])),
        ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"),
    ),
    (
        type(reversed([])),
        ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"),
    ),
    (
        type(iter(())),
        ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"),
    ),
    (
        type(iter(b"")),
        ("__iter__", "__next__", "__reduce__", "__length_hint__", "__setstate__"),
    ),
    (
        type(iter(bytearray())),
        ("__iter__", "__next__", "__reduce__", "__length_hint__", "__setstate__"),
    ),
    (
        type(iter("")),
        ("__iter__", "__next__", "__reduce__", "__length_hint__", "__setstate__"),
    ),
    (
        type(iter("é")),
        ("__iter__", "__next__", "__reduce__", "__length_hint__", "__setstate__"),
    ),
    (
        type(iter(range(1))),
        ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"),
    ),
    (
        type(iter(range(2**100))),
        ("__iter__", "__next__", "__length_hint__", "__reduce__", "__setstate__"),
    ),
    (type(iter(set())), ("__iter__", "__next__", "__length_hint__", "__reduce__")),
    (type(iter({})), ("__iter__", "__next__", "__length_hint__", "__reduce__")),
    (
        type(iter({}.values())),
        ("__iter__", "__next__", "__length_hint__", "__reduce__"),
    ),
    (type(iter({}.items())), ("__iter__", "__next__", "__length_hint__", "__reduce__")),
    (type(reversed({})), ("__iter__", "__next__", "__length_hint__", "__reduce__")),
    (
        type(reversed({}.values())),
        ("__iter__", "__next__", "__length_hint__", "__reduce__"),
    ),
    (
        type(reversed({}.items())),
        ("__iter__", "__next__", "__length_hint__", "__reduce__"),
    ),
    (type(iter(lambda: None, None)), ("__iter__", "__next__", "__reduce__")),
    (type(iter(memoryview(b""))), ("__iter__", "__next__")),
    (
        type(iter(array.array("i"))),
        ("__iter__", "__next__", "__reduce__", "__setstate__"),
    ),
]

for typ, names in ITERATOR_METHODS:
    for name in names:
        parameter = (
            "state"
            if typ.__name__ in ("arrayiterator", "range_iterator", "longrange_iterator")
            else "object"
        )
        expected = (
            f"($self, {parameter}, /)" if name == "__setstate__" else "($self, /)"
        )
        assert typ.__dict__[name].__text_signature__ == expected, (typ, name)

callable_iterator = type(iter(lambda: None, None))
assert "__doc__" in callable_iterator.__dict__
assert callable_iterator.__dict__["__doc__"] is None
assert str(inspect.signature(type(iter([])).__setstate__)) == "(self, object, /)"
assert (
    str(inspect.signature(type(iter(array.array("i"))).__setstate__))
    == "(self, state, /)"
)


# int_text_signatures_python315

VALUE_BINARY = {
    name: "($self, value, /)"
    for name in (
        "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
        "__mod__", "__rmod__", "__divmod__", "__rdivmod__", "__lshift__",
        "__rlshift__", "__rshift__", "__rrshift__", "__and__", "__rand__",
        "__xor__", "__rxor__", "__or__", "__ror__", "__floordiv__",
        "__rfloordiv__", "__truediv__", "__rtruediv__",
    )
}
SELF_ONLY = {
    name: "($self, /)"
    for name in (
        "__repr__", "__hash__", "__neg__", "__pos__", "__abs__", "__bool__",
        "__invert__", "__int__", "__float__", "__index__", "conjugate",
        "bit_length", "bit_count", "as_integer_ratio", "__trunc__", "__floor__",
        "__ceil__", "__getnewargs__", "__sizeof__", "is_integer",
    )
}
EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    **VALUE_BINARY,
    **SELF_ONLY,
    "__pow__": "($self, value, mod=None, /)",
    "__rpow__": "($self, value, mod=None, /)",
    "to_bytes": "($self, /, length=1, byteorder='big', *, signed=False)",
    "from_bytes": "($type, /, bytes, byteorder='big', *, signed=False)",
    "__round__": "($self, ndigits=None, /)",
    "__format__": "($self, format_spec, /)",
}

check_descriptors(int, EXPECTED)

assert str(inspect.signature(int.to_bytes)) == (
    "(self, /, length=1, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.from_bytes)) == (
    "(bytes, byteorder='big', *, signed=False)"
)
assert str(inspect.signature(int.__pow__)) == "(self, value, mod=None, /)"


# list_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__len__": "($self, /)",
    "__getitem__": "($self, index, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__iadd__": "($self, value, /)",
    "__imul__": "($self, value, /)",
    "__reversed__": "($self, /)",
    "__sizeof__": "($self, /)",
    "clear": "($self, /)",
    "copy": "($self, /)",
    "append": "($self, object, /)",
    "insert": "($self, index, object, /)",
    "extend": "($self, iterable, /)",
    "pop": "($self, index=-1, /)",
    "remove": "($self, value, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "reverse": "($self, /)",
    "sort": "($self, /, *, key=None, reverse=False)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(list, EXPECTED)

assert str(inspect.signature(list.__getitem__)) == "(self, index, /)"
assert str(inspect.signature(list.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(list.sort)) == (
    "(self, /, *, key=None, reverse=False)"
)
assert str(inspect.signature(list.__class_getitem__)) == "(object, /)"


# memoryview_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__buffer__": "($self, flags, /)",
    "__release_buffer__": "($self, buffer, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__setitem__": "($self, key, value, /)",
    "__delitem__": "($self, key, /)",
    "release": "($self, /)",
    "tobytes": "($self, /, order='C')",
    "hex": "($self, /, sep=<unrepresentable>, bytes_per_sep=1)",
    "tolist": "($self, /)",
    "cast": "($self, /, format, shape=<unrepresentable>)",
    "toreadonly": "($self, /)",
    "_from_flags": "($type, /, object, flags)",
    "count": "($self, value, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "__enter__": "($self, /)",
    "__exit__": "($self, /, *exc_info)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(memoryview, EXPECTED)

assert str(inspect.signature(memoryview.tobytes)) == "(self, /, order='C')"
assert str(inspect.signature(memoryview.index)) == (
    f"(self, value, start=0, stop={sys.maxsize}, /)"
)
assert str(inspect.signature(memoryview._from_flags)) == "(object, flags)"


# method_wrapper_text_signatures_python315

EXPECTED = {
    staticmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__call__": "($self, /, *args, **kwargs)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
    classmethod: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__get__": "($self, instance, owner=None, /)",
        "__init__": "($self, /, *args, **kwargs)",
        "__class_getitem__": "($type, object, /)",
    },
}

for typ, signatures in EXPECTED.items():
    for name, signature in signatures.items():
        assert typ.__dict__[name].__text_signature__ == signature, (typ, name)

assert str(inspect.signature(staticmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(staticmethod.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(classmethod.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(classmethod.__class_getitem__)) == "(object, /)"


# property_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__get__": "($self, instance, owner=None, /)",
    "__set__": "($self, instance, value, /)",
    "__delete__": "($self, instance, /)",
    "getter": "($self, object, /)",
    "setter": "($self, object, /)",
    "deleter": "($self, object, /)",
    "__set_name__": "($self, owner, name, /)",
}

check_descriptors(property, EXPECTED)

assert str(inspect.signature(property.__get__)) == (
    "(self, instance, owner=None, /)"
)
assert str(inspect.signature(property.setter)) == "(self, object, /)"
assert str(inspect.signature(property.__set_name__)) == "(self, owner, name, /)"


# range_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__bool__": "($self, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__reversed__": "($self, /)",
    "__reduce__": "($self, /)",
    "count": "($self, object, /)",
    "index": "($self, object, /)",
}

check_descriptors(range, EXPECTED)

assert str(inspect.signature(range.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(range.__contains__)) == "(self, key, /)"
assert str(inspect.signature(range.count)) == "(self, object, /)"


# set_text_signatures_python315

COMMON = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__sub__": "($self, value, /)",
    "__rsub__": "($self, value, /)",
    "__and__": "($self, value, /)",
    "__rand__": "($self, value, /)",
    "__xor__": "($self, value, /)",
    "__rxor__": "($self, value, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__len__": "($self, /)",
    "__contains__": "($self, object, /)",
    "copy": "($self, /)",
    "difference": "($self, /, *others)",
    "intersection": "($self, /, *others)",
    "isdisjoint": "($self, other, /)",
    "issubset": "($self, other, /)",
    "issuperset": "($self, other, /)",
    "__reduce__": "($self, /)",
    "__sizeof__": "($self, /)",
    "symmetric_difference": "($self, other, /)",
    "union": "($self, /, *others)",
    "__class_getitem__": "($type, object, /)",
}

SET_ONLY = {
    "__init__": "($self, /, *args, **kwargs)",
    "__isub__": "($self, value, /)",
    "__iand__": "($self, value, /)",
    "__ixor__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "add": "($self, object, /)",
    "clear": "($self, /)",
    "discard": "($self, object, /)",
    "difference_update": "($self, /, *others)",
    "intersection_update": "($self, /, *others)",
    "pop": "($self, /)",
    "remove": "($self, object, /)",
    "symmetric_difference_update": "($self, other, /)",
    "update": "($self, /, *others)",
}

for cls, expected in (
    (set, {**COMMON, **SET_ONLY}),
    (frozenset, {**COMMON, "__hash__": "($self, /)"}),
):
    for name, signature in expected.items():
        assert cls.__dict__[name].__text_signature__ == signature, (cls, name)

assert str(inspect.signature(set.difference)) == "(self, /, *others)"
assert str(inspect.signature(set.update)) == "(self, /, *others)"
assert str(inspect.signature(frozenset.symmetric_difference)) == "(self, other, /)"
assert str(inspect.signature(frozenset.__class_getitem__)) == "(object, /)"


# slice_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "indices": "($self, object, /)",
    "__reduce__": "($self, /)",
}

check_descriptors(slice, EXPECTED)

assert str(inspect.signature(slice.__eq__)) == "(self, value, /)"
assert str(inspect.signature(slice.indices)) == "(self, object, /)"
assert str(inspect.signature(slice.__reduce__)) == "(self, /)"


# str_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__str__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__mod__": "($self, value, /)",
    "__rmod__": "($self, value, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "encode": "($self, /, encoding='utf-8', errors='strict')",
    "replace": "($self, old, new, /, count=-1)",
    "split": "($self, /, sep=None, maxsplit=-1)",
    "rsplit": "($self, /, sep=None, maxsplit=-1)",
    "join": "($self, iterable, /)",
    "capitalize": "($self, /)",
    "casefold": "($self, /)",
    "title": "($self, /)",
    "center": "($self, width, fillchar=' ', /)",
    "count": "($self, sub[, start[, end]], /)",
    "expandtabs": "($self, /, tabsize=8)",
    "find": "($self, sub[, start[, end]], /)",
    "partition": "($self, sep, /)",
    "index": "($self, sub[, start[, end]], /)",
    "ljust": "($self, width, fillchar=' ', /)",
    "lower": "($self, /)",
    "lstrip": "($self, chars=None, /)",
    "rfind": "($self, sub[, start[, end]], /)",
    "rindex": "($self, sub[, start[, end]], /)",
    "rjust": "($self, width, fillchar=' ', /)",
    "rstrip": "($self, chars=None, /)",
    "rpartition": "($self, sep, /)",
    "splitlines": "($self, /, keepends=False)",
    "strip": "($self, chars=None, /)",
    "swapcase": "($self, /)",
    "translate": "($self, table, /)",
    "upper": "($self, /)",
    "startswith": "($self, prefix[, start[, end]], /)",
    "endswith": "($self, suffix[, start[, end]], /)",
    "removeprefix": "($self, prefix, /)",
    "removesuffix": "($self, suffix, /)",
    "isascii": "($self, /)",
    "islower": "($self, /)",
    "isupper": "($self, /)",
    "istitle": "($self, /)",
    "isspace": "($self, /)",
    "isdecimal": "($self, /)",
    "isdigit": "($self, /)",
    "isnumeric": "($self, /)",
    "isalpha": "($self, /)",
    "isalnum": "($self, /)",
    "isidentifier": "($self, /)",
    "isprintable": "($self, /)",
    "zfill": "($self, width, /)",
    "format": "($self, /, *args, **kwargs)",
    "format_map": "($self, mapping, /)",
    "__format__": "($self, format_spec, /)",
    "__sizeof__": "($self, /)",
    "__getnewargs__": "($self, /)",
}

check_descriptors(str, EXPECTED)

raw_maketrans = str.__dict__["maketrans"]
assert not hasattr(raw_maketrans, "__text_signature__")
assert str.maketrans.__text_signature__ == (
    "(x, y=<unrepresentable>, z=<unrepresentable>, /)"
)

assert str(inspect.signature(str.encode)) == (
    "(self, /, encoding='utf-8', errors='strict')"
)
assert str(inspect.signature(str.replace)) == "(self, old, new, /, count=-1)"
assert str(inspect.signature(str.format)) == "(self, /, *args, **kwargs)"


# super_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__getattribute__": "($self, name, /)",
    "__get__": "($self, instance, owner=None, /)",
}

check_descriptors(super, EXPECTED)

assert str(inspect.signature(super.__repr__)) == "(self, /)"
assert str(inspect.signature(super.__getattribute__)) == "(self, name, /)"
assert str(inspect.signature(super.__get__)) == "(self, instance, owner=None, /)"


# tuple_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__iter__": "($self, /)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__add__": "($self, value, /)",
    "__mul__": "($self, value, /)",
    "__rmul__": "($self, value, /)",
    "__contains__": "($self, key, /)",
    "__getnewargs__": "($self, /)",
    "index": "($self, value, start=0, stop=sys.maxsize, /)",
    "count": "($self, value, /)",
    "__class_getitem__": "($type, object, /)",
}

check_descriptors(tuple, EXPECTED)

assert str(inspect.signature(tuple.__getitem__)) == "(self, key, /)"
assert str(inspect.signature(tuple.index)) == (
    "(self, value, start=0, stop=9223372036854775807, /)"
)
assert str(inspect.signature(tuple.__class_getitem__)) == "(object, /)"


# type_text_signatures_python315

EXPECTED = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__call__": "($self, /, *args, **kwargs)",
    "__getattribute__": "($self, name, /)",
    "__setattr__": "($self, name, value, /)",
    "__delattr__": "($self, name, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "mro": "($self, /)",
    "__subclasses__": "($self, /)",
    "__prepare__": "($cls, name, bases, /, **kwds)",
    "__instancecheck__": "($self, instance, /)",
    "__subclasscheck__": "($self, subclass, /)",
    "__dir__": "($self, /)",
    "__sizeof__": "($self, /)",
}

check_descriptors(type, EXPECTED)

assert str(inspect.signature(type.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(type.__prepare__)) == "(name, bases, /, **kwds)"
assert str(inspect.signature(type.__instancecheck__)) == "(self, instance, /)"


# internal_type_text_signatures_python315
#
# The signature sits on the type object rather than on a `__new__` descriptor,
# so `inspect.signature` resolves against it instead of falling back to
# `(*args, **kwargs)`.

EXPECTED_TYPES = {
    types.EllipsisType: "()",
    types.GenericAlias: "(origin, args, /)",
    types.NotImplementedType: "()",
    types.NoneType: "()",
    types.ModuleType: "(name, doc=None)",
    types.MappingProxyType: "(mapping)",
    types.TracebackType: "(tb_next, tb_frame, tb_lasti, tb_lineno)",
    types.CellType: "([contents])",
    types.MethodType: "(function, instance, /)",
    types.FunctionType: (
        "(code, globals, name=None, argdefs=None, closure=None,\n"
        "         kwdefaults=None)"
    ),
    types.CodeType: (
        "(argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize,\n"
        "     flags, codestring, constants, names, varnames, filename, name,\n"
        "     qualname, firstlineno, linetable, exceptiontable, freevars=(),\n"
        "     cellvars=(), /)"
    ),
}

for owner, signature in EXPECTED_TYPES.items():
    assert owner.__text_signature__ == signature, (owner, owner.__text_signature__)


# singleton_type_text_signatures_python315

EXPECTED_SINGLETON_METHODS = {
    types.EllipsisType: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__reduce__": "($self, /)",
    },
    types.NotImplementedType: {
        "__new__": "($type, *args, **kwargs)",
        "__repr__": "($self, /)",
        "__reduce__": "($self, /)",
        "__bool__": "($self, /)",
    },
    types.NoneType: {
        "__new__": "($type, *args, **kwargs)",
        "__bool__": "($self, /)",
        "__repr__": "($self, /)",
        "__hash__": "($self, /)",
        "__eq__": "($self, value, /)",
        "__ne__": "($self, value, /)",
        "__lt__": "($self, value, /)",
        "__le__": "($self, value, /)",
        "__gt__": "($self, value, /)",
        "__ge__": "($self, value, /)",
    },
}
for owner, methods in EXPECTED_SINGLETON_METHODS.items():
    check_descriptors(owner, methods)

GENERIC_NEW_DOC = "Create and return a new object.  See help(type) for accurate signature."
EXPECTED_SINGLETON_DOCS = {
    types.EllipsisType: {
        "__new__": GENERIC_NEW_DOC,
        "__repr__": "Return repr(self).",
        "__reduce__": None,
    },
    types.NotImplementedType: {
        "__new__": GENERIC_NEW_DOC,
        "__repr__": "Return repr(self).",
        "__reduce__": None,
        "__bool__": "True if self else False",
    },
    types.NoneType: {
        "__new__": GENERIC_NEW_DOC,
        "__bool__": "True if self else False",
        "__repr__": "Return repr(self).",
        "__hash__": "Return hash(self).",
        "__eq__": "Return self==value.",
        "__ne__": "Return self!=value.",
        "__lt__": "Return self<value.",
        "__le__": "Return self<=value.",
        "__gt__": "Return self>value.",
        "__ge__": "Return self>=value.",
    },
}
for owner, methods in EXPECTED_SINGLETON_DOCS.items():
    for name, doc in methods.items():
        assert getattr(owner, name).__doc__ == doc, (owner, name)


# generator_type_text_signatures_python315

EXPECTED_GENERATOR_METHODS = {
    "__repr__": "($self, /)",
    "__next__": "($self, /)",
    "send": "($self, object, /)",
    "close": "($self, /)",
    "__iter__": "($self, /)",
    "__del__": "($self, /)",
    "__sizeof__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}
check_descriptors(types.GeneratorType, EXPECTED_GENERATOR_METHODS)
assert types.GeneratorType.throw.__text_signature__ is None

EXPECTED_GENERATOR_DOCS = {
    "__repr__": "Return repr(self).",
    "__next__": "Implement next(self).",
    "send": "send(value) -> send 'value' into generator,\nreturn next yielded value or raise StopIteration.",
    "throw": (
        "throw(value)\nthrow(type[,value[,tb]])\n\n"
        "Raise exception in generator, return next yielded value or raise\n"
        "StopIteration.\nthe (type, val, tb) signature is deprecated, \n"
        "and may be removed in a future version of Python."
    ),
    "close": "close() -> raise GeneratorExit inside generator.",
    "__iter__": "Implement iter(self).",
    "__del__": "Called when the instance is about to be destroyed.",
    "__sizeof__": "gen.__sizeof__() -> size of gen in memory, in bytes",
    "__class_getitem__": "generators are generic over the types of their yield, send, and return values",
}
for name, doc in EXPECTED_GENERATOR_DOCS.items():
    assert getattr(types.GeneratorType, name).__doc__ == doc, name


# coroutine_type_text_signatures_python315

EXPECTED_COROUTINE_METHODS = {
    "__repr__": "($self, /)",
    "__del__": "($self, /)",
    "__await__": "($self, /)",
    "send": "($self, object, /)",
    "close": "($self, /)",
    "__sizeof__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}
check_descriptors(types.CoroutineType, EXPECTED_COROUTINE_METHODS)
assert types.CoroutineType.throw.__text_signature__ is None

EXPECTED_COROUTINE_DOCS = {
    "__repr__": "Return repr(self).",
    "__del__": "Called when the instance is about to be destroyed.",
    "__await__": "Return an iterator to be used in await expression.",
    "send": "send(arg) -> send 'arg' into coroutine,\nreturn next iterated value or raise StopIteration.",
    "throw": (
        "throw(value)\nthrow(type[,value[,traceback]])\n\n"
        "Raise exception in coroutine, return next iterated value or raise\n"
        "StopIteration.\nthe (type, val, tb) signature is deprecated, \n"
        "and may be removed in a future version of Python."
    ),
    "close": "close() -> raise GeneratorExit inside coroutine.",
    "__sizeof__": "gen.__sizeof__() -> size of gen in memory, in bytes",
    "__class_getitem__": "coroutines are generic over the types of their yield, send, and return values",
}
for name, doc in EXPECTED_COROUTINE_DOCS.items():
    assert getattr(types.CoroutineType, name).__doc__ == doc, name


# async_generator_type_text_signatures_python315

EXPECTED_ASYNC_GENERATOR_METHODS = {
    "__repr__": "($self, /)",
    "__del__": "($self, /)",
    "__aiter__": "($self, /)",
    "__anext__": "($self, /)",
    "asend": "($self, object, /)",
    "aclose": "($self, /)",
    "__sizeof__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
}
check_descriptors(types.AsyncGeneratorType, EXPECTED_ASYNC_GENERATOR_METHODS)
assert types.AsyncGeneratorType.athrow.__text_signature__ is None

EXPECTED_ASYNC_GENERATOR_DOCS = {
    "__repr__": "Return repr(self).",
    "__del__": "Called when the instance is about to be destroyed.",
    "__aiter__": "Return an awaitable, that resolves in asynchronous iterator.",
    "__anext__": "Return a value or raise StopAsyncIteration.",
    "asend": "asend(v) -> send 'v' in generator.",
    "athrow": (
        "athrow(value)\nathrow(type[,value[,tb]])\n\n"
        "raise exception in generator.\nthe (type, val, tb) signature is deprecated, \n"
        "and may be removed in a future version of Python."
    ),
    "aclose": "aclose() -> raise GeneratorExit inside generator.",
    "__sizeof__": "gen.__sizeof__() -> size of gen in memory, in bytes",
    "__class_getitem__": "async generators are generic over the types of their yield and send values",
}
for name, doc in EXPECTED_ASYNC_GENERATOR_DOCS.items():
    assert getattr(types.AsyncGeneratorType, name).__doc__ == doc, name


# generator_awaitable_type_text_signatures_python315

async def _metadata_coroutine():
    return None


async def _metadata_async_generator():
    yield None


_metadata_coroutine_object = _metadata_coroutine()
_metadata_coroutine_wrapper = _metadata_coroutine_object.__await__()
_metadata_coroutine_wrapper_type = type(_metadata_coroutine_wrapper)
check_descriptors(
    _metadata_coroutine_wrapper_type,
    {
        "__iter__": "($self, /)",
        "__next__": "($self, /)",
        "send": "($self, object, /)",
        "close": "($self, /)",
    },
)
assert _metadata_coroutine_wrapper_type.throw.__text_signature__ is None
assert _metadata_coroutine_wrapper_type.__doc__ == (
    "A wrapper object implementing __await__ for coroutines."
)

_metadata_asend_generator = _metadata_async_generator()
_metadata_asend = _metadata_asend_generator.__anext__()
_metadata_athrow_generator = _metadata_async_generator()
_metadata_athrow = _metadata_athrow_generator.athrow(ValueError)
_metadata_async_awaitable_types = (type(_metadata_asend), type(_metadata_athrow))

EXPECTED_ASYNC_AWAITABLE_METHODS = {
    "__iter__": "($self, /)",
    "__next__": "($self, /)",
    "__del__": "($self, /)",
    "__await__": "($self, /)",
    "send": "($self, object, /)",
    "close": "($self, /)",
}
for owner in _metadata_async_awaitable_types:
    check_descriptors(owner, EXPECTED_ASYNC_AWAITABLE_METHODS)
    assert owner.throw.__text_signature__ is None

EXPECTED_COROUTINE_WRAPPER_DOCS = {
    "__iter__": "Implement iter(self).",
    "__next__": "Implement next(self).",
    "send": "send(arg) -> send 'arg' into coroutine,\nreturn next iterated value or raise StopIteration.",
    "throw": (
        "throw(value)\nthrow(type[,value[,traceback]])\n\n"
        "Raise exception in coroutine, return next iterated value or raise\n"
        "StopIteration.\nthe (type, val, tb) signature is deprecated, \n"
        "and may be removed in a future version of Python."
    ),
    "close": "close() -> raise GeneratorExit inside coroutine.",
}
for name, doc in EXPECTED_COROUTINE_WRAPPER_DOCS.items():
    assert getattr(_metadata_coroutine_wrapper_type, name).__doc__ == doc, name

EXPECTED_ASYNC_AWAITABLE_DOCS = {
    "__iter__": "Implement iter(self).",
    "__next__": "Implement next(self).",
    "__del__": "Called when the instance is about to be destroyed.",
    "__await__": "Return an iterator to be used in await expression.",
    "send": "send(value) -> send 'value' into generator,\nreturn next yielded value or raise StopIteration.",
    "throw": (
        "throw(value)\nthrow(type[,value[,tb]])\n\n"
        "Raise exception in generator, return next yielded value or raise\n"
        "StopIteration.\nthe (type, val, tb) signature is deprecated, \n"
        "and may be removed in a future version of Python."
    ),
    "close": "close() -> raise GeneratorExit inside generator.",
}
for owner in _metadata_async_awaitable_types:
    for name, doc in EXPECTED_ASYNC_AWAITABLE_DOCS.items():
        assert getattr(owner, name).__doc__ == doc, (owner, name)

_metadata_coroutine_wrapper.close()
_metadata_asend.close()
_metadata_athrow.close()
_metadata_asend_generator.__del__()
_metadata_athrow_generator.__del__()


# module_text_signatures_python315

EXPECTED_MODULE_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__init__": "($self, /, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__getattribute__": "($self, name, /)",
    "__dir__": "($self, /)",
}
check_descriptors(types.ModuleType, EXPECTED_MODULE_METHODS)

EXPECTED_MODULE_DOCS = {
    "__new__": "Create and return a new object.  See help(type) for accurate signature.",
    "__init__": "Initialize self.  See help(type(self)) for accurate signature.",
    "__repr__": "Return repr(self).",
    "__getattribute__": "Return getattr(self, name).",
    "__dir__": "__dir__() -> list\nspecialized dir() implementation",
}
for name, doc in EXPECTED_MODULE_DOCS.items():
    assert getattr(types.ModuleType, name).__doc__ == doc, name

assert str(inspect.signature(types.ModuleType.__new__)) == "(*args, **kwargs)"
assert str(inspect.signature(types.ModuleType.__init__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(types.ModuleType.__getattribute__)) == "(self, name, /)"
assert str(inspect.signature(types.ModuleType.__dir__)) == "(self, /)"


# cell_text_signatures_python315

EXPECTED_CELL_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__repr__": "($self, /)",
}
check_descriptors(types.CellType, EXPECTED_CELL_METHODS)

EXPECTED_CELL_DOCS = {
    "__new__": "Create and return a new object.  See help(type) for accurate signature.",
    "__eq__": "Return self==value.",
    "__ne__": "Return self!=value.",
    "__lt__": "Return self<value.",
    "__le__": "Return self<=value.",
    "__gt__": "Return self>value.",
    "__ge__": "Return self>=value.",
    "__repr__": "Return repr(self).",
}
for name, doc in EXPECTED_CELL_DOCS.items():
    assert getattr(types.CellType, name).__doc__ == doc, name

assert str(inspect.signature(types.CellType.__new__)) == "(*args, **kwargs)"
assert str(inspect.signature(types.CellType.__eq__)) == "(self, value, /)"
assert str(inspect.signature(types.CellType.__repr__)) == "(self, /)"


# mappingproxy_text_signatures_python315

EXPECTED_MAPPINGPROXY_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__len__": "($self, /)",
    "__getitem__": "($self, key, /)",
    "__contains__": "($self, key, /)",
    "__iter__": "($self, /)",
    "__str__": "($self, /)",
    "__repr__": "($self, /)",
    "__hash__": "($self, /)",
    "__or__": "($self, value, /)",
    "__ror__": "($self, value, /)",
    "__ior__": "($self, value, /)",
    "__reversed__": "($self, /)",
    "__class_getitem__": "($type, object, /)",
    "get": "($self, key, default=None, /)",
    "keys": "($self, /)",
    "values": "($self, /)",
    "items": "($self, /)",
    "copy": "($self, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
}
check_descriptors(types.MappingProxyType, EXPECTED_MAPPINGPROXY_METHODS)

EXPECTED_MAPPINGPROXY_DOCS = {
    "__new__": "Create and return a new object.  See help(type) for accurate signature.",
    "__len__": "Return len(self).",
    "__getitem__": "Return self[key].",
    "__contains__": "Return bool(key in self).",
    "__iter__": "Implement iter(self).",
    "__str__": "Return str(self).",
    "__repr__": "Return repr(self).",
    "__hash__": "Return hash(self).",
    "__or__": "Return self|value.",
    "__ror__": "Return value|self.",
    "__ior__": "Return self|=value.",
    "__reversed__": "D.__reversed__() -> reverse iterator",
    "__class_getitem__": (
        "mappingproxy objects are generic over two types, signifying "
        "(respectively) the types of their keys and values"
    ),
    "get": "Return the value for key if key is in the mapping, else default.",
    "keys": "D.keys() -> a set-like object providing a view on D's keys",
    "values": "D.values() -> an object providing a view on D's values",
    "items": "D.items() -> a set-like object providing a view on D's items",
    "copy": "D.copy() -> a shallow copy of D",
    "__eq__": "Return self==value.",
    "__ne__": "Return self!=value.",
    "__gt__": "Return self>value.",
    "__ge__": "Return self>=value.",
    "__lt__": "Return self<value.",
    "__le__": "Return self<=value.",
}
for name, doc in EXPECTED_MAPPINGPROXY_DOCS.items():
    assert getattr(types.MappingProxyType, name).__doc__ == doc, name

assert str(inspect.signature(types.MappingProxyType.get)) == (
    "(self, key, default=None, /)"
)
assert str(inspect.signature(types.MappingProxyType.__reversed__)) == "(self, /)"
assert str(inspect.signature(types.MappingProxyType.__class_getitem__)) == "(object, /)"


# traceback_text_signatures_python315

EXPECTED_TRACEBACK_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__dir__": "($self, /)",
}
check_descriptors(types.TracebackType, EXPECTED_TRACEBACK_METHODS)

assert types.TracebackType.__doc__ == "Create a new traceback object."
assert types.TracebackType.__new__.__doc__ == (
    "Create and return a new object.  See help(type) for accurate signature."
)
assert types.TracebackType.__dir__.__doc__ is None
assert str(inspect.signature(types.TracebackType.__new__)) == "(*args, **kwargs)"
assert str(inspect.signature(types.TracebackType.__dir__)) == "(self, /)"


assert types.CodeType.__dict__["__new__"].__text_signature__ == "($type, *args, **kwargs)"
assert str(inspect.signature(types.CodeType.__new__)) == "(*args, **kwargs)"
assert types.CodeType.__new__.__doc__ == (
    "Create and return a new object.  See help(type) for accurate signature."
)

EXPECTED_CODE_METHODS = {
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__hash__": "($self, /)",
    "__repr__": "($self, /)",
    "replace": "($self, /, **changes)",
    "__replace__": "($self, /, **changes)",
    "_varname_from_oparg": "($self, /, oparg)",
    "co_positions": "($self, /)",
    "co_lines": "($self, /)",
    "co_branches": "($self, /)",
}
check_descriptors(types.CodeType, EXPECTED_CODE_METHODS)

EXPECTED_CODE_DOCS = {
    "__eq__": "Return self==value.",
    "__ne__": "Return self!=value.",
    "__lt__": "Return self<value.",
    "__le__": "Return self<=value.",
    "__gt__": "Return self>value.",
    "__ge__": "Return self>=value.",
    "__hash__": "Return hash(self).",
    "__repr__": "Return repr(self).",
    "replace": (
        "Return a copy of the code object with new values for the specified fields."
    ),
    "__replace__": "The same as replace().",
    "_varname_from_oparg": (
        "(internal-only) Return the local variable name for the given oparg.\n\n"
        "WARNING: this method is for internal use only and may change or go\naway."
    ),
}
for name, doc in EXPECTED_CODE_DOCS.items():
    assert getattr(types.CodeType, name).__doc__ == doc, name

assert str(inspect.signature(types.CodeType.replace)) == "(self, /, **changes)"
assert str(inspect.signature(types.CodeType.co_positions)) == "(self, /)"


# frame_text_signatures_python315

EXPECTED_FRAME_METHODS = {
    "clear": "($self, /)",
    "__repr__": "($self, /)",
}
check_descriptors(types.FrameType, EXPECTED_FRAME_METHODS)

assert types.FrameType.clear.__doc__ == "Clear all references held by the frame."
assert types.FrameType.__repr__.__doc__ == "Return repr(self)."
assert str(inspect.signature(types.FrameType.clear)) == "(self, /)"
assert str(inspect.signature(types.FrameType.__repr__)) == "(self, /)"


# method_text_signatures_python315

EXPECTED_METHOD_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__call__": "($self, /, *args, **kwargs)",
    "__get__": "($self, instance, owner=None, /)",
    "__getattribute__": "($self, name, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__hash__": "($self, /)",
    "__repr__": "($self, /)",
    "__reduce__": "($self, /)",
}
check_descriptors(types.MethodType, EXPECTED_METHOD_METHODS)

EXPECTED_METHOD_DOCS = {
    "__new__": "Create and return a new object.  See help(type) for accurate signature.",
    "__call__": "Call self as a function.",
    "__get__": "Return an attribute of instance, which is of type owner.",
    "__getattribute__": "Return getattr(self, name).",
    "__eq__": "Return self==value.",
    "__ne__": "Return self!=value.",
    "__lt__": "Return self<value.",
    "__le__": "Return self<=value.",
    "__gt__": "Return self>value.",
    "__ge__": "Return self>=value.",
    "__hash__": "Return hash(self).",
    "__repr__": "Return repr(self).",
}
for name, doc in EXPECTED_METHOD_DOCS.items():
    assert getattr(types.MethodType, name).__doc__ == doc, name

assert str(inspect.signature(types.MethodType.__call__)) == "(self, /, *args, **kwargs)"
assert str(inspect.signature(types.MethodType.__get__)) == "(self, instance, owner=None, /)"


assert str(inspect.signature(types.MethodType)) == "(function, instance, /)"
assert str(inspect.signature(types.ModuleType)) == "(name, doc=None)"


# simple_namespace_text_signatures_python315

assert types.SimpleNamespace.__doc__ == "A simple attribute-based namespace."
assert types.SimpleNamespace.__text_signature__ == (
    "(mapping_or_iterable=(), /, **kwargs)"
)
EXPECTED_SIMPLE_NAMESPACE_METHODS = {
    "__new__": "($type, *args, **kwargs)",
    "__repr__": "($self, /)",
    "__lt__": "($self, value, /)",
    "__le__": "($self, value, /)",
    "__eq__": "($self, value, /)",
    "__ne__": "($self, value, /)",
    "__gt__": "($self, value, /)",
    "__ge__": "($self, value, /)",
    "__init__": "($self, /, *args, **kwargs)",
    "__reduce__": "($self, /)",
    "__replace__": "($self, /, **changes)",
}
check_descriptors(types.SimpleNamespace, EXPECTED_SIMPLE_NAMESPACE_METHODS)

EXPECTED_SIMPLE_NAMESPACE_DOCS = {
    "__new__": "Create and return a new object.  See help(type) for accurate signature.",
    "__repr__": "Return repr(self).",
    "__lt__": "Return self<value.",
    "__le__": "Return self<=value.",
    "__eq__": "Return self==value.",
    "__ne__": "Return self!=value.",
    "__gt__": "Return self>value.",
    "__ge__": "Return self>=value.",
    "__init__": "Initialize self.  See help(type(self)) for accurate signature.",
    "__reduce__": "Return state information for pickling",
    "__replace__": (
        "Return a copy of the namespace object with new values for the "
        "specified attributes."
    ),
}
for name, doc in EXPECTED_SIMPLE_NAMESPACE_DOCS.items():
    assert getattr(types.SimpleNamespace, name).__doc__ == doc, name

assert str(inspect.signature(types.SimpleNamespace)) == (
    "(mapping_or_iterable=(), /, **kwargs)"
)


# module_doc_python315
#
# Every module publishes `__doc__` in its own dict, so the key is listed by
# `dir()` and the read never falls through to the `module` type's docstring.

assert "__doc__" in builtins.__dict__
assert "__doc__" in dir(builtins)
assert isinstance(builtins.__doc__, str)

for name in ("sys", "gc", "_ast", "_io"):
    assert "__doc__" in __import__(name).__dict__, name

print("OK")
