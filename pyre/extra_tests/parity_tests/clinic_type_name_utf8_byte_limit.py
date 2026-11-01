# pyre-check: pypy-diverges: PyPy's _typed_unwrap_error prints the full type name.
# CPython-suite gap: byteorder TypeError checks do not cover a type name whose
# UTF-8 encoding straddles the 50-byte argument-clinic precision limit.
# parity-tests reason: CPython 3.15t _PyArg_BadArgument caps the rejected type's
# UTF-8 name at 50 bytes, dropping an incomplete final code point.


def rejected_type_name(value, expected):
    for method, call in (
        ("to_bytes", lambda: (1).to_bytes(1, value)),
        ("from_bytes", lambda: int.from_bytes(b"\x01", value)),
    ):
        try:
            call()
        except TypeError as exc:
            assert str(exc) == (
                method + "() argument 'byteorder' must be str, not " + expected
            ), str(exc)
        else:
            raise AssertionError("expected TypeError")


rejected_type_name(None, "None")
for name, expected in (
    ("short", "short"),
    ("a" * 50, "a" * 50),
    ("a" * 60, "a" * 50),
    ("a" * 48 + "é" + "z", "a" * 48 + "é"),
    ("a" * 49 + "é" + "z", "a" * 49),
    ("a" * 49 + "한" + "z", "a" * 49),
    ("a" * 49 + "🐍" + "z", "a" * 49),
):
    rejected_type_name(type(name, (), {})(), expected)
print("OK")
