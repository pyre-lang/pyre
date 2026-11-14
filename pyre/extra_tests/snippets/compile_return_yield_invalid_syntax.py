# pyre-check: gate=1
# `return yield 42` matches no expression alternative, so the diagnostic is
# the generic one pointing at the `yield` token. test_repl asserts that
# shape; this snippet keeps compile() on it after a parser Display change.

src = "def f():\n    return yield 42\n"
try:
    compile(src, "<stdin>", "exec")
except SyntaxError as error:
    assert error.msg == "invalid syntax", error.msg
    assert error.text == "    return yield 42\n", repr(error.text)
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (
        2,
        12,
        2,
        17,
    ), (error.lineno, error.offset, error.end_lineno, error.end_offset)
else:
    raise AssertionError("return yield 42 compiled")
