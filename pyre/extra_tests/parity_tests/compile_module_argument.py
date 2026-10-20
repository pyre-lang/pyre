# pyre-check: pypy-diverges: PyPy 3.11 compile() has no module= / _feature_version=.
# CPython-suite gap: test_compile.test_filter_syntax_warnings_by_module covers
# filter matching but not the TypeError text or WarningMessage.module for
# omitted / None / str.
# parity-tests reason: importlib/_bootstrap_external.source_to_code passes
# module=fullname; pin the 3.15 builtin surface against the 3.15.0 oracle.

import warnings

assert compile("1+1", "<s>", "eval", module=None)
assert compile("1+1", "<s>", "eval", module="pkg.mod")
assert compile("1+1", "<s>", "eval", _feature_version=-1)
code = compile("1+1", "<s>", "eval", module="m")
assert code.co_filename == "<s>"
assert eval(code) == 2

for bad, name in ((1, "int"), (b"x", "bytes"), (object(), "object")):
    try:
        compile("1", "<s>", "eval", module=bad)
    except TypeError as e:
        assert str(e) == (
            f"compile() argument 'module' must be str or None, not {name}"
        ), e
    else:
        raise AssertionError(f"module={bad!r} accepted")

src = r'"\q"'
warning_text = (
    '"\\q" is an invalid escape sequence. '
    "Such sequences will not work in the future. "
    'Did you mean "\\\\q"? A raw string is also an option.'
)


def capture(mod):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mod is Ellipsis:
            compile(src, "file.py", "eval")
        else:
            compile(src, "file.py", "eval", module=mod)
        return [
            (
                type(w.message).__name__,
                str(w.message),
                w.filename,
                w.lineno,
                w.module,
            )
            for w in caught
        ]


omitted = capture(Ellipsis)
none = capture(None)
named = capture("pkg.mod")
empty = capture("")
assert omitted == [
    ("SyntaxWarning", warning_text, "file.py", 1, None)
], omitted
assert none == omitted, none
assert named == [
    ("SyntaxWarning", warning_text, "file.py", 1, "pkg.mod")
], named
assert empty == [
    ("SyntaxWarning", warning_text, "file.py", 1, "")
], empty

print("OK")
