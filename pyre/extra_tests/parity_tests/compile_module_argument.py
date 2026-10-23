# pyre-check: pypy-diverges: PyPy 3.11 compile() has no module= / _feature_version=.
# CPython-suite gap: test_compile.test_filter_syntax_warnings_by_module covers
# filter matching but not the TypeError text or WarningMessage.module for
# omitted / None / str.
# parity-tests reason: importlib/_bootstrap_external.source_to_code passes
# module=fullname; pin the 3.15 builtin surface against the 3.15.0 oracle.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

import gc
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

c = compile("1+1", "e402_name.py", "eval")
print(c.co_filename is c.co_filename)
junk = [object() for _ in range(4000)]
gc.collect()
print(c.co_filename is c.co_filename)
print(c.co_filename)
c2 = c.replace(co_filename="other.py")
print(c2.co_filename)
print(c2.co_filename is c2.co_filename)
print(c.co_filename)


def _nested_filenames(code):
    names = set()
    stack = [code]
    while stack:
        item = stack.pop()
        names.add(item.co_filename)
        for const in item.co_consts:
            if hasattr(const, "co_code"):
                stack.append(const)
    return names


# Many sibling code objects so the compilation-unit filename walk keeps
# indexing `co_consts` after a nested replacement that can collect.
_src = "\n".join("def f%d():\n    return (lambda: %d)()" % (i, i) for i in range(200))
_unit = compile(_src, "orig.py", "exec")
print(sorted(_nested_filenames(_unit)))
import _imp

_imp._fix_co_filename(_unit, "renamed.py")
print(sorted(_nested_filenames(_unit)))

import os
import py_compile
import shutil
import sys
import tempfile

_tmp = tempfile.mkdtemp()
_renamed_dir = None
try:
    _src_dir = os.path.join(_tmp, "src")
    os.makedirs(_src_dir)
    _mod_path = os.path.join(_src_dir, "unit_filenames.py")
    with open(_mod_path, "w", encoding="utf-8") as _fh:
        _fh.write(_src + "\n")
    py_compile.compile(_mod_path, doraise=True)
    _renamed_dir = os.path.join(_tmp, "renamed")
    shutil.move(_src_dir, _renamed_dir)
    sys.path.insert(0, _renamed_dir)
    import unit_filenames

    _imported_names = _nested_filenames(unit_filenames.f0.__code__)
    print(len(_imported_names) == 1)
    print(_imported_names == {unit_filenames.f0.__code__.co_filename})
finally:
    sys.modules.pop("unit_filenames", None)
    if _renamed_dir and sys.path and sys.path[0] == _renamed_dir:
        sys.path.pop(0)
    shutil.rmtree(_tmp)

print("OK")
