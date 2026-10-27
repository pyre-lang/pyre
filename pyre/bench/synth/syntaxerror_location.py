# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# A compile-time SyntaxError carries its parser location: msg, lineno, offset,
# text, filename and end_lineno populate the instance (args is
# (msg, (filename, lineno, offset, text, end_lineno, end_offset))), and __str__
# renders 'msg (filename, line N)'.  The message text, offset and end positions
# differ between parsers, so this fixture asserts only the facts stable across
# CPython, PyPy and pyre's parser: attribute presence and types, the args
# shape, and the __str__ structure.
# SyntaxError.__str__ also renders 'msg (filename, line lineno)', degrading
# through the filename-only, lineno-only and bare-msg shapes; a non-str msg
# falls back to str(msg). (The end_lineno range form is not exercised here:
# PyPy renders 'lines N-M' while CPython keeps 'line N', so the two references
# diverge. A filename carrying directory separators is likewise avoided:
# basename stripping is os-sep-aware and diverges between the references on
# Windows.)
def main():
    try:
        compile("x = (1 +\n", "prog.py", "exec")
    except SyntaxError as e:
        print(type(e).__name__)
        print(e.lineno == 1)
        print(isinstance(e.offset, int) and e.offset >= 1)
        print(e.filename == "prog.py")
        print(isinstance(e.text, str) and "x = (1 +" in e.text)
        print(isinstance(e.end_lineno, int))
        print(len(e.args) == 2)
        print(e.args[0] == e.msg)
        print(isinstance(e.args[1], tuple) and len(e.args[1]) == 6)
        print(isinstance(e.msg, str) and str(e).startswith(e.msg))
        print("prog.py" in str(e) and "line" in str(e))


main()


def syntaxerror_str_main():
    cases = [
        SyntaxError("bad", ("f.py", 3, 5, "code")),
        SyntaxError("bad"),
        SyntaxError(),
        SyntaxError("only msg", (None, 7, 1, "x")),
    ]
    for e in cases:
        print(repr(str(e)))


syntaxerror_str_main()
