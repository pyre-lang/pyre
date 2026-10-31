# pyre-check: gate=1
# pyre-check: pypy-diverges: pins 3.14's SyntaxError column offsets, which pypy3 reports differently
# CPython-suite gap: multibyte SyntaxError offsets are not exhaustively asserted.
# parity-tests reason: guard pyre's CPython 3.14 character-column semantics.

"""Python 3.14 SyntaxError positions use character, not UTF-8 byte, columns."""


def syntax_error(source, mode="exec"):
    try:
        compile(source, "<fragment>", mode)
    except SyntaxError as error:
        return error
    raise AssertionError("source unexpectedly compiled")


error = syntax_error('Python = "Ṕýţĥòñ" +')
assert (error.lineno, error.offset) == (1, 20), (
    error.lineno,
    error.offset,
)

error = syntax_error('α = 0xI')
assert (error.lineno, error.offset) == (1, 6), (
    error.lineno,
    error.offset,
)

error = syntax_error(b'\n\n\nPython = "\xcf\xb3\xf2\xee\xed" +')
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (4, 12, 4, 12), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)
assert error.text == 'Python = "ϳ���" +', error.text

error = syntax_error(b"\xef\xbb\xbf#coding: utf8\nprint('\xe6\x88\x91')\n")
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 0, 1, 13), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)
assert error.text == "#coding: utf8", error.text

error = syntax_error('return "ä"')
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 1, 1, 12), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

error = syntax_error('x = rb"é"')
assert error.msg == "bytes can only contain ASCII literal characters", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 5, 1, 10), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

for source, expected in (
    ("f'{6 0}'", (1, 4, 1, 7)),
    ('f"""\n\n\n            {\n            6\n            0="""', (5, 13, 6, 14)),
):
    error = syntax_error(source)
    assert error.msg == "invalid syntax. Perhaps you forgot a comma?", error.msg
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == expected, (
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

error = syntax_error('if True:\nprint "No indent"')
assert isinstance(error, IndentationError), type(error)
assert error.msg == "expected an indented block after 'if' statement on line 1", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (2, 1, 2, 6), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

error = syntax_error('if True:\n        print()\n\texec "mixed tabs and spaces"')
assert isinstance(error, TabError), type(error)
assert error.msg == "inconsistent use of tabs and spaces in indentation", error.msg

error = syntax_error("def f():\n  global x\n  nonlocal x")
assert error.msg == "name 'x' is nonlocal and global", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (2, 3, 2, 11), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

# An unparenthesized generator argument keeps the parser's message and its whole range.
error = syntax_error('"┬ó┬ó┬ó┬ó┬ó┬ó" + f(4, x for x in range(1))')
assert error.msg == "Generator expression must be parenthesized", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 23, 1, 42), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

for operator, message in (
    ("=", "cannot assign to dict comprehension here. Maybe you meant '==' instead of '='?"),
    ("+=", "'dict comprehension' is an illegal expression for augmented assignment"),
):
    error = syntax_error(f"{{x: y for y, x in ((1, 2), (3, 4))}} {operator} 5")
    assert error.msg == message, error.msg
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 1, 1, 36), (
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

# "Perhaps you forgot a comma?" spans both atoms, so it ends past the whole
# second one.  A width measured in bytes rather than in atoms agrees only when
# that atom is a single ASCII character, which `(a b)` is and none of these
# are: the ASCII rows below catch a length assumption and the non-ASCII ones
# catch a byte-vs-character one.
for source, offset, end_offset in (
    ("(a bb)", 2, 6),
    ("(a bbb)", 2, 7),
    ("(1 22)", 2, 6),
    ("(a \u03b2)", 2, 5),
    ("(a \u03b2\u03b2)", 2, 6),
    ("(\u03b1\u03b1 \u03b2)", 2, 6),
    ("[\u03b1 \u03b2]", 2, 5),
    # The f-string forms reach the same rule through a recursive parse of the
    # replacement field, and the span has to survive being mapped back onto
    # the original source.
    ("f'{a bb}'", 4, 8),
    ("f'{\u03b1\u03b1 \u03b2}'", 4, 8),
    ("f'{a \u03b2}'", 4, 7),
):
    error = syntax_error(source)
    assert error.msg == "invalid syntax. Perhaps you forgot a comma?", (
        source,
        error.msg,
    )
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (
        1,
        offset,
        1,
        end_offset,
    ), (
        source,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

# `python.gram` names an assignment target with `_PyPegen_get_expr_name`
# and picks between three messages: `invalid_named_expression`'s form
# suggesting `==`, `RAISE_SYNTAX_ERROR_INVALID_TARGET`'s form naming the
# innermost element of a compound target, and the augmented-assignment
# form.  Which one is reached turns on the target's precedence, on whether
# the target or the value was parenthesized, and on how many targets the
# statement has, so the rows below vary exactly those.
for source, message, position in (
    ('*x = 1', 'starred assignment target must be in a list or tuple', (1, 1, 1, 3)),
    ('lambda: 0 = 1', 'cannot assign to lambda', (1, 1, 1, 10)),
    ('f() = 1', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('a or b = 1', 'cannot assign to expression', (1, 1, 1, 7)),
    ('a and b = 1', 'cannot assign to expression', (1, 1, 1, 8)),
    ('not a = 1', 'cannot assign to expression', (1, 1, 1, 6)),
    ('a + b = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 1, 1, 6)),
    ('-a = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 1, 1, 3)),
    ('~a = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 1, 1, 3)),
    ('(x for x in y) = 1', 'cannot assign to generator expression', (1, 1, 1, 15)),
    ('((x for x in y)) = 1', "cannot assign to generator expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 16)),
    ('def f(): (yield) = 1', "cannot assign to yield expression here. Maybe you meant '==' instead of '='?", (1, 11, 1, 16)),
    ('def f(): (yield a) = 1', "cannot assign to yield expression here. Maybe you meant '==' instead of '='?", (1, 11, 1, 18)),
    ('def f(): (yield from a) = 1', "cannot assign to yield expression here. Maybe you meant '==' instead of '='?", (1, 11, 1, 23)),
    ('def f(): yield a = 1', 'assignment to yield expression not possible', (1, 10, 1, 17)),
    ('def f(): yield from a = 1', 'assignment to yield expression not possible', (1, 10, 1, 22)),
    ('def f(): yield = 1', 'assignment to yield expression not possible', (1, 10, 1, 15)),
    ('async def f(): (await a) = 1', "cannot assign to await expression here. Maybe you meant '==' instead of '='?", (1, 17, 1, 24)),
    ('def f(): (await a) = 1', "cannot assign to await expression here. Maybe you meant '==' instead of '='?", (1, 11, 1, 18)),
    ('def f(): await a = 1', "cannot assign to await expression here. Maybe you meant '==' instead of '='?", (1, 10, 1, 17)),
    ('async def f(): await a = 1', "cannot assign to await expression here. Maybe you meant '==' instead of '='?", (1, 16, 1, 23)),
    ('[x for x in y] = 1', "cannot assign to list comprehension here. Maybe you meant '==' instead of '='?", (1, 1, 1, 15)),
    ('{x for x in y} = 1', "cannot assign to set comprehension here. Maybe you meant '==' instead of '='?", (1, 1, 1, 15)),
    ('{k:v for k,v in y} = 1', "cannot assign to dict comprehension here. Maybe you meant '==' instead of '='?", (1, 1, 1, 19)),
    ('{1:2} = 1', "cannot assign to dict literal here. Maybe you meant '==' instead of '='?", (1, 1, 1, 6)),
    ('{1,2} = 1', "cannot assign to set display here. Maybe you meant '==' instead of '='?", (1, 1, 1, 6)),
    ("f'{x}' = 1", "cannot assign to f-string expression here. Maybe you meant '==' instead of '='?", (1, 1, 1, 7)),
    ("t'{x}' = 1", "cannot assign to t-string expression here. Maybe you meant '==' instead of '='?", (1, 1, 1, 7)),
    ('None = 1', 'cannot assign to None', (1, 1, 1, 5)),
    ('True = 1', 'cannot assign to True', (1, 1, 1, 5)),
    ('False = 1', 'cannot assign to False', (1, 1, 1, 6)),
    ('(None) = 1', "cannot assign to None here. Maybe you meant '==' instead of '='?", (1, 2, 1, 6)),
    ('(True) = 1', "cannot assign to True here. Maybe you meant '==' instead of '='?", (1, 2, 1, 6)),
    ('(False) = 1', "cannot assign to False here. Maybe you meant '==' instead of '='?", (1, 2, 1, 7)),
    ('... = 1', "cannot assign to ellipsis here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('1 = 1', "cannot assign to literal here. Maybe you meant '==' instead of '='?", (1, 1, 1, 2)),
    ("'s' = 1", "cannot assign to literal here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('a < b = 1', 'cannot assign to comparison', (1, 1, 1, 6)),
    ('(a < b) = 1', "cannot assign to comparison here. Maybe you meant '==' instead of '='?", (1, 2, 1, 7)),
    ('(a if b else c) = 1', "cannot assign to conditional expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 15)),
    ('x if y else z = 1', 'cannot assign to conditional expression', (1, 1, 1, 14)),
    ('(a := b) = 1', "cannot assign to named expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 8)),
    ('(lambda: 0) = 1', "cannot assign to lambda here. Maybe you meant '==' instead of '='?", (1, 2, 1, 11)),
    ('(a or b) = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 8)),
    ('(not a) = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 7)),
    ('(-a) = 1', "cannot assign to expression here. Maybe you meant '==' instead of '='?", (1, 2, 1, 4)),
    ('f() = g() = 1', 'cannot assign to function call', (1, 1, 1, 4)),
    ('1 = 2 = 3', 'cannot assign to literal', (1, 1, 1, 2)),
    ('x[0] = f() = 1', 'cannot assign to function call', (1, 8, 1, 11)),
    ('a = f() = 1', 'cannot assign to function call', (1, 5, 1, 8)),
    ('f() = a if b else c', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('f() = a or b', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('f() = yield x', 'cannot assign to function call', (1, 1, 1, 4)),
    ('f() = a, b', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('f() = *a,', 'cannot assign to function call', (1, 1, 1, 4)),
    ('f() = a := 1', 'cannot assign to function call', (1, 1, 1, 4)),
    ('f() = (a := 1)', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 4)),
    ('f() = not a', 'cannot assign to function call', (1, 1, 1, 4)),
    ('[a, f()] = 1', 'cannot assign to function call', (1, 5, 1, 8)),
    ('(a, f()) = 1', 'cannot assign to function call', (1, 5, 1, 8)),
    ('[a, 1] = 1', 'cannot assign to literal', (1, 5, 1, 6)),
    ('f() += 1', "'function call' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('a + b += 1', "'expression' is an illegal expression for augmented assignment", (1, 1, 1, 6)),
    ('(a,) += 1', "'tuple' is an illegal expression for augmented assignment", (1, 1, 1, 5)),
    ('[a] += 1', "'list' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('*x += 1', "'starred' is an illegal expression for augmented assignment", (1, 1, 1, 3)),
    ('def f(): (yield) += 1', "'yield expression' is an illegal expression for augmented assignment", (1, 11, 1, 16)),
    ('def f(): (yield a) += 1', "'yield expression' is an illegal expression for augmented assignment", (1, 11, 1, 18)),
    ('def f(): (yield from a) += 1', "'yield expression' is an illegal expression for augmented assignment", (1, 11, 1, 23)),
    ('def f(): yield a += 1', 'invalid syntax', (1, 18, 1, 20)),
    ('async def f(): (await a) += 1', "'await expression' is an illegal expression for augmented assignment", (1, 17, 1, 24)),
    ('async def f(): await a += 1', "'await expression' is an illegal expression for augmented assignment", (1, 16, 1, 23)),
    ('[x for x in y] += 1', "'list comprehension' is an illegal expression for augmented assignment", (1, 1, 1, 15)),
    ('{x for x in y} += 1', "'set comprehension' is an illegal expression for augmented assignment", (1, 1, 1, 15)),
    ('{k:v for k,v in y} += 1', "'dict comprehension' is an illegal expression for augmented assignment", (1, 1, 1, 19)),
    ('{1:2} += 1', "'dict literal' is an illegal expression for augmented assignment", (1, 1, 1, 6)),
    ('{1,2} += 1', "'set display' is an illegal expression for augmented assignment", (1, 1, 1, 6)),
    ("f'{x}' += 1", "'f-string expression' is an illegal expression for augmented assignment", (1, 1, 1, 7)),
    ("t'{x}' += 1", "'t-string expression' is an illegal expression for augmented assignment", (1, 1, 1, 7)),
    ('None += 1', "'None' is an illegal expression for augmented assignment", (1, 1, 1, 5)),
    ('True += 1', "'True' is an illegal expression for augmented assignment", (1, 1, 1, 5)),
    ('... += 1', "'ellipsis' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('1 += 1', "'literal' is an illegal expression for augmented assignment", (1, 1, 1, 2)),
    ("'s' += 1", "'literal' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('(a if b else c) += 1', "'conditional expression' is an illegal expression for augmented assignment", (1, 2, 1, 15)),
    ('a < b += 1', "'comparison' is an illegal expression for augmented assignment", (1, 1, 1, 6)),
    ('lambda: 0 += 1', "'lambda' is an illegal expression for augmented assignment", (1, 1, 1, 10)),
    ('(x for x in y) += 1', "'generator expression' is an illegal expression for augmented assignment", (1, 1, 1, 15)),
    ('(a := b) += 1', "'named expression' is an illegal expression for augmented assignment", (1, 2, 1, 8)),
    ('f() //= 1', "'function call' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('f() **= 1', "'function call' is an illegal expression for augmented assignment", (1, 1, 1, 4)),
    ('async def f(): [x async for x in y] = 1', "cannot assign to list comprehension here. Maybe you meant '==' instead of '='?", (1, 16, 1, 36)),
    ('x = 1\nf() = 2\n', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (2, 1, 2, 4)),
    ('if 1:\n    f() = 2\n', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (2, 5, 2, 8)),
    ('z = [1]\nz[0] = 2\nf() = 3\n', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (3, 1, 3, 4)),
    ('f() = ', 'cannot assign to function call', (1, 1, 1, 4)),
    ('x = 1 +', 'invalid syntax', (1, 8, 1, 9)),
    ('x = *a', "can't use starred expression here", (1, 5, 1, 7)),
    ('x = )', "unmatched ')'", (1, 5, 1, 5)),
    ('f() = )', "unmatched ')'", (1, 7, 1, 7)),
    ('f(): int = 1', 'illegal target for annotation', (1, 1, 1, 4)),
    ('[a]: int = 1', 'only single target (not list) can be annotated', (1, 1, 1, 4)),
    ('for f() in x: pass', 'cannot assign to function call', (1, 5, 1, 8)),
    ('del f()', 'cannot delete function call', (1, 5, 1, 8)),
    ('del (a, f())', 'cannot delete function call', (1, 9, 1, 12)),
    ('with a as f(): pass', 'cannot assign to function call', (1, 11, 1, 14)),
    ('f() = 1 = 2 = 3', 'cannot assign to function call', (1, 1, 1, 4)),
    ('a = b = f() = 1', 'cannot assign to function call', (1, 9, 1, 12)),
    ('class C:\n    f() = 1\n', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (2, 5, 2, 8)),
    ('def g():\n    f() = 1\n', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (2, 5, 2, 8)),
    ("f'{x}' = f'{y}' = 1", 'cannot assign to f-string expression', (1, 1, 1, 7)),
    ('a.b() = 1', "cannot assign to function call here. Maybe you meant '==' instead of '='?", (1, 1, 1, 6)),
    ('x = y = f() = z = 1', 'cannot assign to function call', (1, 9, 1, 12)),
    ('async def f():\n    async with a as f(): pass\n', 'cannot assign to function call', (2, 21, 2, 24)),
):
    error = syntax_error(source)
    assert error.msg == message, (source, error.msg)
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == position, (
        source,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

# The neighbouring shapes that do compile, so none of the diagnostics above
# can be reached by widening what counts as an invalid target.
for source in (
    'x.y = 1',
    'x[0] = 1',
    '[a] = 1',
    '(a,) = 1',
    'x.y += 1',
    'x: int = f()',
    'x.y: int = 1',
    'x = 1\ny = 2\n',
    'lambda x=f(): 0',
    '(a) = 1',
    '((a)) = 1',
    '(a) += 1',
    '((a)) += 1',
    'x[0:1] = 1',
    'x[0:1] += 1',
    'a.b.c = 1',
    '@f()\ndef g(): pass',
    'print = 1',
):
    compile(source, "<fragment>", "exec")

# `invalid_assignment` is a grammar alternative, so it is reached only by a
# statement whose tokens lexed and parsed.  A literal whose own content is wrong
# names that failure instead, wherever in the statement it sits, and so does a
# call's unparenthesized generator expression.
for source, message, position in (
    ("b'\u00e4' = 1", "bytes can only contain ASCII literal characters", (1, 1, 1, 5)),
    ("b'\u00e4' += 1", "bytes can only contain ASCII literal characters", (1, 1, 1, 5)),
    ("'\\N{' = 1", "(unicode error) 'unicodeescape' codec can't decode bytes in position 0-2: malformed \\N character escape", (1, 1, 1, 6)),
    ("b'\\x' = 1", "(value error) invalid \\x escape at position 0", (1, 1, 1, 6)),
    ("f'{x!z}' = 1", "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'", (1, 6, 1, 7)),
    ("f(x for x in y, 1) = 2", "Generator expression must be parenthesized", (1, 3, 1, 15)),
    ("bu'x' = 1", "'u' and 'b' prefixes are incompatible", (1, 1, 1, 3)),
    ("ub'x' = 1", "'u' and 'b' prefixes are incompatible", (1, 1, 1, 3)),
    ("bu'x' += 1", "'u' and 'b' prefixes are incompatible", (1, 1, 1, 3)),
    # The tokenizer check precedes every grammar alternative, so the literal
    # can stand after the token the parser reported as well as before it.
    ("f() = bu'x'", "'u' and 'b' prefixes are incompatible", (1, 7, 1, 9)),
    ("f() = ub'x'", "'u' and 'b' prefixes are incompatible", (1, 7, 1, 9)),
    # Like PyPy's `fstring_find_literal`, a backslash leaves a following brace
    # visible to f-string parsing.  CPython 3.14's tokenizer therefore reads
    # the first replacement field, while doubled braces below stay literal.
    ("f'\\{bu\"x\"}'", "'u' and 'b' prefixes are incompatible", (1, 5, 1, 7)),
    # A nested replacement inside a format specification switches back from
    # literal format text to tokenized expression text.
    ("f'''{1:{bu\"x\"}}'''", "'u' and 'b' prefixes are incompatible", (1, 9, 1, 11)),
    # The tokenizer does not read a comment, so neither does the scan.
    ("x = = 1  # bu'z'", "invalid syntax", (1, 5, 1, 6)),
):
    error = syntax_error(source)
    assert error.msg == message, (source, error.msg)
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == position, (
        source,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

# An escaped doubled brace is literal text rather than a replacement field, so
# its prefix-shaped contents cannot override the later assignment diagnostic.
source = "f'\\{{ bu\"x\" }}'; 1=2"
error = syntax_error(source)
assert error.msg == "cannot assign to literal here. Maybe you meant '==' instead of '='?"
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 18, 1, 19)

# Format specifications are literal text, and comments disappear from the
# token stream of a replacement expression.  Prefix-shaped text in either one
# cannot override the later assignment diagnostic.
for source, position in (
    ("f'''{1:bu\"x\"}'''; 1 = 2", (1, 19, 1, 20)),
    ("f'''{1 # bu\"x\"\n}'''; 1 = 2", (2, 7, 2, 8)),
):
    error = syntax_error(source)
    assert error.msg == "cannot assign to literal here. Maybe you meant '==' instead of '='?"
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == position

# A fatal tokenizer error stops before the later prefix-shaped literal.  A
# grammar error does not: its whole token stream was already read, as the
# `f() = bu'x'` cases above demonstrate.
for source, message, position in (
    ("0x\nbu'x'", "invalid hexadecimal literal", (1, 2, 1, 2)),
    (
        "(]\nbu'x'",
        "closing parenthesis ']' does not match opening parenthesis '('",
        (1, 2, 1, 2),
    ),
):
    error = syntax_error(source)
    assert error.msg == message, (source, error.msg)
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == position


# `eval` is `expressions NEWLINE* ENDMARKER` and has no assignment rule at all,
# so the parse stops at the operator.  A starred element of an unparenthesized
# tuple and an unparenthesized `yield` fail earlier still, at their own first
# token; a display keeps its own brackets, so unpacking inside one does not.
for source, position in (
    ("f() = 1", (1, 5, 1, 6)),
    ("f() += 1", (1, 5, 1, 7)),
    ("x = 1", (1, 3, 1, 4)),
    ("x.y = 1", (1, 5, 1, 6)),
    ("1 = 2", (1, 3, 1, 4)),
    ("(a, b) = 1", (1, 8, 1, 9)),
    ("[x for x in y] = 1", (1, 16, 1, 17)),
    ("{1:2} = 1", (1, 7, 1, 8)),
    ("a if b else c = 1", (1, 15, 1, 16)),
    ("a = b = 1", (1, 3, 1, 4)),
    ("x[0] = 1", (1, 6, 1, 7)),
    ("a @= b", (1, 3, 1, 5)),
    ("*x = 1", (1, 1, 1, 2)),
    ("*a, b = 1", (1, 1, 1, 2)),
    ("a, *b = 1", (1, 4, 1, 5)),
    ("(a, *b) = 1", (1, 9, 1, 10)),
    ("(a) = 1", (1, 5, 1, 6)),
    ("((a)) = 1", (1, 7, 1, 8)),
    ("(a := 1) = 2", (1, 10, 1, 11)),
    ("[*a, b] = 1", (1, 9, 1, 10)),
    ("yield = 1", (1, 1, 1, 6)),
    ("(yield) = 1", (1, 9, 1, 10)),
    ("class C:\n    f() = 1\n", (1, 1, 1, 6)),
    ("x: int = f()", (1, 2, 1, 3)),
    # An assignment inside a statement stands behind the keyword that opens it,
    # which is where the expression grammar stopped.
    ("def f(): (yield a) = 1", (1, 1, 1, 4)),
    ("async def f(): (await a) += 1", (1, 1, 1, 6)),
    ("with a: x = 1", (1, 1, 1, 5)),
    ("@d\ndef f(): x = 1", (1, 1, 1, 2)),
):
    error = syntax_error(source, "eval")
    assert error.msg == "invalid syntax", (source, error.msg)
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == position, (
        source,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )


print("OK")
