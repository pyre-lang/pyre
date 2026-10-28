# pyre-check: pypy-diverges: PyPy 3.11 compile() of a hand-built tree has no 3.15 RESUME prefix in co_positions(), and drops end/column fields when col_offset is near the C int limit.
# CPython-suite gap: test_ast / test_compile compile hand-built trees only
# with small coordinates, never with lineno or col_offset near the C int
# limit, and never compare co_positions of a tree shifted by
# ast.increment_lineno with the source compile.
# parity-tests reason: the text pyre builds for an object tree must not
# scale with the coordinate values.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

import ast


def iter_codes(code):
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            yield from iter_codes(const)


def shifted_position(pos, shift):
    start, end, col, end_col = pos
    if start is None or start == 0:
        return pos
    return (
        start + shift,
        None if end is None else end + shift,
        col,
        end_col,
    )


for line, col in (
    (2**31 - 1, 0),
    (1, 2**31 - 2),
    (2**31 - 1, 2**31 - 2),
    (10**9, 5),
):
    tree = ast.Module(
        body=[
            ast.Expr(
                ast.Constant(
                    1,
                    lineno=line,
                    col_offset=col,
                    end_lineno=line,
                    end_col_offset=col + 1,
                ),
                lineno=line,
                col_offset=col,
                end_lineno=line,
                end_col_offset=col + 1,
            )
        ],
        type_ignores=[],
    )
    code = compile(tree, "large_positions.py", "exec")
    assert list(code.co_positions())[2:] == [(line, line, col, col + 1)] * 2


src = """\
@dec
def f(x=1):
    return x


class C:
    def m(self):
        return 1

# a comment between statements

xs = [i for i in range(3)]
g = (i for i in range(3))
s = f'{xs}'

try:
    h()
except Exception:
    pass

with ctx() as fh:
    fh.read()

value = ident(
    1,
    2,
)
"""
assert len(src.splitlines()) < 100

filename = "large_positions.py"
from_src = compile(src, filename, "exec")
from_src_codes = list(iter_codes(from_src))

for shift in (0, 10**6, 2**31 - 100):
    tree = ast.parse(src)
    ast.increment_lineno(tree, shift)
    from_tree = compile(tree, filename, "exec")
    from_tree_codes = list(iter_codes(from_tree))
    assert len(from_tree_codes) == len(from_src_codes)
    for original, compiled in zip(from_src_codes, from_tree_codes):
        assert original.co_name == compiled.co_name
        if original.co_name == "<module>":
            assert compiled.co_firstlineno == original.co_firstlineno
        else:
            assert compiled.co_firstlineno == original.co_firstlineno + shift
        expected = [shifted_position(pos, shift) for pos in original.co_positions()]
        assert list(compiled.co_positions()) == expected

print("OK")
