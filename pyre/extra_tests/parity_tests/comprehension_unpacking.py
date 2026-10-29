# pyre-check: pypy-diverges: pypy3 implements 3.11, which rejects unpacking in comprehensions
# CPython-suite gap: test_unpack_ex covers these doctests but its row stays FAIL for
# parser error messages, so nothing gates the runtime results.
#
# parity-tests reason: PEP 798 compiles `[*x for ...]` to LIST_EXTEND, `{*x for ...}` to
# SET_UPDATE, `{**d for ...}` to DICT_UPDATE and `(*x for ...)` to an inner iteration.
import ast

print([*x for x in [[1, 2], [3], []]])
print(sorted({*x for x in [[1, 2], [2, 3]]}))
print({**d for d in [{"a": 1}, {"b": 2, "a": 3}]})
print(list(*x for x in [(1, 2), (3,)]))
print([*x for x in ["ab", "c"] if x])
print([*row for rows in [[[1], [2, 3]]] for row in rows])
g = (*x for x in [[1, 2], [3]])
print(next(g), next(g), list(g))


def f(xs):
    return [*x for x in xs]


total = 0
xs = [[1, 2], [3], []]
for _ in range(3000):
    total += len(f(xs))
print(total)

try:
    [*x for x in [1]]
except Exception as e:
    print(type(e).__name__, e)

try:
    {**x for x in [1]}
except Exception as e:
    print(type(e).__name__, e)

print(ast.dump(ast.parse("{**d for d in x}").body[0].value))
print(ast.DictComp(key=ast.Name("a"), generators=[]).value)
print("OK")
