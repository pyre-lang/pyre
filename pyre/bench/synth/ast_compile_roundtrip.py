# compile() accepts an `ast` tree as well as source text, and the two must
# produce the same code. Each source below is compiled both ways and executed,
# and the outputs are folded together, so a statement or expression kind that
# the tree path converts wrongly -- or silently drops, as a decorator list read
# as empty would -- changes the number.
# The tree path is also the only one that sees hand-built nodes, so the cases
# cover the shapes a parser never emits: a missing optional field, a container
# constant, and `orelse` chains that a parser would have written as `elif`.
# Syntax is held to what every compared runtime parses. Output verified against
# CPython/PyPy.
import ast

SOURCES = (
    ("assign", "x = 1\ny = x + 1\nprint(x, y, x < y <= 2, x is not None)"),
    ("augassign", "x = 1\nx += 4\nx *= 2\nx //= 3\nprint(x)"),
    ("annassign", "x: int = 5\ndef f():\n    y: int\n    return x\nprint(x, f())"),
    ("delete", "d = {'k': 1}\ndel d['k']\nprint(d)"),
    ("ifchain", "for i in range(4):\n    if i == 0:\n        print('zero')\n    elif i == 1:\n        print('one')\n    elif i == 2:\n        print('two')\n    else:\n        print('many')"),
    ("loops", "i = 0\nwhile i < 3:\n    i += 1\nelse:\n    print('while-else', i)\nfor j in range(5):\n    if j == 2:\n        break\nelse:\n    print('unreached')\nprint('j', j)\nfor k in range(2):\n    continue\nelse:\n    print('for-else', k)"),
    ("try", "try:\n    raise ValueError('v')\nexcept KeyError:\n    print('no')\nexcept ValueError as e:\n    print('caught', e)\nelse:\n    print('no else')\nfinally:\n    print('finally')\ntry:\n    1 / 0\nexcept:\n    print('bare')"),
    ("trystar", "try:\n    raise ExceptionGroup('g', [ValueError('a'), TypeError('b')])\nexcept* ValueError as e:\n    print('star-v', len(e.exceptions))\nexcept* TypeError as e:\n    print('star-t', len(e.exceptions))"),
    ("raise_from", "try:\n    try:\n        raise KeyError('k')\n    except KeyError as e:\n        raise ValueError('v') from e\nexcept ValueError as e:\n    print('cause', type(e.__cause__).__name__)"),
    ("assert", "assert 1 == 1\ntry:\n    assert 0, 'boom'\nexcept AssertionError as e:\n    print('assert', e)"),
    ("with", "import io\nwith io.StringIO('a') as f, io.StringIO('b') as g:\n    print(f.read(), g.read())"),
    ("imports", "import sys\nimport sys as alias\nfrom sys import path\nfrom sys import path as p\nprint(type(path) is list, p is path, alias is sys)"),
    ("scopes", "g = 0\ndef setg():\n    global g\n    g = 7\nsetg()\ndef outer():\n    v = 1\n    def inner():\n        nonlocal v\n        v = 9\n    inner()\n    return v\nprint(g, outer())"),
    ("funcargs", "def f(a, b, /, c, d=4, *rest, e, g=7, **kw):\n    return (a, b, c, d, rest, e, g, sorted(kw.items()))\nprint(f(1, 2, 3))\nprint(f(1, 2, 3, 4, 5, 6, e=7, h=8))"),
    ("decorators", "def deco(tag):\n    def wrap(fn):\n        def call(*a):\n            return tag + ':' + str(fn(*a))\n        return call\n    return wrap\n@deco('x')\n@deco('y')\ndef f(v):\n    return v\nprint(f(3))"),
    ("annotations", "def f(a: int, b: str = 'x') -> bool:\n    return True\nprint(f(1), sorted(f.__annotations__))"),
    ("classes", "def tag(cls):\n    cls.t = 'T'\n    return cls\nclass A:\n    def m(self):\n        return 'A'\n@tag\nclass B(A):\n    def m(self):\n        return 'B' + super().m()\nprint(B().m(), B.t, B.__mro__[1].__name__)"),
    ("lambdas", "f = lambda a, b=2, *r, k=3, **kw: (a, b, r, k, sorted(kw))\nprint(f(1), f(1, 5, 6, k=7, z=8))"),
    ("boolops", "print(1 and 2, 0 or 3, 1 and 0 or 4, not 0, 'a' in 'abc', 'z' not in 'abc')"),
    ("ifexp", "print(['no', 'yes'][1], 'yes' if 1 else 'no', (lambda v: 'p' if v > 0 else 'n')(-1))"),
    ("namedexpr", "print([y := 3, y + 1], y)"),
    ("containers", "a = {'x': 1}\nd = {**a, 'y': 2}\nprint(sorted(d.items()), sorted({3, 1, 2}), [1, 2][::-1], (1, 2, 3)[1:])"),
    ("comprehensions", "print([i * i for i in range(5) if i % 2 == 0])\nprint(sorted({i % 3 for i in range(7)}))\nprint(sorted({i: i * 2 for i in range(3)}.items()))\nprint(sum(i for i in range(5) if i != 2))\nprint([(i, j) for i in range(2) for j in range(2) if i != j])"),
    ("generators", "def g():\n    x = yield 1\n    yield x\ndef outer():\n    yield from g()\n    yield 'end'\nit = outer()\nprint(next(it))\nprint(it.send(5))\nprint(next(it))"),
    ("subscripts", "s = list(range(10))\nprint(s[2], s[1:4], s[::2], s[::-1], s[1:8:3])\ns[0] = 99\ns[1:3] = [7, 8]\ndel s[9]\nprint(s)"),
    ("starred", "a, *rest = [1, 2, 3]\n(b, (c, d)) = (1, (2, 3))\nargs = [1, 2]\nprint(a, rest, b, c, d, max(*args), [*args, 3])"),
    ("match", "def f(v):\n    match v:\n        case 0:\n            return 'zero'\n        case [1, 2]:\n            return 'onetwo'\n        case [1, *tail]:\n            return 'head1 ' + str(tail)\n        case {'k': val, **extra}:\n            return 'map ' + str(val) + str(sorted(extra))\n        case str() as s if len(s) > 1:\n            return 'str ' + s\n        case (int() | float()) as n:\n            return 'num ' + str(n)\n        case None:\n            return 'none'\n        case _:\n            return 'other'\nfor v in (0, [1, 2], [1, 5, 6], {'k': 9, 'z': 1}, 'hi', 4.5, None, object):\n    print(f(v))"),
    ("matchclass", "class P:\n    __match_args__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x, self.y = x, y\ndef f(p):\n    match p:\n        case P(0, y=b):\n            return 'originx ' + str(b)\n        case P(a, b):\n            return 'point ' + str(a + b)\nprint(f(P(0, 5)), f(P(2, 3)))"),
    ("constants", "print(1, 1.5, 'a', b'b', True, False, None, ..., 3 + 4j, -0.0, 10 ** 30)"),
    ("fstrings", "x, w = 5, 8\nprint(f'', f'plain', f'a{x}b', f'{x}{x}')\nprint(f'{x!r} {x!s} {x!a}')\nprint(f'{x:05d}|{x:>8}|{x:+.3f}|{x:>{w}}|{x:0{w}d}')\nprint(f'{x!r:>10}', f'{{literal}} {x}')"),
    ("fstring_nesting", "x = 3\nd = {'k': [1, 2]}\nprint(f'{f\"{x}\"}', f'{d[\"k\"][1] + 1}', f'{len(\"abc\")}')\nprint(f'a' 'b' f'{x}' 'c')\nprint(f'{x=}', f'{x = }', f'{x=:04d}', f'{x=!r}')\ndef g(v):\n    return f'<{v!r}>'\nprint(g(1), g('s'), [f'{i}:{i * i}' for i in range(3)])"),
)


def fold(acc, value):
    for ch in str(value):
        acc = (acc * 31 + ord(ch)) & 0xFFFFFFFF
    return acc


def run(code, out):
    ns = {"__name__": "__main__", "_out": out}
    exec("def _p(*a):\n    _out.append(' '.join(str(x) for x in a))\n", ns)
    ns["print"] = ns["_p"]
    try:
        exec(code, ns)
    except BaseException as e:
        out.append("!!" + type(e).__name__ + ":" + str(e))
    return out


def both_ways(src):
    # Same source, two compile paths; the outputs have to match each other.
    direct = run(compile(src, "<s>", "exec"), [])
    viaast = run(compile(ast.parse(src), "<s>", "exec"), [])
    assert direct == viaast, (direct, viaast)
    return direct


def handbuilt():
    # Shapes the parser never produces, so only the tree path can reach them.
    out = []
    # An `If` whose alternative is a nested `If`: the parser would have written
    # `elif`, and both have to compile to the same branch chain.
    tree = ast.parse("if a == 1:\n    print('one')\nelse:\n    print('other')")
    inner = ast.parse("if a == 2:\n    print('two')\nelse:\n    print('many')").body[0]
    tree.body[0].orelse = [inner]
    ast.fix_missing_locations(tree)
    code = compile(tree, "<s>", "exec")
    for a in (1, 2, 3):
        ns = {"a": a, "_out": out}
        exec("def _p(*x):\n    _out.append(' '.join(str(v) for v in x))\n", ns)
        ns["print"] = ns["_p"]
        exec(code, ns)
    # Optional fields left off a hand-built node.
    tree = ast.parse("def f():\n    return 1")
    tree.body[0].body = [ast.Return()]
    ast.fix_missing_locations(tree)
    ns = {}
    exec(compile(tree, "<s>", "exec"), ns)
    out.append("bare-return " + str(ns["f"]()))
    # Container and complex constants, which no parser emits.
    for value in ((1, "a", 2.5), (1, (2, 3)), frozenset({1, 2}), 3 + 4j, ()):
        tree = ast.parse("x = 0")
        tree.body[0].value = ast.Constant(value=value)
        ast.fix_missing_locations(tree)
        ns = {}
        exec(compile(tree, "<s>", "exec"), ns)
        got = ns["x"]
        shown = sorted(got) if isinstance(got, frozenset) else got
        out.append("const " + type(got).__name__ + " " + str(shown) + " " + str(got == value))
    # An f-string reaches the tree path as the values it joins, not as the
    # literal and interpolated parts the parser split it into, so the shapes
    # below only exist here: no values at all, a lone `FormattedValue`, a
    # nested `JoinedStr`, and a spec that is itself formatted.
    name = ast.Name("x", ast.Load())
    for label, node in (
        ("empty", ast.JoinedStr(values=[])),
        ("const", ast.JoinedStr(values=[ast.Constant("ab")])),
        ("lone-fv", ast.FormattedValue(value=name, conversion=-1)),
        ("fv-repr", ast.FormattedValue(value=name, conversion=114)),
        ("fv-str", ast.FormattedValue(value=name, conversion=115)),
        ("fv-ascii", ast.FormattedValue(value=ast.Constant("\xe9"), conversion=97)),
        ("mixed", ast.JoinedStr(values=[
            ast.Constant("a"), ast.FormattedValue(value=name, conversion=-1), ast.Constant("b")])),
        ("spec", ast.JoinedStr(values=[ast.FormattedValue(
            value=name, conversion=-1,
            format_spec=ast.JoinedStr(values=[ast.Constant("07.2f")]))])),
        ("spec-formatted", ast.JoinedStr(values=[ast.FormattedValue(
            value=name, conversion=-1,
            format_spec=ast.JoinedStr(values=[
                ast.Constant(">"),
                ast.FormattedValue(value=ast.Name("w", ast.Load()), conversion=-1)]))])),
        ("nested", ast.JoinedStr(values=[
            ast.JoinedStr(values=[ast.FormattedValue(value=name, conversion=-1)]),
            ast.Constant("!")])),
        ("non-str-const", ast.JoinedStr(values=[ast.Constant(1)])),
    ):
        tree = ast.Expression(body=node)
        ast.fix_missing_locations(tree)
        out.append("fstring " + label + " " + repr(eval(compile(tree, "<s>", "eval"),
                                                       {"x": 42.5, "w": 9})))
    # Only the class is compared: the wording differs between runtimes.
    def without(node, field):
        # Dropped after construction rather than left off it, which warns.
        delattr(node, field)
        return node

    # A tree the parser could never have produced. Only the class is compared:
    # the wording differs between runtimes.
    L, S, D = ast.Load(), ast.Store(), ast.Del()
    args = lambda **kw: ast.arguments(
        posonlyargs=kw.get("po", []), args=kw.get("a", []), kwonlyargs=kw.get("ko", []),
        kw_defaults=kw.get("kd", []), defaults=kw.get("d", []))
    pat = lambda p: ast.Match(subject=ast.Constant(1),
                              cases=[ast.match_case(pattern=p, body=[ast.Pass()])])
    RANGE = {"lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 1}

    class Index:
        # An integer field is not asked for `__index__`.
        def __index__(self):
            return 1

    for label, node in (
        ("missing-conversion", ast.JoinedStr(values=[
            without(ast.FormattedValue(value=name, conversion=-1), "conversion")])),
        ("missing-value", ast.JoinedStr(values=[
            without(ast.FormattedValue(value=name, conversion=-1), "value")])),
        ("values-not-a-list", ast.JoinedStr(values=ast.Constant("a"))),
        ("none-in-values", ast.JoinedStr(values=[None])),
        ("none-in-body", ast.Module(body=[None], type_ignores=[])),
        ("more-defaults-than-args", ast.Lambda(
            args=args(d=[ast.Constant(1), ast.Constant(2)]), body=ast.Constant(1))),
        ("kw-defaults-length", ast.Lambda(
            args=args(ko=[ast.arg(arg="a"), ast.arg(arg="b")], kd=[ast.Constant(1)]),
            body=ast.Constant(1))),
        ("compare-no-comparators", ast.Compare(left=ast.Constant(1), ops=[], comparators=[])),
        ("compare-ops-mismatch", ast.Compare(left=ast.Constant(1), ops=[ast.Lt()], comparators=[])),
        ("boolop-one-value", ast.BoolOp(op=ast.And(), values=[ast.Constant(1)])),
        ("boolop-no-values", ast.BoolOp(op=ast.And(), values=[])),
        ("comprehension-no-generators", ast.ListComp(elt=ast.Constant(1), generators=[])),
        ("name-is-a-constant", ast.Name("None", L)),
        ("store-in-load-position", ast.Name("x", S)),
        ("empty-body-functiondef", ast.Module(body=[ast.FunctionDef(
            name="f", args=args(), body=[], decorator_list=[], type_params=[])], type_ignores=[])),
        ("empty-body-classdef", ast.Module(body=[ast.ClassDef(
            name="C", bases=[], keywords=[], body=[], decorator_list=[], type_params=[])],
            type_ignores=[])),
        ("empty-targets-assign", ast.Module(body=[ast.Assign(targets=[], value=ast.Constant(1))],
                                            type_ignores=[])),
        ("load-target-assign", ast.Module(body=[ast.Assign(targets=[ast.Name("x", L)],
                                                           value=ast.Constant(1))],
                                          type_ignores=[])),
        ("load-target-delete", ast.Module(body=[ast.Delete(targets=[ast.Name("x", L)])],
                                          type_ignores=[])),
        ("empty-names-import", ast.Module(body=[ast.Import(names=[])], type_ignores=[])),
        ("negative-import-level", ast.Module(body=[ast.ImportFrom(
            module="x", names=[ast.alias(name="y")], level=-1)], type_ignores=[])),
        ("raise-cause-no-exc", ast.Module(body=[ast.Raise(exc=None, cause=ast.Name("x", L))],
                                          type_ignores=[])),
        ("try-without-handlers", ast.Module(body=[ast.Try(
            body=[ast.Pass()], handlers=[], orelse=[], finalbody=[])], type_ignores=[])),
        ("empty-except-body", ast.Module(body=[ast.Try(body=[ast.Pass()], handlers=[
            ast.ExceptHandler(type=None, name=None, body=[])], orelse=[], finalbody=[])],
            type_ignores=[])),
        ("empty-items-with", ast.Module(body=[ast.With(items=[], body=[ast.Pass()])],
                                        type_ignores=[])),
        ("empty-cases-match", ast.Module(body=[ast.Match(subject=ast.Constant(1), cases=[])],
                                         type_ignores=[])),
        ("match-or-one-pattern", ast.Module(body=[pat(ast.MatchOr(
            patterns=[ast.MatchValue(value=ast.Constant(1))]))], type_ignores=[])),
        ("match-star-alone", ast.Module(body=[pat(ast.MatchStar(name="a"))], type_ignores=[])),
        ("match-as-no-name", ast.Module(body=[pat(ast.MatchAs(
            pattern=ast.MatchValue(value=ast.Constant(1)), name=None))], type_ignores=[])),
        ("match-capture-underscore", ast.Module(body=[pat(ast.MatchAs(
            pattern=ast.MatchValue(value=ast.Constant(1)), name="_"))], type_ignores=[])),
        ("match-class-bad-cls", ast.Module(body=[pat(ast.MatchClass(
            cls=ast.Constant(1), patterns=[], kwd_attrs=[], kwd_patterns=[]))], type_ignores=[])),
        ("match-singleton-bad", ast.Module(body=[pat(ast.MatchSingleton(value=42))],
                                           type_ignores=[])),
        ("match-value-bad-constant", ast.Module(body=[pat(ast.MatchValue(
            value=ast.Constant(value=None)))], type_ignores=[])),
        # Only trees the two oracles agree on belong here: every backend's
        # output is compared byte for byte against PyPy's. Where 3.14 and PyPy
        # disagree the check lives in `astcompiler::validate`'s own tests.
        ("match-mapping-store-key", ast.Module(body=[pat(ast.MatchMapping(
            keys=[ast.Attribute(value=ast.Name("o", S), attr="a", ctx=ast.Load())],
            patterns=[ast.MatchAs(pattern=None, name="v")], rest=None))], type_ignores=[])),
        ("type-ignores-not-a-list", ast.Module(body=[ast.Pass()], type_ignores=42)),
        ("type-ignores-bad-element", ast.Module(body=[ast.Pass()], type_ignores=[1])),
        ("lineno-not-an-int", ast.Module(body=[ast.Pass(lineno=Index(), col_offset=0)],
                                         type_ignores=[])),
        ("constant-kind-not-str", ast.Constant(value="a", kind=1)),
        ("expr-context-not-a-node", ast.Name("x", 1)),
        ("boolop-not-a-node", ast.BoolOp(op=1, values=[ast.Constant(1), ast.Constant(2)])),
        ("operator-not-a-node", ast.BinOp(left=ast.Constant(1), op=1, right=ast.Constant(2))),
        ("unaryop-not-a-node", ast.UnaryOp(op=1, operand=ast.Constant(1))),
        ("cmpop-not-a-node", ast.Compare(left=ast.Constant(1), ops=[1],
                                         comparators=[ast.Constant(2)])),
    ):
        tree = node if isinstance(node, ast.Module) else ast.Expression(body=node)
        try:
            ast.fix_missing_locations(tree)
            compile(tree, "<s>", "exec" if isinstance(node, ast.Module) else "eval")
            out.append("reject " + label + " none")
        except BaseException as e:
            out.append("reject " + label + " " + type(e).__name__)
    # A source range is required on the node kinds below, so these trees are
    # the ones `fix_missing_locations` would have repaired.
    for label, node in (
        ("stmt-no-range", ast.Module(body=[ast.Pass()], type_ignores=[])),
        ("expr-no-range", ast.Module(body=[ast.Expr(value=ast.Name("x", L), **RANGE)],
                                     type_ignores=[])),
        ("pattern-no-range", ast.Module(body=[ast.Match(
            subject=ast.Constant(1, **RANGE),
            cases=[ast.match_case(pattern=ast.MatchAs(pattern=None, name="v"),
                                  body=[ast.Pass(**RANGE)])], **RANGE)], type_ignores=[])),
        ("excepthandler-no-range", ast.Module(body=[ast.Try(
            body=[ast.Pass(**RANGE)],
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass(**RANGE)])],
            orelse=[], finalbody=[], **RANGE)], type_ignores=[])),
        ("alias-no-range", ast.Module(body=[ast.Import(names=[ast.alias(name="os")], **RANGE)],
                                      type_ignores=[])),
        ("arg-no-range", ast.Module(body=[ast.FunctionDef(
            name="f", args=args(a=[ast.arg(arg="a")]), body=[ast.Pass(**RANGE)],
            decorator_list=[], type_params=[], **RANGE)], type_ignores=[])),
        ("keyword-no-range", ast.Module(body=[ast.Expr(value=ast.Call(
            func=ast.Name("f", L, **RANGE), args=[],
            keywords=[ast.keyword(arg="k", value=ast.Constant(1, **RANGE))], **RANGE), **RANGE)],
            type_ignores=[])),
    ):
        try:
            compile(node, "<s>", "exec")
            out.append("reject " + label + " none")
        except BaseException as e:
            out.append("reject " + label + " " + type(e).__name__)
    return out


def warm(reps):
    # The conversion runs per compile, so keep the hot path the traced one.
    src = "def f(a, b=2, *r, k=3):\n    if a > b:\n        return a - b\n    return sum(i for i in range(a) if i != b)"
    acc = 0
    for _ in range(reps):
        acc += len(compile(ast.parse(src), "<s>", "exec").co_names)
    return acc


def main():
    print("warm", warm(400))
    acc = 0
    for name, src in SOURCES:
        lines = both_ways(src)
        for line in lines:
            acc = fold(acc, line)
        print(name, len(lines), acc)
    for line in handbuilt():
        print("hand", line)


main()
