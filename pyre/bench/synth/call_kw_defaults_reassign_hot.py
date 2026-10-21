# CALL_KW filling omitted positional and keyword-only defaults from
# `Function.defs_w` / `Function.w_kw_defs`. Both are quasi-immutable
# (`function.py Function._immutable_fields_ = [..., 'defs_w?[*]', ...,
# 'w_kw_defs?']`); `PyCode.funcrun` reads them so a later
# `f.__defaults__ = ...` / `f.__kwdefaults__ = ...` retires the compiled
# loop. A binder that copies the recording-time values into const_refs
# keeps returning the old sum.
#
# `round_robin_defaults` is the other half of that bake: four Function
# objects from one `def` (same code, same globals, different defaults).
# Code/globals guards pass across the list, so a getfield of `defs_w` /
# `w_kw_defs` off a recording-time `const_ref` answers every later
# `fs[i % 4](1, c=0)` with the traced function's defaults.
N = 3000
ROUND_ROBIN_N = 4000


def f(a, b, c=10, *, d=20):
    return a + b + c + d


def call_hot(want):
    acc = 0
    for i in range(N):
        got = f(1, b=2)
        assert got == want, (i, got, want)
        acc += got
    return acc


def after_compile():
    acc = call_hot(33)
    f.__defaults__ = (100,)
    f.__kwdefaults__ = {"d": 200}
    acc += call_hot(303)
    return acc


def midloop():
    def g(a, b, c=10, *, d=20):
        return a + b + c + d

    acc = 0
    for i in range(N):
        if i == 1500:
            g.__defaults__ = (100,)
            g.__kwdefaults__ = {"d": 200}
        got = g(1, b=2)
        want = 33 if i < 1500 else 303
        assert got == want, (i, got, want)
        acc += got
    return acc


def round_robin_defaults():
    def make(d):
        def f(a, b=d, *, c=d):
            return a + b + c

        return f

    fs = [make(i) for i in range(4)]
    acc = 0
    for i in range(ROUND_ROBIN_N):
        acc += fs[i % 4](1, c=0)
    return acc


def round_robin_layout_and_length():
    # Four Functions from one `def`. The first has `__defaults__ is None`;
    # the others are a Cls_ii pair, a Cls_oo pair, and a length-1
    # array-backed tuple. All three positionals are passed so the
    # None-defaults function binds. It is first so the loop records a
    # zero-defaults binder; the live `defs_w` getfield has to re-prove
    # null/None (or the tuple class and length) before the recording-time
    # item reads (`argument.py _match_signature` branches on
    # `len(defaults_w)`).
    def make(d):
        def f(a, b=d, c=d, *, k=0):
            bv = len(b) if type(b) is str else b
            cv = len(c) if type(c) is str else c
            return a + bv + cv + k

        return f

    f_none = make(0)
    f_none.__defaults__ = None
    f_ii = make(0)
    f_ii.__defaults__ = (1, 2)
    f_oo = make(0)
    f_oo.__defaults__ = (1.5, "x")
    f_block = make(0)
    f_block.__defaults__ = (7,)
    fs = [f_none, f_ii, f_oo, f_block]
    acc = 0
    for i in range(ROUND_ROBIN_N):
        acc += fs[i % 4](0, 0, 0, k=0)
    return (
        fs[0](0, 0, 0, k=0),
        fs[1](0, 0, 0, k=0),
        fs[2](0, 0, 0, k=0),
        fs[3](0, 0, 0, k=0),
        acc,
    )


print(after_compile(), midloop(), round_robin_defaults(), round_robin_layout_and_length())
