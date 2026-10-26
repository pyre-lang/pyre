# CPython-suite gap: test_call checks keyword TypeErrors on plain functions.
# parity-tests reason: `callmethod.py CALL_METHOD_KW` and
# `function.py call_obj_args` both reach `pycode.py funcrun_obj`, so a
# method-form call and a bound method stored in a variable report
# `func.qualname` and count the receiver in `_match_signature`.

"""Keyword errors on a method form and a bound method name the qualname."""

N = 3000


class C:
    def m(self, a, b):
        return (self.n, a, b)


def check(got, expected):
    if got != expected:
        raise AssertionError(repr(got))
    print(got)


def expect_type_error(label, call, expected):
    try:
        call()
    except TypeError as exc:
        check(str(exc), expected)
    else:
        raise AssertionError(label)


def exercise():
    obj = C()
    obj.n = 7
    bm = obj.m
    total = 0
    i = 0
    while i < N:
        if i == 0 or i == N - 1:
            expect_type_error(
                "method missing",
                lambda: obj.m(b=2),
                "C.m() missing 1 required positional argument: 'a'",
            )
            expect_type_error(
                "bound unexpected",
                lambda: bm(1, b=2, c=3),
                "C.m() got an unexpected keyword argument 'c'",
            )
            expect_type_error(
                "bound duplicate",
                lambda: bm(1, a=2),
                "C.m() got multiple values for argument 'a'",
            )
        else:
            got = obj.m(1, b=2)
            if got != (7, 1, 2):
                raise AssertionError(repr(got))
            got = bm(1, b=2)
            if got != (7, 1, 2):
                raise AssertionError(repr(got))
            total += 1
        i += 1
    if total != N - 2:
        raise AssertionError("calls did not run: %s" % total)


exercise()
print("OK")
