# CPython-suite gap: test_call checks missing, duplicate, and unexpected
# keyword TypeErrors, but not the bound-method / keyword-only / **kwargs
# success path together with those errors on a plain user function.
# parity-tests reason: CALL_KW on a plain user function binds through
# `argument.py Arguments.parse_into_scope` (`pycode.py PyCode.funcrun`).

"""CALL_KW binds keywords on a user function and reports signature errors."""


def f_kwonly(a, *, k):
    return (a, k)


r_kwonly = f_kwonly(1, k=2)
assert r_kwonly == (1, 2)


def f_kwdef(*, k=3):
    return k


r_kwdef = f_kwdef()
r_kwover = f_kwdef(k=4)
assert r_kwdef == 3
assert r_kwover == 4


def f_varkw(a, **kw):
    return (a, kw["b"], tuple(sorted(kw)))


r_varkw_head = f_varkw(1, c=3, b=2)
assert r_varkw_head == (1, 2, ("b", "c"))


def f_dup(a):
    return a


try:
    f_dup(1, a=2)
except TypeError as e:
    r_dup = str(e)
assert r_dup == "f_dup() got multiple values for argument 'a'"


def f_unexpected(a):
    return a


try:
    f_unexpected(a=1, b=2)
except TypeError as e:
    r_unexpected = str(e)
assert r_unexpected == "f_unexpected() got an unexpected keyword argument 'b'"


def f_missing(a):
    return a


try:
    f_missing()
except TypeError as e:
    r_missing = str(e)
assert r_missing == "f_missing() missing 1 required positional argument: 'a'"


def f_missing_kw(*, k):
    return k


try:
    f_missing_kw()
except TypeError as e:
    r_missing_kw = str(e)
assert r_missing_kw == "f_missing_kw() missing 1 required keyword-only argument: 'k'"


class A:
    def m(self, a, *, k):
        return (self.n, a, k)


obj = A()
obj.n = 7
bm = obj.m
r_bound = bm(1, k=2)
r_method = obj.m(1, k=2)
assert r_bound == (7, 1, 2)
assert r_method == (7, 1, 2)
try:
    bm(k=2)
except TypeError as e:
    r_bound_missing = str(e)
assert r_bound_missing == "A.m() missing 1 required positional argument: 'a'"
print("OK")
