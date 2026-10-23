# `f(*(i, 1), **mapping)` on a callee the inline takes: the specialised-pair
# path of `fbw_unpack_call_function_ex_args` unpacks the two-element star
# tuple, then `argument.py` `_match_keywords` binds the mapping.  A
# one-element tuple is array-backed; `(i, 1)` is the specialised-pair layout.
# The hot loop pins a successful pair call, an unexpected keyword, and a
# keyword that collides with a star-tuple positional, and the printed
# messages pin both TypeErrors.


def f2(a, b):
    return a + b


def main(n):
    unexpected = None
    multiple = None
    ok = 0
    for i in range(n):
        ok += f2(*(i, 1))
        try:
            unexpected = f2(*(i, 1), **{'c': 1})
        except TypeError as ex:
            unexpected = str(ex)
        try:
            multiple = f2(*(i, 1), **{'a': 1})
        except TypeError as ex:
            multiple = str(ex)
    print('ok', ok)
    print('unexpected', unexpected)
    print('multiple', multiple)


main(3000)
