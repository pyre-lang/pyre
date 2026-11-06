# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# CALL_KW whose callee the inline path refuses (zero positional parameters
# plus a `for` body: `try_multiframe` is false and the body is not strict).
# `f(**d)` is CALL_FUNCTION_EX; this fixture uses CALL_KW `declined(a=1)` so
# the matcher descent is the path under test.
#
# [3.15-spec] `PyDict_GetItemRef(func->func_kwdefaults, varname, &def)`
# (ceval.c, v3.15.0rc3) ↔ pypy3 `space.finditem_str`.  The residual
# binder reads dict storage, so a subclass `__getitem__` stays at 0.
# Synth has no `pypy-diverges` header; this fixture asserts the 3.15
# count itself.
N = 2000


class CountDict(dict):
    gets = 0

    def __getitem__(self, key):
        CountDict.gets += 1
        return dict.__getitem__(self, key)


def declined(*, a, b=1):
    acc = 0
    for x in (a, b):
        acc += x
    return acc


declined.__kwdefaults__ = CountDict(b=1)


def main():
    total = 0
    for i in range(N):
        total += declined(a=1)
    if total != N * 2 or CountDict.gets != 0:
        print(f"FAIL total={total} getitem={CountDict.gets}")
        return 1
    print("PASS call_kw binder storage lookup")
    return 0


import sys

sys.exit(main())
