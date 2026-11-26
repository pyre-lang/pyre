# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# `_match_signature` fills positional defaults (recording
# `missing_positional`), then always runs the kwonly-defaults loop
# before raising `ArgErrMissing(missing_positional, True)` or the
# kwonly form.
#
# [3.15-spec] `PyDict_GetItemRef(func->func_kwdefaults, varname, &def)`
# (ceval.c, v3.15.0rc3) ↔ pypy3 `space.finditem_str`.  Storage lookup
# never dispatches a subclass `__getitem__`, so a raising mapping still
# yields the missing-argument TypeError.  Synth has no `pypy-diverges`
# header; this fixture asserts the 3.15 counts itself.
N = 500


class CountDict(dict):
    gets = 0

    def __getitem__(self, key):
        CountDict.gets += 1
        return dict.__getitem__(self, key)


class RaiseDict(dict):
    def __getitem__(self, key):
        raise RuntimeError("from getitem")


def f(a, *, k):
    return a, k


def main():
    f.__kwdefaults__ = CountDict()
    pos = []
    pos_kw = []
    kwonly = []
    for i in range(N):
        try:
            f()
        except TypeError as e:
            pos.append(str(e))
    gets_empty = CountDict.gets
    CountDict.gets = 0
    for i in range(N):
        try:
            f(k=1)
        except TypeError as e:
            pos_kw.append(str(e))
    gets_kw = CountDict.gets
    CountDict.gets = 0
    for i in range(N):
        try:
            f(1)
        except TypeError as e:
            kwonly.append(str(e))
    gets_pos = CountDict.gets

    raised = []
    f.__kwdefaults__ = RaiseDict()
    for i in range(N):
        try:
            f()
        except Exception as e:
            raised.append("%s: %s" % (type(e).__name__, e))
        try:
            f(1)
        except Exception as e:
            raised.append("%s: %s" % (type(e).__name__, e))

    pos_msg = "f() missing 1 required positional argument: 'a'"
    kw_msg = "f() missing 1 required keyword-only argument: 'k'"
    if (
        len(pos) != N
        or set(pos) != {pos_msg}
        or gets_empty != 0
        or len(pos_kw) != N
        or set(pos_kw) != {pos_msg}
        or gets_kw != 0
        or len(kwonly) != N
        or set(kwonly) != {kw_msg}
        or gets_pos != 0
        or len(raised) != N * 2
        or set(raised) != {"TypeError: " + pos_msg, "TypeError: " + kw_msg}
    ):
        print(
            "FAIL",
            (len(pos), len(set(pos)), gets_empty),
            (len(pos_kw), len(set(pos_kw)), gets_kw),
            (len(kwonly), len(set(kwonly)), gets_pos),
            (len(raised), sorted(set(raised))),
        )
        return 1
    print("PASS call_kw missing-order storage lookup")
    return 0


import sys

sys.exit(main())
