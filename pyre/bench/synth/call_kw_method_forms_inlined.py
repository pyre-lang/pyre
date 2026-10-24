# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,entry-bridge:m
# `callmethod.py CALL_METHOD_KW` and a bound `Method` (`function.py
# call_obj_args`) both reach `pycode.py funcrun_obj`. The receiver is
# `w_firstarg` of `argument.py _match_signature`, so the keyword call
# inlines through the same binder as a plain function.
N = 200000


class C:
    def m(self, a, b):
        return self.x + a + b


def main():
    obj = C()
    obj.x = 0
    bm = obj.m
    total = 0
    i = 0
    while i < N:
        obj.x = i
        total += obj.m(1, b=2)
        total += bm(1, b=2)
        i += 1
    # sum_{i=0}^{N-1} 2 * (i + 1 + 2) = N * (N + 5)
    expect = N * (N + 5)
    if total != expect:
        raise AssertionError("%s != %s" % (total, expect))
    print(total)
    print("PASS")


main()
