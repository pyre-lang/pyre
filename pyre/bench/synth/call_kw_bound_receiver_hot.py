# CALL_KW whose callable is a stored bound method or an instance `__call__`.
# `Method.call_args` prepends `w_instance` (`space.call_obj_args` →
# `args.prepend`); `descroperation.py call_args` passes the instance the
# same way. The recorded matcher array has to carry that receiver: the
# CALL_KW `self_or_null` operand is the original NULL sentinel for these
# shapes, and overwriting `callee_args[0]` after the matcher would replace
# `self` or, when the function has no positional slot, the `*args` tuple.
N = 3000


class Box:
    def __init__(self, x):
        self.x = x

    def f(self, a, b=0):
        return self.x + a + b

    def __call__(self, a, *rest, b=0):
        total = self.x + a + b
        for r in rest:
            total += r
        return total

    def g(self, *a, **k):
        extra = 0
        for v in a:
            extra += v
        for v in k.values():
            extra += v
        return self.x + extra

    def d(self, *, k=0):
        return self.x + k

    def z(*a, **k):
        self = a[0]
        extra = 0
        for v in a[1:]:
            extra += v
        for v in k.values():
            extra += v
        return self.x + extra


class Holder:
    pass


def main():
    obj = Box(3)
    m = obj.f
    holder = Holder()
    holder.m = obj.f
    mg = obj.g
    md = obj.d
    mz = obj.z
    checksum = 0

    for i in range(N):
        r_var = m(1, b=2)
        r_attr = holder.m(1, b=2)
        assert r_var == 6
        assert r_attr == 6
        checksum += r_var + r_attr

    for i in range(N):
        r_call = obj(1, i % 3, b=2)
        assert r_call == 6 + (i % 3)
        checksum += r_call

    for i in range(N):
        r_star = mg(i, b=2)
        assert r_star == 5 + i
        checksum += r_star

    for i in range(N):
        r_kw = md(k=i)
        assert r_kw == 3 + i
        checksum += r_kw

    for i in range(N):
        r_zero = mz(i, b=2)
        assert r_zero == 5 + i
        checksum += r_zero

    print(checksum)


main()
