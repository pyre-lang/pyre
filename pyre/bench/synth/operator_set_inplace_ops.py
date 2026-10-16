# pyre-check: max-pypy-ratio=13
# operator's in-place functions (iadd/iand/ior/... = space.inplace_X) and the
# mutable set's in-place operator slots (__iand__/__ior__/__ixor__/__isub__,
# each a set/frozenset-only op that mutates self and returns it). Sets are
# printed via sorted() since internal iteration order is implementation
# defined. A warmup loop exercises the int in-place fast paths first.
import operator


def warm(n):
    acc = 0
    for i in range(n):
        acc = operator.iadd(acc, i % 7)
    return acc


def s(x):
    return sorted(x)


def main():
    print("warm", warm(75000))

    # operator in-place arithmetic / bitwise on ints
    print("iadd", operator.iadd(3, 4))
    print("isub", operator.isub(10, 3))
    print("imul", operator.imul(3, 4))
    print("ipow", operator.ipow(2, 10))
    print("ilshift", operator.ilshift(1, 4))
    print("irshift", operator.irshift(256, 2))
    print("imod", operator.imod(17, 5))
    print("ifloordiv", operator.ifloordiv(17, 5))
    print("itruediv", operator.itruediv(9, 2))
    print("iand_int", operator.iand(0b1100, 0b1010))
    print("ior_int", operator.ior(0b1100, 0b1010))
    print("ixor_int", operator.ixor(0b1100, 0b1010))

    # operator on mutable objects returns the same (mutated) object
    a = [1, 2]
    print("iadd_list_ident", operator.iadd(a, [9]) is a, a)
    b = [1, 2]
    print("iconcat_ident", operator.iconcat(b, [3]) is b, b)
    print("imul_list", operator.imul([1], 3))

    # operator set in-place functions (value-equivalent to & | ^ -)
    print("iand_set", s(operator.iand({1, 2, 3}, {2, 3, 4})))
    print("ior_set", s(operator.ior({1, 2}, {3})))
    print("ixor_set", s(operator.ixor({1, 2}, {2, 3})))
    print("isub_set", s(operator.isub({1, 2, 3}, {2})))

    # set in-place dunders: mutate self, return self, set/frozenset only
    print("dunder_iand", s({1, 2, 3}.__iand__({2, 3, 4})))
    print("dunder_ior", s({1, 2}.__ior__({3})))
    print("dunder_ixor", s({1, 2}.__ixor__({2, 3})))
    print("dunder_isub", s({1, 2, 3}.__isub__({2})))
    print("dunder_iand_frozen", s({1, 2, 3}.__iand__(frozenset({2, 3}))))
    print("dunder_iand_nonset_NI", {1, 2}.__iand__([1, 2]) is NotImplemented)

    st = {1, 2, 3}
    print("stmt_iand_ident", st.__iand__({2, 3}) is st, s(st))

    # augmented-assignment statement mutates in place
    u = {1, 2}
    v = u
    u |= {3, 4}
    print("stmt_ior", u is v, s(u))
    w = {1, 2, 3}
    w -= {1}
    print("stmt_isub", s(w))

    print("has_iadd", hasattr(operator, "iadd"))
    print("has_set_iand", hasattr(set, "__iand__"))
    print("op_dunder_alias", s(operator.__iand__({1, 2}, {2, 3})))


main()
