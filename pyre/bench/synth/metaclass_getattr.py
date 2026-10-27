# pyre-check: max-pypy-ratio=130
# A metaclass __getattr__ is consulted when an attribute is missing on a
# type and its MRO, mirroring _handle_getattribute looking the hook up on
# type(cls). A present attribute resolves normally without the hook.

N = 400000


class Meta(type):
    def __getattr__(cls, name):
        return "meta:" + name


class C(metaclass=Meta):
    own = 1


def main():
    # A present attribute resolves from the type's own dict, no hook.
    print("own", C.own)
    # A missing attribute falls back to the metaclass __getattr__.
    print("missing", C.nope)

    # Hot loop so the JIT-compiled type getattribute path exercises the
    # metaclass __getattr__ dispatch every iteration.
    acc = 0
    i = 0
    while i < N:
        v = C.k
        if v == "meta:k":
            acc = acc + 1
        i = i + 1
    print("loop", acc)


main()
