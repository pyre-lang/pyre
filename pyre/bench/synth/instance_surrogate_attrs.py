# pyre-check: max-pypy-ratio=50
# pyre-check: trace-shape=main:absent=CallMayForceR
# PyPy traces through builtin getattr for lone-surrogate names just as it does
# for ordinary names.  The pyre fold keeps the WTF-8 name intact, virtualizes
# the bound Method into its CALL, and inlines the property's Python getter.
# Losing either path leaves a per-iteration CALL_MAY_FORCE and crosses this
# ceiling; the folded dynasm path is about 6x faster than that residual path.
N = 1200000

METH = '\udc81'   # lone surrogate naming a method on the class
PROP = '\udc82'   # lone surrogate naming a property (data descriptor)
A1 = '\udc83'            # lone surrogate naming a per-instance attribute
A2 = '\udc84\udc85'      # multi-surrogate per-instance attribute name
A3 = 'ascii_attr'        # plain name stored alongside the surrogate nodes


def _meth(self):
    return 1


class P:
    pass


setattr(P, METH, _meth)
setattr(P, PROP, property(lambda self: 2))


def main():
    p = P()
    acc = 0
    i = 0
    # Surrogate-named attribute access through the full descriptor protocol
    # in a JIT-compiled hot loop: a non-data descriptor (function bound
    # through the type MRO) and a data descriptor (property __get__).
    while i < N:
        acc = acc + getattr(p, METH)()
        acc = acc + getattr(p, PROP)
        i = i + 1

    # Post-loop tail running in the already-compiled `main` frame:
    # per-instance surrogate-named attributes are stored as mapdict nodes
    # (keyed by their full WTF-8 name), interleaved with a plain-named one,
    # then read back, summed through __dict__, deleted and re-added.
    setattr(p, A1, 5)
    setattr(p, A2, 7)
    setattr(p, A3, 11)
    acc = acc + getattr(p, A1) + getattr(p, A2) + getattr(p, A3)
    acc = acc + sum(p.__dict__.values())
    acc = acc + (1 if A1 in p.__dict__ else 0)
    acc = acc + (1 if A2 in p.__dict__ else 0)
    delattr(p, A1)
    acc = acc + (0 if hasattr(p, A1) else 1)
    setattr(p, A1, 13)
    acc = acc + getattr(p, A1)
    print(acc)


main()
