# pyre-check: max-pypy-ratio=110
# A match statement whose subject pattern is a SEQUENCE of CLASS patterns, each
# carrying a literal sub-pattern plus a capture (`[Node('lit', a), Node('lit', b)]`).
# MATCH_CLASS has no dedicated arm in the JIT trace walker and fell through to the
# catch-all, which aborts the (unsupported) structural-match trace WITHOUT modelling
# MATCH_CLASS's -2 stack effect. The resulting stale operand-stack depth made the
# enclosing block's return link carry extra pattern temporaries, tripping the flatten
# pass's `make_return` arity assert (a JIT-codegen crash). Output verified against
# CPython/PyPy.
M = 1000003


class Node:
    __match_args__ = ('kind', 'payload')

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload


def classify(o):
    match o:
        case [Node('lit', a), Node('lit', b)]:
            return 2 + ((a + b) & 15)
        case Node('lit', v):
            return 11 + (v & 7)
        case _:
            return 19


def run(n):
    subjects = [
        [Node('lit', 6), Node('lit', 1)],
        Node('lit', 5),
        Node('other', 9),
        [Node('lit', 2), Node('lit', 3)],
    ]
    acc = 0
    for i in range(n):
        acc = (acc + classify(subjects[i % 4])) % M
    return acc


print(run(270000))
