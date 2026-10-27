# pyre-check: max-pypy-ratio=7.1
# A conditional expression selects one of two loop locals as the receiver
# (`q = o if cond else p`) whose attribute is then read/written inside a hot
# `for i in range(...)` loop.  The two arms of the conditional are distinct
# JitCode source blocks reached without a Python CFG edge; replaying the
# previous opcode across that layout switch overwrote the loop-carried
# FOR_ITER iterator with the selected local, so a later guard serialized the
# foreign box into the resume image and the range iterator was lost
# (`TypeError: not an iterator`).  A mid-loop swap of the two locals keeps the
# selection genuinely dynamic.  Deterministic, terminating, int checksum;
# jit == nojit.
M = 1000000007


class Base:
    __slots__ = ("a",)


class Mid(Base):
    __slots__ = ("b",)


class Leaf(Mid):
    __slots__ = ("c",)


def run():
    acc = 0
    o = Leaf()
    o.a = 0
    o.b = 0
    o.c = 0
    p = Base()
    p.a = 0
    for i in range(1, 6276020):
        q = o if (i & 1) else p
        q.a = (q.a + i) % 91
        if i == 238260:
            o, p = p, o
        acc = (acc + q.a) % M
    return acc


print(run())
