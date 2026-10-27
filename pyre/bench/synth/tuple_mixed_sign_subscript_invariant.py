# pyre-check: max-pypy-ratio=12
# A loop-invariant tuple item read through a PURE `getarrayitem_gc` becomes a
# short-preamble `used_box` -- the extended LABEL carries both the boxed item
# and its unboxed `intval`.  Reading a SECOND item at a negative index in the
# same iteration leaves that producer unresolvable when the short preamble is
# set up against the next Label, and the box it names is a LABEL slot, so the
# ops that read it survive the producer's drop and resolve the slot to the
# back-edge value of whatever else occupies it -- here the accumulator.  The
# invariant is then clobbered once per iteration and `total` runs away.
#
# `c` is a parameter so the items stay real boxes: a module-level tuple
# literal constant-folds and never reaches a LABEL slot at all.


def run(c):
    total = 0
    n = 0
    while n < 28800000:
        total += c[0] + c[-1]
        n += 1
    return total


print(run((10, 20, 30, 40)))
