# pyre-check: trace-shape=rebound_between_reads:present=Jump
# Three `locals()` calls in ONE loop body, separated by two rebinds of the same
# cell.
#
# The expansion reads a cell slot as `getfield_gc_r(cell, Cell.contents)` -- the
# only place a cell is read as a foldable field rather than through the
# `load_deref` residual.  The write between two of those reads is the
# `store_deref` residual, whose `EffectInfo` is `CannotRaise` with empty
# write-descr sets, so neither the `clean_caches` arm nor `force_from_effectinfo`
# names `Cell.contents`.  Whatever keeps the second read from being folded onto
# the first is therefore not that invalidation, and this fixture is what says so
# out loud: each `locals()` must report the value the cell holds AT that point.
#
# `total` is a sum of differences, so a read folded onto an earlier one drives it
# to zero rather than off by a constant.
N = 100000


def rebound_between_reads():
    c = 0

    def peek():
        return c

    total = 0
    for i in range(N):
        c = i
        a = locals()["c"]
        c = i + 3
        b = locals()["c"]
        c = i + 7
        e = locals()["c"]
        total += (b - a) + (e - b)
    return total, peek()


print(rebound_between_reads())
