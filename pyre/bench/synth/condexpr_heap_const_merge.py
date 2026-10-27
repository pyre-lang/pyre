# pyre-check: max-pypy-ratio=2.7
# main CI 10-24..10-26 (6 runs, all native legs): 1.0x-2.0x; ceiling = max x 1.35.
# The trip count answers two ends at once.  The loop has to run long enough to
# reach the JIT, or the merge this fixture exists for never happens; and pypy's
# own execution has to clear FLOOR_GATE_MIN_BASELINE_S, or the ratio divides
# the floor constant instead of a measurement and neither bound arms.
#
# `insert_renamings` skipped any arg whose `as_variable()` equalled
# `last_exception` / `last_exc_value`.  A Constant has `as_variable() == None`,
# and a non-exception link has both fields `None`, so the bare `==` matched and
# dropped the constant before emitting its `ref_copy` -> the merge register was
# left unmaterialised and the heap arm read a stale box.  Small-int arms are
# unaffected because they materialise through box_int_fn into a Variable.
#
# `ce_big` (taken arm const 1000000, a heap int) miscompiled to 333001602
# instead of 666001002; `ce_small` (both arms small ints) stayed correct.  Both
# shapes are kept for the small-vs-heap contrast, and `ce_big` is correct even
# when the branch never diverges (the merge structure alone triggered the
# drop).  Pure arithmetic -> deterministic checksum.
N = 70193549


def ce_small():
    acc = 0
    i = 0
    while i < N:
        c = i % 3
        x = 5 if c else 3
        acc = acc + x
        i = i + 1
    return acc


def ce_big():
    acc = 0
    i = 0
    while i < N:
        c = i % 3
        x = 1000000 if c else 3
        acc = acc + x
        i = i + 1
    return acc


def main():
    print(ce_small(), ce_big())


main()
