# pyre-check: max-pypy-ratio=24
# Regression guard: the frame an inlined callee's traceback node names must
# report the callee's NON-PARAMETER locals, not just the arguments it was
# entered with.
#
# A `STORE_FAST` on the callee's own fresh frame is folded to an SSA register
# and emits no store into `locals_cells_stack_w`, so the array holds only the
# parameters seeded at the inline. Storing that frame into `PyTraceback.frame`
# lets it outlive the trace, and without replaying the fold first every
# `tb_frame.f_locals` consumer — `traceback` formatting and debuggers among
# them — silently loses `marker`.
#
# The loop has to pass the compile threshold: the interpreted iterations write
# the live frame directly and read correctly either way, so a short run cannot
# see this. Before the fix, dynasm and cranelift both lost `marker` from the
# moment the loop compiled — 1242 of 4000 iterations still answered
# `('i', 'marker')`, the rest `('i',)`.
#
# Expected output: 32000 ('i', 'marker')

N = 32000


def mid(i):
    marker = i * 2
    raise ValueError(marker)


def drive():
    kinds = {}
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            tb = e.__traceback__.tb_next
            names = tuple(sorted(tb.tb_frame.f_locals)) if tb is not None else None
            kinds[names] = kinds.get(names, 0) + 1
        k += 1
    return kinds


for names, count in sorted(drive().items(), key=lambda kv: -kv[1]):
    print(count, names)
