# `del <cellvar>` lowers to `load_deref_value` (the bound check) plus
# `store_deref_value(cell, NULL)`, where the NULL is the clear-the-cell
# sentinel `bh_store_deref_value_fn` hands straight to `w_cell_set` without
# ever reading it.
#
# The walker's NULL-Ref-arg refusal must not decline that residual: a decline
# marks the walk as carrying a recorded-but-unexecuted effect, so every walk
# over this loop fails the walk-end flush ("unjournaled effect — legacy replay
# kept") and hands the region back to a replay that re-runs the residuals the
# walk already executed concretely.  `bump` is one of them, so `log` grew
# by exactly one entry per walk: 20048 instead of 20000, on both backends and
# at stock thresholds.
#
# `grab` is what makes the cell a real closure cell rather than a plain local.
N = 1600000

log = []


def bump(v):
    log.append(v)
    return v


def run(n):
    acc = 0
    i = 0
    while i < n:
        x = i

        def grab():
            return x

        acc += grab() & 7
        bump(i)
        del x
        i += 1
    return acc, len(log)


print(run(N))
