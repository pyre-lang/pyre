# A loop body fat enough that one traced iteration runs past `trace_limit`, so
# the trace overflows before it can close. `blackhole_if_trace_too_long`
# (`pyjitpl.py`) finds no inlinable huge function here, so it calls
# `prepare_trace_segmenting` (`pyjitpl.py`), which marks the green key
# with the force-finish flag and asks for the next iteration to be traced again.
#
# On that retry the marked key is supposed to CUT: once the trace passes 0.8x
# `trace_limit` at a `jit_merge_point`, `pyjitpl.py` appends
# `GUARD_ALWAYS_FAILS`, compiles what it has, and blackholes with
# `ABORT_SEGMENTED_TRACE`. The cut arm is what turns the key's forever-retracing
# into compiled code.
#
# Without the cut, every retry runs into the same overflow and aborts again:
# `loops_aborted` read 64 with `bridges_compiled` 0 and `guard_failures` 11918.
# With it the key segments and the workload runs out of compiled arms instead.
#
# The 0.8x check only has something to fire at if a merge point is CROSSED while
# the trace sits in the 0.8x..1.0x band, so the body size and `trace_limit` are
# load-bearing together, not independently: at `trace_limit=200` the same body
# jumps the band in one crossing and never segments.  The band follows the RAW
# op count of one traced iteration, so a change in what the walker records for
# the body moves it.  The swept band is recorded once, at the `set_param`
# call below, and 250 sits inside it.
#
# Which means the number tracks what one traced iteration COSTS in this tree,
# and has to be re-fit whenever that moves — by sweeping, not by nudging.
#
# GET_ITER of exact `range` now records `W_Range.descr_iter` (`functional.py
# descr_iter` / `w_range_iter`) instead of the GetIter fold. That extra raw
# cost pushed one more retry into the 0.8x band: `prepare_trace_segmenting`
# (`pyjitpl.py`) cuts a bridge with `GUARD_ALWAYS_FAILS` and blackholes
# `ABORT_SEGMENTED_TRACE`. PyPy's `PYPYLOG=jit-summary` on this fixture is
# 4 loops / 11 bridges / 13 aborts (4 too-long + 9 segmenting); pyre lands
# 3 loops / 13 bridges / 13 aborts (3 too-long + 10 segmenting). The extra
# abort and its compiled cut (and the matching `fbw_blackhole_adopted_
# single_frame`) are that segmenting abort, not a new inline admit.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the params only bind where a
# JIT exists. `set_param` rather than an environment variable because the wasm
# guest sees no environment.
try:
    import pypyjit

    # The 0.8x..1.0x band is a raw-op-count property; every backend records
    # the same raw op count for this body, and the sweep put the band at
    # 220..260.
    pypyjit.set_param("trace_limit=250")
    pypyjit.set_param("threshold=20")
except ImportError:
    pass


def body(a, b, c, d, e):
    t = 0
    for _ in range(4):
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


s = 0
for i in range(4000):
    s += body(i, i + 1, i + 2, i + 3, i + 4)
print(s % 1000)
