# The nested-loop twin of `retrace_accumulator_type_flip`: the accumulator flips
# int -> float in the OUTER loop, so the retrace grown for it does not close onto
# a token of its own compilation but onto the one the inner loop left behind.
#
# pyre DECLINES that close today, and the recorded `loops_aborted=2` with
# `retraces_compiled=0` is what says so: `jump_to_existing_trace` finds the match,
# and the unroll pass then discards it because the JUMP names an external target
# token, falling back to `jump_to_preamble` (unroll.py:228). pypy takes it, and
# emits the two `new_with_vtable`s plus a `jump(..., descr=TargetToken(<loop>))`.
#
# Measured 2026-08-09, admitting the close on dynasm — the backend that relocates
# every LABEL and refuses a JUMP below the first page, so the branch does name a
# real address — SIGSEGVs `exception_escape_inlined_midframe_tb_node` and adds a
# bridge plus ~200 guard failures each to `global_reassign`,
# `mapdict_unboxed_type_change_attr`, `math_isqrt_compare_bridge_resume` and
# `method_reassign_after_warmup`. Naming a live address is therefore not the whole
# precondition. This fixture is the acceptance test for whatever closes that gap:
# `loops_aborted` goes to 0 and `retraces_compiled` to 1 when it does.
#
# `pypyjit` is absent on the CPython oracle and present on PyPy and pyre, and
# `retrace_limit` defaults to 0 (`rpython/rlib/jit.py`), so no retrace is
# attempted at all without raising it. Guarding the import keeps the printed value
# identical across all three. `set_param` rather than an environment variable
# because the wasm guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("retrace_limit=5")
except ImportError:
    pass


def f(n, m):
    s = 0
    o = 0
    while o < n:
        j = 0
        while j < m:
            j += 1
        if o > 200:
            s = s + 0.5
        else:
            s = s + 1
        o += 1
    return s


print(f(3600, 100))
