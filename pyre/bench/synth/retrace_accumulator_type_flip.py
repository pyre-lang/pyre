# A loop-carried accumulator that flips int -> float partway through, so the
# guard at that position fails repeatedly and the bridge grown there closes with
# a virtual state matching no existing target. That makes the optimizer ask for a
# retrace (`compile.py:1085 retrace_needed`), and `retrace_limit` is what lets
# one be BUILT: it defaults to 0 (`rpython/rlib/jit.py`), so no other fixture
# in this corpus reaches `compile_retrace` at all.
#
# It covers a retrace that is ASSEMBLED: `retrace_needed` -> `cut_retrace_from`
# -> the unroll pass -> the fresh target token, which the closing JUMP then
# matches. `loops_aborted` in the recorded jit-stats is what says so — it read 5
# while every attempt gave up, and reads 0 now.
#
# The closure crosses bytecode offsets (the guard resumes deeper than the loop
# header it closes onto), so `close_loop_args_at` has to publish the merge
# point's own static stack depth rather than the depth the resumed frame still
# advertises. Publishing the stale deeper one pins a loop-VARIANT
# `valuestackdepth` as a constant, and the retrace then fails to match even its
# own label; the fallback is `jump_to_preamble` (`unroll.py:156/171`), which
# pyre refuses on an arity mismatch — MC_DIAG slot 57.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. Guarding the import
# keeps the output identical across all three while the param only binds where a
# JIT exists. `set_param` rather than an environment variable because the wasm
# guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("retrace_limit=5")
except ImportError:
    pass


def f(n):
    s = 0
    i = 0
    while i < n:
        if i > 100000:
            s = s + 0.5
        else:
            s = s + 1
        i += 1
    return s


print(f(9000000))
