# pyre-check: max-pypy-ratio=30
# macOS reads 13.2-15.7x on dynasm (19.4x once); the ceiling is about twice
# that. Generatorentry guards compile bridges the way `compile.py handle_fail` does.
# pyre-check: jitstats-band=bridges_compiled=1,guard_failures=8
# dynasm, cranelift and wasm read 10/12/73543 at PYPY_GC_NURSERY 1/2/4/8 MB
# after `_age_out_of_nursery`. The band is leftover schedule jitter, not a
# nursery split. One TY_NONE guard's `must_compile` crosses the threshold a
# second time only after its `JitCounter` slot decays.
# `decay_all_counters` runs every 32 minor collections (`counter.py`
# `invoke_after_minor_collection`), so a minor inside a failure burst — the
# 1 MB rec compile used to land on that boundary, wasm 4 MB used to split a
# GUARD_VALUE by moving young W_Code — decides whether the force at
# `LoadFastBorrow` closes a loop (`reached_loop_header` -> `compile_loop`)
# or finishes an entry bridge.
# Regression oracle: a callee entered through a CALL_ASSEMBLER fold reads a
# local out of a frame the trace is being recorded for, forcing it.  The
# recording stops there and execution goes on from behind the call, with the
# call's result, in every frame the recording had open -- never from in front
# of the call, and never in the outermost frame alone.
#
#   rec  -- `rec` and `step` call each other; the recursive call is folded
#           while both are inlined levels of the loop in `run_rec`, and the
#           innermost `rec` reads `step`'s and `rec`'s locals;
#   gen  -- a generator with several yields is resumed by the `for` in
#           `run_gen`, and its body reads the locals of that loop's frame.
#
# The call counter must equal what an interpreter counts, and the checksums
# over the results and the forced locals match.
import sys

N = 36000
SWITCH = 17000

calls = [0]
seen = [0]


def force(where, name, on):
    # Count, and when selected read a local out of the named frame above.
    calls[0] += 1
    if on:
        frame = sys._getframe(1)
        while frame.f_code.co_name != where:
            frame = frame.f_back
        seen[0] += frame.f_locals[name]


def step(d, i, on):
    k = d * 2 + i
    r = rec(d - 1, i, on)
    return r + k


def rec(d, i, on):
    if d == 0:
        force('step', 'k', on)
        force('rec', 'd', on)
        return i
    x = step(d, i, on)
    return x + d


def run_rec(n):
    acc = 0
    for i in range(n):
        acc += rec(3, i, i >= SWITCH and i % 7 == 0)
    return acc


def gen(i, on):
    yield i
    force('run_gen', 'i', on)
    yield i + 1
    force('run_gen', 'acc', on)
    yield i + 2


def run_gen(n):
    acc = 0
    for i in range(n):
        for v in gen(i, i >= 5 and i % 2 == 0):
            acc += v
    return acc % 1000003


def report(label, fn):
    calls[0] = 0
    seen[0] = 0
    print(label, fn(N), calls[0], seen[0] % 1000003)


def _age_out_of_nursery():
    # Promote live W_Code (GUARD_VALUE / make_a_counter_per_value, keyed by
    # address) out of the nursery before the measured loops. A minor that
    # moves a young object inside the failure burst splits the count, and
    # `decay_all_counters` (counter.py invoke_after_minor_collection) then
    # decides whether a TY_NONE guard's second `must_compile` compiles a
    # bridge. cranelift core has no gc; 600 discarded ~16 KB tuples (below
    # large_object_threshold) exceed an 8 MB nursery without compiling
    # (loop threshold 1039).
    try:
        import gc as _gc
        # Eight minors: one leaves a 1 MB run's `run_rec` compile on a
        # 32-minor decay boundary (`invoke_after_minor_collection`). Eight
        # shifts that remainder off 0.
        i = 0
        while i < 8:
            _gc.collect(0)
            i += 1
        return
    except ImportError:
        pass
    i = 0
    while i < 600:
        dropped = (None,) * 2000
        i += 1


def main():
    # `gen` first: with `rec` ahead of it no recording of `run_gen` meets the
    # force.
    _age_out_of_nursery()
    report('gen', run_gen)
    report('rec', run_rec)
    # A second pass enters the loops and bridges the first one compiled.
    report('gen', run_gen)
    report('rec', run_rec)


main()
