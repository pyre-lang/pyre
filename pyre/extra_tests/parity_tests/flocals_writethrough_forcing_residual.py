# pyre-check: pypy-diverges: pins the PEP 667 write-through f_locals proxy, which pypy3 does not have -- its f_locals is a snapshot dict, so the callee's write never reaches the caller's fast local
# CPython-suite gap: the suite exercises FrameLocalsProxy writes only against
# cold frames, never against a frame whose surrounding loop is being traced at
# the moment the write happens.
# parity-tests reason: the write is made by a call the tracer runs for real,
# against the frame the tracer is stepping a copy of, so it is the tracer's
# handling of that call -- not the proxy -- that decides whether the write
# survives.
# pyre-check: regresses-under: PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS=1 -- without the adopt the walk keeps the box the local held before the call, and both traced iterations lose the write

"""A callee's write through the caller's ``f_locals`` survives tracing.

``setter`` forces the caller and writes its fast-local array while the tracer
holds a boxed copy. ``adopt_residual_locals_writes`` must read the slot back for
both the concretely called and later inlined recordings. The bound reaches both;
the ``regresses-under`` arm proves the shape remains live.

``callee_sorted`` is the inlined shape: its local ``x`` is folded, and
``sorted``'s key writes that local through ``FrameLocalsProxy``. The forcing
stores are recorded before ``CALL_MAY_FORCE`` (``vrefs_after_residual_call`` /
``optimize_VIRTUAL_REF_FINISH``), so the compiled loop reloads ``x``.

``callee_helper`` is that fold one frame out. ``helper_write`` is itself
inlined, so ``sys._getframe(1)`` is the caller whose vref the call forced.

``callee_reforce`` splits that into two residuals. ``force_then_write``
forces the callee and then writes ``x`` through ``f_locals``. ``force_read``
forces the same frame again. The second force must leave the write in place:
the callee reads ``x`` and returns it.
"""

import sys

ROUNDS = 4000


def bump_callee_local(v):
    frame = sys._getframe(1)
    cur = frame.f_locals["x"]
    frame.f_locals["x"] = cur + 1000
    return v


def callee_sorted(i):
    x = i
    sorted((0,), key=bump_callee_local)
    return x


def helper_write():
    frame = sys._getframe(1)
    frame.f_locals["x"] = frame.f_locals["x"] + 1000


def callee_helper(i):
    x = i
    helper_write()
    return x


def force_then_write(value):
    frame = sys._getframe(1)
    frame.f_locals["x"] = frame.f_locals["x"]
    frame.f_locals["x"] = value


def force_read():
    frame = sys._getframe(1)
    return frame.f_locals["x"]


def callee_reforce(i):
    x = i
    force_then_write(i + 1000)
    force_read()
    return x


def run(rounds):
    def setter(value):
        sys._getframe(1).f_locals["x"] = value

    x = -1
    lost = []
    for i in range(rounds):
        setter(i)
        if x != i:
            lost.append((i, x))
    return lost


def run_inlined(rounds):
    lost = []
    for i in range(rounds):
        got = callee_sorted(i)
        if got != i + 1000:
            lost.append((i, got))
    return lost


def run_helper(rounds):
    lost = []
    for i in range(rounds):
        got = callee_helper(i)
        if got != i + 1000:
            lost.append((i, got))
    return lost


def run_reforce(rounds):
    lost = []
    for i in range(rounds):
        got = callee_reforce(i)
        if got != i + 1000:
            lost.append((i, got))
    return lost


lost = run(ROUNDS)
# Report a few, not all of them: a runtime without the write-through loses
# every iteration, and the whole list buries the runner's output.
assert not lost, f"{len(lost)} of {ROUNDS}, first: {lost[:4]}"

lost_inlined = run_inlined(ROUNDS)
assert not lost_inlined, f"inlined {len(lost_inlined)} of {ROUNDS}, first: {lost_inlined[:4]}"

lost_helper = run_helper(ROUNDS)
assert not lost_helper, f"helper {len(lost_helper)} of {ROUNDS}, first: {lost_helper[:4]}"

lost_reforce = run_reforce(ROUNDS)
assert not lost_reforce, f"reforce {len(lost_reforce)} of {ROUNDS}, first: {lost_reforce[:4]}"
print("OK")
