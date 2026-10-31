# pyre-check: max-pypy-ratio=2.9
# Readings 0.3x-0.7x. A ceiling under 3 derives no floor.
# A trace that outgrows `trace_limit` must not re-apply effects the walk has
# already executed. The walker's length check fires at an arbitrary opcode,
# unlike every other abort, which declines before executing an effect it
# cannot roll back; an abort resumes by re-interpreting the iteration from the
# trace entry, so an executed-but-unrollbackable effect would land twice.
#
# `pypyjit.set_param` lowers `trace_limit` far enough that the check binds on
# every backend (the param hook reaches the wasm guest, which sees no
# environment). CPython, the oracle, has no `pypyjit`; guarding the import
# keeps the output identical across all three while the limit only binds where
# a JIT exists.
try:
    import pypyjit
except ImportError:
    pypyjit = None


def set_trace_limit(n):
    if pypyjit is not None:
        pypyjit.set_param("trace_limit=%d,threshold=1,function_threshold=1" % n)


N = 63415

# Phase A — an effect-free body under a limit small enough to trip it. The
# committed `.jitstats` baseline records the resulting abort, so the length
# check itself cannot silently stop firing.
set_trace_limit(8)

i = 0
free_total = 0
while i < N:
    x = i
    x = x + 1
    x = x + 2
    x = x + 3
    x = x + 4
    x = x + 5
    free_total = free_total + x
    i = i + 1

# The 100/101 post-step cutoff differs between native and wasm: wasm
# takes two extra aborts and one bridge. 96..99 exercises the same abort
# path on all three backends; use 98 within that shared interval.
# Phase B — the same shape with the limit raised past the body's own length,
# so the check lands mid-body, and with one effect per rollback class: a list
# slot store is journaled, a dict store and a list append historically were
# not. Each counter must read exactly N; a replayed iteration shows up as
# N + 1.
set_trace_limit(98)

box = [0]
d = {"k": 0}
obj = []

i = 0
total = 0
while i < N:
    x = i
    x = x + 1
    x = x + 2
    x = x + 3
    box[0] += 1
    d["k"] = d["k"] + 1
    obj.append(x)
    total = total + x
    i = i + 1

print(free_total, total, box[0], d["k"], len(obj), sum(obj) % 1000003)


# Phase C — the same exactly-once contract for a function-entry trace. Its
# trace root is pc 0 rather than the loop header below, so the root key is
# never revisited and only ABORT_TOO_LONG's forward blackhole handoff bounds
# the recording. Returning the three independently observable mutation
# classes also checks that the handoff does not replay from function entry.
def function_entry_effects(n):
    box = [0]
    d = {"k": 0}
    obj = []
    total = 0
    for i in range(n):
        box[0] += 1
        d["k"] = d["k"] + 1
        obj.append(i)
        total += i
    return total, box[0], d["k"], len(obj)


set_trace_limit(98)
print(function_entry_effects(N))


# Scope the small trace limit and threshold=1 to this fixture. Otherwise
# importlib/weakref shutdown callbacks become unrelated, target-dependent
# traces in its counters (rlib/jit.py set_user_param, "default").
if pypyjit is not None:
    pypyjit.set_param("default")
