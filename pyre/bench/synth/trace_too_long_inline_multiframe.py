# ABORT_TOO_LONG while the authoritative walk is inside an inlined Python
# callee must continue from that callee's own MIFrame.  The callee mutates
# three independently observable containers before the limit lands; replaying
# from the caller's CALL applies an iteration twice, while collapsing the
# callee onto the caller loses its locals/operand stack.
# The limit is chosen so the cut lands inside leaf after its three mutations,
# past the zero-effect call_trace prologue walk. The wasm guest has Signed =
# 8 bytes and isize/usize = 4, so `_int_to_int_cast` records `int_signext` /
# `int_and` that are the identity on 64-bit native; traces are a few ops
# longer on wasm and a tiny `trace_limit` can cut one frame earlier or later.
# At 136 that extra length sat on a knife-edge (native 5 multi-frame blackhole
# adoptions, wasm 4). Measured on dynasm, cranelift and wasm together, 109-126
# is an 18-wide plateau of identical counters (multi=3, single=68) including
# every ±2 neighbour, so 124 is robust to the wasm length difference. 141-147
# is a narrower 7-wide plateau.
import atexit

# Keep lazy shutdown imports outside this fixture's tiny trace limit. The
# subject's entry/leaf and raising_entry/module traces then determine the
# blackhole handoff counts, rather than importlib's initialization frames.
atexit._ncallbacks()

try:
    import pypyjit
except ImportError:
    pypyjit = None


if pypyjit is not None:
    pypyjit.set_param("trace_limit=124,threshold=1,function_threshold=1")


def leaf(box, mapping, values, x):
    box[0] += 1
    mapping["count"] = mapping["count"] + 1
    values.append(x)
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    i = h + 9
    j = i + 10
    return j + box[0] + mapping["count"]


def entry(box, mapping, values, x):
    return leaf(box, mapping, values, x)


box = [0]
mapping = {"count": 0}
values = []
total = 0
for n in range(200):
    total += entry(box, mapping, values, n)


# The same handoff must preserve the innermost frame and pending exception
# when the blackhole finishes by unwinding instead of returning a value.
def raising_leaf(values, x):
    values.append(x)
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    i = h + 9
    j = i + 10
    raise ValueError(j)


def raising_entry(values, x):
    return raising_leaf(values, x)


raised_values = []
raised_total = 0
for n in range(100):
    try:
        raising_entry(raised_values, n)
    except ValueError as exc:
        raised_total += exc.args[0]

# Keep the tiny trace limit inside this workload. Otherwise shutdown
# importlib/atexit helpers become extra segmentation tests whose frame
# boundaries depend on the target layout (interp_jit.set_param).  The
# checksums print after the reset for the same reason: app-level `sum` ->
# `_int_sum` under the tiny limit cuts its own traces, inside the list
# iterator helper or in the `_int_sum` frame depending on the target and on
# how its callers are entered, and those cuts were counted with the
# subject's handoffs.
if pypyjit is not None:
    pypyjit.set_param("default")

print(total, box[0], mapping["count"], len(values), sum(values))
print(raised_total, len(raised_values), sum(raised_values))
