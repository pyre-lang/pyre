# ABORT_TOO_LONG while the authoritative walk is inside an inlined Python
# callee must continue from that callee's own MIFrame.  The callee mutates
# three independently observable containers before the limit lands; replaying
# from the caller's CALL applies an iteration twice, while collapsing the
# callee onto the caller loses its locals/operand stack.
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
    pypyjit.set_param("trace_limit=70,threshold=1,function_threshold=1")


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

print(total, box[0], mapping["count"], len(values), sum(values))


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

print(raised_total, len(raised_values), sum(raised_values))
