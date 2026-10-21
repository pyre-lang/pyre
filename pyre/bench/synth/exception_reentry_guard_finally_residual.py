# pyre-check: jitstats-band=guard_failures=1
# `guard_failures` / `bridges_compiled` follow the minor-collection schedule
# (`counter.py` `decay_all_counters`, every 32 minors), and so does the split
# between compiled loops and `force_qmut` aborts: the module loop's global
# stores invalidate it, and whether a retrace starts on an iteration that
# stores is schedule-dependent. The oracle moves too: `abort: force
# quasi-immut` reads 17..29 across PYPY_GC_NURSERY 1MB..8MB. Under check.py's
# environment N is chosen so that dynasm and cranelift both read loops=110
# bridges=131 loops_aborted=25 guard_failures=27423 at the 4MB pin and at
# 6MB; N=64000 and N=68000 split the backends at the pin by one loop. The
# extra `RAISE_WARM` trips through `boom` push the last fail=2 hop past
# `trace_eagerness` (200 / `increment_trace_eagerness`).
# A frame whose residual call raises must still run its own `finally`, and must
# contribute exactly one traceback node.
#
# `run` is the `_contextvars_app.py` `Context.run` shape: a re-entry guard ahead
# of a `try`, two non-idempotent stores inside it, and a `CALL_FUNCTION_EX` on an
# arbitrary callable as the protected region's only statement.  The callee forces
# the caller's frame (`sys._getframe(1).f_locals`), so the call escapes the
# virtualizable mid-opcode and the walk ends on that escape.
#
# Three separate ways to get this wrong, each visible in the output:
#   - the escape falls back to a legacy replay of the already-executed prefix, so
#     the guard sees its own store and raises `RuntimeError` (`reentry`);
#   - the walk's terminal raise skips the frame's exception table, so `finally`
#     never restores the two stores (`leaked`);
#   - the frame leaves without recording its traceback node (`shapes`).
#
# The `except` calls `names()`, which bears a loop, so the callee inline is
# unsupported and the abort has to be adopted rather than retraced.
import sys

N = 60000
RAISE_WARM = 50
_cur = [None]


class Ctx:
    def __init__(self):
        self.entered = False

    def run(self, callable, *args, **kwargs):
        if self.entered:
            raise RuntimeError('cannot enter context')
        prev = _cur[0]
        try:
            self.entered = True
            _cur[0] = self
            return callable(*args, **kwargs)
        finally:
            _cur[0] = prev
            self.entered = False


def names(traceback):
    out = []
    while traceback is not None:
        out.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return tuple(out)


def boom(a, b, k=0):
    sys._getframe(1).f_locals
    raise ValueError(a + b + k)


def fine(a, b, k=0):
    sys._getframe(1).f_locals
    return a + b + k


ctx = Ctx()
leaked = 0
reentry = 0
total = 0
shapes = set()
for i in range(N + RAISE_WARM):
    try:
        if i >= N or i & 1:
            ctx.run(boom, 1, 2, k=3)
        else:
            total += ctx.run(fine, 1, 2, k=3)
    except ValueError as e:
        shapes.add(names(e.__traceback__))
    except RuntimeError:
        reentry += 1
    if ctx.entered:
        leaked += 1
        ctx.entered = False
    if _cur[0] is not None:
        leaked += 1
        _cur[0] = None

print("leaked", leaked, "reentry", reentry, "total", total, "shapes", sorted(shapes))
