# pyre-check: jitstats-band=guard_failures=1
# Pin counter decay, not the collector schedule. counter.py JitCounter.set_decay
# with zero leaves hot counters unchanged after minor collections, so different
# frame layouts cannot swap a compiled trace with a force_qmut abort. Collections
# still move the frame/traceback objects this fixture exercises. RAISE_WARM keeps
# the final fail=2 hop beyond trace_eagerness.
# A frame whose residual call raises must still run its own `finally`, and must
# contribute exactly one traceback node.
#
# `run` is the `_contextvars_app.py` `Context.run` shape: a re-entry guard ahead
# of a `try`, two non-idempotent stores inside it, and a `CALL_FUNCTION_EX` on an
# arbitrary callable as the protected region's only statement.  The callee reads
# the caller's frame locals (`sys._getframe(1).f_locals`) from inside the
# protected call, so the caller's state must survive that introspection.
#
# Three separate ways to get this wrong, each visible in the output:
#   - a guard exit replays the already-executed prefix, so
#     the guard sees its own store and raises `RuntimeError` (`reentry`);
#   - a resumed raise skips the frame's exception table, so `finally`
#     never restores the two stores (`leaked`);
#   - the frame leaves without recording its traceback node (`shapes`).
#
# The `except` calls `names()`, whose loop also exercises traceback traversal
# after the protected call raises.
import sys

try:
    import pypyjit

    pypyjit.set_param("decay=0")
except ImportError:
    pass

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
