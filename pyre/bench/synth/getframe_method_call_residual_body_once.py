# pyre-check: spec-folds=load_type_attr
# A method-form `LOAD_ATTR` + `CALL` whose callee forces the calling frame and
# then raises, caught in the loop body. The callee's own body is outside every
# journal, so a walk that cannot commit makes the legacy entry replay run it a
# second time per trace attempt: `Force.hits` climbs past the iteration count
# while `total` and `len(seen)` — both journaled — stay correct. That asymmetry
# is the whole oracle; a bench that only checked the loop result saw nothing.
#
# The shape matters, not the effect. `f.raiser()` lowers to `LOAD_ATTR` (method
# form) followed by `CALL`, and the CALL's lowering re-enters the LOAD_ATTR
# source segment, so the walk's py_pc floor reports the earlier opcode again
# after the later one. That backward step is no CFG successor, so it arms the
# out-of-order region and every following boundary reseeds the operand mirror
# from the virtualizable shadow — which mid-expression holds the NULLs the
# in-flight opcode already popped, and so cannot return the callable. The
# mirror came back one slot short with the callable in the `self_or_null` slot,
# the escape flush declined on the hole, and the portal replayed the frame from
# its entry.
#
# N is sized to reach the tracing thresholds on its own rather than for
# throughput: at 200 the loop never gets hot and the bench passes with the
# defect in place. Measured against a binary without the fix, N=2000 reports
# 2001 hits and N=20000 reports 20005 — the overshoot is the trace-attempt
# count, so only the fixed value (exactly N) is stable enough to pin. No
# `max-pypy-ratio` for the same reason `exception_reused_object_tb_not_doubled`
# has none — this is a shape oracle, not a workload.
#
# `raiser`'s `_getframe(1).f_lasti` now folds on the portal red box, so the
# method body no longer forces a virtualizable escape.  The oracle is still
# that the body runs once per iteration (`Force.hits == N`).
#
# Expected output: (160000, 160000, 160000)
import sys

N = 160000


class Force:
    hits = 0

    def raiser(self):
        _ = sys._getframe(1).f_lasti
        Force.hits += 1
        raise ValueError(1)


f = Force()


def main():
    seen = []
    total = 0
    for i in list(range(N)):
        try:
            total += f.raiser()
        except ValueError:
            total += 1
        seen.append(i)
    return total, len(seen), Force.hits


print(main())
