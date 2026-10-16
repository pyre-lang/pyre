# Regression oracle: nested try/except where each handler reads
# `sys.exc_info()`.  After an inner handler unwinds, POP_EXCEPT must restore the
# slot to the prev its matching PUSH_EXC_INFO saved (the outer ValueError), and
# after the outer handler to None.  Expected per-iteration signature
# 2*1 + 1*10 + 0*100 = 12 → 360000.
#
# Was `_pending/`: the JIT answered 3320000 (trait) / 3360000 (FBW walker)
# because the nested handlers' POP_EXCEPT restores were not lowered to the EC
# `sys_exc_value` slot — the in-handler `sys.exc_info()` may-force ended the
# authoritative walk, so the slot kept the inner exception after the handler
# exited.  Both tracers now answer 360000, matching the interpreter and CPython.
import sys

N = 300000


def classify(t):
    if t is ValueError:
        return 1
    if t is KeyError:
        return 2
    if t is None:
        return 0
    return 9


def run(n):
    acc = 0
    i = 0
    while i < n:
        try:
            raise ValueError("outer")
        except ValueError:
            try:
                raise KeyError("inner")
            except KeyError:
                acc += classify(sys.exc_info()[0]) * 1
            acc += classify(sys.exc_info()[0]) * 10
        acc += classify(sys.exc_info()[0]) * 100
        i += 1
    return acc


def main():
    print(run(N))


main()
