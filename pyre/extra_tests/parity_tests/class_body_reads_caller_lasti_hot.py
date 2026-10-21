# CPython-suite gap: the frame tests never read a caller's `f_lasti` from a
# class body that a compiled loop enters.
# parity-tests reason: a class statement in a traced loop stops the walk on a
# quasi-immutable force inside the `__build_class__` CALL, and the blackhole
# resumes past that opcode's `last_instr` store; the caller frame must still
# report the CALL as its current instruction.
import sys


def main(n):
    seen = {}
    for i in range(n):
        class C:
            lasti = sys._getframe(1).f_lasti
        seen[C.lasti] = seen.get(C.lasti, 0) + 1
    # A loop whose result is not constant keeps the fresh class reachable.
    return sorted(seen.items())


got = main(3000)
if len(got) != 1 or got[0][1] != 3000:
    print("caller f_lasti changed across iterations:", got)
else:
    print("OK")
