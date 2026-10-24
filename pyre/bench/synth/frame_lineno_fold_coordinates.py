# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# pyre-check: trace-shape=main:present=Jump
# Self-checking guard for the line an app-level `f_lineno` read reports for the
# frame that is running it.
#
# `pyframe.py fget_f_lineno` decodes `last_instr` against the line table, and
# `last_instr` is a virtualizable field.  A generic reader residualizes
# `space.getattr` as one CALL_MAY_FORCE, and that force is the only reason the
# frame's own copy of the field is current enough to decode.  The fold hands
# the same getter body the coordinate the walk already holds instead, so the
# decode runs with no force -- which means a wrong coordinate surfaces here as
# a wrong LINE rather than as a crash.
#
# One iteration cannot see that.  The loop compiles part-way through, so an
# interpreted answer and a compiled one that disagree appear as a SECOND
# element of a set rather than as one shifted value; every site collects across
# the whole run for that reason.
#
# Sites, and what each one pins:
#   A  the receiver is a fresh local from this statement -- the common shape.
#   B  the receiver was hoisted into a local BEFORE the loop, so the read is
#      not on the frame's own box and the fold owes a `ptr_eq` guard.  Its
#      answer is still the line being EXECUTED, not the line it was hoisted on,
#      which is what separates the frame's current coordinate from the box the
#      receiver arrived in.
#   C  a read spread across physical lines, so the attribute lands on a
#      different line from the statement that starts it.
#   D  a callee reading its OWN frame.  The coordinate source differs by
#      construction there: the portal's virtualizable describes the portal, so
#      a callee read that took it would report the caller's CALL boundary.
#
# Not covered here: `f_lineno` with a trace function installed.  That branch of
# the getter is the same leaf body, but arming local tracing on a compiled
# frame is a separate open defect, so a fixture that armed it would be pinning
# that defect rather than this fold.
import sys

N = 20000

FIRST = sys._getframe().f_lineno


def read_own_line():
    return sys._getframe().f_lineno - FIRST                  # +4


def main():
    hoisted = sys._getframe()
    site_a = set()
    site_b = set()
    site_c = set()
    site_d = set()
    total = 0
    for i in range(N):
        site_a.add(sys._getframe().f_lineno - FIRST)         # +15
        total += i
        site_b.add(hoisted.f_lineno - FIRST)                 # +17
        site_c.add(
            hoisted
            .f_lineno                                        # +20
            - FIRST
        )
        site_d.add(read_own_line())

    for label, seen, want in (
        ("A", site_a, 15),
        ("B", site_b, 17),
        ("C", site_c, 20),
        ("D", site_d, 4),
    ):
        if len(seen) != 1:
            print(f"FAIL site {label} diverged across iterations: {sorted(seen)}")
            return 1
        got = next(iter(seen))
        if got != want:
            print(f"FAIL site {label} f_lineno +{got} != +{want}")
            return 1
    if total != sum(range(N)):
        print(f"FAIL dropped iteration: total={total}")
        return 1
    print("PASS f_lineno fold coordinates")
    return 0


sys.exit(main())
