# pyre-check: max-pypy-ratio=2.2
# The recorded cranelift 2.0x maximum plus 10% headroom gives 2.2x.
# Windows median reruns measure dynasm 0.6x-0.7x and cranelift 1.1x;
# the old 3.5x ceiling's derived 0.875x floor rejected the faster dynasm run.
# pyre-check: skip-cpython
# cpython 2.30s vs pyre 0.42s (5.5x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Bare `raise` inside a nested handler whose re-raise must reach the OUTER
# handler, exercised under a bridge resume that lands inside the inner handler.
#
# Unlike exception_bare_reraise_restore.py (which only restores the current-
# exception slot), this shape adds a third inner branch (i % 3) so the catch
# landing of the inner handler shares a resume color with the virtualizable
# frame `f_code` scalar.  At a bridge resuming into the inner handler at the
# bare `raise`, the walker reconstructs the PUSH_EXC_INFO operand-stack slot
# from the per-PC resume map; if it aliases the const-folded `f_code` the
# published current exception becomes a code object and the later blackhole
# re-raise reports `TypeError: exceptions must derive from BaseException`
# instead of routing ValueError to the outer handler.
#
#   i % 3 == 0 -> raise ValueError -> inner except (inner++)
#                   then i % 2 == 0 -> bare raise -> outer except (outer++)
#   i % 3 != 0 -> clean path (clean++)
#
# With N=200000: inner=66667, outer=33334, clean=133333; inner+clean=200000.
# Deterministic.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 130892039


def main():
    inner = 0
    outer = 0
    clean = 0
    i = 0
    while i < N:
        try:
            try:
                if i % 3 == 0:
                    raise ValueError("v")
                else:
                    clean += 1
            except ValueError:
                inner += 1
                if i % 2 == 0:
                    raise
        except ValueError:
            outer += 1
        i += 1
    print(inner, outer, clean)
    print(inner + clean)


main()
