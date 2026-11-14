# pyre-check: max-pypy-ratio=4
# The measured reading stays under 4, so the ceiling is 4.
# At the 100 it was recorded with, the loop never reached the JIT --
# `loops_compiled=0` on every backend -- and pypy's side sat on the
# execution floor, so the gate compared startup. At 1600 the loop compiles
# and pypy's own execution is a measurement.
#
# wasm keeps its own `surrogate_dir.wasm.jitstats`. The guest's posix
# module has a different surface (it builds `stat_result` first, and
# `dir(posix)` has 72 names, not 256), so `sorted()` in
# `structseqtype.__new__` during startup feeds the list_extend jitdriver
# (`_do_extend_from_iterable`) a different sequence. Its trace starts on
# the last item and reaches the portal return, which compiles a FINISH
# trace (`compile_done_with_this_frame`): one more loop, and 200 more
# guard failures before that green is retraced.

import sys

S1 = '\udc81'            # lone surrogate name
S2 = '\udc84'            # another lone surrogate name


class C:
    pass


# The walks are repeated so the measured body is larger than the process
# startup floor; only the last round's answers are printed, so the output is
# identical to a single round.
REPEAT = 1600


def main():
    for _ in range(REPEAT):
        # Module global named by a lone surrogate appears in dir(module).
        setattr(sys, S1, 1)
        dm = dir(sys)
        mod_s1 = S1 in dm
        mod_argv = 'argv' in dm

        # Type attribute named by a surrogate appears in dir(type).
        setattr(C, S2, 3)
        dt = dir(C)
        type_s2 = S2 in dt

        # Instance attribute named by a surrogate appears in dir(instance),
        # alongside the surrogate name inherited from the type.
        c = C()
        setattr(c, S1, 4)
        dc = dir(c)
        inst_s1 = S1 in dc
        inst_s2 = S2 in dc

        # dir() output stays sorted with surrogate names present.
        type_sorted = dt == sorted(dt)

    print('mod_s1', mod_s1)
    print('mod_argv', mod_argv)
    print('type_s2', type_s2)
    print('inst_s1', inst_s1)
    print('inst_s2', inst_s2)
    print('type_sorted', type_sorted)


main()
