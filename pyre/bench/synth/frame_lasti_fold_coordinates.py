# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# Self-checking guard for the coordinate an app-level `f_lasti` read reports
# for the frame that is running it.
#
# `pyframe.py fget_f_lasti` is loop-free and carries no hint, so
# `policy.py look_inside_graph` traces through it, `jtransform.py
# rewrite_op_jit_force_virtualizable` deletes the injected force, and
# `pyjitpl.py opimpl_getfield_vable_i` answers the field out of
# `virtualizable_boxes` -- a constant, because the bytecode dispatch stored one
# there.  The read neither forces the virtualizable nor needs a residual, and
# pyre folds it to the constant its own LOAD_ATTR coordinate names.
#
# A wrong constant is invisible in one iteration, so every site collects a SET
# across the run: the loop compiles part-way through, so an interpreted answer
# and a compiled one that disagree appear as a SECOND element rather than as
# one shifted value.
#
# `co_positions()` is the oracle for the coordinate itself.  Indexing it with
# `f_lasti // 2` is what a `dis` consumer does, so it pins BOTH halves of the
# adaptation the getset performs (`typedef.rs` returns `fget_f_lasti() * 2`
# over an instruction-unit field): a missing factor lands on the wrong row, and
# a pc short by one lands on the previous instruction.  Site C is where that
# second failure becomes a line difference -- its `.f_lasti` is the FIRST
# instruction of its own line, so the row before it belongs to the line above.
import sys

N = 20000

FIRST = sys._getframe().f_lineno


def main():
    code = sys._getframe().f_code
    site_a = set()
    site_b = set()
    site_c = set()
    total = 0
    for i in range(N):
        f = sys._getframe()
        site_a.add((f.f_lasti, f.f_lineno - FIRST))          # +11
        total += i
        g = sys._getframe()
        site_b.add((g.f_lasti, g.f_lineno - FIRST))          # +14
        h = sys._getframe()
        site_c.add(
            h
            .f_lasti                                         # +18
        )

    positions = list(code.co_positions())
    for label, seen, want_line in (
        ("A", site_a, 11),
        ("B", site_b, 14),
        ("C", site_c, 18),
    ):
        if len(seen) != 1:
            print(f"FAIL site {label} diverged across iterations: {sorted(seen)}")
            return 1
        entry = next(iter(seen))
        lasti = entry[0] if isinstance(entry, tuple) else entry
        if lasti < 0 or lasti % 2 != 0:
            print(f"FAIL site {label} f_lasti not an even byte offset: {lasti}")
            return 1
        row = positions[lasti // 2][0]
        if row is None or row - FIRST != want_line:
            print(f"FAIL site {label} f_lasti={lasti} names line {row} not +{want_line}")
            return 1
        if isinstance(entry, tuple) and entry[1] != want_line:
            print(f"FAIL site {label} f_lineno +{entry[1]} != +{want_line}")
            return 1
    a = next(iter(site_a))[0]
    b = next(iter(site_b))[0]
    c = next(iter(site_c))
    if not (a < b < c):
        print(f"FAIL f_lasti not increasing across sites: {a} {b} {c}")
        return 1
    if total != sum(range(N)):
        print(f"FAIL dropped iteration: total={total}")
        return 1
    print("PASS f_lasti fold coordinates")
    return 0


sys.exit(main())
