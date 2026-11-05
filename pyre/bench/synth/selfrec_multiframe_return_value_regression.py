# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:rec,loop:<module>
# `step` inlines into the module loop once `rec`'s self-rec CALL residualizes
# instead of aborting the enclosing walk. Require both that caller loop and
# `rec`'s FINISH-only procedure. In upstream compile.py,
# ResumeFromInterpDescr.compile_and_attach publishes the latter as an
# "entry bridge" after pyjitpl.py MetaInterp.compile_done_with_this_frame;
# the census follows that classification rather than calling it a root loop.
# A self-recursive callee reached through a multi-frame adopt returned `None`
# from a function whose every path returns an `int` (gh#1444).
#
#     TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
#
# `rec` below returns `1`, `0`, or `rec(...) + 1`.  Never `None`.  The caller
# reported the JIT's answer as a Python-level type error, which is what made
# the shape easy to dismiss as a mistake in the script rather than a wrong
# answer from the compiler.
#
# ## What it was
#
# The defect sits in the multi-frame blackhole-adopt path rather than in any
# writer of `last_instr`.  The issue title proposes the opposite --
# that a resume restores `last_instr` without its partner `valuestackdepth` --
# and that was implemented and measured: pairing the inline-callee coordinate
# publish with the forward analysis's depth at `callee_py_pc + 1` changed
# nothing here.  `PYRE_FBW_DEBUG_ABORT=1` named one adoption per run:
#
#     [walk-abort] err=LoopBearingCalleeInlineUnsupported { pc: 258,
#                  blackhole_required: true } claimed=true
#     [mf-adopt] frame=0 name="<module>" / 1 "step" / 2,3,4 "rec"
#     [fbw-blackhole] adopted multi-frame terminal depth=5
#
# `MAJIT_LOG=1` showed the terminal is a `DoneWithThisFrameRef`, so neither
# `adopt_blackhole_crn` nor the `BailToInterpreter` arm was the path taken.
#
# ## Why the shape is exactly this
#
# The trigger is narrow, and every widening below was measured to stop
# reproducing, so none of it is decoration:
#
#     rec + id(x)      FAIL      rec + hash(x)   pass
#     rec + id(0)      FAIL      rec + abs(x)    pass
#     rec + repr(x)    FAIL      rec + len([])   pass
#     id(x) in the recursive arm rather than the base case: pass
#     no recursion, same id(x): pass
#
# So the callee has to be SELF-RECURSIVE and its base case has to reach a
# may-force builtin.  BOTH loops are load-bearing -- neither alone reproduces
# at any size -- and so is the plain shape: adding `is None` checks changes
# what compiles, and the instrumented variant passed on binaries the plain one
# failed.  That is why the assertion below sits after both loops and touches
# nothing inside them.
#
# ## What makes this a guard rather than a hope
#
# It fails on the buggy binary.  Two saved pre-fix `pyre-dynasm` binaries both
# die here with the `TypeError` above, while the fixed tree prints the same
# `592304421450` CPython does, on both backends.
EXPECTED = 592304421450


def ck(seq):
    h = 7
    for v in seq:
        h = (h * 1000003 + v) & 0xFFFFFFFFFF
    return h


def rec(x):
    if x <= 0:
        return 1 if id(x) else 0
    return rec(x - 1) + 1


class W:
    def __init__(self):
        self.seen = []

    def step(self, x):
        self.seen.append(x)
        return x * 1000003 + rec(x % 4)


hh = 7
for trial in range(50):
    n = trial % 50
    w = W()
    res = [w.step(i) for i in range(n)]
    hh = (hh * 1000003 + ck(res) + ck(w.seen)) & 0xFFFFFFFFFF
for trial in range(60):
    n = trial % 20
    w = W()
    res = []
    for i in range(n):
        res.append(w.step(i))
    hh = (hh * 1000003 + ck(res) + ck(w.seen)) & 0xFFFFFFFFFF

if hh == EXPECTED:
    print("PASS selfrec multiframe return value")
else:
    print(f"FAIL checksum {hh} against {EXPECTED}")
    raise SystemExit(1)
