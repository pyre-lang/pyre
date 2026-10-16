# pyre-check: max-pypy-ratio=34
# pyre-check: ungated-jitstats=bridges_compiled,guard_failures
# pypy JITs these tiny list loops to a near-zero denominator, so the ratio is
# dominated by measurement noise on the CI runner (measured 35-43x); the gate
# is set above that band rather than tightened onto the collapsed baseline.
# A hot call whose positional argument is a literal (`build(1000)`) into a
# callee that grows a list over `range(n)`. The list's realloc-boundary append
# declines the FBW fold, so a guard resumes the caller frame at the CALL site.
# The literal argument is loaded into the walker's unboxed int bank, so the
# Ref-only operand mirror left its stack slot a NONE hole; `stack_sync` then
# omitted it and the single-frame bridge rebuilt the argument slot NULL. The
# re-executed CALL passed NULL, so the callee's parameter `n` reconstructed
# unbound -> `UnboundLocalError`. The comprehension and explicit-append forms
# and int / object element strategies each exercise the same caller-frame
# resume; every form must reach `n` (never a dropped element or an unbound
# local).
#
# `bridges_compiled` and `guard_failures` are ungated here because they are not
# a function of the tree on this fixture. The same binary on the same source
# with the same argv, the same child environment and PYTHONHASHSEED all held
# fixed reads six bridges / 1204 or seven / 1404, run to run -- 1 of 6 runs,
# then 2 of 6, then 0 of 8 an hour later. CI has seen it too:
# `dynasm/synth/const_arg_call_resume` came up `jit-stats unstable (not gated)`
# on the windows runner. A machine settles on one value and repeats it many
# times over, which is what the stability re-runs cannot see through -- they
# ask whether a reading reproduces itself, and this one does, until it does
# not. Whichever way it is settled locally, it will stay that way long enough
# to look like a property of the tree.
#
# The two readings are one bridge generation apart. Under MAJIT_GUARDLOG both
# show the same three guard sites and differ only in their failure totals:
# 402/402/400 sums to 1204 and floors to 2+2+2 = six bridges, 602/402/400 to
# 1404 and 3+2+2 = seven. That arithmetic relates the two values; what decides
# which one a run lands on is upstream of it.
#
# What decides it is whether the short preamble exports the list's
# array-bearing boxes at all. `ArraylenGc` is the op that carries an array
# descr, so exporting it builds an `ArrayPtrInfo` whose `make_guards` demands
# a GC layout tid; not exporting it means nothing ever asks. When it is
# exported the resolution always fails, because the descr `make_array_descr`
# mints for these carries `cache_key = 0` -- documented at its producer as
# "no identity carrier, no cache slot" -- and `resolve_gc_tid` performs a real
# lookup on that sentinel anyway. The failure signals InvalidLoop, the unroll
# is abandoned, and the unroll-free retry compiles a simple loop instead --
# which is the cheaper mode, by one bridge and 200 failures. So the low
# reading is the rescued one.
#
# MAJIT_LOG=1 places this fixture in that family directly: it aborts three
# times with `short preamble GC layout tid is unresolved`, on three distinct
# trace keys, one per helper, matching `unroll_free_retry_rescued=3`. Reading
# the abort out beats waiting for the flip, because a host settles on one mode
# and repeats it -- 32 runs here across all three backends drew 6/1204 every
# time. But note what that line does and does not say: it names the path taken,
# not why the path was entered. Three separate analyses of these fixtures were
# wrecked by reading a mechanism out of it; the split is upstream, in short-box
# export. Three other fixtures here are bimodal the same way, and issue #1297
# tracks it.
#
# Two things that do not settle it, both measured, so neither is worth trying
# again:
#
# * The trip count. 220, 240, 260, 280, 300, 320, 360 and 400 read the same
#   counters, so there is no value that lands away from the boundary; only at
#   200 does a guard stop reaching the threshold at all, and that alternates
#   too (1199 / 1397). Halving the file is what settled `str_fstring`, whose
#   counters moved with a major-collection crossing -- `back_edge_polls` reads
#   0 in every run here at every trip count, which is what says this is not
#   that class.
# * Re-recording. It pins whichever side the recording machine was on, and
#   everything moves it: adding these comment lines alone -- text the parser
#   drops, changing nothing the JIT can see -- moved a wasm reading from 1404
#   to 1204 for eight runs in a row.
#
# Everything else the [jit-stats] line carries stays gated, including
# `loops_compiled`, `loops_aborted` and the descr-universe invariants, and the
# printed sum is still compared against cpython and pypy -- which is what
# actually catches the bug described above.


def comp(n):
    return [i for i in range(n)]


def obj_comp(n):
    return [(i, i) for i in range(n)]


def build(n):
    r = []
    for i in range(n):
        r.append(i)
    return r


total = 0
k = 0
while k < 1600:
    total += len(comp(1000))
    total += len(obj_comp(1000))
    total += len(build(1000))
    k += 1
print(total)
