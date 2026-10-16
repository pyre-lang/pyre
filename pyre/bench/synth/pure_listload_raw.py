# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# #171/#11 Approach C read-after-write miscompile guard: LISTS stay NON-pure.
#
# If a list element load were ever folded to GetarrayitemGcPure{R,I,F}, OptPure
# would CSE the `b = lst[0]` read against the earlier `a = lst[0]` read across
# the intervening `lst[0] = a + 1` write (mutable list body), so `b - a` would
# collapse to 0 and the sum would print 0. Lists are mutable, so their element
# loads MUST remain non-pure (heap-cache invalidated on SetarrayitemGc) and this
# MUST print 17000.


def main():
    lst = [0]
    s = 0
    for _ in range(17000):
        a = lst[0]
        lst[0] = a + 1
        b = lst[0]
        s += b - a
    print(s)


main()
