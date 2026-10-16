# pyre-check: max-pypy-ratio=45
# The ceiling bounds a reading that is not a measurement, so it is twice the
# worst native ratio seen rather than twice a typical one. pypy's execution here
# sits on the startup-subtraction floor and crosses it between runs: on one host
# with pyre's own time identical at 0.11s, back-to-back runs read 6.4x/8.4x and
# 11.9x/22.2x, the swing coming entirely from which side of the floor pypy
# landed on. The trip count cannot fix it -- pypy's time does not respond to it.
# A nested `for` loop whose inner `break` lands on the secondary edge of its
# guard — reached from `if not cond: break` (a POP_JUMP_IF_TRUE fall-through) or
# from a break-check that is the last statement in the loop body (a
# POP_JUMP_IF_FALSE that jumps forward to the break) — must reconstruct the
# operand stack correctly when the inner iterator is popped and control resumes
# at the outer FOR_ITER. Both the break-check-first and accumulate-first forms,
# plus a compound `and` guard, are exercised. Output is verified against
# CPython/PyPy.
N = 36000


def break_check_first(n):
    t = 0
    for i in range(n):
        for j in range(2, 4):
            if not i % 3 != 0:
                break
            t += j
    return t


def accumulate_first(n):
    t = 0
    for i in range(n):
        for j in range(2, 4):
            t += j
            if not i % 5 != 0:
                break
    return t


def compound_guard(n):
    t = 0
    for i in range(n):
        for j in range(2, 4):
            if not i % 3 != 0 and i % 2 == 0:
                break
            t += j
    return t


def plain_break(n):
    t = 0
    for i in range(n):
        for j in range(2, 4):
            if i % 3 >= 2:
                break
            t += j
    return t


print(break_check_first(N), accumulate_first(N), compound_guard(N), plain_break(N))
