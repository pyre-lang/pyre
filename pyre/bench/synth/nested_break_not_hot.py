# pyre-check: max-pypy-ratio=86
# A nested `for` loop whose inner `break` lands on the secondary edge of its
# guard — reached from `if not cond: break` (a POP_JUMP_IF_TRUE fall-through) or
# from a break-check that is the last statement in the loop body (a
# POP_JUMP_IF_FALSE that jumps forward to the break) — must reconstruct the
# operand stack correctly when the inner iterator is popped and control resumes
# at the outer FOR_ITER. Both the break-check-first and accumulate-first forms,
# plus a compound `and` guard, are exercised. Output is verified against
# CPython/PyPy.
N = 72000


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
