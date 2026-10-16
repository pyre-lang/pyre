# A loop nested inside a yield-bearing loop. The inner loop's body holds no
# suspension — it runs to completion between two yields — so it is an ordinary
# counted loop and compiles like one. The enclosing loop is a different matter
# and still declines, correctly, because a trace cannot cross a yield.
#
# The shape exists because the loop-body region for the inner loop used to be
# closed by the ENCLOSING loop's back edge, which stretched the region over the
# yield and declined a body that never goes near one. A sibling loop was
# unaffected, so nesting is the variable this fixture pins.
ROUNDS = 1800
INNER = 400


def summer(m, rounds):
    r = 0
    while r < rounds:
        total = 0
        i = 0
        while i < m:
            total += i * 3
            i += 1
        yield total
        r += 1


def run(rounds, m):
    acc = 0
    for v in summer(m, rounds):
        acc += v
    return acc


total = run(ROUNDS, INNER)
assert total == 430920000, total
print(total)
