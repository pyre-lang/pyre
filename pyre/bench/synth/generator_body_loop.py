# A loop that lives inside a generator body and never crosses the yield.
# The suspension is the only part of the frame the tracer has to refuse; the
# loop before it is an ordinary counted loop and compiles like one. This
# fixture exists because the whole generator frame used to be unreachable —
# resumption bypassed the eval override, so nothing in a generator body was
# ever offered to the tracer, yield in the loop or not.
N = 800
ROUNDS = 3600


def summer(m):
    # Hot: runs to completion on every resume, entirely before the yield.
    total = 0
    i = 0
    while i < m:
        total += i * 3
        i += 1
    yield total
    # A second suspension, so the generator is resumed rather than exhausted
    # at the first yield and the resume path is exercised on every round.
    yield total + 1


def run(rounds, m):
    acc = 0
    r = 0
    while r < rounds:
        g = summer(m)
        acc += next(g)
        acc += next(g)
        r += 1
    return acc


total = run(ROUNDS, N)
assert total == 6903363600, total
print(total)
