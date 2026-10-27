# pyre-check: max-pypy-ratio=21
# A nested loop over a sliced list whose length varies between outer
# iterations must preserve both iterators and the accumulator when a
# compiled exhaustion guard resumes in the interpreter. The exact
# aggregate makes corruption of either iterator or the accumulator
# visible in the output.
N = 1800000


def main():
    total = 0
    base = [1, 2]
    for k in range(N):
        for value in base[:(k % 2 + 1)]:
            total += value
    return total


print(main())
