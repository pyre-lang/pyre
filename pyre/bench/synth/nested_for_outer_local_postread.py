# pyre-check: max-pypy-ratio=9
# Readings 2.1x-5.7x (macOS dynasm 2.1x-3.1x, Linux cranelift 4.2x-4.9x,
# Windows up to 8.7x); the floor, a fourth of the ceiling, sits under the fastest.
# pyre-check: spec-folds=for_iter_next
# pyre-check: trace-shape=three_levels:absent=CallMayForceR
# The iteration fold, 487 firings across the corpus and undeclared. 22 here.
def variable_inner(n):
    total = 0
    for i in range(n):
        outer = i % 5
        for j in range(outer):
            pass
        total += outer
    return total


def constant_inner(n):
    total = 0
    for i in range(n):
        outer = i % 5
        for j in range(3):
            pass
        total += outer
    return total


def single_loop(n):
    total = 0
    for i in range(n):
        local = i % 5
        total += local
    return total


def three_levels(n):
    total = 0
    for i in range(n):
        outer = i % 7
        for j in range(outer):
            middle = j % 3
            for k in range(middle):
                pass
        total += outer
    return total


print(variable_inner(1333334))
print(constant_inner(1333334))
print(single_loop(1333334))
print(three_levels(800000))
