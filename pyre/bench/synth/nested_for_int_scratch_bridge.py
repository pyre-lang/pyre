# pyre-check: max-pypy-ratio=8.4
# Nested FOR_ITER with a branch-local Int condition and a kept stack at the
# bridge resume point. The resolved-offset bridge decodes a non-empty Int bank,
# so the Int register bank is reconstructed concretely at bridge resume and the
# nested-loop branch guard folds instead of aborting GotoIfNotValueNotConcrete.
def run(n):
    total = 0
    marker = None
    other = object()
    for i in range(n):
        limit = (i & 3) + 1
        for j in range(limit):
            scratch = j + i
            obj = marker if (j & 1) else other
            if obj is None:
                total += scratch
            else:
                total += j
        total += limit
    return total


print(run(3000000))
