# pyre-check: gate=1
"""FOR_ITER over zip of short tuples at the default JIT threshold.

Unequal lengths stop at the shortest iterator; equal lengths yield every
pair. strict=True raises ValueError when the lengths differ. Until a
guard inside a descended helper can resume in the helper frame, zip next
stays a residual call so the first iterator's cursor is not stored twice.
"""

N = 5000


def zip_sum(a, b, strict=False):
    s = 0
    for _ in range(N):
        for x, y in zip(a, b, strict=strict):
            s += x + y
    return s


first_longer = zip_sum((1, 2, 3), (10, 20))
equal2 = zip_sum((1, 2), (10, 20))
second_longer = zip_sum((1, 2), (10, 20, 30))
equal2_strict = zip_sum((1, 2), (10, 20), strict=True)
print("first_longer", first_longer)
print("equal2", equal2)
print("second_longer", second_longer)
print("equal2_strict", equal2_strict)
assert first_longer == 165000, first_longer
assert equal2 == 165000, equal2
assert second_longer == 165000, second_longer
assert equal2_strict == 165000, equal2_strict

for label, args in (
    ("first_longer_strict", ((1, 2, 3), (10, 20))),
    ("second_longer_strict", ((1, 2), (10, 20, 30))),
):
    try:
        zip_sum(*args, strict=True)
    except ValueError as e:
        print(label, type(e).__name__)
    else:
        raise AssertionError("%s: expected ValueError" % (label,))
