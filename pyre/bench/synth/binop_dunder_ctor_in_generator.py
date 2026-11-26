# A user `__sub__` inlined from a generator body calls a constructor whose
# helper descent suspends the sub-walk driver; the result must resume in
# the dunder frame, not in the caller's helper.


class Span:
    def __init__(self, s):
        q, r = divmod(s, 7)
        self.q = q
        self.r = r
        self.s = s

    def __sub__(self, other):
        if isinstance(other, Span):
            return Span(self.s - other.s)
        return NotImplemented


def shifts(spans):
    prev = spans[0]
    for t in spans[1:]:
        shift = t - prev
        yield shift
        prev = t


def main():
    spans = [Span((i * i) % 977) for i in range(400)]
    total = 0
    for rep in range(10):
        for s in shifts(spans):
            total += s.q * 1000 + s.r + s.s
    print(total)


main()
