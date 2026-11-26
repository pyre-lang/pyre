# A user `__next__` inlined from a generator body calls a constructor whose
# helper descent suspends the sub-walk driver; the item must resume in the
# dunder frame, not in the caller's helper.


class Box:
    def __init__(self, s):
        q, r = divmod(s, 7)
        self.q = q
        self.r = r
        self.s = s

class Walker:
    def __init__(self, items):
        self.items = items
        self.i = 0
    def __iter__(self):
        self.mark = Box(len(self.items))
        return self
    def __next__(self):
        i = self.i
        if i >= len(self.items):
            raise StopIteration
        self.i = i + 1
        return Box(self.items[i])

def gen(items):
    for b in Walker(items):
        yield b.q * 1000 + b.r + b.s

def main():
    items = [(i * i) % 977 for i in range(400)]
    total = 0
    for rep in range(10):
        for v in gen(items):
            total += v
    print(total)

main()
