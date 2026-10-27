# pyre-check: max-pypy-ratio=17
# gh#495 guard: mutating call that raises uncaught must deliver the exception once.
N = 25200000
class C:
    def __init__(self): self.pos = 0
    def tick(self):
        self.pos = self.pos + 1
        if self.pos == N: raise ValueError(self.pos)
def run():
    c = C(); i = 0
    try:
        while i < N + 10:
            c.tick(); i += 1
    except ValueError as e:
        return c.pos, i, str(e)
print(run())
