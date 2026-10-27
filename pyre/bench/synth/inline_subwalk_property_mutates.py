# pyre-check: max-pypy-ratio=94
# gh#495 guard: inlined property residual mutates before branch and caught miss.
# @property value-returning mutating + try/except-inside-callee raising branch
N = 240000
class C:
    def __init__(self): self.pos = 0
    @property
    def tick(self):
        self.pos = self.pos + 1
        return self.pos
def step(c, d, k):
    t = c.tick
    if k < 0: return 0
    try:
        return d[k]
    except KeyError:
        return -1
def run():
    d = {1:100,2:200,3:300}; acc=0; c=C(); i=0
    while i < N:
        k = i % 5
        v = step(c, d, k)
        if v == -1: acc -= 1
        else: acc += v
        i += 1
    return acc, c.pos
print(run())
