# pyre-check: max-pypy-ratio=35
# gh#495 guard: mutating residual raises and is caught inside the inlined callee.
# V2: mutate-then-RAISE, caught inside callee. Tests fixed path under exc churn.
N = 240000
class C:
    def __init__(self):
        self.pos = 0
    def tick(self):
        self.pos = self.pos + 1
        raise ValueError
def step(c, d, k):
    try:
        c.tick()             # void residual mutating then raising
    except ValueError:
        pass
    if k < 0:
        return 0
    if k in d:
        return d[k]
    return -1
def run():
    d = {1:100,2:200,3:300}; acc=0; c=C(); i=0
    while i < N:
        k = i % 5
        v = step(c, d, k)
        acc += (v if v != -1 else 0)
        i += 1
    return acc, c.pos
print(run())
