# pyre-check: max-pypy-ratio=8
# Readings 2.3x-5.1x (macos dynasm 2.5x under floor 3.25 against 13, run 40570473141). Floor 2x.
# gh#495 guard: reflected-add residual mutation feeds the inlined branch result.
# radd result feeds the branch condition; branch varies each iter -> sub-walk churn
N = 7200000
class Acc:
    def __init__(self): self.pos = 0
    def __radd__(self, o):
        self.pos = self.pos + 1
        return self.pos + o
def step(c, k):
    t = k + c                 # c.__radd__ residual, returns int depending on pos
    if t & 1: return 1        # branch depends on mutating residual result
    if k < 0: return 0
    return -1
def run():
    acc=0; c=Acc(); i=0
    while i < N:
        v = step(c, i % 7)
        acc += v
        i += 1
    return acc, c.pos
print(run())
