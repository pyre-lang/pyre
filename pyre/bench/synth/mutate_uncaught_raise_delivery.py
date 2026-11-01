# pyre-check: max-pypy-ratio=10
#
# Floor is 10 / 4 = 2.5x. Ceiling 17 derived floor 4.25x; PR macos dynasm
# run 40876302465 read exec 0.20s vs pypy 0.05s (ratio 3.8x) and missed
# that floor. Fleet (runs 40876302465 / 40867606800 / 40823694446 /
# 40795199813 / 40780530862): macos 2.9x-4.1x, ubuntu 4.8x-7.8x, windows
# 3.7x-9.0x (often `?` or `~`). Highest gated reading 7.8x × 1.15 = 8.97;
# 10 is that 15% headroom. Floor 2.5x still sits under macos 3.8x.
# `_performance_gate_passed` compare_buffer is 0.005 + C * 0.005 off Windows.
#
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
