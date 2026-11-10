# CPython-suite gap: the threading suite never waits on a JIT-compiled
# busy loop to hand the GIL to a sleeping worker.
# parity-tests reason: PyPy's compiled jump_absolute decrements the action
# ticker; when it crosses zero, GILReleaseAction lets the sleeping worker run.
import _thread
import time

flag = [False]


def worker():
    time.sleep(0.05)
    flag[0] = True


_thread.start_new_thread(worker, ())
n = 0
while not flag[0]:
    n += 1
assert n > 0
print("OK")
