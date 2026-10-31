# pyre-check: skip-backends=wasm
# wasm32 has no OS-thread launcher. On native backends, threadlocals.py
# constructs ExecutionContext(space), sharing space.builtin with its parent.
# A worker must not replace the shared builtin functions' defining module.
import builtins
import _thread

assert len.__self__ is builtins
completed = _thread.allocate_lock()
completed.acquire()
seen = []

def worker():
    try:
        seen.append(len.__self__ is builtins)
    finally:
        completed.release()

_thread.start_new_thread(worker, ())
completed.acquire()
assert seen == [True]
assert len.__self__ is builtins
print('PASS')
