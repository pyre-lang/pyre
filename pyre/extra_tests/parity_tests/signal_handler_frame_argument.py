# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: pypy3 accepts setitimer keyword first=; 3.15 is positional-only
# CPython-suite gap: `test_signal` declares `frame` in the signature of every
# handler it defines and then asserts nothing about it -- no test in the module
# reads `f_code`, `f_locals` or `f_back` off the argument -- so a runtime that
# hands every handler `None` passes the whole file.
#
# parity-tests reason: the second argument is the interpreter's own call stack
# handed to application code, which is the point a JIT owes a writeback of the
# fields it was keeping out of the frame. Handing over `None` instead is
# invisible to the suite and takes every `f_back` walk with it, which is how a
# signal-driven debugger or stack dumper finds out where it interrupted.
import os
import signal
import threading

seen = []


def handler(signum, frame):
    seen.append(frame)


def target():
    marker = 'target-local'
    os.kill(os.getpid(), signal.SIGUSR1)
    # Delivery happens at a bytecode boundary rather than inside `kill`, so
    # the handler has not run yet on the line above.
    for _ in range(1000):
        if seen:
            break
    return marker


signal.signal(signal.SIGUSR1, handler)
assert target() == 'target-local'
assert len(seen) == 1, seen

frame = seen[0]
assert frame is not None, 'the handler was handed no frame'
# The frame is the one that was interrupted, not the handler's own.
assert frame.f_code.co_name == 'target', frame.f_code.co_name
# Its locals are readable, which is what a frame the JIT was running owes
# before application code reads it.
assert frame.f_locals['marker'] == 'target-local', frame.f_locals
# The walk a debugger makes from there.
assert frame.f_back is not None, 'the interrupted frame has no caller'
assert frame.f_back.f_code.co_name == '<module>', frame.f_back.f_code.co_name
assert isinstance(frame.f_lineno, int), frame.f_lineno

# interp_signal.py `setitimer` `@unwrap_spec(which=int, first=float, interval=float)`
# and gateway arity `setitimer(which, first, interval=0)`.
class F:
    def __init__(self, *_):
        pass

    def __float__(self):
        return 0.0

old = signal.setitimer(signal.ITIMER_REAL, F(0))
assert isinstance(old, tuple) and len(old) == 2, old
assert all(isinstance(x, float) for x in old), old

try:
    signal.setitimer(signal.ITIMER_REAL, "x")
except TypeError:
    pass
else:
    raise AssertionError('setitimer string seconds')

try:
    signal.setitimer(signal.ITIMER_REAL, 0, 0, 0)
except TypeError:
    pass
else:
    raise AssertionError('setitimer 4 positionals')

# interp_signal.py `setitimer` unwrap `float` / `W_IntObject.float_w`:
# an int subclass converts by payload, never `__float__`.
class I(int):
    def __float__(self):
        raise RuntimeError('float called')


old = signal.setitimer(signal.ITIMER_REAL, I(0))
assert isinstance(old, tuple) and len(old) == 2, old
signal.setitimer(signal.ITIMER_REAL, 0)

try:
    signal.setitimer(which=signal.ITIMER_REAL, seconds=0)
except TypeError as e:
    print(type(e).__name__)
else:
    raise AssertionError('setitimer seconds= keyword')

try:
    signal.setitimer(signal.ITIMER_REAL, first=0)
except TypeError as e:
    print(type(e).__name__)
else:
    raise AssertionError('setitimer first= keyword')

# interp_signal.py `set_wakeup_fd` — `threadlocals.signals_enabled()`.
box = []


def wakeup_from_thread():
    try:
        signal.set_wakeup_fd(-1)
        box.append(None)
    except ValueError as exc:
        box.append(exc)
    except BaseException as exc:
        box.append(exc)


t = threading.Thread(target=wakeup_from_thread)
t.start()
t.join()
assert box and isinstance(box[0], ValueError), box

print('OK')
