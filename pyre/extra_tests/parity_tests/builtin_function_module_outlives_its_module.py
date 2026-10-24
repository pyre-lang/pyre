# pyre-check: pypy-diverges: ObjSpace.finish does not run a __main__ global's __del__, so the child prints no del_* lines; PyPy also runs no __del__ for an object held in a module-level list at shutdown
# CPython-suite gap: `test_pickle` reaches this only by accident.  Its
# `load_tests` runs `doctest.DocTestSuite(pickle)`, which hashes
# `struct.pack.__module__` long after `_struct` may have been dropped from
# `sys.modules`; whether that crashes depends on what reused the freed memory,
# so the suite goes red on one runner layout and stays green on another.
#
# parity-tests reason: a builtin function stays reachable after the module that
# defined it is collected, and it owns its `__module__` string and its module
# object.  Those fields must keep their referents alive for as long as the
# function is reachable from any holder, not only while the defining module's
# dict is.  A plain container holding the function is enough to observe it.
import gc
import sys

import _struct

kept = [_struct.pack, _struct.unpack, _struct.calcsize]
for name in ('struct', '_struct'):
    sys.modules.pop(name, None)
del _struct
for _ in range(5):
    gc.collect()

# Reuse whatever the collections freed.
junk = ['%s.%s.%d' % ('__main__', 'CUnpicklerTests', i) for i in range(200000)]
del junk

for f in kept:
    assert f.__module__ == '_struct', (f.__name__, f.__module__)
    assert f.__self__.__name__ == '_struct', (f.__name__, f.__self__)
assert len({f.__module__: None for f in kept}) == 1

# finalize_modules empties sys.modules before any module dict (including
# `__main__`) is cleared, so a `__main__` global's `__del__` sees those names
# gone while `os.name` still resolves.
import os
import subprocess

SRC = r'''
import atexit
import os
import sys

print("startup_os", "os" in sys.modules)


def late():
    print("atexit_os", "os" in sys.modules)
    alive = 0
    for k, v in list(sys.modules.items()):
        if v is not None:
            alive += 1
    print("atexit_n", alive)


class Watch:
    def __del__(self):
        print("del_os", "os" in sys.modules)
        print("del_sys", "sys" in sys.modules)
        print("del_builtins", "builtins" in sys.modules)
        try:
            print("os_name", os.name)
        except Exception as e:
            print("os_use", type(e).__name__)


w = Watch()
atexit.register(late)
'''

done = subprocess.run([sys.executable, "-c", SRC], capture_output=True, text=True)
assert done.returncode == 0, done.stderr
for line in done.stdout.splitlines():
    if line.startswith(("del_os ", "del_sys ", "del_builtins ", "os_name ")):
        print(line)
assert "del_os False" in done.stdout
assert "del_sys False" in done.stdout
assert "del_builtins False" in done.stdout
assert f"os_name {os.name}" in done.stdout

# A finalizable held only in a module-level list still runs at shutdown, and
# a name bound after the holder is still visible from its `__del__`.
HOLDER_SRC = r'''
class F:
    def __del__(self):
        try:
            print("F_later %r" % (later,))
        except Exception as e:
            print("F_later_err %s %r" % (type(e).__name__, e))
holder = [F()]
later = "alive"
print("setup_ok")
'''

done = subprocess.run([sys.executable, "-c", HOLDER_SRC], capture_output=True, text=True)
assert done.returncode == 0, done.stderr
for line in done.stdout.splitlines():
    print(line)
assert "setup_ok" in done.stdout
assert "F_later 'alive'" in done.stdout
print('OK')
