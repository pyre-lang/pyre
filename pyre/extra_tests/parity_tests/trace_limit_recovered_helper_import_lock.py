# CPython-suite gap: import-lock tests do not trace a callback across a
# recovered helper descent and then cross the trace limit in the same opcode.
# parity-tests reason: blackhole handoff must resume after the completed call
# instead of replaying its acquire_lock side effect.

import _imp

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("trace_limit=70,threshold=1,function_threshold=1")

_module_locks = {}


def cb(ref, name="callback"):
    _imp.acquire_lock()
    try:
        assert _module_locks.get(name) is ref, "callback replay ran cb twice"
        del _module_locks[name]
    finally:
        _imp.release_lock()


assert not _imp.lock_held()
ref = object()
for _ in range(100):
    _module_locks["callback"] = ref
    cb(ref)
    assert not _imp.lock_held(), "callback replay leaked the import lock"
assert not _module_locks
print("OK")
