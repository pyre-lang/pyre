# pyre-check: platforms=win32
# pyre-check: requires-modules=_multiprocessing,_winapi,mmap
# pyre-check: pypy-diverges: PyPy's Windows shared_memory attachment calls
# the absent _winapi.UnmapViewOfFile; its spawn/Pipe/Semaphore path works.
# CPython-suite gap: test_multiprocessing_spawn is not in the Windows CI gate.
# parity-tests reason: interp_semaphore.W_SemLock.acquire binds block/timeout
# by name; multiprocessing synchronization and spawned children rely on it.
import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import os


def child(connection, gate, name):
    memory = shared_memory.SharedMemory(name=name)
    try:
        assert gate.acquire(timeout=10)
        assert connection.recv_bytes() == b'parent'
        memory.buf[:5] = b'child'
        connection.send_bytes(b'child')
    finally:
        memory.close()
        connection.close()


class FalseBlock:
    def __bool__(self):
        gc.collect()
        return False


class ZeroTimeout:
    def __float__(self):
        gc.collect()
        return 0.0


if __name__ == '__main__':
    context = mp.get_context('spawn')
    gate = context.Semaphore(1)
    raw = gate._semlock
    assert raw.acquire(block=FalseBlock(), timeout=ZeroTimeout())
    assert not raw.acquire(block=False, timeout=None)
    raw.release()
    assert raw.acquire(timeout=0)
    raw.release()
    for args, kwargs in (((False,), {'block': False}), ((), {'blocking': False})):
        try:
            raw.acquire(*args, **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError('invalid acquire binding accepted')

    parent, other = context.Pipe()
    gate = context.Semaphore(0)
    memory = shared_memory.SharedMemory(create=True, size=5)
    process = context.Process(target=child, args=(other, gate, memory.name))
    try:
        process.start()
        other.close()
        parent.send_bytes(memoryview(b'parent'))
        gate.release()
        assert parent.poll(20)
        assert parent.recv_bytes() == b'child'
        process.join(20)
        assert process.exitcode == 0, process.exitcode
        assert bytes(memory.buf) == b'child'
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        memory.close()
        memory.unlink()
    print('OK')
