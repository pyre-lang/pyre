# pyre-check: requires-modules=_socket
# CPython-suite gap: socket creation is not paired with another thread forcing
# collection at the blocking-mode syscall and the boxed state-field stores.
# parity-tests reason: the socket receiver must be reloaded after GC moves it;
# a pinned caller alone does not update a Rust argument evaluated before GC.
# Sockets are created unconnected. A connected pair is emulated over loopback
# TCP on Windows and exhausts its buffers.

import gc
import socket
import threading
import time

barrier = threading.Barrier(11)
errors = []


def work():
    try:
        barrier.wait()
        for _ in range(100):
            left = socket.socket()
            right = socket.socket()
            for sock in (left, right):
                sock.settimeout(0.125)
                assert sock.gettimeout() == 0.125
                sock.settimeout(None)
                sock.close()
    except BaseException as exc:
        errors.append(exc)


def collect():
    barrier.wait()
    for _ in range(100):
        gc.collect()
        time.sleep(0.001)


threads = [threading.Thread(target=work) for _ in range(10)]
threads.append(threading.Thread(target=collect))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert not errors, errors
print("OK")
