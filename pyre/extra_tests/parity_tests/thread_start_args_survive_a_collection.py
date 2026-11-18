# CPython-suite gap: test_weakref's threaded copy tests reach this only when a
# collection happens to land while one of their threads is starting.
# parity-tests reason: `_thread.start_new_thread` and `start_joinable_thread`
# hand the new thread its callable and arguments while the starter waits with
# the GIL released.  A collection another thread runs in that window moves
# nursery-born objects, so the hand-off travels through roots the collector
# forwards (os_thread.py `Bootstrapper`) rather than addresses the new thread
# reads afterwards.
# parity-env: PYPY_GC_NURSERY=64k

"""A thread starts with its own callable and arguments while another collects.

A churn thread allocates without pause, so collections keep running while the
main thread starts workers.  Each worker's callable is a fresh bound method and
its arguments are fresh objects, all born in the nursery, and each worker
records what it received.  Some workers receive their text as a keyword
argument.
"""

import _thread
import threading

STARTS = 60

stop = False


def churn():
    junk = []
    while not stop:
        junk.append([object() for _ in range(64)])
        if len(junk) > 32:
            junk.clear()


class Worker:
    def __init__(self, n):
        self.n = n
        self.got = None
        self.done = _thread.allocate_lock()
        self.done.acquire()

    def run(self, items, text):
        try:
            self.got = (self.n, list(items), text)
        finally:
            self.done.release()


def expected(n):
    return (n, [n, n + 1], "w%d" % n)


churner = threading.Thread(target=churn)
churner.start()
try:
    for n in range(STARTS):
        worker = Worker(n)
        case = n % 3
        if case == 1:
            _thread.start_new_thread(worker.run, ([n, n + 1], "w%d" % n))
            assert worker.done.acquire(True, 60.0), n
        elif case == 2:
            _thread.start_new_thread(
                worker.run, ([n, n + 1],), {"text": "w%d" % n}
            )
            assert worker.done.acquire(True, 60.0), n
        else:
            thread = threading.Thread(
                target=worker.run, args=([n, n + 1], "w%d" % n)
            )
            thread.start()
            thread.join(60.0)
            assert not thread.is_alive(), n
        assert worker.got == expected(n), (n, worker.got)
finally:
    stop = True
    churner.join()

print("OK")
