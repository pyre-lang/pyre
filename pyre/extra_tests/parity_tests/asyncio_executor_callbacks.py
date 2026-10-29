# pyre-check: requires-modules=_socket,select
# CPython-suite gap: repeated executor completion must get hot enough to trace
# Handle._run inside _ThreadSafeHandle._run and the event loop together.
# parity-tests reason: an inner helper rewind used to retain a fresh argument
# list's length change, adding a NULL argument and losing a completion callback.

import asyncio


def work(value):
    return value + 1


async def main():
    loop = asyncio.get_running_loop()
    for _ in range(100):
        pending = [loop.run_in_executor(None, work, value) for value in range(64)]
        values = await asyncio.wait_for(asyncio.gather(*pending), 5)
        assert values == list(range(1, 65)), values


asyncio.run(main())
print("OK")
