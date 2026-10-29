# pyre-check: requires-modules=gc
# CPython-suite gap: frame identity and explicit finalization tests do not
# check PyPy's GC hooks while inspecting a suspended coroutine.
# parity-tests reason: cr_frame used to force a full collection on every
# read, even when automatic major collections were disabled.
import gc


class Pause:
    def __await__(self):
        yield


async def pending():
    await Pause()


coro = pending()
coro.send(None)
frame = coro.cr_frame
gc.collect()
enabled = gc.isenabled()
gc.disable()
hooks = getattr(gc, 'hooks', None)
collections = []
if hooks is not None:
    old_hook = hooks.on_gc_collect
    hooks.on_gc_collect = lambda stats: collections.append(stats.count)
try:
    for _ in range(10):
        assert coro.cr_frame is frame
    assert sum(collections) == 0, collections
finally:
    if hooks is not None:
        hooks.on_gc_collect = old_hook
    if enabled:
        gc.enable()
    coro.close()
assert coro.cr_frame is None
print('OK')
