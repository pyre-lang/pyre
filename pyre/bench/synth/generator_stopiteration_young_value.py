# Generator return of a young object. `stop_iteration_with_value` allocates
# before the StopIteration args list is stamped, so the return object must
# be reloaded from a pin.
# Expected: object
def g():
    yield
    return object()


gg = g()
gg.send(None)
try:
    gg.send(None)
except StopIteration as e:
    print(type(e.value).__name__)
