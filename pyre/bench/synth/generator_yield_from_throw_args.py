# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# Original throw() arguments live on SApplicationException and are
# re-read for a non-generator yield-from delegate (`delegate_to_nongen`).
class Delegate:
    def __init__(self):
        self.seen = None

    def __iter__(self):
        return self

    def __next__(self):
        return 1

    def throw(self, *args):
        self.seen = args
        raise StopIteration


def drive(kind, payload):
    d = Delegate()

    def outer():
        yield from d

    g = outer()
    assert next(g) == 1
    try:
        if kind == 1:
            g.throw(payload)
        else:
            g.throw(ValueError, payload)
    except StopIteration:
        pass
    return d.seen


ok = 0
n = 0
while n < 40:
    payload = [n, n + 1]
    seen = drive(2, payload)
    if seen is not None and seen[0] is ValueError and seen[1] is payload and len(seen) == 2:
        ok = ok + 1
    inst = ValueError(payload)
    seen = drive(1, inst)
    if seen is not None and seen[0] is inst and len(seen) == 1:
        ok = ok + 1
    n = n + 1

assert ok == 80, ok
print("PASS")
