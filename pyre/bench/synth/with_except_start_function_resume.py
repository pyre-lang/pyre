# pyre-check: max-pypy-ratio=21
# A function-entry trace records only the early-return arm.  The first true
# call then fails that guard and blackhole-replays the previously unvisited
# `with` arm.  WITH_EXCEPT_START must read the live handler stack and call
# `__exit__`; a disconnected graph result treats the caught TypeError as the
# exit callable and leaks another TypeError instead.
try:
    import pypyjit

    pypyjit.set_param("threshold=-1,function_threshold=20")
except ImportError:
    pass

seen = []


class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        seen.append(exc_type)
        return True


def run(raise_inside):
    if not raise_inside:
        return
    with Context():
        abs()


for _ in range(126000):
    run(False)
run(True)
print("ok", len(seen))
