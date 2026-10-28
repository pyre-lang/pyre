# CPython-suite gap: `test_pickle`'s framing tests dump through the
# pure-Python pickler with the JIT's default trace limit, so whether a trace
# runs out inside a builtin helper depends on how long the surrounding trace
# already is; nothing there pins a limit that puts the abort inside one.
#
# parity-tests reason: when a trace grows too long inside a builtin helper the
# walk entered transparently (the binary-op helper under `save_long` here),
# the blackhole has to finish that helper from the abort point. A helper body
# that reaches a residual call with no host address cannot be finished there,
# and resuming the interpreter at the outer frame instead skipped the
# in-flight `save(x)`: the pickle came out one element short. A trace limit
# low enough to abort inside the helper makes the case occur on every run.
import io
import pickle

try:
    import pypyjit

    pypyjit.set_param(trace_limit=400)
except ImportError:
    pass

N = 20000
obj = list(range(N))
for it in range(30):
    f = io.BytesIO()
    pickle._Pickler(f, 4).dump(obj)
    got = pickle.loads(f.getvalue())
    assert got == obj, (
        f"iteration {it}: {len(got)} elements, first difference at "
        f"{next(i for i, (a, b) in enumerate(zip(got, obj)) if a != b)}"
    )
print("OK")
