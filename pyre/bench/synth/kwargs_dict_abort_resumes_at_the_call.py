# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=run
# A CALL to `helper(*args, **kwds)` builds the callee's `**kwds` dict
# (`argument.py _match_signature`, `space.newdict(kwargs=True)`) inside the
# CALL itself.  The walker descends that allocation as a helper whose return
# it takes back; the return is not the CALL's result.  When the trace limit
# fell inside that helper, two abort images resumed past the CALL:
#
# - the helper latched as its own frame, with its caller paused past the
#   CALL, so the blackhole wrote the empty kwargs dict into the CALL's result
#   (`cm` came back as `{}`);
# - an earlier helper of the same CALL had marked the CALL's result written,
#   so the caller's CALL-entry image resumed past the CALL with the result
#   register still holding an unrelated object (`cm` came back as `helper`).
#
# Where the limit falls depends on the op count, so the fixture sweeps it.
# Each limit compiles the body afresh, so every limit traces new code.
try:
    import pypyjit
except ImportError:
    pypyjit = None

SRC = '''
class Manager:
    def __init__(self, func, args, kwds):
        self.gen = func(*args, **kwds)

    def __enter__(self):
        return next(self.gen)

    def __exit__(self, typ, value, tb):
        try:
            next(self.gen)
        except StopIteration:
            return False
        raise RuntimeError("generator didn't stop")


def manager(func):
    def helper(*args, **kwds):
        return Manager(func, args, kwds)
    return helper


class U:
    def __init__(self):
        self.out = []

    def write(self, s):
        self.out.append(s)

    @manager
    def delimit(self, start, end):
        self.write(start)
        yield
        self.write(end)

    def leaf(self):
        cm = self.delimit("[", "]")
        if type(cm) is not Manager:
            return cm
        with cm:
            self.write("x")
        return None

    def visit(self):
        self.write("a")
        self.write("b")
        return self.leaf()


def run(n):
    u = U()
    for k in range(n):
        u.out = []
        bad = u.visit()
        if bad is not None:
            return k, bad
        if u.out != ["a", "b", "[", "x", "]"]:
            return k, u.out
    return None
'''

failures = []
for limit in range(20, 420, 10):
    if pypyjit is not None:
        pypyjit.set_param(trace_limit=limit, threshold=50, function_threshold=50)
    ns = {}
    exec(compile(SRC, "<trace_limit %d>" % limit, "exec"), ns)
    got = ns["run"](200)
    if got is not None:
        failures.append((limit, got[0], repr(got[1])[:80]))

if failures:
    for failure in failures:
        print("FAIL trace_limit=%d iteration=%d result=%s" % failure)
    raise SystemExit(1)
print("PASS")
