# pyre-check: max-pypy-ratio=8.8
# dynasm 3.8-4.4x (darwin-arm64); the ceiling is twice the slower reading.
# wasm has its own baseline: it reads guard_failures 12664 against 12635 on
# dynasm, with the same loops and bridges. The cause of the extra exits is
# not identified.
# A guard-failure bridge resumes `pickle._Unpickler.load` in the middle of its
# `while True` loop, inlines the dispatched opcode handler and reaches the loop
# header again.  It finishes that frame there and runs the rest of the loop with
# `CALL_ASSEMBLER` (`opimpl_jit_merge_point`'s recursive arm: `finishframe`,
# then `do_recursive_call(assembler_call=True)` on the caller).  The guards
# after that call resume the caller after the call, holding the call's result
# (`do_residual_call` runs `make_result_of_lastop` before `generate_guard`).
#
# They resumed the finished `load` frame at its loop header instead.  When the
# frame was forced inside the call, `GUARD_NOT_FORCED` failed after the call
# had already run the loop to STOP, so the loop ran again: `read(1)` returned
# `b''` and `load` raised `EOFError` for a pickle it had read completely.
#
# Which guard gets that bridge depends on the order bridges are compiled in,
# so the wrapper around `load`, its writes to a `StringIO` and the loop body
# below are kept as they were when the failure was found, and the jit-stats
# baseline pins the compiled shape: a change that moves it may stop this
# fixture from reaching the path.  With those guards capturing the finished
# callee, the first `EOFError` came at iteration 34; 50 iterations leave room.
import pickle, io, sys
out = io.StringIO()

vals = [b'', b'xyz', b'abc' * 100, 'str', 1, (1, 2), [1, 'a'], {'k': b'v'}, None, 2**70, 1.5]
bad = 0
orig = pickle._Unpickler.load
enters = [0]

def wrapped(self):
    enters[0] += 1
    n = enters[0]
    tell = None
    try:
        tell = self._file_read.__self__.tell()
    except Exception:
        pass
    stack = getattr(self, 'stack', 'no')
    print("ENTER", n, "id", id(self), "tell", tell, "stack", stack, file=out)
    try:
        r = orig(self)
        print("EXIT", n, "id", id(self), "r", r, file=out)
        return r
    except Exception as e:
        tell2 = None
        try:
            tell2 = self._file_read.__self__.tell()
        except Exception:
            pass
        print("RAISE", n, type(e).__name__, "id", id(self), "tell", tell2, "stack", getattr(self, 'stack', 'no'), file=out)
        raise

pickle._Unpickler.load = wrapped

for it in range(50):
    for proto in range(6):
        for v in vals:
            f = io.BytesIO()
            pickle._Pickler(f, proto).dump(v)
            p = f.getvalue()
            try:
                bio = io.BytesIO(p); u = pickle._Unpickler(bio); r = u.load()
            except EOFError as e:
                bad += 1
                if bad == 1:
                    print("first EOFError at", it, proto, repr(v), p, "tell", bio.tell(), "len", len(p), "stack", u.stack, "meta", u.metastack, "enters", enters[0], file=out)
                    tb = e.__traceback__
                    while tb is not None:
                        print("tb", tb.tb_frame.f_code.co_name, "lasti", tb.tb_lasti, "lineno", tb.tb_lineno, "unit", tb.tb_lasti // 2, file=out)
                        tb = tb.tb_next
                continue
            assert r == v, (it, proto, v, r)
print("PASS" if bad == 0 else "FAIL bad=%d enters=%d" % (bad, enters[0]))
