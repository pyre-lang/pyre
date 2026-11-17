# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:cand,entry-bridge:g
# An inlined `C()` whose result is stored to a module global must not
# re-execute the CALL when that store's ForceQuasiImmutable aborts the
# walk.
#
# The walker recognises `C(...)` while decoding the residual CALL
# (`try_walker_inline_type_call`), writes dest through `write_ref_reg`,
# and continues to STORE_GLOBAL. Abort convert must copy the already
# advanced `MIFrame.pc` (`_get_opimpl_method` / `make_result_of_lastop`
# in `pyjitpl.py`). Rewinding to the CALL leaves the instance in dest
# (the callable colour) and the interpreter calls it:
# `TypeError: 'R' object is not callable`.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class R:
    def __init__(self):
        self.x = 1


_seq = None


def cand():
    global _seq
    if _seq is None:
        _seq = R()
    return _seq


def g():
    return cand()


def main():
    try:
        got = type(g()).__name__
    except TypeError as e:
        print("FAIL TypeError: %s" % e)
        return 1
    if got != "R":
        print("FAIL type name %r, owed R" % got)
        return 1
    print("R")
    print("PASS inlined constructor then store-global did not recall")
    return 0


sys.exit(main())
