# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=<module>
# pyre-check: requires-modules=_tokenize
# A residual FileType call inside argparse.parse_args raises into a hot
# try/except. The loop trace records GUARD_EXCEPTION on a raising iteration;
# later failures of that guard carry no pending JIT exception, so
# `prepare_resume_from_failure` records GUARD_NO_EXCEPTION. The no-exception
# bridge must continue the fallthrough, not GETFIELD `w_traceback` off the
# SAVE_EXCEPTION box (NULL on that arm).
import argparse

N = 5000


class Err(Exception):
    pass


class P(argparse.ArgumentParser):
    def error(self, message):
        raise Err(message)


def once():
    p = P()
    p.add_argument("-x", type=argparse.FileType("xb"))
    p.add_argument("spam", type=argparse.FileType("xb"))
    try:
        p.parse_args(["-x"])
    except Err:
        pass
    try:
        p.parse_args([""])
    except (Err, SystemExit, TypeError, ValueError, OSError):
        pass


for i in range(N):
    once()
print("PASS")
