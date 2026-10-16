# pyre-check: max-pypy-ratio=26
# Every traceback node keeps the line its frame was executing, including for
# a frame the JIT compiled and exited with an uncaught exception.
#
# 26 is the macos cranelift exec-only reading (24.2× vs pypy 0.01s, gate was
# 11). Ubuntu dynasm stays at 9.2×. The walk forces each compiled-raise
# frame to read `tb_lineno` / `f_code.co_name` on every iteration.
#
# `handle_exception` stamps the node with `frame.last_instr` and compiled code
# never runs the interpreter's per-opcode store, so an intermediate frame that
# only forwards the call reports its `def` line unless the trace publishes the
# raise coordinate before finishing. The pre-compile iterations report the call
# line, so the shape has to be surveyed over the whole run rather than sampled
# once: a single miscompiled frame turns one (name, lineno) tuple into two.
N = 16000


def chain(traceback):
    out = []
    while traceback is not None:
        out.append((traceback.tb_frame.f_code.co_name, traceback.tb_lineno))
        traceback = traceback.tb_next
    return tuple(out)


def a_inner(i):
    raise ValueError(i)


def a_outer(i):
    a_inner(i)


def shape_a(i):
    try:
        a_outer(i)
    except ValueError as e:
        return chain(e.__traceback__)


def c3(i):
    raise TypeError(i)


def c2(i):
    c3(i)


def c1(i):
    c2(i)


def shape_c(i):
    try:
        c1(i)
    except TypeError as e:
        return chain(e.__traceback__)


def survey(name, fn):
    seen = {}
    for i in range(N):
        t = fn(i)
        seen[t] = seen.get(t, 0) + 1
    print(name, sorted(seen))


survey("two_frame  ", shape_a)
survey("three_frame", shape_c)
