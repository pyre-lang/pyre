# Regression guard: a callee that reads its CALLER's frame via sys._getframe(1)
# must see the caller's live locals, even though the callee has a hot loop of its
# own and so takes its own compiled-loop entry from inside the caller's running
# compiled loop.
#
# Two defects met here and produced a JIT-only wrong answer (acc frozen at the
# value it held when the caller's loop was compiled):
#   1. the force was skipped, because it required the frame to equal
#      `MetaInterp::vable_ptr` and the callee's compiled entry had re-pointed
#      that cell at the callee frame with no restore;
#   2. once the force ran it wrote a null over `acc`, because the resume decode
#      went through an allocator that could not materialize the virtual the
#      slot named -- and a null slot reads back as an ABSENT name, so
#      `f_locals['acc']` raised KeyError instead of returning a value.
#
# wasm note. This fixture carried `skip-backends=wasm` while the wasm backend
# had no force protocol at all: `OpCode::ForceToken` lowered to a literal `0`,
# which is `virtualref.py`'s `TOKEN_NONE`, so the compiled loop parked a zero in
# `vable_token` and `force_virtualizable_if_necessary` never forced;
# `GUARD_NOT_FORCED` always passed; and `Backend::force` /
# `is_force_token_armed` kept their `None` / `false` defaults, so there was no
# deadframe to decode even had the force run. wasm read 3691 on all nine
# `f_locals['acc']` reads and printed 104995.
#
# The backend now implements it -- `ForceToken` names the JITFRAME,
# `emit_force_bracket_before_call` arms the following `GUARD_NOT_FORCED`'s exit
# before a may-force call, that guard tests the mark `force` leaves, and `force`
# rebuilds a deadframe from the armed exit's spilled slots. Measured here: wasm
# prints 105005 on three consecutive runs, matching dynasm, cranelift and
# CPython, so the exemption is retired.
import sys


def inner(k):
    s = 0
    for j in range(3):        # callee's own hot loop -> its own compiled entry
        s += j * k
    if k > 149990:
        g = sys._getframe(1)  # the CALLER's frame, mid-activation
        return s + g.f_locals['acc']
    return s


def outer(n):
    acc = 0                   # loop-carried, virtual inside the compiled loop
    for i in range(n):
        acc += inner(i) & 7
    return acc


print(outer(150000))
