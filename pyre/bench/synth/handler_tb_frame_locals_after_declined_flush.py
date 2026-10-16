# A frame read from INSIDE a Python expression. Forcing it would make the
# merge-point escape flush decline and write only the locals region.
#
# `'i' in tb.tb_frame.f_locals` reads the portal frame through a different box.
# The alias is promoted onto the standard virtualizable, so the read does not
# force the frame.  A residual force would land while the operand stack holds
# `[seen, add, <bool being computed>]`.  A mid-expression stack slot reads NULL
# from the virtualizable shadow, so `flush_walk_end_state_to_frame_inner`
# declines the whole write and `flush_locals_region_to_frame` writes slots
# `0..nlocals` on their own -- an unforced array would render `f_locals` as an
# EMPTY mapping, a wrong answer rather than a stale one.
#
# That leg claims no resume pc, so nothing sets `COMMITTED_FRAME_ESCAPE_PC` and
# the walk-end block gated on it never runs.  Its deferred undo restore has to
# be armed at the residual instead: `LiveLastInstrGuard` declines to put
# `last_instr` back while an undo capture is live, so without the arming the
# frame keeps the EXECUTING pc over an operand stack no flush ever wrote, and
# the replay re-enters one opcode late on an empty stack -- `value-stack
# underflow: depth=N base=N`, a JIT-only panic with no output at all.
#
# The handler is what puts the read inside an expression whose stack is deep
# enough to notice: the `seen.add(...)` receiver and its bound method are both
# live below the value being computed.
import sys

N = 180000


def run():
    seen = set()
    i = 0
    while i < N:
        try:
            raise ValueError(i)
        except ValueError:
            tb = sys.exc_info()[2]
            seen.add('i' in tb.tb_frame.f_locals)
        i = i + 1
    return sorted(seen)


print(run())
