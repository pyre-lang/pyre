# A kept-stack short-circuit inside a pending method call aborts tracing after
# the authoritative walk has already advanced a stateful reader.
#
# RPython's `MIFrame.run_one_step` has selected the POP_JUMP successor and
# advanced `frame.pc` before a tracing abort is converted to blackhole frames.
# The blackhole therefore continues with the post-branch stack.  Rewinding to
# the branch opcode needs its already-popped truth operand; rewinding farther
# to the trace entry re-runs `fp.read(1)` and loses one byte.  Iteration 2 is
# where the function-entry trace becomes hot enough to expose that replay.
import io


BLOB = b"a" * 500


def parse(encoding):
    fp = io.BytesIO(BLOB)
    total = 0
    count = 0
    while total < len(BLOB):
        data = fp.read(1)
        text = data.decode(encoding or "ascii")
        count += len(text)
        total += 1
    return count, fp.tell()


for run in range(60):
    print(run, parse(None))
