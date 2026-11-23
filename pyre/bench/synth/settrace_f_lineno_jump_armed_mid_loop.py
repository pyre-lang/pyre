# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=forward,out_of_loop,backward
# Self-checking guard for a line tracer that assigns `frame.f_lineno` while the
# frame is running a loop that is already compiled.
#
# `dispatch_bytecode` (pyopcode.py) stores the instruction start in
# `last_instr`, runs `ec.bytecode_only_trace`, and then continues at
# `next_instr = r_uint(self.last_instr)`: a trace callback that jumps
# (`PyFrame.fset_f_lineno`) moves `last_instr` to the target's start and pops
# the operand stack down to the target's depth, and dispatch resumes there with
# the frame's stack and locals as the callback left them.  `jump_absolute`
# (interp_jit.py) does the same on a back edge.  Dispatching from the value the
# hook was entered with instead runs the skipped line anyway, keeps the popped
# `for` iterator alive, and ignores a local the callback changed.
#
# Three jumps, each armed from a callee via `sys._getframe(1)` after the loop
# has been compiled, so no `call` event fires for the traced frame:
#   forward      skips one body line on every remaining iteration;
#   out_of_loop  leaves the `for` body for the line after the loop, dropping
#                the iterator from the operand stack;
#   backward     after the loop, changes `n` through `f_locals` and jumps back
#                to the `for` line, so the loop runs again over the new range.
import sys

N = 20000  # past `threshold` (1039) many times over before arming
ARM_AT = N - 2000


def arm(tracer):
    frame = sys._getframe(1)
    frame.f_trace = tracer
    frame.f_trace_lines = True
    sys.settrace(tracer)


def relative_line(frame):
    return frame.f_lineno - frame.f_code.co_firstlineno


forward_jumps = []


def forward_tracer(frame, event, arg):
    if event == "line" and relative_line(frame) == 5:
        frame.f_lineno = frame.f_code.co_firstlineno + 6
        forward_jumps.append(1)
    return forward_tracer


def forward(n, arm_at):                 # +0
    acc = 0                             # +1
    for i in range(n):                  # +2
        if i == arm_at:                 # +3
            arm(forward_tracer)         # +4
        acc += 100                      # +5
        acc += 1                        # +6
    sys.settrace(None)                  # +7
    return acc                          # +8


def out_of_loop_tracer(frame, event, arg):
    if event == "line" and relative_line(frame) == 5:
        frame.f_lineno = frame.f_code.co_firstlineno + 7
    return out_of_loop_tracer


def out_of_loop(n, arm_at):             # +0
    acc = 0                             # +1
    for i in range(n):                  # +2
        if i == arm_at:                 # +3
            arm(out_of_loop_tracer)     # +4
        acc += 1                        # +5
    acc += 1000000                      # +6
    sys.settrace(None)                  # +7
    return acc                          # +8


backward_jumps = []


def backward_tracer(frame, event, arg):
    if event == "line" and relative_line(frame) == 6 and not backward_jumps:
        backward_jumps.append(1)
        frame.f_locals["n"] = 3
        frame.f_lineno = frame.f_code.co_firstlineno + 2
    return backward_tracer


def backward(n, arm_at):                # +0
    acc = 0                             # +1
    for i in range(n):                  # +2
        if i == arm_at:                 # +3
            arm(backward_tracer)        # +4
        acc += 1                        # +5
    sys.settrace(None)                  # +6
    return acc                          # +7


got = forward(N, ARM_AT)
tail = N - ARM_AT
assert len(forward_jumps) == tail, (len(forward_jumps), tail)
assert got == 101 * N - 100 * tail, (got, 101 * N - 100 * tail)

got = out_of_loop(N, ARM_AT)
assert got == ARM_AT, (got, ARM_AT)

got = backward(N, ARM_AT)
assert backward_jumps == [1], backward_jumps
assert got == N + 3, (got, N + 3)

print("PASS")
