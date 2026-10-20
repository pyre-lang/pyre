# pyre-check: gate=1
# CPython-suite gap: `test_monitoring` and `test_code` consume `co_branches()`
# values but do not pin its lazy, distinct iterator object.

import dis

# Mirrors `branches_iter_next`: FOR_ITER skips END_FOR+POP_ITER, POP_JUMP_*
# skips NOT_TAKEN, END_ASYNC_FOR reports the matching END_SEND.  Offsets come
# from `dis` so a bytecode-numbering change does not hardcode new numbers.
_POP_JUMPS = (
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE",
)


def expected_branches(code):
    rows = []
    for instr in dis.get_instructions(code):
        name = instr.opname
        if name == "FOR_ITER":
            rows.append(
                (instr.offset, instr.end_offset, instr.end_offset + instr.arg * 2 + 4)
            )
        elif name in _POP_JUMPS:
            rows.append(
                (instr.offset, instr.end_offset + 2, instr.end_offset + instr.arg * 2)
            )
        elif name == "END_ASYNC_FOR":
            src = instr.end_offset - instr.arg * 2
            rows.append((src, src + 4, instr.end_offset))
    return rows


def branchy(value):
    for i in range(3):
        if i:
            value += i
        elif value:
            value -= 1
    while value > 0:
        value -= 1
    return value


code = branchy.__code__
iterator = code.co_branches()
assert type(iterator).__name__ == "line_iterator"
assert type(iterator) is not type(code.co_lines())
assert iter(iterator) is iterator
assert list(iterator) == expected_branches(code)
assert list(iterator) == []

first = code.co_branches()
assert next(first) == next(iter(code.co_branches()))
try:
    type(first)()
except TypeError:
    pass
else:
    raise AssertionError("branches iterator type became directly constructible")


class Sub(type(first)):
    pass


assert Sub.__mro__[1] is type(first)
try:
    type(code.co_lines()).__next__(code.co_branches())
except TypeError:
    pass
else:
    raise AssertionError("co_lines iterator accepted a co_branches receiver")

namespace = {}
source = "async def drain(xs):\n    async for item in xs:\n        pass\n"
exec(compile(source, "<branches>", "exec"), namespace)
drain_code = namespace["drain"].__code__
assert list(drain_code.co_branches()) == expected_branches(drain_code)
assert list((lambda: 1).__code__.co_branches()) == []
assert expected_branches((lambda: 1).__code__) == []
