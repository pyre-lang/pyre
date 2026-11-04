# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=_mock_add_spec,_mock_set_magics
# pyre-check: requires-modules=_tokenize
# MagicMock class-spec then `_mock_set_magics` in a hot loop.
#
# A multi-frame guard resume whose innermost section is an inlined callee
# paused at its loop header. The rebuilt frame stored the live operand
# stack in `locals_cells_stack_w` but left `valuestackdepth` at the empty
# prefix, so `FOR_ITER` peeked a local (NULL / leftover builtin) instead
# of the iterator (`peekvalue` / `consume_boxes`).
from unittest.mock import MagicMock

N = 80


class _One(object):
    one = 1


def hot(n):
    c = 0
    for i in range(n):
        mock = MagicMock()
        mock._mock_add_spec(_One, False)
        mock._mock_set_magics()
        c += int(hasattr(mock, "one"))
    return c


print("PASS", hot(N))
