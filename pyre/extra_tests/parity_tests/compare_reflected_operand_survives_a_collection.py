# CPython-suite gap: test_collections.validate_comparison reaches this only
# when a collection happens to land between the forward and reflected
# comparison calls.
# parity-tests reason: `_make_comparison_impl` / `_invoke_comparison` run the
# forward comparison then the reflected one.  The first call is arbitrary
# Python and can collect, so both operands and both resolved methods must
# travel through roots the collector forwards (`gctransform/shadowstack.py`
# `push_roots`/`pop_roots`) rather than addresses the second call reads
# afterwards.  A function born in a class body inside a function is
# nursery-allocated and moves too.
# parity-env: PYPY_GC_NURSERY=128k

"""A reflected comparison operand must survive a collection in the forward call.

`collections.abc.Set` comparison returns NotImplemented, so the other
operand's reflected method runs next.  Junk allocation in the loop fills
the nursery so that first call collects.
"""

import operator
from collections.abc import Set


def validate_comparison(instance):
    ops = ["lt", "gt", "le", "ge", "ne", "or", "and", "xor", "sub"]
    operators = {}
    for op in ops:
        name = "__" + op + "__"
        operators[name] = getattr(operator, name)

    class Other:
        def __init__(self):
            self.right_side = False

        def __eq__(self, other):
            self.right_side = True
            return True

        __lt__ = __eq__
        __gt__ = __eq__
        __le__ = __eq__
        __ge__ = __eq__
        __ne__ = __eq__
        __ror__ = __eq__
        __rand__ = __eq__
        __rxor__ = __eq__
        __rsub__ = __eq__

    for name, op in operators.items():
        if not hasattr(instance, name):
            continue
        other = Other()
        op(instance, other)
        assert other.right_side, name


class MySet(Set):
    def __contains__(self, x):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])


n = 400
junk = []
for i in range(n):
    junk.append([i] * (i % 7))
    if len(junk) > 50:
        junk = []
    validate_comparison(MySet())
print("OK")
