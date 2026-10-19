# CPython-suite gap: descriptor-call tests do not force collection while binding.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.
#
# descr_get_and_call_gc_roots:
# no test binds a binary special to a descriptor whose __get__ collects. this is a
# pyre/PyPy moving-GC root-liveness regression.
#
# match_keys_gc_roots:
# no match statement uses a mapping whose `get` runs Python. this is a pyre/PyPy moving-GC
# root-liveness regression.
#
# stdout_sink_gc_roots:
# print and displayhook tests do not collect while a write callback replaces the active
# stdout sink. this guards pyre/PyPy moving-GC roots across the two stdout callback paths.
#
# tuple_compare_gc_roots:
# the vendored suite reaches this shape only by accident -- `test.test_datetime` crashed
# on Linux and passed on macOS at the same commit, so whether it covers the walk depends
# on where a collection happens to land. this is a pyre/PyPy moving-GC root-liveness
# regression.
#
# except_star_projection_gc_roots:
# the except* corpus never forces a collection inside the projection walk, so nothing
# there reaches a relocated child. this guards a pyre/PyPy moving-GC allocation invariant.

"""A custom ``__call__`` descriptor must not invalidate call arguments."""

import gc


class CallDescriptor:
    def __get__(self, instance, owner):
        # Descriptor binding is an arbitrary Python callback.  Force a moving
        # collection before the call dispatcher consumes its original args.
        garbage = [[index] for index in range(2000)]
        assert len(garbage) == 2000
        gc.collect()

        def bound(*args, **kwargs):
            return instance, args, kwargs

        return bound


class Callable:
    __call__ = CallDescriptor()


class Payload:
    pass


callable_object = Callable()
for _ in range(20):
    positional = Payload()
    keyword = Payload()

    receiver, args, kwargs = callable_object(positional)
    assert receiver is callable_object
    assert args == (positional,)
    assert kwargs == {}

    receiver, args, kwargs = callable_object(positional, keyword=keyword)
    assert receiver is callable_object
    assert args == (positional,)
    assert kwargs == {"keyword": keyword}

def descr_get_and_call_gc_roots():
    """A special method bound through a descriptor must survive its own ``__get__``.

    ``get_and_call_function`` receives the raw class attribute and the operand
    list.  When that attribute is a descriptor rather than a plain function, the
    ``__get__`` dispatch is application-level Python and can run a collection; the
    receiver and the argument slice — the latter living on the Rust stack, where no
    root reaches it — must therefore be read back afterwards rather than reused.
    """

    import gc


    def _churn():
        # Enough allocation to fill a nursery and force a moving collection while
        # the caller's operand addresses are the only copies in existence.
        garbage = [[index] for index in range(20000)]
        assert len(garbage) == 20000
        gc.collect()


    class BoundThroughDescriptor:
        """A non-function class attribute, so the descriptor arm of `get` runs."""

        def __init__(self, impl):
            self.impl = impl

        def __get__(self, obj, objtype=None):
            _churn()
            impl = self.impl
            return lambda *args: impl(obj, *args)


    def _take_rhs(self, other):
        return other


    def _take_rhs_inplace(self, other):
        return other


    class Operand:
        __add__ = BoundThroughDescriptor(_take_rhs)
        __iadd__ = BoundThroughDescriptor(_take_rhs_inplace)


    for round_index in range(20):
        # A `list` right-hand side is a kind whose header a minor collection
        # relocates, so a stale operand address is observable rather than benign.
        rhs = [[value] for value in range(round_index + 1)]
        expected = [[value] for value in range(round_index + 1)]

        got = Operand() + rhs
        assert got is rhs, (round_index, "binary special lost its operand")
        assert got == expected, (round_index, got)

        accumulator = Operand()
        accumulator += rhs
        assert accumulator is rhs, (round_index, "in-place special lost its operand")
        assert accumulator == expected, (round_index, accumulator)
descr_get_and_call_gc_roots()

def match_keys_gc_roots():
    """``MATCH_KEYS`` must survive a mapping ``get`` that moves the subject.

    The opcode probes the subject once per key pattern, and each probe is an
    arbitrary Python call.  The subject, the seen-key set, the sentinel and every
    value collected so far all have to outlive it.
    """

    import gc


    def churn():
        garbage = [[index, index + 1] for index in range(20000)]
        assert len(garbage) == 20000
        gc.collect()


    class Probed(dict):
        def get(self, key, default=None):
            churn()
            return dict.get(self, key, default)


    def classify(subject):
        match subject:
            case {"kind": kind, "left": left, "right": right}:
                return ("triple", kind, left, right)
            case {"kind": kind}:
                return ("single", kind)
        return ("none",)


    for _ in range(10):
        full = Probed(kind=["op"], left=["l"], right=["r"])
        assert classify(full) == ("triple", ["op"], ["l"], ["r"]), classify(full)

        partial = Probed(kind=["op"])
        assert classify(partial) == ("single", ["op"]), classify(partial)

        assert classify(Probed(other=1)) == ("none",)
match_keys_gc_roots()

def stdout_sink_gc_roots():
    """The stdout sink remains live across displayhook and print callbacks."""

    import gc
    import sys
    from io import StringIO


    def check_displayhook_sink():
        log = []

        def make_sink(tag):
            def write(text):
                log.append((tag, text))
                if tag == "first":
                    sys.stdout = make_sink("second")
                    gc.collect()
                return len(text)

            class Sink:
                pass

            sink = Sink()
            sink.write = write
            sink.flush = lambda: None
            return sink

        saved = sys.stdout
        try:
            sys.stdout = make_sink("first")
            sys.displayhook(42)
        finally:
            sys.stdout = saved
        assert [text for _tag, text in log] == ["42", "\n"]
        assert log[0][0] == "first"


    def check_print_sink(*args, **kwargs):
        written = []

        def write(text):
            written.append(text)
            sys.stdout = StringIO()
            gc.collect()
            return len(text)

        class Sink:
            pass

        sink = Sink()
        sink.write = write
        sink.flush = lambda: None
        saved = sys.stdout
        try:
            sys.stdout = sink
            del sink
            print(*args, **kwargs)
        finally:
            sys.stdout = saved
        assert len(written) > 1


    check_displayhook_sink()
    check_print_sink("a", "b")
    check_print_sink("a", "b", sep="-", end="!\n")
    check_print_sink("a", "b", flush=True)
stdout_sink_gc_roots()

def tuple_compare_gc_roots():
    """Tuple comparison must survive an element ``__eq__`` that collects.

    ``_compare_tuples`` walks both operands element by element, and each step
    runs ``space.eq_w`` on one pair of items -- arbitrary Python, so a collection
    point.  The next step reads both tuples again, and the pair the walk stops on
    is read once more to produce the ordering answer.
    """

    import gc


    def churn():
        garbage = [[index, index + 1] for index in range(8000)]
        assert len(garbage) == 8000
        gc.collect()


    class Cell:
        """Comparisons that collect, so every element boundary moves both operands."""

        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            churn()
            return self.value == other.value

        def __lt__(self, other):
            churn()
            return self.value < other.value

        def __gt__(self, other):
            churn()
            return self.value > other.value


    for _ in range(5):
        # Array-backed tuples: the walk crosses a collection between every pair, so
        # items 2 and 3 are fetched out of tuples that have since moved.
        left = (Cell(0), Cell(1), Cell(2), Cell(3))
        right = (Cell(0), Cell(1), Cell(2), Cell(3))
        assert left == right
        assert not (left != right)

        # Arity two with object slots is the specialised representation, which
        # compares its two raw slots in a loop of its own.
        assert (Cell(4), Cell(5)) == (Cell(4), Cell(5))
        assert (Cell(4), Cell(5)) != (Cell(4), Cell(6))

        # The walk stops on an unequal pair, and the ordering answer comes from
        # reading that pair back out of the operands.
        assert (Cell(0), Cell(1), Cell(2)) < (Cell(0), Cell(1), Cell(3))
        assert (Cell(0), Cell(9), Cell(2)) > (Cell(0), Cell(1), Cell(3))

        # Every shared item agrees, so the answer is the size comparison, taken
        # from the lengths read before the walk.
        assert (Cell(0), Cell(1)) < (Cell(0), Cell(1), Cell(2))
tuple_compare_gc_roots()

def except_star_projection_gc_roots():
    """`_exception_group_projection` holds live leaf objects, not raw addresses.

    `app_group.py _exception_group_projection` keeps the leaves to project in an
    ``identity_dict`` -- the objects themselves -- and then walks the original group
    testing each child against that set.  pyre's ``ExceptionGroupCondition::Identity``
    is that same set as a rooted list of leaves, so a collection inside ``derive``
    forwards every not-yet-tested child instead of dropping it.

    This script forces a collection from inside ``derive``, i.e. between the leaf
    set being collected and the last child being tested, and asserts the projection
    is still complete.
    """

    import gc


    class Deriving(ExceptionGroup):
        """A group whose `derive` collects, the way an arbitrary override may."""

        def derive(self, excs):
            # The split rebuilds a subgroup through this hook, which is reached
            # after the identity set is collected and before the children that
            # follow this subgroup are tested.
            garbage = [[index] for index in range(2000)]
            assert len(garbage) == 2000
            gc.collect()
            return Deriving(self.message, excs)


    def leaves(exc):
        found = []

        def walk(node):
            if isinstance(node, BaseExceptionGroup):
                for child in node.exceptions:
                    walk(child)
            else:
                found.append(node)

        walk(exc)
        return found


    for _ in range(20):
        # Empty the nursery first, so every leaf below is young -- and therefore
        # moved -- when `derive` collects mid-walk.
        gc.collect()
        inner_kept = ValueError("inner-kept")
        inner_dropped = TypeError("inner-dropped")
        outer_kept = ValueError("outer-kept")
        outer_dropped = TypeError("outer-dropped")
        group = Deriving(
            "outer",
            [Deriving("inner", [inner_kept, inner_dropped]), outer_kept, outer_dropped],
        )

        caught = None
        try:
            try:
                raise group
            except* ValueError:
                raise
            except* TypeError:
                pass
        except BaseExceptionGroup as reraised:
            caught = reraised

        assert caught is not None, "the reraised ValueError half must propagate"
        got = leaves(caught)
        # `inner_kept` is tested before any `derive` runs; `outer_kept` is tested
        # after the inner subgroup's two `derive` calls have each collected.
        assert any(leaf is inner_kept for leaf in got), got
        assert any(leaf is outer_kept for leaf in got), got
        assert not any(leaf is inner_dropped for leaf in got), got
        assert not any(leaf is outer_dropped for leaf in got), got
except_star_projection_gc_roots()
print("OK")
