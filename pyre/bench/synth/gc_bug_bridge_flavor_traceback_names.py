# Regression oracle. Was `_pending/`: cranelift aborted 8/10 to 9/10 runs with
#   GC BUG: invalid type_id=... site=object_total_size
# from `gc_alloc_nursery_shim` -> `alloc_with_type` -> `do_collect_nursery` ->
# `incremental_mark_step`, i.e. a dangling nursery pointer on the MAJOR gray
# stack, while dynasm and the no-JIT run stayed clean.  Now 20/20 clean on
# cranelift with the pinned output below.
#
# REFUTED lead, do not re-attempt: the unbarriered `(*frame).f_backref =
# saved_topframeref` in ResidualFrameChainGuard::enter looks exactly like the
# known producer of this signature - old-gen FrameBox taking a possibly-nursery
# caller, while its sibling store in the walker carries a barrier and a comment
# saying why.  Adding the same barrier there changes nothing: 18/20 aborts with
# it versus 8/10 without.
#
# Ingredients, each of which removing made it stop aborting:
#   * TWO exception classes raised into one hot try/except, so a bridge is
#     compiled for the exception edge (the `exc_mixed_classes_bridge_flavor`
#     shape);
#   * an INLINED intermediate frame between the loop and the raise;
#   * reading `tb_frame.f_code.co_name` off every node in the handler;
#   * retaining the resulting name tuples in a set across iterations.
# Counting traceback depth instead of reading `co_name`, or dropping the set,
# or a single exception class, all run clean.
#
# Expected output: A [('T', 'a_bridge_two_classes', 'mid_two', 'leaf_two'),
#                     ('V', 'a_bridge_two_classes', 'mid_two', 'leaf_two')]
N = 120000


def chain(e):
    names = []
    tb = e.__traceback__
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return tuple(names)


def leaf_two(i):
    if i % 3 == 1:
        raise ValueError(i)
    if i % 3 == 2:
        raise TypeError(i)
    return i


def mid_two(i):
    return leaf_two(i)


def a_bridge_two_classes():
    shapes = set()
    acc = 0
    for i in range(N):
        try:
            acc += mid_two(i)
        except ValueError as e:
            shapes.add(("V",) + chain(e))
        except TypeError as e:
            shapes.add(("T",) + chain(e))
    return sorted(shapes)


def _age_out_of_nursery():
    # Promote live W_Code (GUARD_VALUE / make_a_counter_per_value, keyed by
    # address) out of the nursery before the measured loops. A minor that
    # moves a young object inside the failure burst splits the count.
    # cranelift core has no gc; 800 discarded 400-tuples fill an 8MB nursery
    # without compiling (loop threshold 1039).
    try:
        import gc as _gc
        _gc.collect(0)
        return
    except ImportError:
        pass
    i = 0
    while i < 800:
        dropped = (
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
            object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(), object(),
        )
        i += 1

_age_out_of_nursery()
print("A", a_bridge_two_classes())
