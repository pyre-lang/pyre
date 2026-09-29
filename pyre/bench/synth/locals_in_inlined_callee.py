# pyre-check: trace-shape=probe_direct:present=Jump
# `locals()` called from a function the tracer inlines must report the CALLEE's
# own fastlocals, not the caller's.
#
# `builtin_locals` reaches its frame through `gettopframe_nohidden`
# (`gettopframe_nohidden`), and the walker can fold that whole read into
# direct virtualizable reads instead of a residual call.  Inside an inlined
# callee the walker emits those reads against the CALLEE's own virtualizable --
# its inline body carries `getarrayitem_vable_r` / `setarrayitem_vable_r` for
# the callee's slots -- so the fold answers for the right frame.  Nothing
# pinned that before this fixture: the three existing `locals()` fixtures
# (`locals_forced_frame`, `getframe_caller_locals_after_resume`,
# `recursive_forced_frame_kept_stack`) all call it in the portal frame's own
# loop, where the caller's answer and the callee's answer coincide, so the fold
# never fires in a callee there and a fold-on/fold-off comparison over them
# records absent coverage rather than agreement.
#
# The discriminator is the NAME SET, not the values.  A fold that answered from
# the caller's frame would hand back the loop's own names (`i`, `last`,
# `outer`) instead of the callee's (`x`, `y`), and each value would still look
# plausible, so an assertion on values alone can miss it.
#
# Three shapes, because the walker treats them differently:
#
# * `probe_direct` -- one call deep.  All three builtins that share the fold
#   are exercised: `locals()` and `vars()` land on its mapping arm, `dir()` on
#   its sorted-names arm.
# * `probe_nested` -- two calls deep, which is the shape whose inline entries
#   the walker reports as a sub-walk rather than a plain inline body.
# * `probe_closure` -- a callee holding a freevar.  The freevar's name is part
#   of the expected set, which is what catches a fold that dropped it.
# * `probe_cell_rebound` -- the same freevar shape, but with the cell REBOUND
#   between reads.  `Cell.contents` is mutable and STORE_DEREF is lowered as
#   `PlainCannotRaise`, whose write-descr sets are analyzer-empty, so nothing
#   tells `OptHeap` that a store invalidates the cell read the expansion emits.
#   A merged read would report the value the recording iteration saw rather
#   than the one just assigned.  Two sites: one store per iteration (the hazard
#   is across the back edge) and a second store mid-iteration (the hazard is
#   inside one trace body).  The wrong answer is a plausible integer, so this
#   counts disagreements against a known sequence rather than printing values.


def helper_locals(x):
    y = x + 1
    d = locals()
    return sorted(d), d["x"], d["y"]


def helper_vars(x):
    y = x + 1
    d = vars()
    return sorted(d), d["y"]


def helper_dir(x):
    y = x + 1
    return dir()


def probe_direct():
    outer = "caller-only"
    last_locals = None
    last_vars = None
    last_dir = None
    for i in range(47620):
        last_locals = helper_locals(i)
        last_vars = helper_vars(i)
        last_dir = helper_dir(i)
    # `outer` is read after the loop so the caller frame really does carry a
    # name the callee does not; an unread local could be optimized away.
    return outer, last_locals, last_vars, last_dir


def inner_locals(x):
    y = x + 1
    d = locals()
    return sorted(d), d["x"], d["y"]


def outer_helper(a):
    b = a + 2
    return inner_locals(b), b


def probe_nested():
    outer = "caller-only"
    last = None
    for i in range(47620):
        last = outer_helper(i)
    return outer, last


def make_closure(bias):
    def helper(x):
        y = x + bias
        d = locals()
        return sorted(d), d["x"], d["y"]

    return helper


closure_helper = make_closure(1)


def probe_closure():
    outer = "caller-only"
    last = None
    for i in range(47620):
        last = closure_helper(i)
    return outer, last


def probe_cell_rebound():
    cap = -1

    def read_cap():
        # The modelled mapping and a plain LOAD_DEREF of the same cell are two
        # different reads of one field; they must agree.
        d = locals()
        return d["cap"], cap

    bad_a = 0
    bad_b = 0
    mismatched = 0
    for i in range(190477):
        cap = i
        seen, direct = read_cap()
        if seen != i:
            bad_a += 1
        if seen != direct:
            mismatched += 1

        cap = i + 20000
        seen_again, direct_again = read_cap()
        if seen_again != i + 20000:
            bad_b += 1
        if seen_again != direct_again:
            mismatched += 1
    return bad_a, bad_b, mismatched


print(probe_direct())
print(probe_nested())
print(probe_closure())
print(probe_cell_rebound())
