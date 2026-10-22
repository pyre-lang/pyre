# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:ForArm,entry-bridge:WhileArm
# Regression guard: a class statement in a loop body used to LOSE the rest of
# the iteration it aborted in, whenever an earlier opcode of that same body had
# already mutated the heap.
#
# The chain, all on the default path:
#   1. the walk consumes the FOR_ITER item and stashes it in flight;
#   2. `c.x = 1` runs its STORE_ATTR residual concretely.  It writes the live
#      heap outside every journal, so it bumps the executed-effect odometer AND
#      raises the R1 "body effect since consume" signal;
#   3. the class statement is one residual (`__build_class__` has no generated
#      gateway to descend into), and the walker asks its
#      `jit_force_quasi_immutable` question at that boundary -- the class body
#      stores into a `newdict(module=True)` namespace, and upstream abandons
#      every trace that executes a class body;
#   4. the `ForceQuasiImmutable` flush leg DECLINED, because the class-body site
#      offered it no operand-stack mirror.  The walk rolled back and left the
#      legacy replay to re-run the region;
#   5. that replay needs the consumed FOR_ITER item pushed back, and
#      `fbw_foriter_inflight_take` refuses to deliver it while R1 stands -- a
#      re-run of the body would apply the step-2 store a second time.
#
# Refusing delivery is the never-double rule, so the item was DROPPED and the
# loop moved on: the `APPENDED.append(1)` after the class statement never ran
# for that iteration.  This file measured 9995 appends for 10000 iterations on
# dynasm; the same shape reduced to one script was equally red on cranelift, so
# it is not backend-specific.  Spelling the tail as `acc += 1` loses the same
# five increments, which is what makes it a lost ITERATION rather than a lost
# store -- `c.x` itself ends correct either way.
#
# Two things had to change, and the fixture needs both:
#   * the class-body `ABORT_FORCE_QUASIIMMUT` site now latches the operand-stack
#     mirror its two sibling qmut sites already latched, so the leg has
#     something to adopt;
#   * the latch's own sanity check only exempted ONE disagreeing slot, the
#     in-progress opcode's TOS.  A `CALL` pops its whole callable/self/args run
#     into registers before its residual -- four operand slots for a class
#     statement -- so every one of them read NULL in the vable shadow and the
#     flush declined on the first.  The qmut leg now exempts the consumed
#     SUFFIX.
#
# The wider exemption is the QMUT LEG'S ALONE, and that is measured, not
# cautious: the escape leg's opcode is IN PROGRESS -- its residual is what
# forced -- so its `last_instr = pc - 1` reads as "about to run the CALL" while
# the callee doing the forcing is already inside it.  Letting it commit the
# deeper run turned `caller_f_lasti_across_residual_call` red, 2 of 20000 reads
# answering 42 (the `LOAD_GLOBAL` ahead of the call) instead of 52.  The qmut
# abort has no such conflict: it fires BEFORE its residual executes.
#
# The while-loop arm is the control that isolates the loss to the dropped item:
# it runs the same store, the same class statement and the same append, aborts
# the same way, and was already correct because it has no in-flight FOR_ITER
# item to lose.
#
# `c.x = 1` is deliberately a constant store: the value never mattered, only
# that a STORE_ATTR residual ran ahead of the class statement.  Ordering does
# matter -- with the class statement FIRST the walk aborts before any effect,
# and both spellings were always right.
#
# The store is one of many doors into it.  Fourteen effect kinds were run ahead
# of the same class statement on the pre-fix binary; NINE lost the identical
# five iterations -- `obj.attr = v` (plain and `__slots__`), a user
# `__setattr__`, `d[k] = v`, `del d[k]`, `dict.update`, `set.add`,
# `list.extend`, `bytearray[i] = v`. The five that stayed green are the ones
# whose effect the walk JOURNALS and therefore rolls back, so the deliver's R1
# check passes: the module-global cell store, `list.append`/`pop`, a list
# `setitem`, and a nested `for` whose inner loop changes the abort shape.

ROUNDS = 200
INNER = 50


class C:
    __slots__ = ("x",)

    def __init__(self):
        self.x = 0


APPENDED = []


def bump(c, n):
    for _ in range(n):
        c.x = 1

        class ForArm:
            pass

        APPENDED.append(1)
    return len(APPENDED)


def bump_while(c, n, sink):
    i = 0
    while i < n:
        c.x = 1

        class WhileArm:
            pass

        sink.append(1)
        i += 1
    return len(sink)


def main():
    c = C()
    total = 0
    for _ in range(ROUNDS):
        total += bump(c, INNER)
    want_len = ROUNDS * INNER
    # Each call returns the running length, so the returned values are
    # INNER, 2*INNER, ... and their sum pins every intermediate one: a single
    # dropped iteration lowers the total by one for every call after it, not
    # just its own.
    want_total = INNER * ROUNDS * (ROUNDS + 1) // 2
    sink = []
    control = 0
    for _ in range(ROUNDS):
        control += bump_while(c, INNER, sink)

    bad = 0
    for arm, appended, running in (
        ("for", len(APPENDED), total),
        ("while", len(sink), control),
    ):
        if appended != want_len:
            print("FAIL %s: appended %d, want %d" % (arm, appended, want_len))
            bad += 1
        if running != want_total:
            print("FAIL %s: total %d, want %d" % (arm, running, want_total))
            bad += 1
    if bad:
        raise SystemExit(1)
    print("PASS class-stmt qmut abort kept every iteration")


main()
