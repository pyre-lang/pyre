# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:ForArm,entry-bridge:WhileArm
# Regression guard: a class statement in a loop body must keep the rest of
# that iteration, even when an earlier opcode of the same body has already
# mutated the heap.
#
# The chain, all on the default path:
#   1. the walk consumes the FOR_ITER item and stashes it in flight;
#   2. `c.x = 1` runs its STORE_ATTR residual concretely.  It writes the live
#      heap outside every journal, so it bumps the executed-effect odometer AND
#      raises the R1 "body effect since consume" signal;
#   3. `JitPolicy.look_inside_graph` residualizes `build_class` because the
#      graph contains loops (`compiling.py build_class` builds keyword-name
#      lists with comprehensions) and is not `unroll_safe`.  The class body's
#      own portal entry may start a separate trace;
#   4. that inner trace aborts on ABORT_FORCE_QUASIIMMUT when its STORE_NAME
#      mutates the class namespace's `ModuleDictStrategy.version?`;
#   5. a declined flush would need the consumed FOR_ITER item pushed back, and
#      `fbw_foriter_inflight_take` refuses to deliver it while R1 stands -- a
#      re-run of the body would apply the step-2 store a second time.
#
# Refusing delivery is the never-double rule, so a dropped item used to skip
# the `APPENDED.append(1)` after the class statement.  History: 9995 appends
# for 10000 iterations on dynasm (cranelift the same shape).  Spelling the
# tail as `acc += 1` lost the same five increments -- a lost ITERATION rather
# than a lost store; `c.x` itself ends correct either way.  The fixture
# guards that a class statement in a loop body never loses the rest of that
# iteration.
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
