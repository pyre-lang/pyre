# pyre-check: requires-modules=gc
# pyre-check: no-cpython
# Immortal modules and builtin types are traced through `prebuilt_root_objects`,
# so `gc.get_rpy_roots()` is a few hundred objects (pypy3 has about 613) and
# the `<module>` list comprehension over it stays below the loop threshold,
# remaining interpreted as on pypy3. The one compiled loop is
# `warm_jit_for_code_stats`.
import gc
import sys


# The generation argument is integer-unwrapped and then ignored: none of these
# selects anything, because every collection here is a full one.  Which values
# are *accepted* is a spec axis -- this oracle takes any int, `gc_collect_impl`
# takes only a generation it reports, and pyre follows the latter -- so only
# the three both accept are probed here, and
# `extra_tests/snippets/gc_generation_argument_is_bounded.py` pins the rest.
# What `collect` answers is a spec axis too: pyre answers an int where this
# oracle answers None, and `extra_tests/snippets/stdlib_gc.py` pins that.
for generation in (0, 1, 2):
    gc.collect(generation)

for value in (1.0, "1"):
    try:
        gc.collect(value)
    except TypeError:
        pass
    else:
        raise AssertionError("gc.collect must integer-unwrap its argument")


# The finalizer lock is recursive and does not change the collection-enabled
# flag.  Its public enable operation rejects an unmatched call.
assert gc.isenabled()
try:
    gc.enable_finalizers()
except ValueError as error:
    assert str(error) == "finalizers are already enabled"
else:
    raise AssertionError("unmatched gc.enable_finalizers must fail")

gc.disable_finalizers()
gc.disable_finalizers()
assert gc.isenabled()
gc.enable_finalizers()
gc.enable_finalizers()
assert gc.isenabled()

try:
    gc.enable_finalizers()
except ValueError as error:
    assert str(error) == "finalizers are already enabled"
else:
    raise AssertionError("finalizer lock must be balanced")


# `referents.py:get_objects` audits -1 whatever it was handed, because its own
# walk takes no generation, and it emits that event before it looks at the
# argument at all.  pyre keeps both of those: nothing pins the payload to the
# argument, so the walk it does have -- one that selects a generation -- widens
# only which arguments are accepted, and
# `extra_tests/snippets/gc_generation_argument_is_bounded.py` is where the
# accepted set is pinned.  What both agree on is the payload, asserted here.
get_objects_events = []
referent_events = []


def audit_hook(event, args):
    if event == "gc.get_objects":
        get_objects_events.append(args)
    elif event in ("gc.get_referents", "gc.get_referrers"):
        referent_events.append((event, args))


sys.addaudithook(audit_hook)
assert isinstance(gc.get_objects(), list)
assert isinstance(gc.get_objects(None), list)

assert get_objects_events == [(-1,), (-1,)]


referent = []
referrer = [referent]
assert gc.get_referents() == []
assert gc.get_referents(referent, referrer) == [referent]
assert gc.get_referrers() == []
assert any(item is referrer for item in gc.get_referrers(referent, referrer))
assert referent_events == [
    ("gc.get_referents", ()),
    ("gc.get_referents", (referent, referrer)),
    ("gc.get_referrers", ()),
    ("gc.get_referrers", (referent, referrer)),
]


# `referents.py` exposes the collector's raw graph separately from the
# app-level `get_referents` walk. Internal nodes are represented by a real
# traced `GcRef` wrapper, and all four raw inspector operations unwrap it.
roots = gc.get_rpy_roots()
assert isinstance(roots, list)
raw_roots = [root for root in roots if type(root) is gc.GcRef]
assert raw_roots

# A list with an item, because the raw walk reports the fields the object
# actually holds: an empty list carries no storage block, so its only referent
# is its type, which is app-level and passes through unwrapped.
raw_list_referents = gc.get_rpy_referents([object()])
assert any(type(referent) is gc.GcRef for referent in raw_list_referents)

# Bootstrap modules are immortal prebuilt objects outside the managed heap,
# but their registered type still describes the GC fields they own.  The raw
# inspector must trace those fields just as it traces a managed object's.
prebuilt_module_referents = gc.get_rpy_referents(gc)
assert prebuilt_module_referents
assert any(referent is gc.__dict__ for referent in prebuilt_module_referents)

raw = raw_roots[0]
memory_usage = gc.get_rpy_memory_usage(raw)
type_index = gc.get_rpy_type_index(raw)
assert memory_usage >= 0
assert type_index >= 1
gc.collect()
assert gc.get_rpy_memory_usage(raw) == memory_usage
assert gc.get_rpy_type_index(raw) == type_index

try:
    gc.GcRef()
except TypeError:
    pass
else:
    raise AssertionError("GcRef wrappers cannot be created by app-level code")


# referents.py:W_GcStats and app_referents.py:GcStats expose the collector's
# incminimark accounting, not a second frontend estimate.
raw_stats = gc._get_stats()
assert type(raw_stats).__name__ == "GcStats"
assert not hasattr(raw_stats, "__dict__")
for field in (
    "total_memory_pressure",
    "total_gc_memory",
    "total_allocated_memory",
    "peak_memory",
    "peak_allocated_memory",
    "jit_backend_allocated",
    "jit_backend_used",
    "total_arena_memory",
    "total_rawmalloced_memory",
    "peak_arena_memory",
    "peak_rawmalloced_memory",
    "nursery_size",
    "total_gc_time",
):
    assert isinstance(getattr(raw_stats, field), int)
assert raw_stats.total_memory_pressure == -1
assert raw_stats.total_gc_memory == (
    raw_stats.total_arena_memory
    + raw_stats.total_rawmalloced_memory
    + raw_stats.nursery_size
)
assert raw_stats.total_allocated_memory >= raw_stats.total_gc_memory
assert raw_stats.peak_memory >= raw_stats.total_gc_memory
assert raw_stats.peak_allocated_memory >= raw_stats.total_allocated_memory
assert raw_stats.peak_arena_memory >= raw_stats.total_arena_memory
assert raw_stats.peak_rawmalloced_memory >= raw_stats.total_rawmalloced_memory
assert gc._get_stats(memory_pressure=True).total_memory_pressure >= 0
try:
    type(raw_stats)()
except TypeError:
    pass
else:
    raise AssertionError("native GcStats cannot be app-level constructed")
try:
    raw_stats.total_gc_memory = 0
except AttributeError:
    pass
else:
    raise AssertionError("native GcStats fields must be readonly")

public_stats = gc.get_stats()
assert type(public_stats).__name__ == "GcStats"
assert hasattr(public_stats, "__dict__")
assert isinstance(public_stats.total_gc_memory, str)
assert isinstance(public_stats.total_gc_time, int)
assert repr(public_stats).startswith("Total memory consumed:\n")
public_copy = type(public_stats)(raw_stats)
# Compare the copy against the snapshot it was built from.  `public_stats`
# carries a `get_stats()` snapshot taken forty-odd lines earlier and
# `total_gc_memory` is a formatted size, so comparing the two asserts that no
# collection ran in between rather than that the constructor converts -- which
# is false whenever the nursery is small enough to collect there.
# `app_referents.py _format` is the conversion under test.
raw_total = raw_stats.total_gc_memory
assert public_copy.total_gc_memory == (
    "%.1fkB" % (raw_total / 1024.0)
    if raw_total < 1000000
    else "%.1fMB" % (raw_total / 1024.0 / 1024.0)
)
# `total_gc_time` is the one field the wrapper passes through unformatted.
assert public_copy.total_gc_time == raw_stats.total_gc_time

# `jit_hooks.stats_asmmemmgr_*` reads the active CPU's assembler memory
# manager. A hot loop must therefore make both native-code counters visible;
# allocated capacity is cumulative and cannot be smaller than live used bytes.
def warm_jit_for_code_stats():
    total = 0
    i = 0
    while i < 5000:
        total += i
        i += 1
    return total


before_jit_stats = gc._get_stats()
jit_total = warm_jit_for_code_stats()
assert jit_total == 12497500
jit_stats = gc._get_stats()
assert jit_stats.jit_backend_allocated >= jit_stats.jit_backend_used > 0
# Tie the counters to this loop rather than to whatever the process emitted
# earlier: a bare `> 0` would still pass if the warm-up compiled nothing.
assert jit_stats.jit_backend_used > before_jit_stats.jit_backend_used
assert jit_stats.jit_backend_allocated >= before_jit_stats.jit_backend_allocated


# `hook.py:LowLevelGcHooks` only aggregates counters and fires three
# nonperiodic actions. Python callbacks therefore run after the collector has
# returned, never from inside its no-collect region.
assert type(gc.hooks).__name__ == "GcHooks"
assert not hasattr(gc.hooks, "__dict__")
assert gc.hooks.on_gc_minor is None
assert gc.hooks.on_gc_collect_step is None
assert gc.hooks.on_gc_collect is None

minor_events = []
step_events = []
collect_events = []


def on_minor(stats):
    assert type(stats).__name__ == "GcMinorStats"
    assert not hasattr(stats, "__dict__")
    assert stats.count >= 1
    assert 0.0 <= stats.duration_min <= stats.duration_max <= stats.duration
    assert stats.total_memory_used >= 0
    assert stats.pinned_objects >= 0
    minor_events.append(stats)


def on_step(stats):
    assert type(stats) is gc.GcCollectStepStats
    assert not hasattr(stats, "__dict__")
    assert stats.count >= 1
    assert 0 <= stats.oldstate < 4
    assert 0 <= stats.newstate < 4
    assert stats.major_is_done == (stats.newstate == stats.STATE_SCANNING)
    step_events.append(stats)


def on_collect(stats):
    assert type(stats).__name__ == "GcCollectStats"
    assert not hasattr(stats, "__dict__")
    assert stats.count >= 1
    assert stats.num_major_collects >= 1
    assert stats.arenas_count_before >= 0
    assert stats.arenas_count_after >= 0
    assert stats.arenas_bytes >= 0
    assert stats.rawmalloc_bytes_before >= 0
    assert stats.rawmalloc_bytes_after >= 0
    assert stats.pinned_objects >= 0
    collect_events.append(stats)


class Hooks:
    on_gc_minor = on_minor
    on_gc_collect_step = on_step
    on_gc_collect = on_collect


gc.hooks.set(Hooks)
gc.collect()
assert minor_events and step_events and collect_events

# `set()` fetches every attribute before changing any callback.
class IncompleteHooks:
    on_gc_minor = 1
    on_gc_collect_step = 2


try:
    gc.hooks.set(IncompleteHooks())
except AttributeError:
    pass
else:
    raise AssertionError("GcHooks.set must fetch all three attributes")
assert gc.hooks.on_gc_minor is on_minor
assert gc.hooks.on_gc_collect_step is on_step
assert gc.hooks.on_gc_collect is on_collect

# Recursive collection re-fires the action for a later opcode instead of
# recursively invoking the same Python callback.
recursive_steps = []
did_recurse = [False]


def recursive_step(stats):
    recursive_steps.append(stats.count)
    if not did_recurse[0]:
        did_recurse[0] = True
        gc.collect_step()


gc.hooks.on_gc_collect_step = recursive_step
for i in range(6):
    gc.collect_step()
    opcode_boundary = i + 1
assert did_recurse[0]
assert recursive_steps

# Setters deliberately accept arbitrary objects; reset restores all three
# slots to None.
gc.hooks.on_gc_minor = 42
assert gc.hooks.on_gc_minor == 42
gc.hooks.reset()
assert gc.hooks.on_gc_minor is None
assert gc.hooks.on_gc_collect_step is None
assert gc.hooks.on_gc_collect is None

# `space.fromcache(W_AppLevelHooks)` owns the singleton independently of the
# module dictionary.  Removing the public module reference must therefore not
# strand the AsyncAction pointers or let the callback owner be collected.
lifetime_events = []


def lifetime_collect(stats):
    lifetime_events.append(stats.count)


gc.hooks.on_gc_collect = lifetime_collect
del lifetime_collect
del gc.hooks
gc.collect()
assert lifetime_events


print("OK")
