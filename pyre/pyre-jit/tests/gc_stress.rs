//! Consolidated GC-stress integration tests (formerly six separate test
//! binaries: `instance_gc_stress`, `list_elements_gc_stress`,
//! `list_young_dict_gc_stress`, `method_cache_gc_stress`,
//! `type_namespace_gc_stress`, `weak_subclasses_gc_stress`). Merging them into
//! one binary avoids six heavy relinks of the `pyre_jit` + interpreter + object
//! stack in CI.
//!
//! Every `#[test]` runs one end-to-end Python program through the shared
//! `run_harness`. `run_on_worker` re-execs this binary so each test is its own
//! process, with that process's own GC. Inside the child, the program runs on
//! a freshly spawned 256 MiB worker thread, which keeps stack-heavy interpreter
//! / tracer recursion off the default test stack.
//!
//! The harness follows the `pyrex` launcher startup (`pyrex::real_main`, then
//! `run_source`). Each program's module body and functions use `while`
//! loops (no `FOR_ITER`), so eval reaches a JIT-eligible frame. `gc.collect()`
//! then forces deterministic collections through the collect hook
//! (`interp_gc.py collect`).
//!
//! Non-vacuity is asserted AFTER eval: the stable instance allocator hook
//! must be live, proving the program's objects routed through the real managed
//! heap rather than the leaking `lltype::malloc` Box fallback (which would make
//! the survival checks meaningless).
//!
//! This binary is also the crate's only end-to-end Python path that runs with
//! `debug_assertions` on. One regression test uses it for the non-GC invariant
//! that the virtualizable shadow and heap agree when
//! `check_synchronized_virtualizable` runs.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

/// Nursery-born rbigint digits move; every long op that allocates while
/// still reading an operand must have that operand on the shadow stack.
#[test]
fn long_ops_keep_operands_across_collecting_digit_allocs() {
    const CHILD: &str = "PYRE_LONG_OPS_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "long_ops_keep_operands_across_collecting_digit_allocs",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated long-op GC-root regression");
        assert!(
            output.status.success(),
            "long op lost an operand across a collecting alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
LEFT = 10 ** 50 + 7
RIGHT = 10 ** 25 + 3
EXPECTED_TRUEDIV = LEFT / RIGHT
EXPECTED_FLOORDIV = LEFT // RIGHT
EXPECTED_MOD = LEFT % RIGHT
EXPECTED_DIVMOD = divmod(LEFT, RIGHT)
EXPECTED_POWMOD = pow(LEFT, 5, RIGHT)
EXPECTED_LSHIFT = LEFT << 7
EXPECTED_RSHIFT = LEFT >> 7
EXPECTED_FLOAT = float(LEFT)
EXPECTED_HASH = hash(LEFT)
EXPECTED_STR = str(LEFT)

def check_once():
    assert LEFT / RIGHT == EXPECTED_TRUEDIV
    assert LEFT // RIGHT == EXPECTED_FLOORDIV
    assert LEFT % RIGHT == EXPECTED_MOD
    assert divmod(LEFT, RIGHT) == EXPECTED_DIVMOD
    assert pow(LEFT, 5, RIGHT) == EXPECTED_POWMOD
    assert (LEFT << 7) == EXPECTED_LSHIFT
    assert (LEFT >> 7) == EXPECTED_RSHIFT
    assert float(LEFT) == EXPECTED_FLOAT
    assert hash(LEFT) == EXPECTED_HASH
    assert str(LEFT) == EXPECTED_STR

i = 0
while i < 80:
    check_once()
    _ = [0] * 32
    i = i + 1

def f_truediv_long(n):
    big = 1 << 200
    s = 0.0
    i = 1
    while i <= n:
        s += big / (big + i)
        i += 1
    return s

got = int(round(f_truediv_long(5000) * 1e6))
assert got == 5000000000, got
"#,
        "long_ops_gc_roots.py",
        "long-op operand roots",
        "long ops must keep both operands alive across Digits::new",
    );
}

/// `math.log10` of a long that does not fit in a float. `tofloat` collects
/// and moves the payload; the logarithm must re-read it afterwards.
#[test]
fn log10_rereads_long_payload_after_tofloat() {
    const CHILD: &str = "PYRE_LOG10_LONG_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "log10_rereads_long_payload_after_tofloat",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated log10 GC-root regression");
        assert!(
            output.status.success(),
            "log10 read a moved long payload:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import math
# Each iteration builds a fresh young payload and takes the overflow arm on
# the next statement, so a payload the conversion moves is read back while it
# is still young.
# The scaled-double arm is not correctly rounded, so compare with a
# tolerance: a payload read after it moved is an exception or a value that
# is wrong by orders of magnitude, never by an ulp.
ln10 = math.log(10.0)
for e in range(309, 1200):
    value = 10 ** e
    got = math.log10(value)
    assert abs(got - e) < 1e-9, (e, got)
    got = math.log(value)
    assert abs(got - e * ln10) < 1e-6, (e, got)
"#,
        "log10_long_gc.py",
        "log10 long payload",
        "math.log10 must re-read a long after tofloat collects",
    );
}

/// `ResumeGuardForcedDescr.handle_async_forcing` runs before the residual
/// returns. Its virtualizable image must not read the pending return value
/// from a register home that compiled code has not written yet.
#[test]
fn pickle_async_forcing_keeps_pending_result_out_of_frame() {
    const CHILD: &str = "PYRE_PICKLE_ASYNC_FORCE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Set nursery poisoning at process startup, not by mutating the
        // environment underneath the other Rust test workers. The ordinary
        // nursery can accidentally give an unwritten result home a zero.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "pickle_async_forcing_keeps_pending_result_out_of_frame",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated async-forcing regression");
        assert!(
            output.status.success(),
            "async forcing read an unavailable call result:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    // test_bytes makes PyPicklerTests.loads hot. NEWOBJ_EX subsequently
    // enters Python from its residual Unpickler.load and forces that frame.
    // Collect after loads has returned; before the fix,
    // its old locals array contained nursery poison above valuestackdepth.
    run_on_worker(
        r#"
import gc
try:
    from test.test_pickle import CPicklerTests

    case = CPicklerTests('test_bytes')
    case.test_bytes()
    case.test_complex_newobj_ex()
    gc.collect()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
"#,
        "pickle_async_forcing.py",
        "async-forcing frame slots",
        "pickle async forcing must preserve a valid frame image",
    );
}

/// Shared harness body for every GC-stress program. Compiles and runs `program`
/// (using `name` as its `sys.argv[0]` / filename) with the same startup as
/// `pyrex::real_main` and `run_source`, on the process's own GC. An uncaught
/// `assert` in the program surfaces here as `Err`, so a successful return
/// means every read-back assertion held. `vacuity_label` names the survival
/// checks for the post-eval non-vacuity error.
fn run_harness(program: &str, name: &str, vacuity_label: &str) -> Result<(), String> {
    // Mirror `pyrex::real_main` startup, then `pyrex::run_source`.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    pyre_module::register();
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd, Some(cwd.as_os_str()));
    // This harness never imports `site`, so perform the post-site `sys.path[0]`
    // insert directly.
    importing::add_sys_path_0();
    importing::set_sys_argv(&[std::ffi::OsString::from(name)]);

    let code = compile_source_with_filename(program, Mode::Exec, name)
        .map_err(|e| format!("compile error: {e}"))?;

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .map_err(|e| format!("frame setup error: {}", e.message_text()))?;

    // Reuse the canonical globals dict as the __main__ module's dict so
    // `globals()` / `function.__globals__` share one identity
    // (`run_source` parity).
    let canonical = frame.get_w_globals();
    let main_module =
        pyre_object::w_module_new_aliasing_dict(pyre_object::w_str_new("__main__"), canonical);
    importing::set_sys_module("__main__", main_module);

    // An uncaught `assert` in the program surfaces here as `Err`, so a
    // successful return means every read-back assertion held.
    eval_with_jit(&mut frame, None)
        .map_err(|e| format!("execution error: {}", e.message_text()))?;

    // Non-vacuity: the stable instance allocator hook is installed by the
    // `JIT_DRIVER` initializer (`driver_pair` -> `init_gc_subsystem` ->
    // `install_gc_standalone`, which registers the hooks against the `gc_sync`
    // singleton and stores no per-thread box). If it is live now, allocations
    // routed through the managed heap rather than the leaking Box fallback, so
    // the survival checks were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_OBJECT_OBJECT_GC_TYPE_ID,
        pyre_object::W_OBJECT_OBJECT_SIZE,
    )
    .ok_or_else(|| {
        format!("stable GC alloc hook was not live; {vacuity_label} would be vacuous")
    })?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null for an instance-sized block".to_string());
    }
    // The probe block is never rooted; zero it so any later sweep reads a
    // well-formed (null map/storage) payload before reclaiming it.
    unsafe {
        std::ptr::write_bytes(probe as *mut u8, 0, pyre_object::W_OBJECT_OBJECT_SIZE);
    }
    Ok(())
}

/// Run one test in its own process, then `run_harness` on a 256 MiB worker.
///
/// Unless `PYRE_GC_STRESS_CHILD` is set, re-exec this binary with `--exact`
/// for the libtest thread name, `--nocapture`, and `--test-threads=1`. The
/// child has the variable set and runs `run_harness` on the worker thread, so
/// deep tracer / interpreter recursion stays off the default test stack. The
/// child builds one GC through the same startup as `pyrex::real_main`.
fn run_on_worker(
    program: &'static str,
    name: &'static str,
    vacuity_label: &'static str,
    fail_msg: &'static str,
) {
    if std::env::var_os("PYRE_GC_STRESS_CHILD").is_none() {
        let current = std::thread::current();
        let test_name = current.name();
        assert!(test_name.is_some(), "libtest names the test thread");
        let test_name = test_name.unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
            .env("PYRE_GC_STRESS_CHILD", "1")
            .output()
            .expect("re-exec gc_stress test in its own process");
        assert!(
            output.status.success(),
            "{fail_msg}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_harness(program, name, vacuity_label))
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect(fail_msg);
}

/// Class and `exec(code, ns)` bodies use dict-scope frames, whose LOAD_NAME /
/// STORE_NAME traffic reads Ref static virtualizable fields. In debug builds,
/// that path checks that the virtualizable shadow and heap copy stay
/// synchronized after tracing mirrors static fields into boxes.
#[test]
fn class_body_and_exec_loops_keep_the_vable_shadow_synchronized() {
    const PROGRAM: &str = r#"
class Namespace(dict):
    pass

CODE = """
i = 0
total = 0
while i < 3000:
    total = total + i
    i = i + 1
"""

def class_bodies():
    acc = 0
    repeat = 0
    while repeat < 3:
        class Probe:
            i = 0
            total = 0
            while i < 3000:
                total = total + i
                i = i + 1
        assert Probe.total == 4498500, Probe.total
        assert Probe.i == 3000, Probe.i
        acc = acc + Probe.total + Probe.i
        repeat = repeat + 1
    return acc

def exec_bodies():
    acc = 0
    repeat = 0
    while repeat < 3:
        plain = {}
        exec(CODE, plain)
        assert plain["total"] == 4498500, plain["total"]
        assert plain["i"] == 3000, plain["i"]
        acc = acc + plain["total"] + plain["i"]

        subclass = Namespace()
        exec(CODE, subclass)
        assert subclass["total"] == 4498500, subclass["total"]
        assert subclass["i"] == 3000, subclass["i"]
        acc = acc + subclass["total"] + subclass["i"]

        repeat = repeat + 1
    return acc

class_result = class_bodies()
exec_result = exec_bodies()
assert class_result == 13504500, class_result
assert exec_result == 27009000, exec_result
assert class_result + exec_result == 40513500
"#;
    run_on_worker(
        PROGRAM,
        "<vable_shadow_debug_gate>",
        "virtualizable shadow synchronization checks",
        "virtualizable shadow debug gate program failed",
    );
}

/// `W_ObjectObject` is GC-managed: an instance's movable attribute values
/// (list / str / dict, reachable only through mapdict storage slots), a
/// devolved instance's stored values (reached via the storage back-edge), and a
/// materialised `__dict__` view all survive repeated full collections, while
/// dead throwaway instances are reclaimed each round.
///
/// `a` is a non-devolved instance whose list / str / dict attr values live in
/// mapdict storage slots; `b` devolves past the attribute limit, with its
/// `__dict__` materialised while live (`view`) and kept rooted across the
/// collections. Each round allocates fresh garbage and a dead throwaway
/// instance. The returned checksum is reachable only if every live value
/// survived the 100 collections.
#[test]
fn instance_attrs_survive_full_collection() {
    const PROGRAM: &str = r#"
import gc

class A:
    pass

class B:
    pass

def run():
    a = A()
    a.lst = [1, 2, 3, 4, 5]
    a.s = "hello" * 10
    a.d = {"k": [9, 8, 7]}

    b = B()
    i = 0
    while i < 85:
        setattr(b, "f%d" % i, [i, i + 1, i + 2])
        i = i + 1
    view = b.__dict__

    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = A()
        tmp.q = [7, 7]
        gc.collect()
        n = n + 1

    total = a.lst[0] + a.lst[4] + len(a.s) + a.d["k"][2]
    i = 0
    while i < 85:
        v = getattr(b, "f%d" % i)
        total = total + v[0] + v[1] + v[2]
        i = i + 1
    total = total + len(view) + view["f0"][0]
    return total

result = run()
assert result == 11113, result
"#;
    run_on_worker(
        PROGRAM,
        "<instance_gc_stress>",
        "instance survival checks",
        "instance gc stress program failed",
    );
}

/// An object-strategy `W_ListObject`'s elements are GC-traced: instances
/// reachable ONLY through a list survive repeated full collections. Regression
/// for the `W_LIST_GC_TYPE_ID` registration that traced `items` as a single
/// non-managed pointer (the `std::alloc`'d `ItemsBlock`) and never reached the
/// elements — a major collection then swept a list element reachable only via
/// the list. Fixed by giving `W_ListObject` a custom trace
/// (`list_object_custom_trace`) that walks the off-GC block, mirroring
/// `W_TupleObject` / `W_SetObject`.
///
/// `objs` (a list literal -> Object strategy) and `grown` (built by `append`,
/// exercising object-strategy growth) hold `Node` instances reachable ONLY
/// through their list. The returned checksum is reachable only if every element
/// survived the 100 collections.
#[test]
fn list_elements_survive_full_collection() {
    const PROGRAM: &str = r#"
import gc

class Node:
    pass

def run():
    objs = [Node(), Node(), Node()]
    objs[0].v = 10
    objs[1].v = 20
    objs[2].v = 30

    grown = []
    i = 0
    while i < 12:
        e = Node()
        e.v = i
        grown.append(e)
        i = i + 1

    n = 0
    while n < 100:
        junk = [0] * 64
        gc.collect()
        n = n + 1

    total = objs[0].v + objs[1].v + objs[2].v
    i = 0
    while i < 12:
        total = total + grown[i].v
        i = i + 1
    return total

result = run()
assert result == 126, result
"#;
    run_on_worker(
        PROGRAM,
        "<list_elements_gc_stress>",
        "list element survival checks",
        "list elements gc stress program failed",
    );
}

/// Storing a *young* (nursery) value into an object-strategy `W_ListObject`
/// keeps it alive across a minor GC. Once the list body has survived and moved
/// to old-gen, a minor collection forwards its later young refs ONLY when the
/// container sits in the remembered set, populated by the write barrier
/// (`try_gc_write_barrier`).
/// Regression for the missing list write barriers in `w_list_append` /
/// `w_list_setitem` / `w_list_insert` / `w_list_setslice` and
/// `w_list_new_with_strategy` creation. Proven non-vacuous: without the
/// barriers the checksum is corrupted / the process faults; with them it is
/// exact.
///
/// Each `{}` is a nursery `W_DICT` reachable only through its list. The four
/// list mutators that store a ref — append, literal creation, setitem, insert —
/// must each barrier a promoted list so the next minor GC forwards the young
/// dict. The checksum is recoverable only if every stored dict survives the 200
/// collections.
#[test]
fn young_dict_list_elements_survive_minor_collection() {
    const PROGRAM: &str = r#"
import gc

def run():
    # (1) append: young dicts into an object-strategy list
    appended = []
    i = 0
    while i < 16:
        d = {}
        d['v'] = i
        appended.append(d)
        i = i + 1

    # (2) creation: object-strategy list literal of young dicts
    literal = [{}, {}, {}]
    literal[0]['v'] = 10
    literal[1]['v'] = 20
    literal[2]['v'] = 30

    # (3) setitem + insert of young dicts
    slots = [{}, {}]
    slots[0]['v'] = 1
    slots[1] = {}
    slots[1]['v'] = 2
    slots.insert(1, {})
    slots[1]['v'] = 3

    n = 0
    while n < 200:
        junk = [0] * 32
        scratch = {}
        scratch['x'] = n
        gc.collect()
        n = n + 1

    total = 0
    i = 0
    while i < 16:
        total = total + appended[i]['v']
        i = i + 1
    total = total + literal[0]['v'] + literal[1]['v'] + literal[2]['v']
    total = total + slots[0]['v'] + slots[1]['v'] + slots[2]['v']
    return total

result = run()
assert result == 186, result
"#;
    run_on_worker(
        PROGRAM,
        "<list_young_dict_gc_stress>",
        "young-dict survival checks",
        "list young dict gc stress program failed",
    );
}

/// Interpreter method cache (`baseobjspace::MethodCache`,
/// `typeobject.py:516-552`) correctness under heavy collection: repeated
/// interpreter-mode method lookups, interleaved with full collections, keep
/// returning the correct functions — proving `version_tag` invalidation and the
/// cached results stay consistent and uncorrupted across `gc.collect()`s. The
/// cache is `not we_are_jitted()`-gated, so `run()`'s early loop iterations run
/// in the interpreter (before the trace threshold) and fill / hit the cache
/// between collections. This guards cache *correctness*, not rooting in
/// isolation.
///
/// `C` is a user heap type with three methods. `run()`'s early (interpreter)
/// iterations cache the `m0` lookup and call it across collections; `tail()`
/// then does interpreter-mode hits on all three methods after the GC is warm. A
/// stale cached function pointer would SIGSEGV on the call, or return a wrong
/// value the assertions catch.
#[test]
fn method_cache_stays_correct_across_collections() {
    const PROGRAM: &str = r#"
import gc

class C:
    def m0(self):
        return 1
    def m1(self):
        return 10
    def m2(self):
        return 100

def run():
    c = C()
    total = 0
    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = C()
        total = total + c.m0()
        gc.collect()
        n = n + 1
    return total

def tail():
    c = C()
    acc = 0
    k = 0
    while k < 30:
        acc = acc + c.m0() + c.m1() + c.m2()
        gc.collect()
        k = k + 1
    return acc

warm = run()
acc = tail()
assert warm == 100, warm
assert acc == 30 * 111, acc
result = warm + acc
assert result == 3430, result
"#;
    run_on_worker(
        PROGRAM,
        "<method_cache_gc_stress>",
        "method-cache survival check",
        "method cache gc stress program failed",
    );
}

/// A heap type's namespace is GC-rooted: a user class's method (a function), its
/// class attribute (a movable list), and the per-type `__dict__` getset
/// descriptor (whose `fget` is a collectable function) all survive repeated
/// full collections, even when first reached *fresh* after the collections.
/// Heap type objects (`w_type_new`) are GC-managed, so their
/// `W_TYPE_GC_TYPE_ID` custom trace reaches the movable values bound in the
/// type's namespace `DictStorage`. Immortal builtin types reach the same
/// custom trace through `prebuilt_root_objects` after their construction
/// write barrier.
///
/// `C`'s namespace dict holds a method (`method`), a class attribute
/// (`KLASS_ATTR`), and — once `c.__dict__` is first read — the copied
/// `__dict__` getset descriptor. The returned checksum is reachable only if
/// every namespace value (and the descriptor's getter) survived.
#[test]
fn type_namespace_survives_full_collection() {
    const PROGRAM: &str = r#"
import gc

class C:
    KLASS_ATTR = [10, 20, 30]
    def method(self):
        return 7

def run():
    c = C()
    c.x = 5
    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = C()
        gc.collect()
        n = n + 1
    total = c.method()
    total = total + C.KLASS_ATTR[1]
    d = c.__dict__
    total = total + len(d) + d["x"]
    return total

result = run()
assert result == 33, result
"#;
    run_on_worker(
        PROGRAM,
        "<type_namespace_gc_stress>",
        "namespace survival checks",
        "type namespace gc stress program failed",
    );
}

/// A heap type's `weak_subclasses` list is GC-rooted: a subclass recorded in its
/// base's `weak_subclasses` survives repeated full collections so the base's
/// `mutated()` walk still reaches it. `w_type_add_subclass` stores
/// `w_weakref_new(subclass)` — a `try_gc_alloc` young WEAKREF GcStruct — in the
/// base's off-GC `weak_subclasses`. Before this fix the type root walk
/// forwarded `bases` and namespace values but NOT `weak_subclasses`, so the
/// first collection reclaimed the WEAKREF and `type.__setattr__`'s `mutated()`
/// walk ran over the freed slot (a UAF that dropped cache invalidation).
///
/// Observable: `mutated()` resets each live subclass's cached
/// `compares_by_identity_status` to UNKNOWN. The program caches `B`'s status as
/// YES (using `B` instances as identity dict keys), churns collections, then
/// defines `A.__eq__` / `A.__hash__`. If `B`'s weakref survived, its status is
/// reset and fresh `B` keys recompute to "compare by value" -> a 2-key dict
/// collapses to 1 entry. A dangling weakref leaves the stale YES (or crashes on
/// deref). `B` is defined *inside* `run()` after the GC is live so its
/// `weak_subclasses` weakref is a real young GC object.
#[test]
fn weak_subclasses_survive_full_collection() {
    const PROGRAM: &str = r#"
import gc

class A:
    pass

def run():
    n = 0
    while n < 40:
        junk = [0] * 50
        gc.collect()
        n = n + 1
    # GC is live now: B's weakref in A.weak_subclasses is a young GC object.
    class B(A):
        pass
    b1 = B()
    b2 = B()
    seed = {}
    seed[b1] = 1
    seed[b2] = 2                 # distinct by identity -> caches B status = YES
    primed = len(seed)           # 2
    m = 0
    while m < 40:
        junk = [0] * 50
        tmp = A()
        gc.collect()
        m = m + 1
    def beq(self, other):
        return True
    def bhash(self):
        return 7
    A.__eq__ = beq               # type.__setattr__ -> A.mutated('__eq__')
    A.__hash__ = bhash           # must reach B via weak_subclasses
    b3 = B()
    b4 = B()
    d = {}
    d[b3] = 10
    d[b4] = 20                   # compare by value (all equal) -> 1 entry
    return primed * 10 + len(d)

result = run()
assert result == 21, result
"#;
    run_on_worker(
        PROGRAM,
        "<weak_subclasses_gc_stress>",
        "weak_subclasses survival check",
        "weak_subclasses gc stress program failed",
    );
}

/// A trace-built exception raised with a bare-class `from` cause survives a
/// collection during construction and cause normalization. The generated walk
/// carries concrete exceptions on recorder boxes and the active exception
/// slots; it no longer has the retired PyreSym.trace_built_exc map. Preserve
/// this end-to-end assertion independently of that former implementation.
/// Under `MAJIT_GC_STRESS`, cause normalization also exercises allocating
/// safepoints before the raised instance reaches the handler.
#[test]
fn trace_built_exc_from_class_cause_survives_collection() {
    const PROGRAM: &str = r#"
import gc

def run():
    total = 0
    n = 0
    while n < 300:
        junk = [0] * 16
        try:
            raise ValueError(n) from KeyError
        except ValueError as e:
            total = total + e.args[0]
        gc.collect()
        n = n + 1
    return total

result = run()
assert result == 44850, result
"#;
    run_on_worker(
        PROGRAM,
        "<trace_built_exc_from_class_cause_gc_stress>",
        "trace-built exception window-A survival check",
        "trace-built exc from-class-cause gc stress program failed",
    );
}

/// A bare-class exception raised with a bare-class `from` cause exercises window
/// B: the normalized cause (`RuntimeError`) lives only in a Rust local across the
/// class-exception instantiation (`call_function(ValueError, &[])`), a GC
/// safepoint, before `attach_raise_cause` reads it. The `push_roots` /
/// `pin_root(cause)` bracket keeps the cause alive so `__cause__` is a live
/// `RuntimeError`; without it that read is against a swept block. Under
/// `MAJIT_GC_STRESS` the instantiation deterministically collects inside the
/// window.
#[test]
fn class_exc_from_class_cause_keeps_cause_across_collection() {
    const PROGRAM: &str = r#"
import gc

def run():
    ok = 0
    n = 0
    while n < 300:
        junk = [0] * 16
        try:
            raise ValueError from RuntimeError
        except ValueError as e:
            if type(e.__cause__) is RuntimeError:
                ok = ok + 1
        gc.collect()
        n = n + 1
    return ok

result = run()
assert result == 300, result
"#;
    run_on_worker(
        PROGRAM,
        "<class_exc_from_class_cause_gc_stress>",
        "trace-built exception window-B cause survival check",
        "class-exc from-class-cause gc stress program failed",
    );
}

/// A `StopIteration` leaking out of a generator body (PEP 479) is converted to
/// `RuntimeError("generator raised StopIteration")` by `leak_stopiteration`. The
/// leaked instance lives only in a Rust local (`w_stopiter`) across the
/// `RuntimeError` allocation, a GC safepoint, before it is stamped onto the
/// `RuntimeError` as `__cause__` / `__context__`. The `push_roots` /
/// `pin_root(w_stopiter)` bracket keeps it alive so `__cause__` is a live
/// `StopIteration`; without it that read is against a swept block. Under
/// `MAJIT_GC_STRESS` the `RuntimeError` instantiation deterministically collects
/// inside the window.
#[test]
fn generator_leaked_stopiteration_cause_survives_collection() {
    const PROGRAM: &str = r#"
import gc

def gen():
    raise StopIteration
    yield

def run():
    ok = 0
    n = 0
    while n < 300:
        junk = [0] * 16
        try:
            next(gen())
        except RuntimeError as e:
            if type(e.__cause__) is StopIteration:
                ok = ok + 1
        gc.collect()
        n = n + 1
    return ok

result = run()
assert result == 300, result
"#;
    run_on_worker(
        PROGRAM,
        "<generator_leaked_stopiteration_gc_stress>",
        "leaked-StopIteration cause survival check",
        "generator leaked-StopIteration gc stress program failed",
    );
}

/// `generator.py _invoke_execute_frame` / `pyframe.py execute_frame` keep
/// `w_arg_or_err` live from entry through `stack_check`, `call_trace` and
/// `resume_execute_frame`. A young send/throw payload is copied into
/// `FrameResumeArgs`; a moving collection in that span leaves the copy stale
/// unless the value is re-read from its shadow-stack slot after every
/// allocating call. The child uses the tree's nursery-stress harness
/// (`PYPY_GC_NURSERY=1`): a one-byte nursery collects on almost every
/// alloc, so a nursery-born list sent or thrown through the generator must
/// come back as the same object with its contents intact.
#[test]
fn generator_send_throw_keeps_young_value_across_collection() {
    const CHILD: &str = "PYRE_GENERATOR_RESUME_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "generator_send_throw_keeps_young_value_across_collection",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated generator-resume GC-root regression");
        assert!(
            output.status.success(),
            "generator send/throw lost a young value across a collecting alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    const PROGRAM: &str = r#"
def echo():
    v = None
    while True:
        try:
            v = yield v
        except ValueError as e:
            v = e.args[0]

def run():
    g = echo()
    g.send(None)
    ok = 0
    n = 0
    while n < 300:
        junk = [0] * 16
        payload = [n, n + 1, n + 2]
        got = g.send(payload)
        if got is payload and got[0] == n and got[2] == n + 2:
            ok = ok + 1
        thrown = [n + 100, n + 101]
        got = g.throw(ValueError(thrown))
        if got is thrown and got[0] == n + 100 and got[1] == n + 101:
            ok = ok + 1
        n = n + 1
    return ok

result = run()
assert result == 600, result
"#;
    run_on_worker(
        PROGRAM,
        "<generator_send_throw_young_value_gc_stress>",
        "generator send/throw young-value survival check",
        "generator send/throw young-value gc stress program failed",
    );
}

/// `pyopcode.py` `FOR_ITER` absorbs `StopIteration`. A generator loop that
/// yields from a nested generator must keep doing that after the loop is
/// traced: the inner generator's return is exhaustion, not
/// `RuntimeError("generator raised StopIteration")`.
#[test]
fn generator_for_iter_absorbs_inner_stopiteration() {
    const PROGRAM: &str = r#"
def inner():
    yield 1
    yield 2

def outer():
    for y in inner():
        yield y

def run():
    n = 0
    while n < 2000:
        got = list(outer())
        if got != [1, 2]:
            return got
        n = n + 1
    return "OK"

result = run()
assert result == "OK", result
"#;
    run_on_worker(
        PROGRAM,
        "<generator_for_iter_absorbs_inner_stopiteration>",
        "nested generator FOR_ITER absorption",
        "generator FOR_ITER leaked StopIteration",
    );
}

/// A `bytes` / `bytearray` object's `data` buffer is a GC-managed leaf storage
/// box (off-GC storage): `bytes_object_custom_trace` /
/// `bytearray_object_custom_trace` grey it through the `data` field slot, and
/// the box tid's drop glue reclaims it on sweep. Regression for two failure
/// modes the migration off `malloc_typed`-immortal holders could introduce: a
/// buffer swept while its still-live object is reachable only through a list
/// (UAF on read-back), and a per-round dead throwaway whose buffer leaks or is
/// double-freed. `bytearray` extension after the collections re-derives the box
/// pointer, proving the box survived and stayed writable.
///
/// `blobs` (bytes reachable only through a list), `ba` (a live bytearray grown
/// after the collections), and `pieces` (bytearray fragments held only through a
/// list) must all survive the 100 collections. `slot_ba.x` additionally lives
/// only in the `W_BytearrayObjectUser` mapdict storage traced by
/// `bytearray_user_object_custom_trace`; its ordinary `z` attribute must remain in
/// the separate instance dict. Each round allocates fresh dead bytes/bytearray
/// garbage. The returned checksum is reachable only if every live edge and
/// buffer survived intact.
#[test]
fn bytes_bytearray_buffers_survive_full_collection() {
    const PROGRAM: &str = r#"
import gc

def run():
    class SlotBA(bytearray):
        __slots__ = ("x", "__dict__")

    blobs = []
    i = 0
    while i < 12:
        blobs.append(b"abcde")
        i = i + 1

    ba = bytearray(b"seed")
    slot_ba = SlotBA(b"slot-owner")
    slot_ba.x = bytearray(b"slot-value")
    slot_ba.z = 17
    pieces = []
    i = 0
    while i < 8:
        pieces.append(bytearray(b"xy"))
        i = i + 1

    n = 0
    while n < 100:
        junk = b"garbage" * 4
        tmp = bytearray(b"throwaway") * 3
        gc.collect()
        n = n + 1

    ba.extend(b"-grown")
    assert slot_ba.__dict__ == {"z": 17}
    assert slot_ba.x == bytearray(b"slot-value")

    total = 0
    i = 0
    while i < 12:
        total = total + len(blobs[i]) + blobs[i][0]
        i = i + 1
    i = 0
    while i < 8:
        total = total + len(pieces[i]) + pieces[i][0]
        i = i + 1
    total = total + len(ba) + ba[0]
    return total

result = run()
assert result == 2325, result
"#;
    run_on_worker(
        PROGRAM,
        "<bytes_bytearray_gc_stress>",
        "bytes/bytearray buffer survival checks",
        "bytes/bytearray gc stress program failed",
    );
}

/// A `str` SUBCLASS instance's WTF-8 `value` buffer is a GC-managed leaf storage
/// box (off-GC storage): the mortal subclass holder is GC-swept and its
/// `value` gc-pointer edge greys the box, whose tid drop glue reclaims the
/// buffer on sweep. Regression for the migration off the per-type
/// `unicode_object_destructor`: a subclass string reachable only through a list
/// must keep its value buffer across collections (UAF on read-back if the box is
/// swept early), and per-round dead subclass strings must be reclaimed without a
/// double-free (the box drop glue is the sole reclaimer). Exact strings keep an
/// immortal `malloc_raw` value and are unaffected — the test mixes both.
///
/// `subs` holds `S` (a str subclass) instances reachable only through a list;
/// each carries its own boxed value. The returned checksum reads them back after
/// 100 collections, so it is reachable only if every subclass value survived.
#[test]
fn str_subclass_value_survives_full_collection() {
    const PROGRAM: &str = r#"
import gc

class S(str):
    pass

def run():
    subs = []
    i = 0
    while i < 12:
        subs.append(S("payload"))
        i = i + 1

    n = 0
    while n < 100:
        junk = S("throwaway garbage")
        plain = "exact" * 3
        gc.collect()
        n = n + 1

    total = 0
    i = 0
    while i < 12:
        s = subs[i]
        total = total + len(s) + ord(s[0])
        i = i + 1
    return total

result = run()
assert result == 1428, result
"#;
    run_on_worker(
        PROGRAM,
        "<str_subclass_value_gc_stress>",
        "str subclass value survival checks",
        "str subclass value gc stress program failed",
    );
}

/// A mortal (user) `Function`'s `name` string and a heap `W_TypeObject`'s `name`
/// string are GC-managed leaf storage boxes shared under one `NameStorage` tid
/// (off-GC storage): the function's `FUNCTION_NAME_OFFSET` gc-pointer
/// edge and the type's `type_object_custom_trace` name-slot greying keep the box
/// live, and the box tid's drop glue is the sole reclaimer. Regression for the
/// migration off `function_object_destructor` / the type destructor's name-free:
/// a function/type reachable only through a list must keep its name across
/// collections (UAF on `__name__` read-back if swept early), the `__name__`
/// setter must rebox on a mortal holder, and per-round dead functions/types must
/// be reclaimed without a double-free.
///
/// `fns` and `types` are reachable only through lists; each function's name is
/// reassigned via the setter. The checksum reads every `__name__` back after
/// 100 collections, so it is reachable only if every name box survived.
#[test]
fn function_type_name_survives_full_collection() {
    const PROGRAM: &str = r#"
import gc

def make_funcs():
    fns = []
    i = 0
    while i < 10:
        def f():
            return 1
        f.__name__ = "renamed"
        fns.append(f)
        i = i + 1
    return fns

def make_types():
    types = []
    i = 0
    while i < 10:
        class C:
            pass
        types.append(C)
        i = i + 1
    return types

def run():
    fns = make_funcs()
    types = make_types()

    n = 0
    while n < 100:
        def junk():
            return 0
        class Junk:
            pass
        gc.collect()
        n = n + 1

    total = 0
    i = 0
    while i < 10:
        total = total + len(fns[i].__name__)
        total = total + len(types[i].__name__)
        i = i + 1
    return total

result = run()
assert result == 80, result
"#;
    run_on_worker(
        PROGRAM,
        "<function_type_name_gc_stress>",
        "function/type name survival checks",
        "function/type name gc stress program failed",
    );
}

/// rbigint.py's process-global `_parts_cache` owns the powers used by recursive
/// non-binary formatting. Its cached digit arrays must be forwarded as roots
/// across moving collections, and extending the shared list later must retain
/// every previously cached rbigint object.
#[test]
fn rbigint_format_parts_cache_survives_and_grows_across_full_collection() {
    const PROGRAM: &str = r#"
import gc

def run():
    x = 10 ** 600 + 123456789
    expected_x = "1" + "0" * (600 - 9) + "123456789"

    # Populate `_parts_cache_10`, then repeatedly consume its cached powers
    # only after a moving full collection has forwarded their digit arrays.
    assert str(x) == expected_x
    i = 0
    while i < 40:
        gc.collect()
        assert str(x) == expected_x
        assert str(-x) == "-" + expected_x
        i = i + 1

    # Grow the same shared parts list after its old prefix has survived those
    # collections, then exercise both the old and new levels after more moves.
    y = 10 ** 2400 + 987654321
    expected_y = "1" + "0" * (2400 - 9) + "987654321"
    assert str(y) == expected_y
    i = 0
    while i < 40:
        gc.collect()
        assert str(x) == expected_x
        assert str(y) == expected_y
        i = i + 1
    return len(str(x)) + len(str(y))

result = run()
assert result == 3002, result
"#;
    run_on_worker(
        PROGRAM,
        "<rbigint_format_parts_cache_gc_stress>",
        "rbigint formatter cache survival checks",
        "rbigint formatter cache gc stress program failed",
    );
}

/// Long consumers keep only the Python wrapper as the GC root while rbigint
/// arithmetic borrows its immutable digit payload.  Repeatedly move those
/// wrappers and digit arrays between operations to catch stale borrowed
/// payloads, missing W_LongObject write barriers, and result digit edges that
/// were not rooted when the payload was boxed.
#[test]
fn rbigint_consumers_survive_moving_collection_between_operations() {
    const PROGRAM: &str = r#"
import gc

def run():
    a = (1 << 300) + 123456789
    b = (1 << 190) + 98765

    # rbigint.abs returns the original payload for a nonnegative value while
    # W_LongObject.descr_abs creates a new old-gen wrapper. Drop the original
    # wrapper so the creation barrier on that shared-payload edge is the only
    # thing keeping the young rbigint alive.
    positive = abs((1 << 333) + 13579)
    gc.collect()
    assert positive == (1 << 333) + 13579
    assert +positive is positive

    class HugeInt(int):
        pass
    subclassed = HugeInt((1 << 345) + 24680)
    base = int(subclassed)
    subclassed = None
    gc.collect()
    assert type(base) is int
    assert base == (1 << 345) + 24680

    i = 0
    while i < 60:
        shifted = a << ((i % 31) + 1)
        gc.collect()
        assert shifted >> ((i % 31) + 1) == a

        product = a * b
        gc.collect()
        q, r = divmod(product, b)
        assert q == a
        assert r == 0
        gc.collect()
        assert product // b == a
        assert product % b == 0

        modular = pow(a, 5, b)
        gc.collect()
        assert modular == (a * a * a * a * a) % b
        negative_modular = pow(a, 3, -b)
        gc.collect()
        assert -b < negative_modular <= 0
        assert negative_modular == (a * a * a) % -b

        mixed = (product ^ shifted) + i
        gc.collect()
        assert (mixed - i) ^ shifted == product

        # Keep producing fresh long payloads; both operands have survived a
        # moving collection before the next borrowed rbigint operation.
        a = a + (1 << (70 + i))
        b = b | (1 << (80 + i))
        i = i + 1
    return a.bit_length() + b.bit_length()

result = run()
assert result > 490, result
"#;
    run_on_worker(
        PROGRAM,
        "<rbigint_consumer_gc_stress>",
        "rbigint consumer survival checks",
        "rbigint consumer gc stress program failed",
    );
}

/// RPython's GC transform roots a by-value `rbigint` local when a later
/// `space.index`, comparison, or wrapping allocation can collect. The Rust
/// interpreter needs the equivalent explicit stable slot: these callbacks
/// force a collection while an earlier machine-int conversion exists only as
/// an unboxed `RBigInt`.
#[test]
fn unboxed_rbigint_consumers_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc
import math

class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        gc.collect()
        return self.value

class Even:
    def __eq__(self, other):
        gc.collect()
        return other % 2 == 0

i = 0
while i < 40:
    assert math.gcd(123456789, Index(3)) == 3
    assert math.lcm(123456789, Index(3)) == 123456789
    assert math.comb(50, Index(3)) == 19600
    assert math.perm(20, Index(3)) == 6840

    sliced = range(20)[slice(Index(1), Index(12), Index(2))]
    assert list(sliced) == [1, 3, 5, 7, 9, 11]
    assert range(20).count(Even()) == 10
    i += 1
"#,
        "<unboxed_rbigint_callback_gc_stress>",
        "unboxed rbigint callback-root checks",
        "unboxed rbigint callback GC stress program failed",
    );
}

/// `min_max_sequence` keeps the current best item and key in shadow-stack
/// slots while the `key=` function runs Python and calls `gc.collect()`.
#[test]
fn min_max_key_values_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class K:
    def __init__(self, v):
        self.v = v

def keyf(o):
    gc.collect()
    return o.v

i = 0
while i < 40:
    lo = min([K(5), K(3), K(9), K(1), K(7)], key=keyf)
    assert lo.v == 1, lo.v
    hi = max([K(5), K(3), K(9), K(1), K(7)], key=keyf)
    assert hi.v == 9, hi.v
    i += 1
"#,
        "<min_max_key_gc_rooting>",
        "min/max key callback-root checks",
        "min/max key callback GC rooting program failed",
    );
}

/// `sort_rooted_items` keeps every item and computed key in shadow-stack
/// slots while the `key=` function runs Python and calls `gc.collect()`.
#[test]
fn sorted_key_values_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class K:
    def __init__(self, v):
        self.v = v

def keyf(o):
    gc.collect()
    return o.v

i = 0
while i < 40:
    ordered = sorted([K(5), K(3), K(9), K(1), K(7)], key=keyf)
    values = (ordered[0].v, ordered[1].v, ordered[2].v, ordered[3].v, ordered[4].v)
    assert values == (1, 3, 5, 7, 9), values
    i += 1
"#,
        "<sorted_key_gc_rooting>",
        "sorted key callback-root checks",
        "sorted key callback GC rooting program failed",
    );
}

/// `list.sort` decodes `reverse=` by calling the object's `__bool__`, which runs
/// Python and can collect.  The receiver and the `key=` callable are read out of
/// the builtin's native argument copy, which the collector does not update, so
/// both have to be pinned across that call rather than after it.
#[test]
fn list_sort_receiver_and_key_survive_the_reverse_bool_callback() {
    run_on_worker(
        r#"
import gc

class Flag:
    def __init__(self, v):
        self.v = v
    def __bool__(self):
        gc.collect()
        return self.v

def keyf(o):
    gc.collect()
    return -o

i = 0
while i < 40:
    items = [5, 3, 9, 1, 7]
    items.sort(reverse=Flag(True))
    assert items == [9, 7, 5, 3, 1], items

    other = [5, 3, 9, 1, 7]
    other.sort(key=keyf, reverse=Flag(False))
    assert other == [9, 7, 5, 3, 1], other

    plain = [5, 3, 9, 1, 7]
    plain.sort(key=None, reverse=Flag(False))
    assert plain == [1, 3, 5, 7, 9], plain
    i += 1
"#,
        "<list_sort_reverse_bool_gc_rooting>",
        "list.sort reverse-callback root checks",
        "list.sort reverse callback GC rooting program failed",
    );
}

/// The reflected binary-operator helper holds both operands and the reflected
/// implementation in Rust locals while `L.__add__` / `L.__sub__` run Python
/// and collect before `R.__radd__` / `R.__rsub__` use those values.
#[test]
fn reflected_binary_operands_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class L:
    def __init__(self, v):
        self.v = v
    def __add__(self, other):
        gc.collect()
        return NotImplemented
    def __sub__(self, other):
        gc.collect()
        return NotImplemented

class R:
    def __init__(self, v):
        self.v = v
    def __radd__(self, other):
        gc.collect()
        return (other.v, self.v)
    def __rsub__(self, other):
        gc.collect()
        return (other.v, self.v)

i = 0
while i < 60:
    assert L(i) + R(i + 1) == (i, i + 1)
    assert L(i) - R(i + 1) == (i, i + 1)
    i += 1
"#,
        "<reflected_binary_gc_rooting>",
        "reflected binary operand callback-root checks",
        "reflected binary operand callback GC rooting program failed",
    );
}

/// The comparison helper holds its two-entry `(ov, recv, other)` order array
/// in Rust locals while the first `P.__eq__` runs Python and collects before
/// the second `Q.__eq__` reads both captured operands.
#[test]
fn comparison_order_operands_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class P:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return NotImplemented

class Q:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return (self.v, other.v)

i = 0
while i < 60:
    assert (P(i) == Q(i + 1)) == (i + 1, i)
    i += 1
"#,
        "<comparison_order_gc_rooting>",
        "comparison order callback-root checks",
        "comparison order callback GC rooting program failed",
    );
}

/// `_regular_sum` holds its accumulator and the remaining collected elements
/// in Rust locals while `Addend.__add__` runs Python and collects.
/// `collect_iterable` / `collect_iterator` root elements only while building
/// the Vec; their `RootScope` is dropped before `sum` consumes that Vec.
#[test]
fn sum_accumulator_and_items_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Addend:
    def __init__(self, v):
        self.v = v
    def __add__(self, other):
        gc.collect()
        return Addend(self.v + other.v)

i = 0
while i < 60:
    total = sum(
        [Addend(1), Addend(2), Addend(3), Addend(4),
         Addend(5), Addend(6), Addend(7), Addend(8)],
        Addend(0),
    )
    assert total.v == 36, total.v
    i += 1
"#,
        "<sum_gc_rooting>",
        "sum accumulator and item callback-root checks",
        "sum callback GC rooting program failed",
    );
}

/// `any` / `all` hold their iterator in a Rust local while `RootingIter.__next__`
/// and `CollectingBool.__bool__` run Python and collect. `collect_iterator`
/// states the required opposite rule: reload the iterator from its
/// post-relocation shadow-stack slot before every `next` call.
#[test]
fn any_all_iterators_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class CollectingBool:
    def __init__(self, value):
        self.value = value
    def __bool__(self):
        gc.collect()
        return self.value

class RootingIter:
    def __init__(self, count, true_count):
        self.i = 0
        self.count = count
        self.true_count = true_count
    def __iter__(self):
        return self
    def __next__(self):
        gc.collect()
        if self.i >= self.count:
            raise StopIteration
        value = self.i < self.true_count
        self.i += 1
        return CollectingBool(value)

i = 0
while i < 60:
    assert any(RootingIter(8, 0)) is False
    assert all(RootingIter(8, 3)) is False
    i += 1
"#,
        "<any_all_gc_rooting>",
        "any/all iterator callback-root checks",
        "any/all callback GC rooting program failed",
    );
}

/// `set_method_union` / `set_method_update` hold their result and each
/// collected item Vec in Rust locals while `RootingIter.__next__` runs Python
/// and collects. Unlike these methods, sibling `set_init_from_iterable_impl`
/// pins its working values for exactly this reason.
#[test]
fn set_union_update_values_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class RootingIter:
    def __init__(self, start, stop):
        self.current = start
        self.stop = stop
    def __iter__(self):
        return self
    def __next__(self):
        gc.collect()
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

i = 0
while i < 60:
    united = {0}.union(RootingIter(1, 5), RootingIter(5, 9))
    assert united == {0, 1, 2, 3, 4, 5, 6, 7, 8}, united

    updated = {0}
    updated.update(RootingIter(1, 5), RootingIter(5, 9))
    assert updated == {0, 1, 2, 3, 4, 5, 6, 7, 8}, updated
    i += 1
"#,
        "<set_union_update_gc_rooting>",
        "set union/update callback-root checks",
        "set union/update callback GC rooting program failed",
    );
}

/// Tracing `a ^ b` with a raising `__eq__` used to SIGSEGV in
/// `array_sanity_load`: the walker presented a non-array GCREF to
/// `do_getarrayitem_gc_r`, `bhimpl_arraylen_gc` read the first payload
/// word (a pointer) as the length, and the item load used that as an
/// index. `executor.py` only ever runs on the live array box.
#[test]
fn set_symdiff_raising_eq_survives_tracing_sanity_load() {
    run_on_worker(
        r#"
class Boom:
    def __init__(self, key, explode=False):
        self.key = key
        self.explode = explode
    def __hash__(self):
        return 7
    def __eq__(self, other):
        if self.explode or getattr(other, "explode", False):
            raise ValueError("boom in __eq__")
        return self.key == other.key

def run_symdiff(a, b):
    return a ^ b

i = 0
while i < 1700:
    assert run_symdiff({1, 2, 3}, {2, 3, 4}) == {1, 4}
    i += 1

j = 0
while j < 80:
    raised = False
    try:
        run_symdiff({Boom(1), Boom(2)}, {Boom(1, explode=True)})
    except ValueError:
        raised = True
    assert raised
    j += 1
"#,
        "<set_symdiff_raising_eq_gc_stress>",
        "set symdiff raising-eq tracing sanity load",
        "set symdiff raising __eq__ crashed the tracer array sanity load",
    );
}

/// `PyCode.co_consts_w` owns the one wrapped object for each constant index
/// (`pycode.py`, `pyopcode.py:498-499`). A large integer constant is a  allow-line-citation
/// managed `W_LongObject`; the Box-immortal PyCode therefore has to expose the
/// filled slot to the raw-root walker across both interpreter and JIT loads.
#[test]
fn rbigint_code_constant_identity_survives_full_collection() {
    run_on_worker(
        r#"
import gc

def load_big():
    return 123456789012345678901234567890123456789012345678901234567890

i = 0
first = load_big()
while i < 80:
    if i % 7 == 0:
        gc.collect()
    assert load_big() is first
    i += 1

gc.collect()
assert load_big() is first
"#,
        "<rbigint_code_constant_gc_stress>",
        "rbigint code constant identity and survival checks",
        "rbigint code constant gc stress program failed",
    );
}

/// `contains`'s tuple membership scan (`baseobjspace::contains`) holds the
/// haystack tuple and the needle in Rust locals while `eq_w` runs the element's
/// `__eq__`, which collects. A minor collection forwards the young tuple, so a
/// raw local reused for the next `w_tuple_getitem` would read a swept slot.
/// Regression for the missing shadow-stack pin on the `is_tuple(haystack)` arm.
#[test]
fn tuple_contains_scan_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Bomb:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return self.v == other.v

hits = 0
i = 0
while i < 60:
    box = (Bomb(1), Bomb(2), Bomb(3), Bomb(7))
    if Bomb(7) in box:
        hits += 1
    junk = [0] * 64
    i += 1
assert hits == 60, hits
"#,
        "<tuple_contains_gc_rooting>",
        "tuple-contains scan callback-root checks",
        "tuple contains callback GC rooting program failed",
    );
}

/// `contains`'s iterable fallback (`descroperation.py sequence_contains`
/// analogue) holds the iterator and needle in Rust locals while `next` and
/// `eq_w` run Python and collect. The sibling `sequence_contains` pins both;
/// this duplicate on the generic-iterable path must too.
#[test]
fn iterable_contains_fallback_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Bomb:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return self.v == other.v

class Seq:
    def __init__(self, vals):
        self.vals = vals
    def __iter__(self):
        return SeqIter(self.vals)

class SeqIter:
    def __init__(self, vals):
        self.vals = vals
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= len(self.vals):
            raise StopIteration
        v = self.vals[self.i]
        self.i += 1
        return v

hits = 0
i = 0
while i < 60:
    seq = Seq([Bomb(1), Bomb(2), Bomb(3), Bomb(7)])
    if Bomb(7) in seq:
        hits += 1
    junk = [0] * 64
    i += 1
assert hits == 60, hits
"#,
        "<iterable_contains_gc_rooting>",
        "iterable-contains fallback callback-root checks",
        "iterable contains fallback callback GC rooting program failed",
    );
}

/// `w_list_find_or_count`'s generic strategy loop holds the list and the needle
/// in Rust locals while `eq_w` runs the element's `__eq__` and collects. The
/// loop re-reads `w_list_len`/`w_list_getitem` from the raw list each iteration,
/// so a forwarded young list would be read from a swept slot.
#[test]
fn list_contains_scan_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Bomb:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return self.v == other.v

hits = 0
i = 0
while i < 60:
    box = [Bomb(1), Bomb(2), Bomb(3), Bomb(7)]
    if Bomb(7) in box:
        hits += 1
    junk = [0] * 64
    i += 1
assert hits == 60, hits
"#,
        "<list_contains_gc_rooting>",
        "list-contains scan callback-root checks",
        "list contains callback GC rooting program failed",
    );
}

/// Integer-strategy `list_eq` / `_compare_unwrappeditems` boxes each element
/// through `w_int_new`.  The second getitem is a collecting malloc, so the
/// first box has to be on the shadow stack before that call, and `list.sort`
/// of tuples has to pin each getitem result the same way (`descr_sort`
/// `sorter.list`).
#[test]
fn list_richcompare_and_sort_boxes_survive_sibling_getitem() {
    const CHILD: &str = "PYRE_LIST_RICHCOMPARE_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "list_richcompare_and_sort_boxes_survive_sibling_getitem",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "65536")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated list richcompare GC-root regression");
        assert!(
            output.status.success(),
            "list compare/sort lost a boxed element across a sibling getitem:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
def check():
    assert [1] < [2]
    assert [1] == [1]
    assert [2, 1] > [2, 0]
    assert max([1], [2]) == [2]
    assert min([3, 1], [3, 0]) == [3, 0]
    xs = [(2, 1), (1, 2), (1, 1)]
    xs.sort()
    assert xs == [(1, 1), (1, 2), (2, 1)]

i = 0
while i < 80:
    check()
    _ = [0] * 32
    i = i + 1
"#,
        "<list_richcompare_gc_rooting>",
        "list richcompare/sort box-root checks",
        "list richcompare/sort GC rooting program failed",
    );
}

/// `isinstance`'s tuple-classinfo recursion holds `obj` and the classinfo tuple
/// in Rust locals while the recursive `isinstance` runs a metaclass
/// `__instancecheck__` that collects. A forwarded young tuple would make the
/// next `w_tuple_getitem(classinfo, i)` read a swept slot.
#[test]
fn isinstance_tuple_classinfo_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Meta(type):
    def __instancecheck__(cls, obj):
        gc.collect()
        return False

class A(metaclass=Meta):
    pass

class B(metaclass=Meta):
    pass

class C(metaclass=Meta):
    pass

class Real:
    pass

r = Real()
hits = 0
i = 0
while i < 60:
    classinfo = (A, B, C, Real)
    if isinstance(r, classinfo):
        hits += 1
    junk = [0] * 64
    i += 1
assert hits == 60, hits
"#,
        "<isinstance_tuple_gc_rooting>",
        "isinstance tuple-classinfo callback-root checks",
        "isinstance tuple classinfo callback GC rooting program failed",
    );
}

/// The `values()`-view membership scan holds the dict-items snapshot's values
/// and the needle in Rust locals while `eq_w` runs the value's `__eq__` and
/// collects. Regression for the un-pinned `w_dict_items` snapshot on the
/// `DictViewKind::Values` arm.
#[test]
fn dict_values_view_contains_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class Bomb:
    def __init__(self, v):
        self.v = v
    def __eq__(self, other):
        gc.collect()
        return self.v == other.v

hits = 0
i = 0
while i < 60:
    d = {0: Bomb(1), 1: Bomb(2), 2: Bomb(3), 3: Bomb(7)}
    if Bomb(7) in d.values():
        hits += 1
    junk = [0] * 64
    i += 1
assert hits == 60, hits
"#,
        "<dict_values_view_gc_rooting>",
        "dict values-view contains callback-root checks",
        "dict values-view contains callback GC rooting program failed",
    );
}

/// `try_hash_value`'s tuple element walk holds the tuple in a Rust local while
/// each element's `__hash__` runs Python and collects; it re-reads
/// `w_tuple_getitem` from the raw tuple in the loop and stores the cached hash
/// on it afterward. A forwarded young tuple would be read/written at a swept
/// slot. The per-element `v` gives a deterministic tuple hash across rounds.
#[test]
fn tuple_hash_element_walk_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc

class H:
    def __init__(self, v):
        self.v = v
    def __hash__(self):
        gc.collect()
        return self.v
    def __eq__(self, other):
        return self is other

expected = None
i = 0
while i < 60:
    t = (H(1), H(2), H(3), H(4))
    h = hash(t)
    if expected is None:
        expected = h
    else:
        assert h == expected, (h, expected)
    junk = [0] * 64
    i += 1
"#,
        "<tuple_hash_gc_rooting>",
        "tuple-hash element walk callback-root checks",
        "tuple hash element walk callback GC rooting program failed",
    );
}

/// `slice_hash_value` captures `[start, stop, step]` into a Rust array, then
/// hashes each while the component `__hash__` runs Python and collects. A
/// forwarded young component would leave a later array entry pointing at a swept
/// slot. The per-component `v` gives a deterministic slice hash across rounds.
#[test]
fn slice_hash_components_survive_python_callbacks() {
    run_on_worker(
        r#"
import gc

class H:
    def __init__(self, v):
        self.v = v
    def __hash__(self):
        gc.collect()
        return self.v
    def __eq__(self, other):
        return self is other

expected = None
i = 0
while i < 60:
    s = slice(H(1), H(2), H(3))
    h = hash(s)
    if expected is None:
        expected = h
    else:
        assert h == expected, (h, expected)
    junk = [0] * 64
    i += 1
"#,
        "<slice_hash_gc_rooting>",
        "slice-hash component callback-root checks",
        "slice hash component callback GC rooting program failed",
    );
}

/// `w_dict_move_to_end_checked`'s module-dict branch falls back to
/// `scan_dict_key_reentrant` when a probing `__eq__` escapes the callback-free
/// ladder. The scan re-derefs the module `object_storage` after each callback;
/// a `__hash__`/`__eq__` that collects would move a young module dict or its
/// entries, so the scan must pin and reload the container. Every `K` shares one
/// hash, forcing a collision bucket where the fresh probe key must compare via
/// `__eq__` (which collects) to locate the stored entry. Before #46 the module
/// branch returned `DictKeyError`, surfacing as a spurious `TypeError`; the
/// reentrant scan now finds and reorders the entry. `delitem_if_value_is` shares
/// the same scan and tail.
#[test]
fn module_dict_move_to_end_reentrant_survives_python_callbacks() {
    run_on_worker(
        r#"
import gc
import __pypy__

ModuleType = type(gc)

class K:
    def __init__(self, tag):
        self.tag = tag
    def __hash__(self):
        gc.collect()
        return 42
    def __eq__(self, other):
        gc.collect()
        return isinstance(other, K) and self.tag == other.tag

i = 0
while i < 40:
    m = ModuleType('probe_mod')
    d = m.__dict__
    a = K('a')
    b = K('b')
    d[a] = 'va'          # non-str key -> module dict switches to object strategy
    d[b] = 'vb'          # same hash -> shares a's collision bucket
    # A fresh, equal, colliding key drives the callback-free probe into the
    # reentrant scan on the module object_storage.
    moved = __pypy__.move_to_end(d, K('a'), last=True)
    assert moved is None or moved, moved
    order = [k.tag for k in d.keys() if isinstance(k, K)]
    assert order == ['b', 'a'], order
    __pypy__.move_to_end(d, K('a'), last=False)
    order = [k.tag for k in d.keys() if isinstance(k, K)]
    assert order == ['a', 'b'], order
    try:
        __pypy__.move_to_end(d, K('absent'), last=True)
        raise AssertionError('missing key did not raise')
    except KeyError:
        pass
    junk = [K(i) for _ in range(32)]
    i += 1
"#,
        "<module_dict_move_to_end_gc_rooting>",
        "module-dict move_to_end reentrant-scan callback-root checks",
        "module dict move_to_end reentrant scan callback GC rooting program failed",
    );
}

/// Builtin iterator `next_w` arms hold the yielded item (and the iterator
/// itself) across a collecting predicate / `is_true` / `append`. A young
/// element that moves during that call must be reloaded from the shadow stack
/// before it is returned and stored. Covers takewhile / dropwhile /
/// filterfalse plus the other callback-style arms that share the hole.
#[test]
fn iterator_next_w_operands_survive_allocating_predicates() {
    run_on_worker(
        r#"
from itertools import (
    takewhile, dropwhile, filterfalse, cycle, accumulate, chain, compress,
    starmap, zip_longest,
)

class Young:
    def __init__(self, v):
        self.v = v
    def __bool__(self):
        junk = [0] * 16
        return self.v != 0
    def __eq__(self, other):
        junk = [0] * 16
        return isinstance(other, Young) and self.v == other.v

def allocating_pred(x):
    junk = [0] * 32
    return x.v < 8

def allocating_odd(x):
    junk = [0] * 32
    return x.v % 2 == 0

def allocating_add(a, b):
    junk = [0] * 16
    return Young(a.v + b.v)

def drain(it):
    out = []
    while True:
        try:
            out.append(next(it))
        except StopIteration:
            return out

i = 0
while i < 40:
    src = [Young(j) for j in range(12)]
    tw = drain(takewhile(allocating_pred, src))
    assert [x.v for x in tw] == [0, 1, 2, 3, 4, 5, 6, 7], [x.v for x in tw]

    src = [Young(j) for j in range(12)]
    dw = drain(dropwhile(allocating_pred, src))
    assert [x.v for x in dw] == [8, 9, 10, 11], [x.v for x in dw]

    src = [Young(j) for j in range(8)]
    ff = drain(filterfalse(allocating_odd, src))
    assert [x.v for x in ff] == [1, 3, 5, 7], [x.v for x in ff]

    src = [Young(j) for j in range(6)]
    fl = drain(filter(allocating_odd, src))
    assert [x.v for x in fl] == [0, 2, 4], [x.v for x in fl]

    src = [Young(j) for j in range(4)]
    en = drain(enumerate(src))
    assert [(idx, x.v) for idx, x in en] == [(0, 0), (1, 1), (2, 2), (3, 3)], [
        (idx, x.v) for idx, x in en
    ]

    src = [Young(j) for j in range(3)]
    cy = cycle(src)
    got = []
    k = 0
    while k < 7:
        got.append(next(cy).v)
        k += 1
    assert got == [0, 1, 2, 0, 1, 2, 0], got

    n = [0]
    def counting():
        junk = [0] * 16
        n[0] += 1
        return Young(n[0])
    sentinel = Young(4)
    ci = drain(iter(counting, sentinel))
    assert [x.v for x in ci] == [1, 2, 3], [x.v for x in ci]

    src = [Young(j) for j in range(1, 5)]
    ac = drain(accumulate(src, allocating_add))
    assert [x.v for x in ac] == [1, 3, 6, 10], [x.v for x in ac]

    z = drain(zip([Young(1)], [Young(2), Young(3)]))
    assert len(z) == 1 and z[0][0].v == 1 and z[0][1].v == 2

    z1 = drain(zip([Young(9)]))
    assert len(z1) == 1 and z1[0][0].v == 9

    ch = drain(chain([Young(1), Young(2)], [Young(3)]))
    assert [x.v for x in ch] == [1, 2, 3], [x.v for x in ch]

    co = drain(compress([Young(1), Young(2), Young(3)], [Young(1), Young(0), Young(1)]))
    assert [x.v for x in co] == [1, 3], [x.v for x in co]

    sm = drain(starmap(allocating_add, [(Young(1), Young(2)), (Young(3), Young(4))]))
    assert [x.v for x in sm] == [3, 7], [x.v for x in sm]

    zl = drain(zip_longest([Young(1)], [Young(2), Young(3)], fillvalue=Young(0)))
    assert [tuple(x.v for x in t) for t in zl] == [(1, 2), (0, 3)]

    junk = [Young(i) for _ in range(32)]
    i += 1
"#,
        "<iterator_next_w_gc_rooting>",
        "iterator next_w callback-root checks",
        "iterator next_w operands did not survive allocating predicates",
    );
}

/// `_k_mul` / `_x_mul` allocate the result digit array while the operand
/// handles still hold `_digits` by value. Those edges have to live on
/// owner-root slots across that malloc, otherwise a later pin publishes the
/// from-space address (`minor_root` GC BUG) or `_normalize` reads zeros and
/// the product collapses to 0.
#[test]
fn karatsuba_mul_keeps_operand_digits_across_collecting_allocs() {
    const CHILD: &str = "PYRE_KARATSUBA_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "karatsuba_mul_keeps_operand_digits_across_collecting_allocs",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated karatsuba GC-root regression");
        assert!(
            output.status.success(),
            "karatsuba mul/pow lost a digit array across a collecting alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 0
while i < 8:
    x = 10 ** 800
    assert x != 0
    assert str(x) == "1" + "0" * 800
    y = (10 ** 40) * (10 ** 40)
    assert y == 10 ** 80
    junk = [0] * 32
    i = i + 1
"#,
        "<karatsuba_mul_gc_rooting>",
        "karatsuba operand-digit roots",
        "karatsuba operands did not survive collecting digit allocs",
    );
}

/// `compiling.py compile` reloads `w_source` after `source_as_str` / the
/// AST type probe collect. The native path imports `_ast` for that probe,
/// so the source/filename/mode locals have to live on slots or
/// `isinstance_w` reads a from-space word (address 0x2).
#[test]
fn compile_keeps_source_across_ast_import() {
    const CHILD: &str = "PYRE_COMPILE_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "compile_keeps_source_across_ast_import",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated compile GC-root regression");
        assert!(
            output.status.success(),
            "compile() lost its source across a collecting _ast import:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
src = "x = 1\n"
i = 0
while i < 8:
    junk = [0] * 64
    code = compile(src, "<stress>", "exec")
    ns = {}
    exec(code, ns)
    assert ns["x"] == 1
    try:
        compile("if:\n", "<bad>", "exec")
        raise AssertionError("expected SyntaxError")
    except SyntaxError as e:
        assert e.lineno == 1
    i = i + 1
"#,
        "<compile_source_gc_rooting>",
        "compile source roots",
        "compile() source did not survive collecting _ast import",
    );
}

/// `functional.py W_Range.descr_eq` runs `space.eq_w` on length / start /
/// step, and `rbigint.py fromint` allocates Digits. The two range objects
/// and their field pointers have to live on slots across those mallocs.
#[test]
fn range_eq_keeps_operands_across_collecting_fromint() {
    const CHILD: &str = "PYRE_RANGE_EQ_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "range_eq_keeps_operands_across_collecting_fromint",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated range-eq GC-root regression");
        assert!(
            output.status.success(),
            "range == lost an operand across a collecting fromint:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 0
while i < 8:
    junk = [0] * 64
    a = range(2, 10, 3)
    b = range(2, 10, 3)
    assert a == b
    c = range(0, 10 ** 40, 1)
    d = range(0, 10 ** 40, 1)
    assert c == d
    assert range(1, 5) != range(1, 6)
    i = i + 1
"#,
        "<range_eq_gc_rooting>",
        "range eq roots",
        "range == operands did not survive collecting fromint",
    );
}

/// `typeobject.py W_TypeObject.ready` allocates a weakref per base, and
/// `__build_class__` holds the `__classcell__` across that plus
/// `__set_name__`. Without those slots, `super()` in a method reads an
/// empty cell (`NameError: __class__`) or `w_type_ready` walks poison.
#[test]
fn classcell_survives_ready_weakref_alloc() {
    const CHILD: &str = "PYRE_CLASSCELL_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "classcell_survives_ready_weakref_alloc",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated classcell GC-root regression");
        assert!(
            output.status.success(),
            "classcell / type.ready lost a livevar across weakref alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 0
while i < 8:
    junk = [0] * 64
    class Meta(type):
        def __new__(mcs, name, bases, ns):
            return super().__new__(mcs, name, bases, ns)
    class C(metaclass=Meta):
        def ident(self):
            return super() is not None
    assert C().ident() is True
    i = i + 1
"#,
        "<classcell_gc_rooting>",
        "classcell roots",
        "classcell did not survive type.ready weakref alloc",
    );
}

/// Host `RBigInt` is a by-value `{_digits,_size}` handle. Boxing
/// (`W_LongObject(num)`), mixed long/int arithmetic, 3-arg `pow` inverse,
/// `chr`, `slice.indices`, and `math.isqrt` all allocate digits while a
/// handle is still live; those edges have to sit on owner-root slots and
/// be re-read before the store into a new payload.
#[test]
fn rbigint_handles_survive_boxing_and_digit_allocs() {
    const CHILD: &str = "PYRE_RBIGINT_HANDLE_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rbigint_handles_survive_boxing_and_digit_allocs",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated rbigint-handle GC-root regression");
        assert!(
            output.status.success(),
            "rbigint handle lost _digits across a collecting alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import math
i = 0
while i < 8:
    junk = [0] * 32
    x = 10 ** 80 + 7
    y = x + 3
    assert y - x == 3
    assert x * 2 == x + x
    assert pow(5, -1, 13) == 8
    assert pow(3, 4, -8) == -7
    i = 0
    while i < 20:
        junk = [0] * 32
        assert pow(i % 7, 13, 1000) == pow(i % 7, 13) % 1000
        i = i + 1
    s = "1"
    i = 0
    while i < 40:
        junk = [0] * 32
        s = s + "9"
        n = int(s)
        assert str(n) == s
        i = i + 1
    assert chr(0x10ffff) == "\U0010ffff"
    assert slice(1, 10, 2).indices(20) == (1, 10, 2)
    root = math.isqrt(x)
    assert root * root <= x < (root + 1) * (root + 1)
    i = i + 1
"#,
        "<rbigint_handle_gc_rooting>",
        "rbigint handle roots",
        "rbigint handles did not survive boxing / digit allocs",
    );
}

/// `_x_add`/`_x_mul` fill used to treat a stack handle address as a GC
/// livevar. After a collecting digit malloc the next `pow`/`isqrt`/`divmod`
/// step then read a from-space `_digits` (poisoned size / exponent) and
/// either looped in GC or returned a wrong value.
#[test]
fn rbigint_pow_isqrt_divmod_survive_tiny_nursery() {
    const CHILD: &str = "PYRE_RBIGINT_POW_ISQRT_DIVMOD_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rbigint_pow_isqrt_divmod_survive_tiny_nursery",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated pow/isqrt/divmod GC-root regression");
        assert!(
            output.status.success(),
            "pow/isqrt/divmod lost a handle across a collecting alloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import math
BIG = (1 << 200) + 12345
expected_pow3 = 1
j = 0
while j < 80:
    expected_pow3 = expected_pow3 * 3
    j = j + 1
i = 0
while i < 8:
    junk = [0] * 32
    assert pow(0, 10 ** 80) == 0
    assert pow(1, 10 ** 80) == 1
    assert pow(-1, 10 ** 80) == 1
    assert pow(3, 80) == expected_pow3
    assert pow(5, 13, 1000) == 125
    assert pow(3, 40, -8) == -7
    n = BIG + i
    q, r = divmod(n, 97 + (i & 7))
    assert q * (97 + (i & 7)) + r == n
    q2, r2 = divmod(n, 1000003)
    assert q2 * 1000003 + r2 == n
    root = math.isqrt(n)
    assert root * root <= n < (root + 1) * (root + 1)
    i = i + 1
"#,
        "<rbigint_pow_isqrt_divmod_gc_rooting>",
        "rbigint pow/isqrt/divmod roots",
        "pow/isqrt/divmod did not survive collecting digit allocs",
    );
}

/// Gateway `args` is a stack copy. `__index__` on a bound can collect, so
/// the receiver and the other bound have to be shadow-stack slots, not
/// that copy — same shape as `list_method_pop` / `index_bounds_not_none`.
#[test]
fn gateway_index_args_survive_collect() {
    const CHILD: &str = "PYRE_GATEWAY_INDEX_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "gateway_index_args_survive_collect",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated gateway-index GC-root regression");
        assert!(
            output.status.success(),
            "gateway index args lost a livevar across __index__:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
class Idx:
    def __init__(self, n):
        self.n = n
    def __index__(self):
        junk = [0] * 64
        return self.n
i = 0
while i < 8:
    junk = [0] * 32
    buf = bytearray(b"abcdefgh")
    assert buf.pop(Idx(2)) == 99
    text = "abcdefgh" * 4
    assert text.find("c", Idx(2)) == 2
    assert text.count("a", Idx(2)) == 3
    raw = b"abcdefgh" * 4
    assert raw.find(b"c", Idx(2)) == 2
    i = i + 1
"#,
        "<gateway_index_gc_rooting>",
        "gateway index roots",
        "gateway index args did not survive collecting __index__",
    );
}

/// Pattern.match/search `pos`/`endpos` run `__index__` after gathering a
/// buffer subject into fresh bytes. The gateway `args` copy and the
/// gathered payload have to be shadow-stack slots (`index_bounds_not_none`).
#[test]
fn sre_buffer_subject_survives_bound_conversion() {
    const CHILD: &str = "PYRE_SRE_BUFFER_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "sre_buffer_subject_survives_bound_conversion",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated sre-buffer GC-root regression");
        assert!(
            output.status.success(),
            "sre buffer subject lost across collecting __index__:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import re
class Idx:
    def __init__(self, n):
        self.n = n
    def __index__(self):
        junk = [0] * 64
        return self.n
i = 0
while i < 8:
    junk = [0] * 32
    data = bytearray(b"ab12cd34")
    found = re.compile(rb"\d+").search(memoryview(data), Idx(0), Idx(len(data)))
    assert found is not None and found.group() == b"12"
    i = i + 1
"#,
        "<sre_buffer_gc_rooting>",
        "sre buffer subject roots",
        "sre buffer subject did not survive collecting __index__",
    );
}

/// `W_Range.descr_getitem` / `_compute_slice`: `fromint` of length and
/// slice bounds collect, so the range, the slice, and start/stop/step
/// field pointers have to be shadow-stack slots (`functional.py
/// _compute_item` / `_compute_slice`).
#[test]
fn range_getitem_slice_survives_fromint() {
    const CHILD: &str = "PYRE_RANGE_GETITEM_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "range_getitem_slice_survives_fromint",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated range-getitem GC-root regression");
        assert!(
            output.status.success(),
            "range getitem/slice lost a livevar across fromint:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
class Idx:
    def __init__(self, n):
        self.n = n
    def __index__(self):
        junk = [0] * 64
        return self.n
i = 0
while i < 8:
    junk = [0] * 32
    r = range(20)
    assert r[Idx(7)] == 7
    assert list(r[Idx(2):Idx(10):Idx(2)]) == [2, 4, 6, 8]
    i = i + 1
"#,
        "<range_getitem_gc_rooting>",
        "range getitem roots",
        "range getitem/slice did not survive collecting fromint",
    );
}

/// `unicode_decode_error` mints encoding/object/reason then allocates the
/// exception. `RootedItems::take` copies those slots; stores after that
/// malloc have to re-read them (`W_UnicodeDecodeError.descr_init`).
#[test]
fn unicode_decode_error_fields_survive_wrapper_alloc() {
    const CHILD: &str = "PYRE_UNICODE_DECODE_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "unicode_decode_error_fields_survive_wrapper_alloc",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated unicode-decode GC-root regression");
        assert!(
            output.status.success(),
            "UnicodeDecodeError fields lost across wrapper malloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 0
while i < 8:
    junk = [0] * 32
    try:
        b"\xff".decode("utf-8")
    except UnicodeDecodeError as e:
        s = str(e)
        assert "utf-8" in s
        assert e.encoding == "utf-8"
    i = i + 1
"#,
        "<unicode_decode_gc_rooting>",
        "unicode decode error roots",
        "UnicodeDecodeError fields did not survive wrapper malloc",
    );
}

/// `build_class` holds the metaclass across `__prepare__` / getattr / call.
/// A callee pin does not rewrite the caller's natives.
#[test]
fn build_class_metaclass_survives_prepare() {
    const CHILD: &str = "PYRE_BUILD_CLASS_METACLASS_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "build_class_metaclass_survives_prepare",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated build-class metaclass GC-root regression");
        assert!(
            output.status.success(),
            "metaclass lost across __prepare__:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
class Prepare(type):
    @classmethod
    def __prepare__(mcls, name, bases, **kwds):
        junk = [0] * 64
        return dict(prepared=True)
    def __new__(mcls, name, bases, ns, **kwds):
        junk = [0] * 64
        return super().__new__(mcls, name, bases, ns)
class Built(metaclass=Prepare):
    seen = prepared
assert Built.seen is True
"#,
        "<build_class_metaclass_gc_rooting>",
        "build_class metaclass roots",
        "metaclass did not survive collecting __prepare__",
    );
}

/// OSError construction mints the wrapper then `args_new`; the exception
/// must be reloaded from its slot before `w_exception_set_args`.
#[test]
fn os_error_args_survive_wrapper_alloc() {
    const CHILD: &str = "PYRE_OS_ERROR_ARGS_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "os_error_args_survive_wrapper_alloc",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated OSError GC-root regression");
        assert!(
            output.status.success(),
            "OSError args lost across wrapper malloc:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 0
while i < 8:
    junk = [0] * 32
    try:
        raise OSError(2, "nope", "file")
    except OSError as e:
        assert e.errno == 2
        assert e.strerror == "nope"
        assert e.filename == "file"
    i = i + 1
"#,
        "<os_error_gc_rooting>",
        "OSError args roots",
        "OSError fields did not survive wrapper malloc",
    );
}

/// `W_Range.descr_new` stores `space.index` of each bound. A bignum stop
/// is already an `int` (`W_LongObject`), so the range must keep that ref.
#[test]
fn range_keeps_original_long_bound() {
    const CHILD: &str = "PYRE_RANGE_LONG_BOUND_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "range_keeps_original_long_bound", "--nocapture"])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated range long-bound identity regression");
        assert!(
            output.status.success(),
            "range lost the original long bound:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
i = 2 ** 64
j = 0
while j < 8:
    junk = [0] * 32
    r = range(i)
    assert r.stop is i
    j = j + 1
"#,
        "<range_long_bound_identity>",
        "range long bound identity",
        "range(i).stop is not i after a collecting alloc",
    );
}

/// Marshal of a tuple constant converts each element through
/// `obj_to_bigint`, which allocates digit arrays and can collect. The
/// tuple (and nested tuples) must be reloaded from the shadow stack
/// between those conversions.
#[test]
fn marshal_tuple_constants_survive_collecting_int_conversion() {
    const CHILD: &str = "PYRE_OBJ_TO_CONSTANT_DATA_GC_ROOT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "marshal_tuple_constants_survive_collecting_int_conversion",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "4096")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated obj_to_constant_data GC-root regression");
        assert!(
            output.status.success(),
            "tuple constant conversion lost its container:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import marshal
T = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (11, 12), 13)
src = compile("x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (11, 12), 13)", "<t>", "exec")
payload = marshal.dumps(src)
i = 0
while i < 16:
    code = marshal.loads(payload)
    assert T in code.co_consts
    i = i + 1
"#,
        "<obj_to_constant_data_tuple>",
        "obj_to_constant_data tuple roots",
        "marshal.loads lost a tuple constant across a collecting int conversion",
    );
}

/// `setobject.py` `W_BaseSetObject.clear` installs a fresh storage box and
/// leaves `d = self.unerase(w_set.sstorage)` iterating the detached one.
/// Young keys in that box are reachable only through the box's own trace
/// plus the write barrier on the box. A minor that runs inside the probing
/// `__eq__` used to move those keys and the next compare read the corpse.
#[test]
fn set_update_traces_storage_orphaned_by_eq_clear() {
    const CHILD: &str = "PYRE_SET_ORPHAN_STORAGE_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "set_update_traces_storage_orphaned_by_eq_clear",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYRE_JIT", "0")
            .env("PYPY_GC_NURSERY", "1048576")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated set orphan-storage regression");
        assert!(
            output.status.success(),
            "set update lost a key in a storage box detached by clear:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
class Bad:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return 0
    def __eq__(self, other):
        global armed, left, junk
        if armed and isinstance(other, Bad):
            armed = False
            left.clear()
            i = 0
            while i < 4000:
                junk.append([i, i, i])
                i += 1
        return self is other

armed = False
junk = []
left = set()
right = set()
i = 0
while i < 30:
    left.add(Bad(i))
    right.add(Bad(1000 + i))
    i += 1
armed = True
left |= right
assert len(left) == 0, len(left)
assert len(right) == 30, len(right)
"#,
        "<set_orphan_storage>",
        "set storage box orphaned by clear",
        "set update crashed or kept elements after clear inside __eq__",
    );
}

/// A builtin `Function` materialises `__name__` on first read and stores the
/// fresh string into itself. Held only by a managed list (the original a
/// `mock.patch` saves), no raw-root walk reaches the builtin, so only the
/// barrier on the box can record that store. Before the builtin carried the
/// prebuilt header, the next minor left `w_name` naming a moved string.
#[test]
fn builtin_name_cached_on_a_builtin_held_only_by_a_list() {
    const CHILD: &str = "PYRE_BUILTIN_NAME_CACHE_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "builtin_name_cached_on_a_builtin_held_only_by_a_list",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYRE_JIT", "0")
            .env("PYPY_GC_NURSERY", "1048576")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated builtin __name__ cache regression");
        assert!(
            output.status.success(),
            "a builtin's cached __name__ went stale after a minor:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import math
holder = [math.gcd]
math.gcd = None
name = holder[0].__name__
del name
junk = []
i = 0
while i < 20000:
    junk.append([i, i])
    i += 1
junk = None
i = 0
while i < 5:
    assert holder[0].__name__ == "gcd", holder[0].__name__
    junk = [[j] for j in range(20000)]
    i += 1
"#,
        "builtin_name_cache.py",
        "builtin __name__ cache checks",
        "a builtin's cached __name__ went stale after a minor",
    );
}

/// An `rbigint` handle keeps its digit array by value, so a handle read after
/// a call that allocates digits must be rooted across it. `left.mul(&w5pow(..))`
/// derefs its receiver before the argument runs; `divmod(-a, b)` keeps `q`
/// and `r` across `invert`; the `i64::MIN` operand fallbacks build it with
/// `fromint` after `self` is read. Each read a moved digit array.
#[test]
fn rbigint_handles_stay_rooted_across_digit_allocations() {
    const CHILD: &str = "PYRE_RBIGINT_ROOTS_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rbigint_handles_stay_rooted_across_digit_allocations",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYRE_JIT", "0")
            .env("PYPY_GC_NURSERY", "1048576")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .output()
            .expect("run isolated rbigint rooting regression");
        assert!(
            output.status.success(),
            "an rbigint handle read a moved digit array:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import sys
sys.set_int_max_str_digits(0)
digits = '8' * 40000
n = int(digits)
assert n % 1000 == 888
assert str(n) == digits
m = -9223372036854775808
x = 10 ** 400 + 12345
i = 0
while i < 200:
    assert x * m == -(x << 63)
    assert x + m == x - (1 << 63)
    assert (x & m) == x - (x % (1 << 63))
    assert (x | m) == -((1 << 63) - (x % (1 << 63)))
    assert x % m == -((1 << 63) - x % (1 << 63))
    i += 1
a = 3 ** 6000 + 7
b = 7 ** 1100 + 1
j = 0
while j < 20:
    q, r = divmod(-a, b)
    assert q * b + r == -a and 0 <= r < b
    j += 1
"#,
        "rbigint_roots.py",
        "rbigint rooting checks",
        "an rbigint handle read a moved digit array",
    );
}

/// A guard-resume bridge once answered `x << n` with a nursery word: the
/// recycled block's bits, which `MAJIT_GC_NURSERY_POISON` fills with `0xAA`
/// (`-6148914691236517206` as a signed field). Collecting on every allocation
/// is what opens that window. The shift must stay the integer product.
#[test]
fn long_shift_identity_survives_poisoned_nursery() {
    const CHILD: &str = "PYRE_LONG_SHIFT_POISON_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "long_shift_identity_survives_poisoned_nursery",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated long-shift nursery regression");
        assert!(
            output.status.success(),
            "a shift read a recycled nursery word:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
poison = -6148914691236517206
i = 0
while i < 2000:
    x = 123456789
    n = (i % 20) + 1
    y = x << n
    assert y != poison, y
    assert y == x * (1 << n), (y, n)
    big = (1 << 100) + 3
    z = big << 5
    assert z != poison, z
    assert (z >> 5) == big, z
    i += 1
"#,
        "long_shift_poison.py",
        "long shift under a poisoned nursery",
        "x << n must not be a recycled nursery word",
    );
}

/// `_match_signature` writes `space.newdict(kwargs=True)` into a `**kwargs`
/// local even when the call passed no keywords. A positional call must see
/// that empty dict, not the positional argument and not an unbound local.
#[test]
fn positional_call_binds_an_empty_varkeywords_dict() {
    const CHILD: &str = "PYRE_VARKEYWORDS_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "positional_call_binds_an_empty_varkeywords_dict",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .output()
            .expect("run isolated varkeywords regression");
        assert!(
            output.status.success(),
            "a positional call mis-bound **kwargs:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
def f(a, **k):
    return a + len(k)

def g(**k):
    return len(k)

i = 0
total = 0
while i < 300:
    total = total + f(i)
    assert g() == 0
    i = i + 1
assert total == 44850

def named(a, **k):
    return a + len(k)

j = 0
named_total = 0
while j < 300:
    named_total = named_total + named(a=j)
    j = j + 1
assert named_total == 44850

def collected(a, **k):
    return a + k["b"]

c = 0
collected_total = 0
while c < 300:
    collected_total = collected_total + collected(c, b=1)
    c = c + 1
assert collected_total == 45150

def only(**k):
    return k["b"]

o = 0
only_total = 0
while o < 300:
    only_total = only_total + only(b=1)
    o = o + 1
assert only_total == 300

def positional_only(a):
    return a

raised = False
try:
    positional_only(b=1)
except TypeError:
    raised = True
assert raised
"#,
        "varkeywords_positional.py",
        "positional **kwargs binding",
        "a positional call must bind an empty **kwargs dict",
    );
}

/// `setup_context` reads `frame.last_instr`. A compiled caller keeps that
/// coordinate in the virtualizable until the consumer forces it, so
/// `warnings.warn(..., stacklevel=2)` reports the call in the loop.
#[test]
fn warning_stacklevel_reads_the_compiled_caller_line() {
    run_on_worker(
        r#"
import warnings
import pypyjit
pypyjit.set_param("threshold=1,function_threshold=1")

def leaf():
    warnings.warn("hot-line", stacklevel=2)

def main():
    i = 0
    while i < 80:
        leaf()
        i = i + 1

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    main()

assert len(caught) == 80
assert caught[-1].lineno == 12
"#,
        "warn_lineno.py",
        "warning stacklevel lineno",
        "a compiled caller warning did not report the call line",
    );
}

/// `frame.__repr__` reads `last_instr` through `descr_repr`. A compiled
/// caller keeps that coordinate in the virtualizable, so both `show()`
/// calls in the loop name their own lines.
#[test]
fn frame_repr_reads_the_compiled_caller_line() {
    run_on_worker(
        r#"import sys
import pypyjit
pypyjit.set_param("threshold=1,function_threshold=1")

def show():
    text = repr(sys._getframe(1))
    return int(text.split("line ", 1)[1].split(",", 1)[0])

def main():
    i = 0
    seen = []
    while i < 40:
        seen.append(show())
        seen.append(show())
        i = i + 1
    return seen

got = main()
assert got.count(13) == 40, got
assert got.count(14) == 40, got
"#,
        "frame_repr_lineno.py",
        "frame repr lineno",
        "a compiled frame repr did not report the call lines",
    );
}

/// `f_generator_wref` is a strong edge to the WEAKREF box
/// `initialize_as_generator` stores; `get_generator` dereferences its
/// `weakptr`. Collecting must forward the box slot and let the collector
/// maintain the `weakptr`; a skipped slot still names the nursery address,
/// which `MAJIT_GC_NURSERY_POISON` fills with `0xAA`. Once the last strong
/// reference to the generator is gone, the frame no longer reports it.
#[test]
fn generator_backlink_survives_minor_after_external_refs_drop() {
    const CHILD: &str = "PYRE_GENERATOR_BACKLINK_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "generator_backlink_survives_minor_after_external_refs_drop",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated generator-backlink regression");
        assert!(
            output.status.success(),
            "frame backlink read freed generator memory:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import gc

def g():
    yield 1

gen = g()
frame = gen.gi_frame
frames = [frame]
gens = [gen]
del gen
del frame
gc.collect()
frame = frames[0]
owner = frame.f_generator
assert owner is gens[0], owner
assert owner.gi_frame is frame
assert owner.gi_code.co_name == "g"
del owner
del gens
gc.collect()
assert frame.f_generator is None, frame.f_generator
"#,
        "generator_backlink.py",
        "generator backlink",
        "f_generator must follow the generator across a minor collection",
    );
}

/// `W_Property.init` copies the new property into a raw local, then
/// `space.findattr(w_fget, '__doc__')` runs Python. A getter whose
/// `__getattribute__` collects forwards the caller's root and leaves that
/// local in from-space; the following doc store must reload it.
#[test]
fn property_init_doc_reloads_after_getter_getattribute() {
    const CHILD: &str = "PYRE_PROPERTY_INIT_DOC_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "property_init_doc_reloads_after_getter_getattribute",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated property-init doc regression");
        assert!(
            output.status.success(),
            "property init stored __doc__ through a moved property:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import gc

class Getter:
    def __getattribute__(self, name):
        gc.collect()
        junk = [object() for _ in range(64)]
        if name == "__doc__":
            return "captured-doc"
        return object.__getattribute__(self, name)
    def __call__(self, obj):
        return 1

g = Getter()
p = property(g)
assert p.__doc__ == "captured-doc", p.__doc__
assert p.fget is g
"#,
        "property_init_doc.py",
        "property init doc",
        "property.__init__ must store a getter doc on the live property",
    );
}

/// `W_Property.descr_isabstract` calls `space.isabstractmethod_w` on each
/// accessor. The first call can collect; the next accessor is read from
/// `self` afterwards.
#[test]
fn property_isabstract_reloads_between_accessors() {
    const CHILD: &str = "PYRE_PROPERTY_ISABSTRACT_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "property_isabstract_reloads_between_accessors",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated property-isabstract regression");
        assert!(
            output.status.success(),
            "property.__isabstractmethod__ read a moved property:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import gc

class Flag:
    def __init__(self, value):
        self.value = value
    def __getattribute__(self, name):
        gc.collect()
        junk = [object() for _ in range(40)]
        if name == "__isabstractmethod__":
            return object.__getattribute__(self, "value")
        return object.__getattribute__(self, name)

p = property(Flag(False), Flag(True))
assert p.__isabstractmethod__ is True
"#,
        "property_isabstract.py",
        "property isabstract",
        "descr_isabstract must re-read self between accessor probes",
    );
}

/// `W_Property._properror` raises through `oefmt` `%R` of `self.w_name`.
/// `OpErrFmt._compute_value` runs `space.repr` for that `%R`, which can
/// collect; a collection must not leave a stale property/type/name pointer
/// (`MAJIT_GC_NURSERY_POISON`).
#[test]
fn property_no_accessor_reloads_after_name_repr() {
    const CHILD: &str = "PYRE_PROPERTY_NO_ACCESSOR_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "property_no_accessor_reloads_after_name_repr",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated property-no-accessor regression");
        assert!(
            output.status.success(),
            "missing-accessor error read a moved name after %R:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import gc

class NameObj:
    def __repr__(self):
        gc.collect()
        junk = [object() for _ in range(40)]
        return "BoomName"

class C:
    pass

p = property()
p.__set_name__(C, NameObj())
try:
    p.__get__(C())
except AttributeError as e:
    msg = str(e)
    assert msg == "property BoomName of 'C' object has no getter", msg
else:
    raise AssertionError("expected AttributeError")
"#,
        "property_no_accessor_name_repr.py",
        "property no accessor name repr",
        "_properror must re-read the name after OpErrFmt._compute_value %R",
    );
}

/// `bufferwrapper_releasebuf`, native-subclass arm: the Python
/// `__release_buffer__` override runs with the returned memoryview as a
/// livevar, then the native backing is released through that view. An
/// override that collects forwards the view; the release must read it back
/// from its root rather than through the pre-callback address, which
/// `MAJIT_GC_NURSERY_POISON` fills with `0xAA`.
#[test]
fn memoryview_buffer_wrapper_release_reloads_after_override() {
    const CHILD: &str = "PYRE_MEMORYVIEW_WRAPPER_RELEASE_GC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "memoryview_buffer_wrapper_release_reloads_after_override",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PYRE_GC_STRESS_CHILD", "1")
            .env("PYPY_GC_NURSERY", "1")
            .env("PYPY_GC_NURSERY_DEBUG", "1")
            .env("MAJIT_GC_NURSERY_POISON", "1")
            .output()
            .expect("run isolated memoryview wrapper-release regression");
        assert!(
            output.status.success(),
            "buffer-wrapper release read a moved memoryview:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }
    run_on_worker(
        r#"
import gc

released = []

class B(bytearray):
    def __buffer__(self, flags):
        return super().__buffer__(flags)
    def __release_buffer__(self, view):
        released.append(view.readonly)
        gc.collect()
        junk = [object() for _ in range(40)]

i = 0
while i < 20:
    b = B(b"abcd")
    m = memoryview(b)
    assert m[0] == 97
    m.release()
    b.append(101)
    assert bytes(b) == b"abcde", bytes(b)
    i += 1
assert released
"#,
        "memoryview_wrapper_release.py",
        "memoryview wrapper release",
        "bufferwrapper_releasebuf must re-read the view after __release_buffer__",
    );
}

/// Derived views (slices, copies, casts) share the root view's export
/// instead of owning one, so only the root carries a finalizer.  A cycle
/// through a bytearray reached by a dead root and a dead intermediate slice
/// is then freed by one collection, weakrefs into it included.  Releasing the
/// root ends the export whatever slices remain, dead or alive, and a live
/// slice still reads the bytearray afterwards.
#[test]
fn memoryview_derived_views_share_the_root_export() {
    run_on_worker(
        r#"
import gc
import _weakref

class MyObject:
    pass

class MySource(bytearray):
    pass

b = MySource(b"abcdef")
m = memoryview(b)[:7][1:]
o = MyObject()
b.m = m
b.o = o
wr = _weakref.ref(o)
b = m = o = None
gc.collect()
assert wr() is None

b = bytearray(b"abcdef")
m = memoryview(b)
s = m[1:]
assert list(m[1:3]) == [98, 99]
m.release()
b.append(1)
assert s[0] == 98
s.release()
b.append(2)
assert b == bytearray(b"abcdef\x01\x02"), b
"#,
        "memoryview_derived_export.py",
        "memoryview derived export",
        "derived memoryviews must share the root's export",
    );
}

/// `sys.intern` on a miss stores a weak managed string. A later collection
/// must keep that string while a strong reference lives, and a builtin
/// `interp2app` reached through the declaration container must still run.
#[test]
fn intern_and_declaration_roots_survive_collection() {
    run_on_worker(
        r#"
import gc
import sys

name = "pyre-intern-survives-collection-9c1e"
first = sys.intern(name)
assert sys.intern(name) is first
gc.collect()
assert sys.intern(name) is first, "intern miss was not the same object after collect"
assert first + "!" == name + "!"

add = int.__add__
gc.collect()
assert int.__add__ is add
assert add(1, 2) == 3
"#,
        "intern_and_declaration_roots.py",
        "intern identity and declaration interior",
        "intern or declaration root did not survive collection",
    );
}
