//! `pypy/module/gc/moduledef.py` — `class Module(MixedModule)`.
//!
//! The bodies live in the submodules upstream splits them into: `interp_gc`
//! (collection control and the finalizer lock), `referents` (raw-heap
//! inspection), `app_referents` (the app-level `GcStats` and `dump_rpy_heap`)
//! and `hook` (`gc.hooks` and the stats objects it reports). Explicit
//! collection runs the complete RPython collection, then drains the finalizer
//! queue synchronously.
//!
//! Part of this module answers to 3.14 alone. `moduledef.py` binds no
//! `get_count`, `set_threshold`/`get_threshold`, `set_debug`/`get_debug` or
//! `freeze`/`unfreeze`/`get_freeze_count`, so those have no implementation to
//! follow and each one in `interp_gc` states what it answers and why. Nothing
//! grades them either: `test_gc` is an implementation-detail module on both
//! axes — `lib-python/conftest.py`'s testmap skips it and
//! `cpython_tests/run.py` carries that skip forward — so the assertions that
//! would pin these live in `extra_tests/snippets/` instead.

use pyre_object::*;

use super::referents::{self, gcref, stats};
use super::{app_referents, hook, interp_gc};

pyre_interpreter::py_module! {
    "gc",
    interpleveldefs: {
        // No `callbacks`.  `moduledef.py` defines none, and the collector-side
        // hook contract keeps its calls allocation-free, so nothing here can
        // run an app-level callback around a collection the way
        // `invoke_gc_callback` does.  Binding an empty list would satisfy
        // `hasattr` and then never call it, which is a silent failure where the
        // missing attribute is a loud one; `gc.hooks` is the notification
        // surface that does fire.
        "garbage"             => w_list_new(vec![]),
        "DEBUG_STATS"         => w_int_new(1),
        "DEBUG_COLLECTABLE"   => w_int_new(2),
        "DEBUG_UNCOLLECTABLE" => w_int_new(4),
        "DEBUG_SAVEALL"       => w_int_new(32),
        "DEBUG_LEAK"          => w_int_new(38),
        "GcCollectStepStats"  => hook::gc_collect_step_stats_type(),
        "GcRef"               => gcref::type_object(),
        "hooks"               => hook::hooks_object(),
    },
    functions: {
        "disable"              / 0 = interp_gc::disable,
        "enable"               / 0 = interp_gc::enable,
        "isenabled"            / 0 = interp_gc::isenabled,
        "get_referrers"        / * = referents::get_referrers,
        "get_referents"        / * = referents::get_referents,
        "get_rpy_roots"        / 0 = referents::get_rpy_roots,
        "get_rpy_referents"    / 1 = referents::get_rpy_referents,
        "set_threshold"        / * = interp_gc::set_threshold,
        "get_threshold"        / 0 = interp_gc::get_threshold,
        "get_count"            / 0 = interp_gc::get_count,
        "get_debug"            / 0 = interp_gc::get_debug,
        "set_debug"            / 1 = interp_gc::set_debug,
        "is_tracked"           / 1 = interp_gc::is_tracked,
        "get_rpy_memory_usage" / 1 = referents::get_rpy_memory_usage,
        "get_rpy_type_index"   / 1 = referents::get_rpy_type_index,
        "_dump_rpy_heap"       / 1 = referents::_dump_rpy_heap,
        "get_typeids_z"        / 0 = referents::get_typeids_z,
        "get_typeids_list"     / 0 = referents::get_typeids_list,
        "is_finalized"         / 1 = interp_gc::is_finalized,
        "freeze"               / 0 = interp_gc::freeze,
        "unfreeze"             / 0 = interp_gc::unfreeze,
        "get_freeze_count"     / 0 = interp_gc::get_freeze_count,
    },
    // moduledef.py binds these by submodule path (`interp_gc.collect`,
    // `referents.get_stats`, `app_referents.dump_rpy_heap`). The function
    // pointer is that submodule function; this table only stores it.
    extra_init: |ns| {
        let mut ns = ns;
        fn install(
            mut ns: PyObjectRef,
            name: &'static str,
            func: pyre_interpreter::BuiltinCodeFn,
            arity: u16,
            sig: Option<pyre_interpreter::Signature>,
        ) -> PyObjectRef {
            let mut value = pyre_object::with_roots!(ns => pyre_interpreter::gateway::with_module(
                "gc",
                pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                    name, func, arity, sig,
                ),
            ));
            pyre_interpreter::__pyre_store!(ns, name, value);
            ns
        }
        ns = install(
            ns,
            "collect",
            interp_gc::collect,
            interp_gc::collect_pyre_arity(),
            interp_gc::collect_pyre_sig(),
        );
        ns = install(
            ns,
            "collect_step",
            interp_gc::collect_step,
            interp_gc::collect_step_pyre_arity(),
            interp_gc::collect_step_pyre_sig(),
        );
        ns = install(
            ns,
            "enable_finalizers",
            interp_gc::enable_finalizers,
            interp_gc::enable_finalizers_pyre_arity(),
            interp_gc::enable_finalizers_pyre_sig(),
        );
        ns = install(
            ns,
            "disable_finalizers",
            interp_gc::disable_finalizers,
            interp_gc::disable_finalizers_pyre_arity(),
            interp_gc::disable_finalizers_pyre_sig(),
        );
        ns = install(
            ns,
            "get_objects",
            referents::get_objects,
            referents::get_objects_pyre_arity(),
            referents::get_objects_pyre_sig(),
        );
        ns = install(
            ns,
            "_get_stats",
            referents::get_stats,
            referents::get_stats_pyre_arity(),
            referents::get_stats_pyre_sig(),
        );
        ns = install(
            ns,
            "get_stats",
            app_referents::get_stats,
            app_referents::get_stats_pyre_arity(),
            app_referents::get_stats_pyre_sig(),
        );
        ns = install(
            ns,
            "dump_rpy_heap",
            app_referents::dump_rpy_heap,
            app_referents::dump_rpy_heap_pyre_arity(),
            app_referents::dump_rpy_heap_pyre_sig(),
        );
        let _ = ns;
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `referents.py W_GcRef`: the wrapper's raw gcref field is a normal traced
    // edge, so an internal object stays live and is forwarded in place.
    types.push(ModuleGcType {
        descriptor: <gcref::W_GcRef as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `hook.py W_AppLevelHooks`: the process-owned hooks singleton keeps the
    // three app callbacks in ordinary traced fields.
    types.push(ModuleGcType {
        descriptor: <hook::W_AppLevelHooks as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
    // `referents.py W_GcStats`: scalar statistics live on the W_Root itself, so
    // the class registers even though it has no trace edges.
    types.push(ModuleGcType {
        descriptor: <stats::W_GcStats as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
