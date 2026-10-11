//! App-level GC hooks — PyPy: `pypy/module/gc/hook.py`.

use majit_gc::GcStepTransition;
use pyre_interpreter::executioncontext::{
    ActionFlagOps, AsyncAction, AsyncActionControl, AsyncActionOps, ExecutionContext,
};
use pyre_interpreter::pyframe::PyFrame;
use pyre_object::*;
use rustpython_wtf8::Wtf8;
use std::sync::OnceLock;

struct GcMinorHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    total_memory_used: usize,
    pinned_objects: usize,
}

impl GcMinorHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            duration: 0.0,
            duration_min: f64::INFINITY,
            duration_max: 0.0,
            total_memory_used: 0,
            pinned_objects: 0,
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.duration = 0.0;
        self.duration_min = f64::INFINITY;
        self.duration_max = 0.0;
    }

    fn do_perform(&mut self) -> Result<(), pyre_interpreter::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let duration = self.duration;
        let duration_min = self.duration_min;
        let duration_max = self.duration_max;
        let total_memory_used = self.total_memory_used;
        let pinned_objects = self.pinned_objects;
        self.reset();

        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_minor });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_minor_stats(
            count,
            duration,
            duration_min,
            duration_max,
            total_memory_used,
            pinned_objects,
        )?;
        let _ = pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcMinorHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, pyre_interpreter::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

struct GcCollectStepHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    oldstate: u8,
    newstate: u8,
}

impl GcCollectStepHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            duration: 0.0,
            duration_min: f64::INFINITY,
            duration_max: 0.0,
            oldstate: 0,
            newstate: 0,
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.duration = 0.0;
        self.duration_min = f64::INFINITY;
        self.duration_max = 0.0;
    }

    fn do_perform(&mut self) -> Result<(), pyre_interpreter::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let duration = self.duration;
        let duration_min = self.duration_min;
        let duration_max = self.duration_max;
        let oldstate = self.oldstate;
        let newstate = self.newstate;
        self.reset();

        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_collect_step });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_collect_step_stats_full(
            count,
            duration,
            duration_min,
            duration_max,
            oldstate,
            newstate,
            super::interp_gc::is_done_states(oldstate, newstate),
        )?;
        let _ = pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcCollectStepHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, pyre_interpreter::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

struct GcCollectHookAction {
    base: AsyncAction,
    depth: usize,
    count: i64,
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
}

impl GcCollectHookAction {
    fn new() -> Self {
        Self {
            base: AsyncAction::default(),
            depth: 0,
            count: 0,
            num_major_collects: 0,
            arenas_count_before: 0,
            arenas_count_after: 0,
            arenas_bytes: 0,
            rawmalloc_bytes_before: 0,
            rawmalloc_bytes_after: 0,
            pinned_objects: 0,
        }
    }

    fn do_perform(&mut self) -> Result<(), pyre_interpreter::PyError> {
        // `self` is a field of this very allocation, so read the callback
        // through the pointer rather than borrowing the whole singleton.
        let Some(hooks) = app_hooks_ptr() else {
            return Ok(());
        };
        let count = self.count;
        let num_major_collects = self.num_major_collects;
        let arenas_count_before = self.arenas_count_before;
        let arenas_count_after = self.arenas_count_after;
        let arenas_bytes = self.arenas_bytes;
        let rawmalloc_bytes_before = self.rawmalloc_bytes_before;
        let rawmalloc_bytes_after = self.rawmalloc_bytes_after;
        let pinned_objects = self.pinned_objects;
        self.count = 0;

        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(unsafe { (*hooks).w_on_gc_collect });
        let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let stats = new_collect_stats(
            count,
            num_major_collects,
            arenas_count_before,
            arenas_count_after,
            arenas_bytes,
            rawmalloc_bytes_before,
            rawmalloc_bytes_after,
            pinned_objects,
        )?;
        let _ = pyre_object::gc_roots::pin_root(stats);
        let stats_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_interpreter::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &[pyre_object::gc_roots::shadow_stack_get(stats_slot)],
        )?;
        Ok(())
    }
}

impl AsyncActionOps for GcCollectHookAction {
    fn perform(
        &mut self,
        _executioncontext: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<AsyncActionControl, pyre_interpreter::PyError> {
        if self.depth != 0 {
            return Ok(AsyncActionControl::Continue);
        }
        self.depth += 1;
        let result = self.do_perform();
        self.depth -= 1;
        result.map(|()| AsyncActionControl::Continue)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// `hook.py W_AppLevelHooks`. The callback references live directly
/// on the singleton owner, so the generated type tracer forwards them; the
/// three action objects are embedded exactly as its `gc_minor`,
/// `gc_collect_step`, and `gc_collect` attributes are upstream.
#[pyre_interpreter::pyre_class("GcHooks")]
pub struct W_AppLevelHooks {
    pub w_on_gc_minor: PyObjectRef,
    pub w_on_gc_collect_step: PyObjectRef,
    pub w_on_gc_collect: PyObjectRef,
    gc_minor_enabled: bool,
    gc_collect_step_enabled: bool,
    gc_collect_enabled: bool,
    gc_minor: GcMinorHookAction,
    gc_collect_step: GcCollectStepHookAction,
    gc_collect: GcCollectHookAction,
}

impl W_AppLevelHooks {
    fn write_barrier(&mut self) {
        pyre_object::gc_hook::try_gc_write_barrier_managed(
            self as *mut Self as pyre_object::gc_hook::GCREF,
        );
    }
}

#[pyre_interpreter::pyre_methods]
impl W_AppLevelHooks {
    #[getter]
    fn on_gc_minor(&self) -> PyObjectRef {
        self.w_on_gc_minor
    }

    #[setter]
    fn set_on_gc_minor(&mut self, w_obj: PyObjectRef) {
        self.gc_minor_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_minor = w_obj;
        self.write_barrier();
    }

    #[getter]
    fn on_gc_collect_step(&self) -> PyObjectRef {
        self.w_on_gc_collect_step
    }

    #[setter]
    fn set_on_gc_collect_step(&mut self, w_obj: PyObjectRef) {
        self.gc_collect_step_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_collect_step = w_obj;
        self.write_barrier();
    }

    #[getter]
    fn on_gc_collect(&self) -> PyObjectRef {
        self.w_on_gc_collect
    }

    #[setter]
    fn set_on_gc_collect(&mut self, w_obj: PyObjectRef) {
        self.gc_collect_enabled = !unsafe { is_none(w_obj) };
        self.w_on_gc_collect = w_obj;
        self.write_barrier();
    }

    fn set(&mut self, w_obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
        // hook.py:100-107 — fetch all three first, so a missing later
        // attribute leaves the existing hook set untouched.
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_a = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_minor",
        )?;
        let _ = pyre_object::gc_roots::pin_root(w_a);
        let a_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_b = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_collect_step",
        )?;
        let _ = pyre_object::gc_roots::pin_root(w_b);
        let b_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_c = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "on_gc_collect",
        )?;
        let _ = pyre_object::gc_roots::pin_root(w_c);
        let c_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        self.set_on_gc_minor(pyre_object::gc_roots::shadow_stack_get(a_slot));
        self.set_on_gc_collect_step(pyre_object::gc_roots::shadow_stack_get(b_slot));
        self.set_on_gc_collect(pyre_object::gc_roots::shadow_stack_get(c_slot));
        Ok(())
    }

    fn reset(&mut self) {
        self.set_on_gc_minor(w_none());
        self.set_on_gc_collect_step(w_none());
        self.set_on_gc_collect(w_none());
    }
}

static HOOKS_OBJECT: OnceLock<usize> = OnceLock::new();

/// Root the `space.fromcache(W_AppLevelHooks)` singleton independently of
/// the `gc` module dictionary.  Upstream ownership is the process-wide object
/// space, so deleting/rebinding `gc.hooks` must not let the action owner (or
/// its callback fields) be swept while the actionflag still stores pointers
/// into it. `allocate_stable` makes relocation impossible; the visitor marks
/// the owner and its registered type trace reaches the three callbacks.
pub fn walk_hook_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let Some(&addr) = HOOKS_OBJECT.get() else {
        return;
    };
    let mut root = majit_ir::GcRef(addr);
    visitor(&mut root);
    debug_assert_eq!(root.0, addr, "allocate_stable GcHooks moved");
}

/// The singleton as a raw pointer, without borrowing it.
///
/// The three hook actions are *fields* of `W_AppLevelHooks`, so every
/// `AsyncActionOps::perform` already holds `&mut` over part of this
/// allocation. `W_AppLevelHooks::from_obj` hands out `&'static mut Self`,
/// which would overlap that borrow — and so would a shared `&`. Callers read
/// and write through this pointer instead, narrowing any reference they do
/// form to the single field they touch. Holding it across an allocating call
/// is safe because `initialize` uses `allocate_stable`, which is what
/// `walk_hook_roots`'s `debug_assert_eq!` above pins down.
fn app_hooks_ptr() -> Option<*mut W_AppLevelHooks> {
    let &addr = HOOKS_OBJECT.get()?;
    let obj = addr as PyObjectRef;
    unsafe { pyre_object::py_type_check(obj, &APPLEVELHOOKS_TYPE) }
        .then_some(obj as *mut W_AppLevelHooks)
}

/// Create the space-owned singleton and bind its three actions to the shared
/// `space.actionflag`. The main ExecutionContext calls this during bootstrap;
/// worker ECs carry [`pyre_interpreter::executioncontext::SpaceActionFlag`] references to
/// that same flag, so a collection performed by a worker fires and dispatches
/// the same action indexes there, matching PyPy's process-owned object space.
pub fn initialize(
    space: PyObjectRef,
    actionflag: &mut (dyn ActionFlagOps + 'static),
) -> PyObjectRef {
    *HOOKS_OBJECT.get_or_init(|| {
        let _ = type_object();
        let none = w_none();
        let obj = W_AppLevelHooks::allocate_stable(W_AppLevelHooks {
            ob: PyObject::default(),
            w_on_gc_minor: none,
            w_on_gc_collect_step: none,
            w_on_gc_collect: none,
            gc_minor_enabled: false,
            gc_collect_step_enabled: false,
            gc_collect_enabled: false,
            gc_minor: GcMinorHookAction::new(),
            gc_collect_step: GcCollectStepHookAction::new(),
            gc_collect: GcCollectHookAction::new(),
        });

        let hooks = W_AppLevelHooks::from_obj(obj).expect("fresh GcHooks layout");
        hooks
            .gc_minor
            .register_nonperiodic_action(space, actionflag);
        hooks
            .gc_collect_step
            .register_nonperiodic_action(space, actionflag);
        hooks
            .gc_collect
            .register_nonperiodic_action(space, actionflag);
        majit_gc::hook::register_gc_hooks(majit_gc::hook::GcHookCallbacks {
            is_gc_minor_enabled,
            is_gc_collect_step_enabled,
            is_gc_collect_enabled,
            on_gc_minor,
            on_gc_collect_step,
            on_gc_collect,
        });
        obj as usize
    }) as PyObjectRef
}

pub fn hooks_object() -> PyObjectRef {
    if let Some(&addr) = HOOKS_OBJECT.get() {
        return addr as PyObjectRef;
    }
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    assert!(
        !ec.is_null(),
        "gc.hooks initialized without an ExecutionContext"
    );
    unsafe {
        initialize(
            (*ec).space,
            &mut (*ec).actionflag as &mut (dyn ActionFlagOps + 'static),
        )
    }
}

fn is_gc_minor_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_minor_enabled })
}

fn is_gc_collect_step_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_collect_step_enabled })
}

fn is_gc_collect_enabled() -> bool {
    app_hooks_ptr().is_some_and(|hooks| unsafe { (*hooks).gc_collect_enabled })
}

fn on_gc_minor(duration: f64, total_memory_used: usize, pinned_objects: usize) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_minor };
    action.count += 1;
    action.duration += duration;
    action.duration_min = action.duration_min.min(duration);
    action.duration_max = action.duration_max.max(duration);
    action.total_memory_used = total_memory_used;
    action.pinned_objects = pinned_objects;
    action.fire();
}

fn on_gc_collect_step(duration: f64, oldstate: u8, newstate: u8) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_collect_step };
    action.count += 1;
    action.duration += duration;
    action.duration_min = action.duration_min.min(duration);
    action.duration_max = action.duration_max.max(duration);
    action.oldstate = oldstate;
    action.newstate = newstate;
    action.fire();
}

#[allow(clippy::too_many_arguments)]
fn on_gc_collect(
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
) {
    let Some(hooks) = app_hooks_ptr() else { return };
    let action = unsafe { &mut (*hooks).gc_collect };
    action.count += 1;
    action.num_major_collects = num_major_collects;
    action.arenas_count_before = arenas_count_before;
    action.arenas_count_after = arenas_count_after;
    action.arenas_bytes = arenas_bytes;
    action.rawmalloc_bytes_before = rawmalloc_bytes_before;
    action.rawmalloc_bytes_after = rawmalloc_bytes_after;
    action.pinned_objects = pinned_objects;
    action.fire();
}

/// `hook.py W_GcCollectStepStats` takes the four states from `incminimark`,
/// where they are declared and compared against it, and numbers its own one
/// past the last.
pub(super) const STATE_SCANNING: u8 = GcStepTransition::STATE_SCANNING;
const STATE_MARKING: u8 = GcStepTransition::STATE_MARKING;
const STATE_SWEEPING: u8 = GcStepTransition::STATE_SWEEPING;
const STATE_FINALIZING: u8 = GcStepTransition::STATE_FINALIZING;
pub(super) const STATE_USERDEL: u8 = GcStepTransition::STATE_FINALIZING + 1;

fn collect_step_stat_value(
    args: &[PyObjectRef],
    name: &'static str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let value = unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_node_getdictvalue(
            args[1],
            Wtf8::new(name),
        )
    };
    value.ok_or_else(|| {
        pyre_interpreter::PyError::attribute_error("uninitialized GcCollectStepStats")
    })
}

macro_rules! collect_step_stat_getter {
    ($function:ident, $name:literal) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            collect_step_stat_value(args, $name)
        }
    };
}

collect_step_stat_getter!(collect_step_count, "_count");
collect_step_stat_getter!(collect_step_duration, "_duration");
collect_step_stat_getter!(collect_step_duration_min, "_duration_min");
collect_step_stat_getter!(collect_step_duration_max, "_duration_max");
collect_step_stat_getter!(collect_step_oldstate, "_oldstate");
collect_step_stat_getter!(collect_step_newstate, "_newstate");
collect_step_stat_getter!(collect_step_major_is_done, "_major_is_done");

/// Whether `name` is one of a stats object's hidden storage slots.
///
/// Both stats types keep their values in mapdict slots under single-underscore
/// names, so the rule is the prefix rather than the seven names that exist
/// today — a slot added later is hidden without a second edit. Dunders are not
/// storage and stay reachable: upstream's `W_GcCollectStepStats` is an ordinary
/// `TypeDef` object, so `__class__` and `__repr__` answer the way they do on
/// any other one. `__dict__` is the exception, because these objects have none.
fn is_hidden_stat_slot(name: &str) -> bool {
    name == "__dict__" || (name.starts_with('_') && !name.starts_with("__"))
}

fn collect_step_stats_getattribute(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `args` is the gateway's native copy; `text_w` can collect.
    let mut w_obj = args[0];
    let name = pyre_object::with_roots!(w_obj => pyre_interpreter::baseobjspace::text_w(args[1]))?;
    if is_hidden_stat_slot(name) {
        return Err(pyre_interpreter::PyError::attribute_error(format!(
            "'GcCollectStepStats' object has no attribute '{name}'"
        )));
    }
    pyre_interpreter::baseobjspace::object_getattribute(w_obj, name)
}

fn collect_step_stats_setattr(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let name = pyre_interpreter::baseobjspace::text_w(args[1])?;
    Err(pyre_interpreter::PyError::attribute_error(format!(
        "readonly attribute '{name}'"
    )))
}

pub(super) fn gc_collect_step_stats_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("GcCollectStepStats", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "__getattribute__",
                pyre_interpreter::make_builtin_function_with_arity(
                    "__getattribute__",
                    collect_step_stats_getattribute,
                    2,
                )
            );
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                "__setattr__",
                pyre_interpreter::make_builtin_function_with_arity(
                    "__setattr__",
                    collect_step_stats_setattr,
                    3,
                )
            );
            for (name, value) in [
                ("STATE_SCANNING", STATE_SCANNING),
                ("STATE_MARKING", STATE_MARKING),
                ("STATE_SWEEPING", STATE_SWEEPING),
                ("STATE_FINALIZING", STATE_FINALIZING),
                ("STATE_USERDEL", STATE_USERDEL),
            ] {
                pyre_interpreter::__pyre_put_new!(ns_slot, name, w_int_new(value as i64));
            }
            {
                let mut names = pyre_object::gc_roots::RootedItems::new();
                for s in ["SCANNING", "MARKING", "SWEEPING", "FINALIZING", "USERDEL"] {
                    names.push(w_str_new(s));
                }
                pyre_interpreter::__pyre_put_new!(ns_slot, "GC_STATES", w_tuple_new(names.take()));
            }
            for (name, getter) in [
                (
                    "count",
                    collect_step_count as pyre_interpreter::gateway::BuiltinCodeFn,
                ),
                ("duration", collect_step_duration),
                ("duration_min", collect_step_duration_min),
                ("duration_max", collect_step_duration_max),
                ("oldstate", collect_step_oldstate),
                ("newstate", collect_step_newstate),
                ("major_is_done", collect_step_major_is_done),
            ] {
                pyre_interpreter::__pyre_put_new!(
                    ns_slot,
                    name,
                    pyre_interpreter::typedef::make_getset_descriptor_named(
                        pyre_interpreter::make_builtin_function_with_arity(name, getter, 2),
                        name,
                    )
                );
            }
        });
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp
    })
}

pub(super) fn new_collect_step_stats(
    oldstate: u8,
    newstate: u8,
    major_is_done: bool,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    new_collect_step_stats_full(1, -1.0, -1.0, -1.0, oldstate, newstate, major_is_done)
}

#[allow(clippy::too_many_arguments)]
fn new_collect_step_stats_full(
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    oldstate: u8,
    newstate: u8,
    major_is_done: bool,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    initialize_stats(
        gc_collect_step_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            ("_duration", StatValue::Float(duration)),
            ("_duration_min", StatValue::Float(duration_min)),
            ("_duration_max", StatValue::Float(duration_max)),
            ("_oldstate", StatValue::Int(oldstate as i64)),
            ("_newstate", StatValue::Int(newstate as i64)),
            ("_major_is_done", StatValue::Bool(major_is_done)),
        ],
        "GcCollectStepStats",
    )
}

fn readonly_stat_value(
    args: &[PyObjectRef],
    name: &'static str,
    typename: &'static str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let value = unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_node_getdictvalue(
            args[1],
            Wtf8::new(name),
        )
    };
    value.ok_or_else(|| {
        pyre_interpreter::PyError::attribute_error(format!("uninitialized {typename}"))
    })
}

macro_rules! readonly_stat_getter {
    ($function:ident, $field:literal, $typename:literal) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            readonly_stat_value(args, $field, $typename)
        }
    };
}

readonly_stat_getter!(minor_count, "_count", "GcMinorStats");
readonly_stat_getter!(minor_duration, "_duration", "GcMinorStats");
readonly_stat_getter!(minor_duration_min, "_duration_min", "GcMinorStats");
readonly_stat_getter!(minor_duration_max, "_duration_max", "GcMinorStats");
readonly_stat_getter!(
    minor_total_memory_used,
    "_total_memory_used",
    "GcMinorStats"
);
readonly_stat_getter!(minor_pinned_objects, "_pinned_objects", "GcMinorStats");

readonly_stat_getter!(collect_count, "_count", "GcCollectStats");
readonly_stat_getter!(
    collect_num_major_collects,
    "_num_major_collects",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_arenas_count_before,
    "_arenas_count_before",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_arenas_count_after,
    "_arenas_count_after",
    "GcCollectStats"
);
readonly_stat_getter!(collect_arenas_bytes, "_arenas_bytes", "GcCollectStats");
readonly_stat_getter!(
    collect_rawmalloc_bytes_before,
    "_rawmalloc_bytes_before",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_rawmalloc_bytes_after,
    "_rawmalloc_bytes_after",
    "GcCollectStats"
);
readonly_stat_getter!(collect_pinned_objects, "_pinned_objects", "GcCollectStats");

fn stats_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let name = pyre_interpreter::baseobjspace::text_w(args[1])?;
    Err(pyre_interpreter::PyError::attribute_error(format!(
        "readonly attribute '{name}'"
    )))
}

fn stats_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `args` is the gateway's native copy; `text_w` can collect.
    let mut w_obj = args[0];
    let name = pyre_object::with_roots!(w_obj => pyre_interpreter::baseobjspace::text_w(args[1]))?;
    if is_hidden_stat_slot(name) {
        return Err(pyre_interpreter::PyError::attribute_error(format!(
            "stats object has no attribute '{name}'"
        )));
    }
    pyre_interpreter::baseobjspace::object_getattribute(w_obj, name)
}

fn make_private_stats_type(
    name: &'static str,
    fields: &[(&'static str, pyre_interpreter::gateway::BuiltinCodeFn)],
) -> PyObjectRef {
    let tp = pyre_interpreter::typedef::make_builtin_type(name, |ns| {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__getattribute__",
            pyre_interpreter::make_builtin_function_with_arity(
                "__getattribute__",
                stats_getattribute,
                2,
            )
        );
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "__setattr__",
            pyre_interpreter::make_builtin_function_with_arity("__setattr__", stats_setattr, 3)
        );
        for &(field, getter) in fields {
            pyre_interpreter::__pyre_put_new!(
                ns_slot,
                field,
                pyre_interpreter::typedef::make_getset_descriptor_named(
                    pyre_interpreter::make_builtin_function_with_arity(field, getter, 2),
                    field,
                )
            );
        }
    });
    unsafe { typeobject::w_type_set_hasdict(tp, true) };
    unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
    tp
}

fn gc_minor_stats_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        make_private_stats_type(
            "GcMinorStats",
            &[
                ("count", minor_count),
                ("duration", minor_duration),
                ("duration_min", minor_duration_min),
                ("duration_max", minor_duration_max),
                ("total_memory_used", minor_total_memory_used),
                ("pinned_objects", minor_pinned_objects),
            ],
        )
    })
}

fn gc_collect_stats_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        make_private_stats_type(
            "GcCollectStats",
            &[
                ("count", collect_count),
                ("num_major_collects", collect_num_major_collects),
                ("arenas_count_before", collect_arenas_count_before),
                ("arenas_count_after", collect_arenas_count_after),
                ("arenas_bytes", collect_arenas_bytes),
                ("rawmalloc_bytes_before", collect_rawmalloc_bytes_before),
                ("rawmalloc_bytes_after", collect_rawmalloc_bytes_after),
                ("pinned_objects", collect_pinned_objects),
            ],
        )
    })
}

/// One private stats field, still unbuilt.
///
/// The point of deferring is rooting: a `Vec<(&str, PyObjectRef)>` built up
/// front holds every value in plain Rust memory while the remaining `w_*_new`
/// calls run, and the collector forwards shadow-stack slots, not Rust locals.
enum StatValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl StatValue {
    fn materialize(&self) -> PyObjectRef {
        match *self {
            StatValue::Int(v) => w_int_new(v),
            StatValue::Float(v) => w_float_new(v),
            StatValue::Bool(v) => w_bool_from(v),
        }
    }
}

/// Allocate a private stats instance of `stats_type` and fill it.
///
/// Both the field constructors and the mapdict transition inside
/// `instance_node_setdictvalue` allocate, so either can move the instance and
/// the value a Rust local names. Pin each on the shadow stack and read them
/// back through their slots for every store, the way `populate_public_gc_stats`
/// below does. The instance is created here rather than passed in so a caller
/// cannot hand over one that was allocated before any root existed.
fn initialize_stats(
    stats_type: PyObjectRef,
    fields: &[(&'static str, StatValue)],
    typename: &'static str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let stats_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_instance_new(stats_type));
    for (name, value) in fields {
        // The value pin lives across this one store. The instance stays on
        // `_roots`.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value.materialize());
        let stored = unsafe {
            pyre_interpreter::objspace::std::mapdict::instance_node_setdictvalue(
                pyre_object::gc_roots::shadow_stack_get(stats_slot),
                Wtf8::new(name),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )
        };
        if !stored {
            return Err(pyre_interpreter::PyError::attribute_error(format!(
                "cannot initialize {typename}"
            )));
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(stats_slot))
}

fn new_minor_stats(
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    total_memory_used: usize,
    pinned_objects: usize,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    initialize_stats(
        gc_minor_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            ("_duration", StatValue::Float(duration)),
            ("_duration_min", StatValue::Float(duration_min)),
            ("_duration_max", StatValue::Float(duration_max)),
            (
                "_total_memory_used",
                StatValue::Int(total_memory_used as i64),
            ),
            ("_pinned_objects", StatValue::Int(pinned_objects as i64)),
        ],
        "GcMinorStats",
    )
}

#[allow(clippy::too_many_arguments)]
fn new_collect_stats(
    count: i64,
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    initialize_stats(
        gc_collect_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            (
                "_num_major_collects",
                StatValue::Int(num_major_collects as i64),
            ),
            (
                "_arenas_count_before",
                StatValue::Int(arenas_count_before as i64),
            ),
            (
                "_arenas_count_after",
                StatValue::Int(arenas_count_after as i64),
            ),
            ("_arenas_bytes", StatValue::Int(arenas_bytes as i64)),
            (
                "_rawmalloc_bytes_before",
                StatValue::Int(rawmalloc_bytes_before as i64),
            ),
            (
                "_rawmalloc_bytes_after",
                StatValue::Int(rawmalloc_bytes_after as i64),
            ),
            ("_pinned_objects", StatValue::Int(pinned_objects as i64)),
        ],
        "GcCollectStats",
    )
}
