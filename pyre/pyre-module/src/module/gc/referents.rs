//! `pypy/module/gc/referents.py`: the raw-heap inspection surface
//! (`GcRef`, `GcStats`, referents, roots, heap dump, typeids).

use pyre_object::*;

use super::interp_gc::NUM_GENERATIONS;

/// `referents.py W_GcRef(W_Root)`: an app-level handle for a raw GC
/// object that is not itself a Python object.  The field is deliberately on
/// the wrapper and participates in normal type tracing; a side table would
/// neither keep the referent alive nor receive forwarding updates.
pub mod gcref {
    use super::*;

    #[pyre_interpreter::pyre_class("GcRef")]
    pub struct W_GcRef {
        pub gcref: PyObjectRef,
    }

    #[pyre_interpreter::pyre_methods]
    impl W_GcRef {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(pyre_interpreter::PyError::type_error(
                "GcRef() takes no arguments",
            ))
        }
    }

    /// Allocate from a raw target already published at `target_slot` on the
    /// shadow stack.  `type_object()` may allocate while it initializes the
    /// TypeDef, so the target is re-read afterwards.
    pub fn wrap_rooted(target_slot: usize) -> PyObjectRef {
        let w_type = type_object();
        let target = pyre_object::gc_roots::shadow_stack_get(target_slot);
        let value = W_GcRef {
            ob: PyObject {
                ob_type: &GCREF_TYPE,
                w_class: w_type,
            },
            gcref: target,
        };
        pyre_object::lltype::malloc_typed_managed(value) as PyObjectRef
    }

    pub fn unwrap(w_obj: PyObjectRef) -> majit_ir::GcRef {
        W_GcRef::from_obj(w_obj)
            .map(|wrapper| majit_ir::GcRef(wrapper.gcref as usize))
            .unwrap_or(majit_ir::GcRef(w_obj as usize))
    }
}

/// `referents.py W_GcStats`.  These are native integer fields on the
/// interpreter object, matching the upstream W_Root rather than a Python dict
/// or a process-global side table.
pub mod stats {
    use super::*;

    #[pyre_interpreter::pyre_class("GcStats")]
    pub struct W_GcStats {
        pub(in crate::module::gc) total_memory_pressure: i64,
        pub(in crate::module::gc) total_gc_memory: i64,
        pub(in crate::module::gc) total_allocated_memory: i64,
        pub(in crate::module::gc) peak_memory: i64,
        pub(in crate::module::gc) peak_allocated_memory: i64,
        pub(in crate::module::gc) jit_backend_allocated: i64,
        pub(in crate::module::gc) jit_backend_used: i64,
        pub(in crate::module::gc) total_arena_memory: i64,
        pub(in crate::module::gc) total_rawmalloced_memory: i64,
        pub(in crate::module::gc) peak_arena_memory: i64,
        pub(in crate::module::gc) peak_rawmalloced_memory: i64,
        pub(in crate::module::gc) nursery_size: i64,
        pub(in crate::module::gc) total_gc_time: i64,
    }

    #[pyre_interpreter::pyre_methods]
    impl W_GcStats {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(pyre_interpreter::PyError::type_error(
                "object.__new__(GcStats) is not safe, use GcStats.__new__()",
            ))
        }

        #[getter]
        fn total_memory_pressure(&self) -> i64 {
            self.total_memory_pressure
        }
        #[getter]
        fn total_gc_memory(&self) -> i64 {
            self.total_gc_memory
        }
        #[getter]
        fn total_allocated_memory(&self) -> i64 {
            self.total_allocated_memory
        }
        #[getter]
        fn peak_memory(&self) -> i64 {
            self.peak_memory
        }
        #[getter]
        fn peak_allocated_memory(&self) -> i64 {
            self.peak_allocated_memory
        }
        #[getter]
        fn jit_backend_allocated(&self) -> i64 {
            self.jit_backend_allocated
        }
        #[getter]
        fn jit_backend_used(&self) -> i64 {
            self.jit_backend_used
        }
        #[getter]
        fn total_arena_memory(&self) -> i64 {
            self.total_arena_memory
        }
        #[getter]
        fn total_rawmalloced_memory(&self) -> i64 {
            self.total_rawmalloced_memory
        }
        #[getter]
        fn peak_arena_memory(&self) -> i64 {
            self.peak_arena_memory
        }
        #[getter]
        fn peak_rawmalloced_memory(&self) -> i64 {
            self.peak_rawmalloced_memory
        }
        #[getter]
        fn nursery_size(&self) -> i64 {
            self.nursery_size
        }
        #[getter]
        fn total_gc_time(&self) -> i64 {
            self.total_gc_time
        }
    }

    pub fn new(memory_pressure: bool) -> PyObjectRef {
        // `#[pyre_class]::allocate` stamps `get_instantiate(PYTYPE)` into the
        // header. Initialize the TypeDef first so that slot is the real class,
        // not the macro static's pre-init name placeholder.
        let _ = type_object();
        let stats = majit_gc::active_gc_memory_stats();
        let (jit_backend_allocated, jit_backend_used) = majit_gc::active_jit_backend_memory_stats();
        // referents.py:192-195: the optional selector performs the collector's
        // root-reachable `inspector.count_memory_pressure` walk; otherwise it
        // preserves the public -1 sentinel.
        let total_memory_pressure = if memory_pressure {
            majit_gc::total_memory_pressure() as i64
        } else {
            -1
        };
        W_GcStats::allocate(W_GcStats {
            ob: PyObject::default(),
            total_memory_pressure,
            total_gc_memory: stats.total_gc_memory as i64,
            total_allocated_memory: stats.total_allocated_memory as i64,
            peak_memory: stats.peak_memory as i64,
            peak_allocated_memory: stats.peak_allocated_memory as i64,
            jit_backend_allocated: jit_backend_allocated as i64,
            jit_backend_used: jit_backend_used as i64,
            total_arena_memory: stats.total_arena_memory as i64,
            total_rawmalloced_memory: stats.total_rawmalloced_memory as i64,
            peak_arena_memory: stats.peak_arena_memory as i64,
            peak_rawmalloced_memory: stats.peak_rawmalloced_memory as i64,
            nursery_size: stats.nursery_size as i64,
            total_gc_time: stats.total_gc_time_ms as i64,
        })
    }
}

fn pin_object(object: majit_ir::GcRef) {
    let _ = pyre_object::gc_roots::pin_root(object.0 as PyObjectRef);
}

/// `[3.14-spec]` PyPy's `referents.py:115-122` exposes every app-level
/// object reached by `rgc.do_get_objects`, while CPython 3.14
/// `Modules/gcmodule.c:319-342` exposes only objects tracked by its cyclic
/// collector. Keep the PyPy/RPython traversal in `majit_gc`; filter only at
/// the public CPython-compatible boundary, using the same tracked-state model
/// as `interp_gc::is_tracked`.
fn pin_cpython_tracked_object(object: majit_ir::GcRef) {
    let w_obj = object.0 as PyObjectRef;
    if pyre_interpreter::typedef::cpython_object_is_gc(w_obj) {
        let _ = pyre_object::gc_roots::pin_root(w_obj);
    }
}

/// CPython's container traversal sees logical entries even where a PyPy list
/// strategy, dict strategy, or specialised tuple stores an unboxed scalar.  The
/// collector walk correctly has no GC edge to report for those fields, so
/// materialise only the missing logical half at the public API boundary.
/// Object-strategy entries remain the collector's responsibility: rebuilding
/// all of them here would both duplicate results and lose the identity of
/// direct referents.
fn pin_unboxed_container_referents(source_slot: usize) {
    let w_obj = pyre_object::gc_roots::shadow_stack_get(source_slot);
    if w_obj.is_null()
        || (pyre_object::tagged_int::CAN_BE_TAGGED && tagged_int::is_tagged_int(w_obj))
    {
        return;
    }
    unsafe {
        if pyre_object::is_list(w_obj) {
            let list = &*(w_obj as *const listobject::W_ListObject);
            if matches!(
                list.strategy,
                listobject::ListStrategy::Empty
                    | listobject::ListStrategy::Size
                    | listobject::ListStrategy::Object
            ) {
                return;
            }
            let len = listobject::w_list_len(w_obj);
            for index in 0..len {
                let list = pyre_object::gc_roots::shadow_stack_get(source_slot);
                if let Some(item) = listobject::w_list_getitem(list, index as i64) {
                    let _ = pyre_object::gc_roots::pin_root(item);
                }
            }
        } else if std::ptr::eq((*w_obj).ob_type, &DICT_TYPE) {
            let kind = dictmultiobject::w_dict_get_strategy(w_obj).strategy_kind();
            if !matches!(
                kind,
                dictmultiobject::StrategyKind::Int | dictmultiobject::StrategyKind::Bytes
            ) {
                return;
            }
            let len = dictmultiobject::w_dict_len(w_obj);
            // A slot walk: a deleted entry leaves a tombstone the count of
            // live pairs does not name.
            let mut cursor = 0;
            for _ in 0..len {
                let dict = pyre_object::gc_roots::shadow_stack_get(source_slot);
                let Some((slot, key, _)) = dictmultiobject::w_dict_next_item(dict, cursor) else {
                    break;
                };
                cursor = slot + 1;
                // The typed strategy's GC walker already reported the
                // boxed value; only its native i64/Vec<u8> key was absent.
                let _ = pyre_object::gc_roots::pin_root(key);
            }
        } else if is_specialised_tuple_ii(w_obj) || is_specialised_tuple_ff(w_obj) {
            // Both items live in inline i64/f64 fields, so these two variants
            // carry no GC-pointer slot at all and the walker reports an empty
            // tuple.  `w_tuple_getitem` re-wraps them the same way
            // `specialisedtupleobject.py:138-141 wraps[i](self.space, value)`
            // does.  `Cls_oo` stores both items as GC pointers and stays the
            // collector's.
            for index in 0..2 {
                let tuple = pyre_object::gc_roots::shadow_stack_get(source_slot);
                if let Some(item) = tupleobject::w_tuple_getitem(tuple, index) {
                    let _ = pyre_object::gc_roots::pin_root(item);
                }
            }
        }
    }
}

/// Remove one temporary root while retaining every result appended after it.
/// Shadow-stack slots, rather than copied addresses, are moved so a collection
/// during scalar materialisation cannot leave a stale result behind.
fn remove_root_slot_preserving_tail(slot: usize) {
    let end = pyre_object::gc_roots::shadow_stack_len();
    // Every caller pushes the root it names, so `slot < end` holds. Keep it a
    // runtime check anyway: a release build would otherwise underflow `end - 1`
    // below and truncate the stack to a nonsense length.
    if slot >= end {
        return;
    }
    for index in slot + 1..end {
        let value = pyre_object::gc_roots::shadow_stack_get(index);
        pyre_object::gc_roots::shadow_stack_set(index - 1, value);
    }
    pyre_object::gc_roots::shadow_stack_cell_truncate(
        pyre_object::gc_roots::shadow_stack_cell(),
        end - 1,
    );
}

/// `referents.py _list_w_obj_referents`: push the app-level objects
/// `w_obj` refers to directly onto the shadow stack. The collector walk looks
/// through interpreter-internal structs; the CPython-facing supplement then
/// restores logical entries hidden by PyPy's unboxed strategies.
///
/// Only managed-heap referents are reported, the same boundary `gc.get_objects`
/// and `gc.is_tracked` draw. An immortal referent carries a GC header but sits
/// outside the collector's ranges, and a slot such as `ob_type` can hold a
/// static that has no header at all, so there is no address the walk could
/// safely widen to.
fn pin_referents(w_obj: PyObjectRef) {
    let source_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    majit_gc::get_referents(majit_ir::GcRef(source as usize), pin_object);
    pin_unboxed_container_referents(source_slot);
    pin_heaptype_referent(source_slot);
    remove_root_slot_preserving_tail(source_slot);
}

/// `subtype_traverse` visits an object's own type when that type is a heap
/// type — "for a heaptype, the instances count as references to the type" — and
/// so does `type_traverse` for a class whose metaclass is one.  The collector
/// has no edge to report for it: `PyObject.ob_type` addresses a
/// `try_gc_alloc_stable_raw` header that never moves and that nothing traces.
/// Materialise it here, the same way `pin_unboxed_container_referents`
/// materialises the logical half of an unboxed container entry.
///
/// Some layouts already arrive with it — a tuple subclass carries its type
/// among the collector's own edges — so the referents pinned so far are scanned
/// and the type is appended only when it is not already among them.  The append
/// is at the end, which is where `subtype_traverse` puts it for an instance
/// with managed-dict values; a class or a tuple subclass reports it first
/// instead, and this module does not reproduce that order.
///
/// Gated on `cpython_object_is_gc` because that is the flag that decides
/// whether `tp_traverse` runs at all: a non-GC object has no referents, which
/// is why `gc.get_referents(1)` is empty.
fn pin_heaptype_referent(source_slot: usize) {
    let w_obj = pyre_object::gc_roots::shadow_stack_get(source_slot);
    if w_obj.is_null() || !pyre_interpreter::typedef::cpython_object_is_gc(w_obj) {
        return;
    }
    let Some(w_type) = pyre_interpreter::typedef::r#type(w_obj) else {
        return;
    };
    let w_type = w_type.as_ptr();
    if !unsafe { pyre_object::w_type_is_cpython_heaptype(w_type) } {
        return;
    }
    for slot in source_slot + 1..pyre_object::gc_roots::shadow_stack_len() {
        if pyre_object::gc_roots::shadow_stack_get(slot) == w_type {
            return;
        }
    }
    let _ = pyre_object::gc_roots::pin_root(w_type);
}

/// Wrap every raw collector node rooted in `[first, last)` as
/// `referents.py wrap`: app-level objects pass through, internal nodes
/// become `W_GcRef`.  Results are rooted as they are made because constructing
/// a later wrapper can initialize a type and allocate.
///
/// Returns the first shadow-stack slot of the wrapped range, which runs to the
/// stack top on return. A slot range rather than a `Vec<PyObjectRef>` because
/// every caller allocates a list next, and only the slots are forwarded.
fn wrap_raw_nodes(first: usize, last: usize) -> usize {
    let result_first = pyre_object::gc_roots::shadow_stack_len();
    for slot in first..last {
        let raw = majit_ir::GcRef(pyre_object::gc_roots::shadow_stack_get(slot) as usize);
        // `inspector.py:get_rpy_roots` returns its non-resizable raw list with
        // spare NULL entries; `referents.py:get_rpy_roots` removes those at
        // the Python boundary. `get_rpy_referents` has no NULL padding, so the
        // same check is harmless for its other caller.
        if raw.is_null() {
            continue;
        }
        let wrapped = if majit_gc::is_app_level_object(raw) {
            // The query enters the collector; the address comes back out of
            // the slot rather than out of the word held across it.
            pyre_object::gc_roots::shadow_stack_get(slot)
        } else {
            gcref::wrap_rooted(slot)
        };
        let _ = pyre_object::gc_roots::pin_root(wrapped);
    }
    result_first
}

/// Build a list holding the objects rooted at `slots`.
///
/// The list is allocated before any slot is read, and both the list and the
/// element are re-read around every append. Gathering the elements into a
/// `Vec<PyObjectRef>` first and allocating afterwards would hand the list
/// pre-copy addresses as soon as one of these allocations collects: the
/// collector forwards shadow-stack entries, not Rust vectors. Being pinned
/// keeps an object alive, which is not the same as keeping a copy of its
/// address valid.
fn list_from_root_slots(slots: impl IntoIterator<Item = usize>) -> PyObjectRef {
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_list_new_empty());
    for slot in slots {
        unsafe {
            w_list_append(
                pyre_object::gc_roots::shadow_stack_get(list_slot),
                pyre_object::gc_roots::shadow_stack_get(slot),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(list_slot)
}

/// `list_from_root_slots` over every slot pinned since `first`.
fn list_from_roots(first: usize) -> PyObjectRef {
    list_from_root_slots(first..pyre_object::gc_roots::shadow_stack_len())
}

#[cfg(feature = "sandbox")]
fn heap_dump_write_via_host(fd: i32, bytes: &[u8]) -> Result<isize, i32> {
    pyre_interpreter::host_seam::raw_heap_dump_write(fd, bytes)
        .map(|written| written as isize)
        // A non-OS seam failure still needs an errno. Use the collector's code
        // for targets and failure modes that cannot supply one.
        .map_err(|error| match error {
            pyre_interpreter::host_seam::SeamError::Os(errno) => errno,
            _ => majit_gc::HEAP_DUMP_EIO,
        })
}

pub(super) fn dump_rpy_heap_fd(fd: i32) -> Result<(), pyre_interpreter::PyError> {
    #[cfg(feature = "sandbox")]
    majit_gc::set_heap_dump_write(Some(heap_dump_write_via_host));
    match majit_gc::dump_rpy_heap(fd) {
        Ok(true) => Ok(()),
        Ok(false) => Err(pyre_interpreter::PyError::not_implemented(
            "operation not implemented by this GC",
        )),
        Err(errno) => Err(pyre_interpreter::PyError::os_error_with_errno(
            errno,
            "raw_os_write failed",
        )),
    }
}

fn typeids_z_bytes() -> Result<Vec<u8>, pyre_interpreter::PyError> {
    use rustpython_common::compression::zlib;

    let text = majit_gc::get_typeids_text().ok_or_else(|| {
        pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
    })?;
    zlib::compress(&text, 9, zlib::MAX_WBITS).map_err(|error| {
        let message = match error {
            zlib::InitError::InvalidOption => "Invalid initialization option".to_owned(),
            zlib::InitError::Zlib(message) => message,
        };
        pyre_interpreter::PyError::os_error(message)
    })
}

/// `referents.py get_stats`.
#[pyre_interpreter::pyre_function]
pub(super) fn get_stats(#[default(false)] memory_pressure: bool) -> PyObjectRef {
    stats::new(memory_pressure)
}

/// `referents.py get_objects`.
#[pyre_interpreter::pyre_function]
pub(super) fn get_objects(
    #[default(w_none())] generation: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // [3.15-spec] `referents.py get_objects` accepts only None and
    // returns every app-level object via `rgc.do_get_objects`.
    // `_PyGC_GetObjects` ignores generation after the same bounds
    // check `gc_get_objects_impl` already applies.  Validate the
    // argument, then walk the None path for every accepted value.
    // The audit event always reports -1 and fires before the
    // argument is examined, so an out-of-range value is still audited.
    let _generation_root = pyre_object::gc_roots::push_roots();
    let generation_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(generation);
    pyre_interpreter::module::sys::vm::audit("gc.get_objects", &[w_int_new(-1)])?;
    let generation = pyre_object::gc_roots::shadow_stack_get(generation_slot);
    if !unsafe { is_none(generation) } {
        if unsafe {
            !is_int_or_long(generation)
                && pyre_interpreter::baseobjspace::lookup(generation, "__index__").is_none()
        } {
            let generation = pyre_object::gc_roots::shadow_stack_get(generation_slot);
            let tp_name = pyre_interpreter::type_methods::arg_type_name(generation);
            return Err(pyre_interpreter::PyError::type_error(format!(
                "argument should be integer or None, not '{tp_name}'"
            )));
        }
        let generation = pyre_object::gc_roots::shadow_stack_get(generation_slot);
        let generation = pyre_interpreter::baseobjspace::int_w(
            pyre_interpreter::baseobjspace::space_index(generation)?,
        )?;
        if generation >= NUM_GENERATIONS {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "generation parameter must be less than the number of \
                 available generations ({NUM_GENERATIONS})"
            )));
        }
        if generation < -1 {
            return Err(pyre_interpreter::PyError::value_error(
                "generation parameter cannot be negative",
            ));
        }
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let first = pyre_object::gc_roots::shadow_stack_len();
    majit_gc::get_objects(-1, pin_cpython_tracked_object);
    Ok(list_from_roots(first))
}

/// `referents.py get_referrers`: list every app-level object, then keep the
/// ones whose direct referents include an argument.
pub(super) fn get_referrers(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // The argument scan at `referents.py` has no `break`, and
    // the multiplicity that follows from that is the contract: an
    // object referring to the same argument twice is reported once
    // (the membership test collapses it), but one that refers to two
    // of the arguments — or one argument passed twice — is reported
    // once per match.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let mut rooted_args = vec![std::ptr::null_mut(); args.len()];
    pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut rooted_args);
    pyre_interpreter::module::sys::vm::audit("gc.get_referrers", &rooted_args)?;
    let all_first = pyre_object::gc_roots::shadow_stack_len();
    majit_gc::get_objects(-1, pin_cpython_tracked_object);
    let all_last = pyre_object::gc_roots::shadow_stack_len();
    // Accumulate the matches as slot indices, not as addresses: the
    // entries stay pinned in `all_first..all_last`, but a copy of one
    // of their addresses goes stale the moment the list allocation
    // below moves the object it names.
    let mut result = Vec::new();
    for slot in all_first..all_last {
        let w_obj = pyre_object::gc_roots::shadow_stack_get(slot);
        let _refs = pyre_object::gc_roots::push_roots();
        let refs_first = pyre_object::gc_roots::shadow_stack_len();
        pin_referents(w_obj);
        let refs_last = pyre_object::gc_roots::shadow_stack_len();
        for index in 0..args.len() {
            let w_arg = pyre_object::gc_roots::shadow_stack_get(args_base + index);
            if (refs_first..refs_last).any(|s| pyre_object::gc_roots::shadow_stack_get(s) == w_arg)
            {
                result.push(slot);
            }
        }
    }
    Ok(list_from_root_slots(result))
}

/// `referents.py get_referents`.
pub(super) fn get_referents(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let mut rooted_args = vec![std::ptr::null_mut(); args.len()];
    pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut rooted_args);
    pyre_interpreter::module::sys::vm::audit("gc.get_referents", &rooted_args)?;
    let first = pyre_object::gc_roots::shadow_stack_len();
    for index in 0..args.len() {
        let w_obj = pyre_object::gc_roots::shadow_stack_get(args_base + index);
        pin_referents(w_obj);
    }
    Ok(list_from_roots(first))
}

/// `referents.py get_rpy_roots`.
pub(super) fn get_rpy_roots(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let first = pyre_object::gc_roots::shadow_stack_len();
    if !majit_gc::get_rpy_roots(pin_object) {
        return Err(pyre_interpreter::PyError::not_implemented(
            "operation not implemented by this GC",
        ));
    }
    let last = pyre_object::gc_roots::shadow_stack_len();
    Ok(list_from_roots(wrap_raw_nodes(first, last)))
}

/// `referents.py get_rpy_referents`.
pub(super) fn get_rpy_referents(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let first = pyre_object::gc_roots::shadow_stack_len();
    if !majit_gc::get_rpy_referents(raw, pin_object) {
        return Err(pyre_interpreter::PyError::not_implemented(
            "operation not implemented by this GC",
        ));
    }
    let last = pyre_object::gc_roots::shadow_stack_len();
    Ok(list_from_roots(wrap_raw_nodes(first, last)))
}

/// `referents.py get_rpy_memory_usage` / `inspector.py get_rpy_memory_usage`.
/// The size is just the translated object itself: no GC header and no
/// reachable internal storage.
pub(super) fn get_rpy_memory_usage(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let size = majit_gc::get_rpy_memory_usage(raw).ok_or_else(|| {
        pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
    })?;
    Ok(w_int_new(size as i64))
}

/// `referents.py get_rpy_type_index`: a positive index into the translated
/// type-info group (index zero is the upstream dummy member).
pub(super) fn get_rpy_type_index(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let index = majit_gc::get_rpy_type_index(raw).ok_or_else(|| {
        pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
    })?;
    Ok(w_int_new(index as i64))
}

/// `referents.py _dump_rpy_heap`.
pub(super) fn _dump_rpy_heap(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let fd = pyre_interpreter::baseobjspace::int_w(args[0])? as i32;
    dump_rpy_heap_fd(fd)?;
    Ok(w_none())
}

/// `referents.py get_typeids_z`.
pub(super) fn get_typeids_z(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &typeids_z_bytes()?,
    ))
}

/// `referents.py get_typeids_list`.
pub(super) fn get_typeids_list(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let list = majit_gc::get_typeids_list().ok_or_else(|| {
        pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
    })?;
    // Each `w_int_new` can collect, so pin as we go: an int built by an
    // earlier iteration would otherwise live only in a `Vec`.
    let _roots = pyre_object::gc_roots::push_roots();
    let first = pyre_object::gc_roots::shadow_stack_len();
    for value in list {
        let _ = pyre_object::gc_roots::pin_root(w_int_new(value as i64));
    }
    Ok(list_from_roots(first))
}
