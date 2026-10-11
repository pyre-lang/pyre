//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-interpreter operation with the
//! correct calling convention and integer-based parameter passing.

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_metainterp::{TraceCtx, default_effect_info};

use pyre_interpreter::{
    jit_binary_value_from_tag, jit_bool_value_from_truth, jit_compare_value_from_tag,
    jit_unary_invert_value,
};
use pyre_interpreter::{jit_range_iter_next_or_null, jit_sequence_getitem};
use pyre_object::*;

pub use pyre_interpreter::{
    FlatBuildKind, callable_call_helper, flat_build_helper, jit_build_list_0, jit_build_list_1,
    jit_build_list_2, jit_build_list_3, jit_build_list_4, jit_build_list_5, jit_build_list_6,
    jit_build_list_7, jit_build_list_8, jit_build_map_0, jit_build_map_1, jit_build_map_2,
    jit_build_map_3, jit_build_map_4, jit_build_tuple_0, jit_build_tuple_1, jit_build_tuple_2,
    jit_build_tuple_3, jit_build_tuple_4, jit_build_tuple_5, jit_build_tuple_6, jit_build_tuple_7,
    jit_build_tuple_8, jit_call_callable_0, jit_call_callable_1, jit_call_callable_2,
    jit_call_callable_3, jit_call_callable_4, jit_call_callable_5, jit_call_callable_6,
    jit_call_callable_7, jit_call_callable_8, jit_load_name_from_namespace,
    jit_make_function_from_globals, jit_store_name_to_namespace, known_builtin_call_helper,
    known_function_call_helper, register_jit_function_caller,
};

/// Residual CALL targets go through [`pyre_interpreter::residual_word_addr!`]:
/// a path-based publisher matching `cpu_word!` / `word_fn_addr!`, so the
/// wasm32 closure is a fn item (`word_publish` `zeroed()`). Native keeps the
/// raw address.

pub fn emit_trace_call_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // pyjitpl.py do_residual_call parity: thread the
    // codewriter-analyzed `EffectInfo` through `record_nospec`. The
    // codewriter's `CallControl::getcalldescr`
    // (`majit-translate/src/codewriter/call.rs`) ports call.py
    // including the raise / random-effects / write / collect /
    // virtualizable / quasi-immut analyzers; the gap is the trace-side
    // plumbing — pyre-jit-trace helpers live outside the codewriter
    // pipeline so the analyzer's per-callee EI never reaches this
    // emit site. Until per-helper EI registration lands, the callee is
    // `graphanalyze.py:109-112`'s no-graph case, so
    // `default_effect_info()` hands out `effectinfo.MOST_GENERAL`.
    ctx.call_int_typed_with_effect(helper, args, arg_types, default_effect_info())
}

/// Float-bank counterpart of [`emit_trace_call_int_typed`], recording a `CallF`
/// for a helper whose C ABI returns its `f64` in a floating-point register.
/// Carries the same conservative `default_effect_info()`, so the two differ only
/// in the result bank.
pub fn emit_trace_call_float_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_float_typed_with_effect(helper, args, arg_types, default_effect_info())
}

/// `pyjitpl.py MIFrame.execute_varargs(opnum, argboxes, descr,
/// exc=False, pure=True)` parity for direct trace emit paths.
///
/// `emit_trace_call_int_typed` calls into the tracer with
/// `default_effect_info()` (`effectinfo.MOST_GENERAL`), so even an
/// `#[elidable_cannot_raise]` callee is recorded as a plain `CallI`.  This wrapper threads an explicit
/// `ElidableCannotRaise` `EffectInfo` through
/// `record_result_of_call_pure` (`pyjitpl.py`) and patches the
/// trace to `CallPureI`.
///
/// `concrete_arg_values` follows `_build_allboxes`
/// (`pyjitpl.py`): the funcbox's concrete value sits in slot 0
/// and the per-argument concrete values follow.  Direct trace-emit paths
/// have no jitcode-dispatch frame-operand-fetch, so the caller supplies
/// the values directly.
pub fn emit_trace_call_int_typed_elidable_cannot_raise(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    concrete_arg_values: &[Value],
    concrete_result: Value,
) -> OpRef {
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper,
        args,
        arg_types,
        Type::Int,
        effect,
        concrete_arg_values,
        concrete_result,
    )
}

pub fn emit_trace_call_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // See emit_trace_call_int_typed for the plumbing-gap rationale.
    ctx.call_ref_typed_with_effect(helper, args, arg_types, default_effect_info())
}

/// Run one strategy's lookup on raw JIT arguments.
///
/// A contended stripe acquire enters `before_external_block`, which is a GC
/// safepoint, so `dict` and `key` arrive as raw arguments that must be
/// shadow-rooted across the lock.  `lookup` opens the `DictOperationGuard` that
/// does it: the guard publishes both refs *before* it touches the stripe, so a
/// second one here would root the same two objects a second time and acquire
/// the same reentrant stripe again.  A miss returns `PY_NULL` for the caller's
/// `guard_nonnull` to side-exit on.
unsafe fn jit_dict_exact_lookup_or_null(
    dict: PyObjectRef,
    key: PyObjectRef,
    lookup: unsafe fn(PyObjectRef, PyObjectRef) -> Option<PyObjectRef>,
) -> PyObjectRef {
    lookup(dict, key).unwrap_or(PY_NULL)
}

pub extern "C" fn jit_dict_exact_int_lookup_or_null(
    dict: PyObjectRef,
    key: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        jit_dict_exact_lookup_or_null(
            dict,
            key,
            pyre_object::dictmultiobject::w_dict_lookup_or_null_int_strategy,
        )
    }
}

/// `rordereddict.py ll_dict_lookup` for `IntDictStrategy` — insertion-order
/// index, or `-1` on a miss.  The four-argument shape is the `dict.lookup`
/// oopspec (`d, key, hash, FLAG_LOOKUP`): `d` is the unerased `dstorage`
/// table, and `key`/`hash` are the unboxed `plain_int_w` word.  Int hash is
/// identity (`ll_int_hash`), which is why PyPy's after-opt is
/// `call_i(lookup, p62, i47, i47, 0)`.
pub extern "C" fn jit_dict_exact_int_lookup_index(
    storage: *const pyre_object::dictmultiobject::IntDictStorage,
    key: i64,
    _hash: i64,
    flag: i64,
) -> i64 {
    debug_assert_eq!(flag, 0, "only FLAG_LOOKUP is implemented");
    if storage.is_null() {
        return -1;
    }
    unsafe {
        let entries = &*storage;
        entries
            .index_of(&key)
            .map(|index| index as i64)
            .unwrap_or(-1)
    }
}

/// Value half of an int-strategy `dict.lookup`: `d.entries[i].value` after
/// the index settled.  Stands in for `getinteriorfield_gc_r` on
/// `odictentry.value` (`ll_dict_getitem_with_hash`); pyre's `RDict` is not
/// a GC array-of-structs, so the load cannot be a raw interior field.
/// A stale index answers `PY_NULL` so the caller's `GuardNonnull` side-exits.
pub extern "C" fn jit_dict_int_value_at(
    storage: *const pyre_object::dictmultiobject::IntDictStorage,
    index: i64,
) -> PyObjectRef {
    if index < 0 || storage.is_null() {
        return PY_NULL;
    }
    unsafe {
        let entries = &*storage;
        entries
            .get_slot(index as usize)
            .map(|(_, &value)| value)
            .unwrap_or(PY_NULL)
    }
}

/// Read a dict value at an entry index a traced `dict.lookup` settled on —
/// `rordereddict.py ll_dict_getitem`'s `d.entries[i].value` after
/// `ll_dict_lookup` returned the slot.  The read goes through the live storage
/// so a value-only overwrite stays visible, which is the whole reason the
/// lookup and the value read are two operations rather than one.
///
/// `w_dict_unicode_value_at_checked` re-validates the index against the key that
/// produced it; see there for why a bounds check alone is not enough.  Anything
/// the index no longer describes answers `PY_NULL`, and the caller's
/// `GuardNonnull` side-exits to the generic residual.
pub extern "C" fn jit_dict_value_at(
    dict: PyObjectRef,
    index: i64,
    key: PyObjectRef,
    hash: i64,
) -> PyObjectRef {
    if index < 0 {
        return PY_NULL;
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_unicode_value_at_checked(
            dict,
            index as usize,
            key,
            hash,
        )
    }
}

/// Force a frame's `f_backref` vref word to the concrete frame behind it —
/// the callee of the residual `jit_force_virtual` the constant-depth
/// `sys._getframe` fold emits per hop (`pyjitpl.py _do_jit_force_virtual
/// _do_jit_force_virtual`).
///
/// `executioncontext::force_vref` is a Rust-ABI function, so it cannot be the
/// callee directly; this is its `extern "C"` entry.
pub extern "C" fn jit_force_vref(
    frame: *mut pyre_interpreter::PyFrame,
) -> *mut pyre_interpreter::PyFrame {
    pyre_interpreter::executioncontext::force_vref(frame)
}

/// `W_DictMultiObject.length()` of a dict whose strategy the trace has
/// already pinned.
pub extern "C" fn jit_dict_len(dict: i64) -> i64 {
    unsafe { pyre_object::dictmultiobject::w_dict_len(dict as PyObjectRef) as i64 }
}

/// `objspace.py space.getexecutioncontext()` as a residual callee: the
/// running thread's ExecutionContext, read out of its thread-local slot.
///
/// `call::getexecutioncontext` is a Rust-ABI function, so it cannot be the
/// callee directly; this is its `extern "C"` entry.
pub extern "C" fn jit_getexecutioncontext() -> *const pyre_interpreter::PyExecutionContext {
    pyre_interpreter::call::getexecutioncontext()
}

/// Emit the [`jit_getexecutioncontext`] residual for a trace that holds no
/// `ec`. `interp_jit.py PyPyJitDriver.reds = ['frame', 'ec']` makes the
/// context a red the `-live-` markers force-keep at every guard, but a guard
/// whose resume data named no value at the red's color resumes without one,
/// and `PyFrame` carries no `execution_context` field to fall back on.
///
/// The call must stay non-elidable. A nullary elidable call has all its
/// arguments constant by construction, so the pure pass would fold it to the
/// recording thread's ExecutionContext and every other thread entering the
/// compiled trace would read that thread's exception state.
///
/// `site` names the caller in the decline census, which is what separates the
/// three recoveries: the live red costs no call and these do.
pub(crate) fn emit_current_execution_context(ctx: &mut TraceCtx, site: &'static str) -> OpRef {
    crate::jitcode_dispatch::census_record(site);
    ctx.call_typed_with_effect(
        OpCode::CallR,
        pyre_interpreter::residual_word_addr!(0, jit_getexecutioncontext),
        &[],
        &[],
        Type::Ref,
        majit_metainterp::cannot_raise_effect_info(),
    )
}

pub extern "C" fn jit_dict_exact_unicode_lookup_or_null(
    dict: PyObjectRef,
    key: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        jit_dict_exact_lookup_or_null(
            dict,
            key,
            pyre_object::dictmultiobject::w_dict_lookup_or_null_unicode_strategy,
        )
    }
}

pub fn emit_trace_call_ref_typed_elidable_cannot_raise(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    concrete_arg_values: &[Value],
    concrete_result: Value,
) -> OpRef {
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    ctx.call_typed_with_effect_pure(
        OpCode::CallR,
        helper,
        args,
        arg_types,
        Type::Ref,
        effect,
        concrete_arg_values,
        concrete_result,
    )
}

/// `celldict.py getdictvalue_no_unwrapping` residual: returns the
/// raw stored value-or-cell at `slot` of the module dict `namespace_ptr`
/// (the object, not its storage), _not_ unwrapped.  The elidable form of
/// this — keyed on `version?` via `QuasiimmutField` — folds to a constant
/// cell pointer; the live `cell.w_value` read is emitted separately as a
/// `GetfieldGcR`.  Null on a non-module dict, missing slot, or after
/// `switch_to_object_strategy`.
#[allow(dead_code)]
pub(crate) extern "C" fn jit_namespace_cell_lookup(
    w_globals: PyObjectRef,
    slot: i64,
) -> PyObjectRef {
    if w_globals.is_null() || slot < 0 {
        return PY_NULL;
    }
    let cell =
        unsafe { pyre_object::dictmultiobject::module_dict_cell_at(w_globals, slot as usize) };
    cell.unwrap_or(PY_NULL)
}

#[allow(dead_code)]
pub(crate) fn namespace_slot_lookup_values(
    func_ptr: *const (),
    w_globals: pyre_object::PyObjectRef,
    slot: usize,
) -> [Value; 3] {
    [
        Value::Int(func_ptr as usize as i64),
        Value::Ref(GcRef(w_globals as usize)),
        Value::Int(slot as i64),
    ]
}

#[allow(dead_code)]
pub(crate) fn namespace_slot_lookup_result(result: PyObjectRef) -> Value {
    Value::Ref(GcRef(result as usize))
}

/// `typeobject.py _pure_lookup_where_with_method_cache` residual: the
/// `@elidable` method-cache lookup keyed on `(w_type, w_name, version_tag)`,
/// recorded as a foldable `CALL_PURE_R` so repeated same-key lookups in a
/// hot loop collapse to a constant `w_descr` pointer (null = `None`).  The
/// promoted `version_tag` and interned immortal `w_name`
/// (`box_str_constant`) are the green tokens the trace folds on.
///
/// A plain `extern "C"` i64-ABI wrapper, like `jit_namespace_cell_lookup`:
/// the `#[elidable]` macro cannot emit a trampoline for the
/// `PyObjectRef`-aliased signature (it recognises only literal pointer /
/// primitive types), so the recordable surface is this wrapper, recorded
/// with an explicit `ElidableCannotRaise` effect — the raw-pointer return
/// (null = `None`) genuinely cannot raise.  Null on a null `w_type` /
/// `w_name`; callers (the front door) already guarded `is_type` /
/// `version_tag == 0`.
pub extern "C" fn jit_lookup_where_with_method_cache(
    w_type: PyObjectRef,
    w_name: PyObjectRef,
    version_tag: i64,
) -> PyObjectRef {
    if w_type.is_null() || w_name.is_null() {
        return PY_NULL;
    }
    let w_descr = unsafe {
        pyre_interpreter::_pure_lookup_where_with_method_cache(w_type, w_name, version_tag as u64)
    };
    w_descr
}

/// The receiver test the mapdict storage wrappers share: `mapdict_carrier`'s
/// own precondition. It admits the generated int/str/tuple user layouts as
/// well as ordinary instances, so a builtin subclass reaches the same storage
/// the interpreter reads. A null receiver answers false.
///
/// # Safety
/// `w_obj` must be null or a live object reference.
unsafe fn is_mapdict_carrier(w_obj: PyObjectRef) -> bool {
    unsafe { pyre_interpreter::objspace::std::mapdict::has_mapdict_layout(w_obj) }
}

/// `mapdict.py getdictvalue` residual: the instance-dict shadowing
/// read the `LOAD_METHOD` fast path performs after the type lookup
/// (`callmethod.py w_value = w_obj.getdictvalue(space, name)`), to make
/// sure no instance attribute shadows the class method.  Returns the
/// shadowing value or `PY_NULL` when the attribute is absent from the
/// instance dict.
///
/// A plain `extern "C"` i64-ABI wrapper (like [`jit_namespace_cell_lookup`]),
/// recorded as a normal residual call — NOT pure: the instance dict mutates,
/// so the result is guarded `null` per iteration rather than folded.  `w_name`
/// is the interned immortal name pointer; the body reads it back via
/// `w_str_get_wtf8`, mirroring [`jit_lookup_where_with_method_cache`].  Null
/// on a null receiver / name or a non-carrier receiver (the fast path
/// already pinned the receiver type with `guard_class`).
pub extern "C" fn jit_instance_getdictvalue(
    w_obj: PyObjectRef,
    w_name: PyObjectRef,
) -> PyObjectRef {
    if w_name.is_null() || !unsafe { is_mapdict_carrier(w_obj) } {
        return PY_NULL;
    }
    let name = unsafe { w_str_get_wtf8(w_name) };
    let w_value = unsafe {
        pyre_interpreter::objspace::std::mapdict::instance_node_getdictvalue(w_obj, name)
    };
    w_value.unwrap_or(PY_NULL)
}

/// mapdict.py `_mapdict_read_storage(storageindex)` — the LOAD_ATTR
/// fast-path storage read.  A plain `extern "C"` i64-ABI wrapper recorded as a
/// residual call, like [`jit_instance_getdictvalue`]: the value changes per
/// instance so it is not folded, but `storageindex` is a green constant (the
/// fast path resolved it off the promoted map) and the surrounding class / map
/// / version_tag guards already established the shape, so this replaces
/// `getattr_str`'s MRO walk + name hash + descriptor dispatch with a single
/// `storage[index]` fetch.  Null receiver / non-carrier returns `PY_NULL`
/// (the fast path pinned the receiver type with `guard_class`, so this only
/// guards against a torn recording).
pub extern "C" fn jit_mapdict_read(w_obj: PyObjectRef, storageindex: i64) -> PyObjectRef {
    if !unsafe { is_mapdict_carrier(w_obj) } {
        return PY_NULL;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::read_boxed_storage(w_obj, storageindex as usize)
    }
}

/// Raw unboxed counterpart of [`jit_mapdict_read`].  The guarded map pins the
/// shared longlong-list coordinates, so this non-forcing helper performs only
/// `_prim_direct_read`'s storage read (mapdict.py); boxing stays in the
/// trace so an immediate consumer can virtualize it away.  Null receiver /
/// non-carrier returns zero only for a torn recording.
pub extern "C" fn jit_mapdict_unboxed_read_raw(
    w_obj: PyObjectRef,
    storageindex: i64,
    listindex: i64,
) -> i64 {
    if !unsafe { is_mapdict_carrier(w_obj) } {
        return 0;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
        )
    }
}

/// Enter a residual leaf's exception into both channels the trailing
/// `GuardNoException` and the blackhole read, and answer the null a leaf
/// returns on the raising arm.
pub(crate) fn publish_leaf_exception(err: &mut pyre_interpreter::PyError) -> i64 {
    let exc = err.to_exc_object() as i64;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc));
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    majit_backend_cranelift::jit_exc_raise(exc);
    #[cfg(all(feature = "dynasm", not(target_arch = "wasm32")))]
    majit_backend_dynasm::jit_exc_raise(exc);
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_raise(exc);
    let _ = exc;
    0
}

/// `format_w(value, spec)` as a CanRaise residual.  Exact `int`/`str`
/// FORMAT_WITH_SPEC records this instead of the MayForce
/// `bh_format_with_spec_fn` residual, so a compiled `f"{i:05d}"` does not
/// force virtualizables.
pub extern "C" fn jit_format_w(value: PyObjectRef, spec: PyObjectRef) -> i64 {
    match pyre_interpreter::type_methods::format_w(value, spec) {
        Ok(s) => s as i64,
        Err(mut err) => publish_leaf_exception(&mut err),
    }
}

/// `normalize_hash_digest` as a JIT residual: normalize a boxed `__hash__`
/// digest to the machine hash, raising for a non-integer.  On error the
/// exception enters both channels for the trailing `GuardNoException`.
pub extern "C" fn jit_hash_normalize_digest(digest: PyObjectRef) -> i64 {
    match pyre_interpreter::builtins::normalize_hash_digest(digest) {
        Ok(h) => h,
        Err(mut err) => publish_leaf_exception(&mut err),
    }
}

/// `descriptor.py _super_from_frame` as a JIT residual: build the zero-argument
/// `super()` proxy from the frame the caller names, rather than from the frame
/// `ExecutionContext::gettopframe()` rediscovers.
///
/// This is the entry point `LOAD_SUPER_ATTR` already reaches through
/// `bh_load_super_attr_fn`; naming it directly is what lets the name-bound
/// spelling take the same route.
pub extern "C" fn jit_bare_super_from_frame(frame: *mut pyre_interpreter::PyFrame) -> i64 {
    match pyre_interpreter::builtins::builtin_super_from_frame(frame) {
        Ok(proxy) => proxy as i64,
        Err(mut err) => publish_leaf_exception(&mut err),
    }
}

/// Float-bank counterpart of [`jit_mapdict_unboxed_read_raw`].  Unboxed float
/// storage already contains the value's IEEE-754 bit pattern, so this helper
/// performs the raw read and reconstructs the float (mapdict.py).
/// Null receiver / non-carrier returns zero only for a torn recording.
pub extern "C" fn jit_mapdict_unboxed_read_f(
    w_obj: PyObjectRef,
    storageindex: i64,
    listindex: i64,
) -> f64 {
    if !unsafe { is_mapdict_carrier(w_obj) } {
        return 0.0;
    }
    unsafe {
        f64::from_bits(
            pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
                w_obj,
                storageindex as usize,
                listindex as usize,
            ) as u64,
        )
    }
}

/// Non-forcing raw write for a mapdict unboxed attribute.  The full-body
/// walker has already guarded the receiver's instance class and exact map,
/// and proved that the incoming value is an integer, so this is only the
/// same-type longlong-list update (mapdict.py).  A torn recording can
/// reach the wrapper with a null/non-carrier receiver; keep that defensive
/// path a no-op.
pub extern "C" fn jit_mapdict_unboxed_write_raw(
    w_obj: PyObjectRef,
    storageindex: i64,
    listindex: i64,
    raw: i64,
) {
    if !unsafe { is_mapdict_carrier(w_obj) } {
        return;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
            raw,
        );
    }
}

/// Float-bank counterpart of [`jit_mapdict_unboxed_write_raw`].  A same-type
/// float update writes its IEEE-754 bit pattern to the existing longlong-list
/// slot (mapdict.py:615-619).  A torn recording with a null/non-carrier
/// receiver is a defensive no-op.
pub extern "C" fn jit_mapdict_unboxed_write_f(
    w_obj: PyObjectRef,
    storageindex: i64,
    listindex: i64,
    value: f64,
) {
    if !unsafe { is_mapdict_carrier(w_obj) } {
        return;
    }
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_unboxed_storage_raw(
            w_obj,
            storageindex as usize,
            listindex as usize,
            value.to_bits() as i64,
        );
    }
}

/// Record a void residual whose hand-written `extern "C"` helper returns a
/// dummy machine word (`-> i64`, value ignored) — the convention of this
/// module's i64-ABI wrappers ([`jit_store_name_to_namespace`],
/// [`jit_list_append`]). The word-ABI descr lets a signature-exact backend
/// lowering (wasm direct `call_indirect`) call the helper in-module. A
/// helper that genuinely returns `()` must use `ctx.call_void_typed`
/// instead.
///
/// `effect_info`: these helpers write the heap (namespace cells, list
/// storage), so the caller must supply the effect — normally
/// `EffectInfo::MOST_GENERAL` (`graphanalyze.py analyze_external_call
/// analyze_external_call` top for an unanalyzed external writer).  The
/// opcode-default empty write set would let optheap CSE a getfield
/// across the call: `acc = acc + a; acc = acc + b` at module level then
/// reuses the pre-store cell value and drops the first term.
pub fn emit_trace_call_void_word_abi(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    effect_info: majit_ir::EffectInfo,
) {
    ctx.call_void_typed_word_abi(helper, args, arg_types, effect_info);
}

pub fn emit_trace_call_may_force_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_ref_typed(helper, args, arg_types)
}

pub fn emit_trace_call_may_force_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_int_typed(helper, args, arg_types)
}

pub fn emit_trace_bool_value_from_truth(ctx: &mut TraceCtx, truth: OpRef, negate: bool) -> OpRef {
    let truth = if negate {
        let one = ctx.const_int(1);
        let neg = ctx.record_op(OpCode::IntSub, &[one, truth]);
        // Box(value) parity: derive negated bool from truth's Box.value.
        if let Some(majit_ir::Value::Int(n)) = ctx.box_value(truth) {
            ctx.set_opref_concrete(neg, majit_ir::Value::Int(1 - n));
        }
        neg
    } else {
        truth
    };
    // `space.newbool` selects the `w_True` / `w_False` singleton: it cannot
    // raise, so the residual is EF_CANNOT_RAISE (no trailing GuardNoException).
    //
    // This residual is the fallback, not the port.  `baseobjspace.py`
    // is plain RPython, so a walk that can capture a resume image traces the
    // `if b:` itself — `walker_newbool_guarded` emits the truth guard and hands
    // back the prebuilt singleton as a constant, and no box op exists at all.
    // Only a walk with no snapshot to resume into reaches this call.
    ctx.call_ref_typed_with_effect(
        pyre_interpreter::residual_word_addr!(1, jit_bool_value_from_truth),
        &[truth],
        &[Type::Int],
        EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None),
    )
}

/// `ll_unboxed_getclass` low-bit test (rtagged.py): IntAnd(CastPtrToInt(obj),1).
/// Caller emits the GuardTrue (tagged leg) / GuardFalse (boxed leg) via its
/// path-native guard mechanism. `observed_tagged` stamps the folded bit.
pub(crate) fn emit_tag_lowbit_test(ctx: &mut TraceCtx, obj: OpRef, observed_tagged: bool) -> OpRef {
    let as_int = ctx.record_op(OpCode::CastPtrToInt, &[obj]);
    let one = ctx.const_int(1);
    let lowbit = ctx.record_op(OpCode::IntAnd, &[as_int, one]);
    ctx.set_opref_concrete(lowbit, majit_ir::Value::Int(observed_tagged as i64));
    lowbit
}

/// `ll_unboxed_to_int` (rtagged.py): arithmetic IntRshift(CastPtrToInt(obj),1).
pub(crate) fn emit_untag_int(ctx: &mut TraceCtx, obj: OpRef, value: i64) -> OpRef {
    let as_int = ctx.record_op(OpCode::CastPtrToInt, &[obj]);
    let one = ctx.const_int(1);
    let raw = ctx.record_op(OpCode::IntRshift, &[as_int, one]);
    ctx.set_opref_concrete(raw, majit_ir::Value::Int(value));
    raw
}

/// Teach the heap cache what a `NewWithVtable` determined about `new_op`.
///
/// The class is known (`opimpl_new_with_vtable` → `class_now_known`;
/// `class_now_known` here takes the vtable address, since pyre tracks the
/// concrete class pointer where upstream only raises `HF_KNOWN_CLASS`), so a
/// later `guard_class` on `new_op` records nothing.  The class word the
/// allocation writes from its size descr (`SizeDescr::w_class_obj`, the
/// same answer `OptVirtualize` folds the header read off a virtual with) is
/// cached as a recorded `setfield` would be: a later `getfield_gc_r` of
/// `w_class` on `new_op` yields the class constant and a `promote` of that
/// read records no guard.  A size descr without a class word, or whose type
/// has no canonical class, seeds only the class.
pub fn note_class_word_after_new(
    ctx: &mut TraceCtx,
    new_op: OpRef,
    size_descr: &majit_ir::DescrRef,
) {
    let Some(size) = size_descr.as_size_descr() else {
        return;
    };
    ctx.heap_cache_mut().class_now_known(new_op);
    let seed = size
        .class_word_field()
        .map(|field| field.index())
        .zip(size.w_class_obj());
    let Some((field_index, w_class)) = seed else {
        return;
    };
    let w_class = ctx.const_ref(w_class);
    ctx.heapcache_getfield_now_known(new_op, field_index, w_class);
}

/// Emit inline W_Int creation (NewWithVtable + SetfieldGc).
///
/// jtransform.py rewrite_op_setfield: setfield on typeptr is dropped
/// — `new_with_vtable` writes the typeptr in the backend (llmodel.py).
pub fn emit_box_int_inline(
    ctx: &mut TraceCtx,
    raw_int: OpRef,
    size_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py: rewrite_op_setfield skips typeptr setfield
    // entirely ("ignore the operation completely -- instead, it's done by
    // 'new'"). rewrite.py handle_malloc_operation emits the vtable
    // setfield via fielddescr_vtable during GC rewrite of NEW_WITH_VTABLE.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
    ctx.heap_cache_mut().new_object(new_op);
    note_class_word_after_new(ctx, new_op, &size_descr);
    // Emit: SetfieldGc(v, intval, raw_int)
    let intval_idx = intval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_int], intval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`raw_int` OpRef); cache-hit readers fetch the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(new_op, intval_idx, raw_int);
    new_op
}

/// Emit inline W_LongObject creation (NewWithVtable + SetfieldGc) for the
/// boxing of a bigint result — the PyPy `W_LongObject(rbigint)` shape
/// (`new_with_vtable` + `setfield_gc('num', z)`). `bigint_ref` is the
/// (Ref-typed) `jit_w_long_*_raw` result; the collecting `NewWithVtable`
/// gcmap-roots it, and the SetfieldGc into the registered `value`
/// gc-pointer field carries the write barrier.
///
/// Like [`emit_box_int_inline`], `w_class` is cleared by the GC rewriter — the
/// JIT int box does the same; `type(x)`/`isinstance` resolve through `ob_type`
/// (the vtable the NewWithVtable writes).
pub fn emit_box_long_inline(
    ctx: &mut TraceCtx,
    bigint_ref: OpRef,
    size_descr: majit_ir::DescrRef,
    value_descr: majit_ir::DescrRef,
) -> OpRef {
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
    ctx.heap_cache_mut().new_object(new_op);
    note_class_word_after_new(ctx, new_op, &size_descr);
    let value_idx = value_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, bigint_ref], value_descr);
    ctx.heapcache_setfield_cached(new_op, value_idx, bigint_ref);
    new_op
}

/// Emit inline `W_BaseException` creation (NewWithVtable + SetfieldGc for
/// `kind` / `w_class` / `args_w` / `suppress_context`), so a builtin exception
/// built by a Python `raise Type(args)` becomes traced New+SetField ops the
/// optimizer can virtualize when the exception never escapes — instead
/// of the opaque residual `jit_call_callable_N` constructor call.
///
/// Mirrors `allocate_instance`: `user` selects the realbase layout or its
/// `_getusercls` layout. `NewWithVtable` writes that layout's vtable.
/// `w_class` is the Python class. `descr_init` stores `args_w`.
/// GC pointer fields such as `w_cause` and `w_context` are cleared by the GC
/// rewriter; the plain `suppress_context` byte is stored explicitly.
///
/// `_mapdict_init_empty` leaves the user layout's `map` as None and `storage`
/// null. `map` is a plain word, so the rewriter does not clear it, and a
/// virtual force would otherwise publish whatever bits the nursery held.
/// `storage` is stored too: a virtual read of an unwritten field reloads the
/// word instead of the null `allocate_exception` wrote.
pub fn emit_exception_new_inline(
    ctx: &mut TraceCtx,
    kind: pyre_object::interp_exceptions::ExcKind,
    w_class: OpRef,
    args_w: OpRef,
    user: bool,
) -> OpRef {
    let (size_descr, kind_descr, w_class_descr, args_w_descr) =
        crate::descr::w_exception_descrs_for(kind, user);
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let kind_const = ctx.const_int(kind as u8 as i64);
    let kind_idx = kind_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, kind_const], kind_descr);
    ctx.heapcache_setfield_cached(new_op, kind_idx, kind_const);
    let w_class_idx = w_class_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_class], w_class_descr);
    ctx.heapcache_setfield_cached(new_op, w_class_idx, w_class);
    let args_w_idx = args_w_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, args_w], args_w_descr);
    ctx.heapcache_setfield_cached(new_op, args_w_idx, args_w);
    let suppress_context = ctx.const_int(0);
    let suppress_context_descr = crate::descr::w_exception_suppress_context_descr_for(kind, user);
    let suppress_context_idx = suppress_context_descr.index();
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, suppress_context],
        suppress_context_descr,
    );
    ctx.heapcache_setfield_cached(new_op, suppress_context_idx, suppress_context);
    if user {
        // `allocate_exception` writes both words. The offsets differ between
        // the slim and extended user structs; the field descr is the one on
        // this kind's user SizeDescr.
        let (map_offset, storage_offset) =
            if pyre_object::interp_exceptions::exc_kind_uses_extended_layout(kind) {
                (
                    pyre_object::interp_exceptions::EXC_EXTENDED_USER_MAP_OFFSET,
                    pyre_object::interp_exceptions::EXC_EXTENDED_USER_STORAGE_OFFSET,
                )
            } else {
                (
                    pyre_object::interp_exceptions::EXC_USER_MAP_OFFSET,
                    pyre_object::interp_exceptions::EXC_USER_STORAGE_OFFSET,
                )
            };
        let map_descr = crate::descr::w_exception_slot_descr_for(kind, map_offset, true)
            .expect("user exception layout has a map field");
        let map_zero = ctx.const_int(0);
        let map_idx = map_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, map_zero], map_descr);
        ctx.heapcache_setfield_cached(new_op, map_idx, map_zero);
        let storage_descr = crate::descr::w_exception_slot_descr_for(kind, storage_offset, true)
            .expect("user exception layout has a storage field");
        let storage_null = ctx.const_ref(pyre_object::PY_NULL as i64);
        let storage_idx = storage_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, storage_null], storage_descr);
        ctx.heapcache_setfield_cached(new_op, storage_idx, storage_null);
    }
    new_op
}

/// Emit inline `Method` creation (NewWithVtable + SetfieldGc for
/// `w_function` / `w_self` / `w_class` and the inherited header
/// `PyObject.w_class`), mirroring `function.rs w_method_new`.
///
/// This is the traced form of the bound method `getattr(obj, name)` builds
/// for a `LOAD_ATTR`-method whose descriptor lives on the type. Emitting it
/// as New+SetField instead of the opaque `bh_load_attr_fn` residual lets the
/// optimizer virtualize the method away when it is immediately consumed by
/// the following `CALL` — the shape PyPy gets for free by tracing through
/// `space.getattr` (its LOAD_METHOD emits no ops at all in a steady-state
/// loop; `pypy/objspace/std/callmethod.py`).
pub fn emit_bound_method_inline(
    ctx: &mut TraceCtx,
    w_function: OpRef,
    w_self: OpRef,
    w_class: OpRef,
    header_w_class: OpRef,
) -> OpRef {
    let new_op = ctx.execute_new_with_vtable(crate::descr::w_method_size_descr());
    for (descr, value) in [
        (crate::descr::method_w_function_descr(), w_function),
        (crate::descr::method_w_self_descr(), w_self),
        (crate::descr::method_w_class_descr(), w_class),
        (crate::descr::method_header_w_class_descr(), header_w_class),
    ] {
        let index = descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
        ctx.heapcache_setfield_cached(new_op, index, value);
    }
    new_op
}

/// Emit inline `W_Super` creation (`NewWithVtable` + `SetfieldGc` for
/// `super_type` / `obj_type` / `obj` and the inherited header
/// `PyObject.w_class`), mirroring `descriptor.rs w_super_new`.
///
/// This is the traced form of the proxy `super(C, self)` builds.  Upstream
/// gets it by tracing straight through `W_Super.__init__`, where the proxy
/// virtualizes and disappears when the following attribute load consumes it;
/// pyre reaches `builtin_super` through an opaque `bh_call_fn`, so the proxy
/// is allocated for real once per iteration AND the call's may-force flavour
/// makes the walk publish a vref for the frame first.  Emitting New+SetField
/// removes both.
pub fn emit_super_inline(
    ctx: &mut TraceCtx,
    super_type: OpRef,
    obj_type: OpRef,
    obj: OpRef,
    header_w_class: OpRef,
) -> OpRef {
    let new_op = ctx.execute_new_with_vtable(crate::descr::w_super_size_descr());
    for (descr, value) in [
        (crate::descr::super_start_type_descr(), super_type),
        (crate::descr::super_obj_type_descr(), obj_type),
        (crate::descr::super_obj_descr(), obj),
        (crate::descr::super_header_w_class_descr(), header_w_class),
    ] {
        let index = descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
        ctx.heapcache_setfield_cached(new_op, index, value);
    }
    new_op
}

/// Emit inline `Cell` creation for one cellvar of a frame the inline-call path
/// builds — the `NewWithVtable` plus the `SetfieldGc` set `w_cell_new`
/// performs (`nestedscope.py Cell.__init__`).
///
/// `family` is the address `(pycode, slot)` names
/// (`pycode.py PyCode._initialize` builds the table once with the code
/// object, so it is a trace-time constant); `header_w_class` is
/// `get_instantiate(&CELL_TYPE)`.  `contents` is not stored: the constructor
/// this mirrors passes `PY_NULL` for a fresh cell, and a virtual reads an
/// unwritten reference field as null.
///
/// Keeping the allocation as `New` + `SetField` rather than a residual lets
/// the optimizer virtualize the cell away for the common shape where the
/// callee both creates and consumes it within one trace.
pub fn emit_new_cell_inline(ctx: &mut TraceCtx, family: OpRef, header_w_class: OpRef) -> OpRef {
    let new_op = ctx.execute_new_with_vtable(crate::descr::w_cell_size_descr());
    for (descr, value) in [
        (crate::descr::cell_family_descr(), family),
        (crate::descr::cell_header_w_class_descr(), header_w_class),
    ] {
        let index = descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
        ctx.heapcache_setfield_cached(new_op, index, value);
    }
    new_op
}

/// Emit inline `Function` creation for `MAKE_FUNCTION` — `NewWithVtable` plus
/// the `SetfieldGc` set `function.py Function.__init__` performs — instead
/// of the opaque `jit_make_function_from_globals` residual.
///
/// `pyopcode.py MAKE_FUNCTION` builds a fresh function on every execution,
/// so a `def` inside a loop allocates once per iteration; emitting the
/// allocation as `New` + `SetField` lets the optimizer virtualize it away when
/// the function never escapes, which is what upstream gets by tracing straight
/// through `Function.__init__`.
///
/// Only the non-null slots are written.  Every other pointer slot of the
/// struct is `PY_NULL` after construction and a virtual reads an unwritten
/// field as null, so the emit does not spell them out; the size descr's census
/// is what makes the materialized form agree (see `FUNCTION_DESCR_GROUP`).
/// `can_change_code` is the one exception — a plain byte gets no NULL store
/// behind the allocation, so its `True` is written explicitly.
#[allow(clippy::too_many_arguments)]
pub fn emit_make_function_inline(
    ctx: &mut TraceCtx,
    header_w_class: OpRef,
    code: OpRef,
    can_change_code: OpRef,
    name: OpRef,
    w_name: OpRef,
    w_func_globals: OpRef,
    w_builtins: OpRef,
    w_qualname: OpRef,
) -> OpRef {
    let new_op = ctx.execute_new_with_vtable(crate::descr::w_function_size_descr());
    for (descr, value) in [
        (
            crate::descr::function_header_w_class_descr(),
            header_w_class,
        ),
        (crate::descr::function_code_descr(), code),
        (
            crate::descr::function_can_change_code_descr(),
            can_change_code,
        ),
        (crate::descr::function_name_descr(), name),
        (crate::descr::function_w_name_descr(), w_name),
        (crate::descr::function_w_globals_descr(), w_func_globals),
        (crate::descr::function_w_builtins_descr(), w_builtins),
        (crate::descr::function_w_qualname_descr(), w_qualname),
    ] {
        let index = descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
        ctx.heapcache_setfield_cached(new_op, index, value);
    }
    new_op
}

/// Emit inline `W_ObjectObject` creation (`NewWithVtable` + `SetfieldGc` for
/// the inherited header `PyObject.w_class` and `map`), mirroring
/// `objectobject.rs w_instance_new`.
///
/// `user` selects `typedef.py` `_getusercls(W_ObjectObject)`
/// (`W_ObjectObjectUserDictWeakrefable`) rather than exact `W_ObjectObject`.
/// Both size descrs share the field Arcs, so the stores below name one
/// index space. `NewWithVtable` writes that layout's vtable.
///
/// `storage` keeps the allocation's zero — `w_instance_new` stores a null
/// there as well (`mapdict.py _mapdict_init_empty`, `storage = None`).
/// `map` is the owning type's terminator, read eagerly for the same reason
/// `w_instance_new` reads it: a deferred install leaves the promoted map guard
/// on the next iteration's fresh instance naming a map the instance does not
/// have yet.
///
/// Emitting the instantiation as New+SetField instead of the opaque
/// `bh_call_fn` residual lets the optimizer virtualize the instance away when
/// it never escapes the loop — the shape PyPy gets by tracing through
/// `typeobject.py descr_call` → `space.allocate_instance`.
pub fn emit_instance_inline(
    ctx: &mut TraceCtx,
    header_w_class: OpRef,
    map: OpRef,
    user: bool,
) -> OpRef {
    let size_descr = if user {
        crate::descr::w_object_object_user_size_descr()
    } else {
        crate::descr::w_object_object_size_descr()
    };
    let new_op = ctx.execute_new_with_vtable(size_descr);
    for (descr, value) in [
        (crate::descr::object_header_w_class_descr(), header_w_class),
        (crate::descr::object_map_descr(), map),
    ] {
        let index = descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
        ctx.heapcache_setfield_cached(new_op, index, value);
    }
    new_op
}

/// Emit `_set_mapdict_increase_storage1` (`mapdict.py`) as trace
/// operations: allocate the one-slot-longer storage block, copy the
/// `old_len` live slots over, append the new value, then install the block
/// and the transition map on the instance.
///
/// `old_len` and `new_map` are trace-time constants, established by the map
/// guard `store_attr_add_fast_path`'s caller emits; a fresh instance has
/// `old_len == 0`, so the loop body disappears and only the allocation and
/// the two field stores remain.
///
/// The point of emitting this rather than calling a residual is virtualization:
/// with the store folded, an instance built and thrown away inside one loop
/// iteration escapes nowhere and the optimizer removes it, its storage block,
/// and the value box along with it.
///
/// The old block is left to the collector.  `grow_instance_items_block` frees
/// the block it replaces, but that free is a no-op for a GC-owned block, which
/// `store_attr_add_fast_path` requires.
///
/// GC: the block is a leaf, so the barrier that matters is the one on the
/// *instance* — the collector reaches the block's slots only through
/// `object_object_custom_trace`.  Storing `new_block` into the instance's
/// `storage` field is a reference store, so the GC rewriter emits
/// `CondCallGcWb(obj)` for it, and it is emitted after the item stores, so a
/// young value already in the block is remembered with the instance.
///
/// Free threading: these are raw stores, so unlike the interpreter's transition
/// they do not hold the striped `instance_lock`, and unlike the single-slot
/// in-place write they publish two fields.  The caller is what makes that
/// sound — it folds only an `is_unescaped` receiver, one this trace allocated
/// and no other thread can name yet.  The two stores are still ordered
/// `storage` before `map`, so the only intermediate a later reader can observe
/// is the longer block under the shorter map, which over-allocates by one slot;
/// the reverse order would publish a map claiming an index the installed block
/// does not have.
pub fn emit_mapdict_add_attr_inline(
    ctx: &mut TraceCtx,
    obj: OpRef,
    old_len: usize,
    new_map: OpRef,
    value: OpRef,
    map_descr: majit_ir::DescrRef,
    storage_descr: majit_ir::DescrRef,
) {
    let new_len = ctx.const_int(old_len as i64 + 1);
    let new_block = ctx.record_op_with_descr(
        OpCode::NewArrayClear,
        &[new_len],
        crate::state::mapdict_storage_gcarray_descr(),
    );
    ctx.heap_cache_mut().new_object(new_block);

    if old_len > 0 {
        let old_block = crate::state::opimpl_getfield_gc_r(ctx, obj, storage_descr.clone());
        for index in 0..old_len {
            let index_op = ctx.const_int(index as i64);
            let item = crate::state::trace_mapdict_storage_getitem(ctx, old_block, index_op);
            crate::state::trace_mapdict_storage_setitem(ctx, new_block, index_op, item);
        }
    }

    let value_index = ctx.const_int(old_len as i64);
    crate::state::trace_mapdict_storage_setitem(ctx, new_block, value_index, value);

    for (field_descr, stored) in [(storage_descr, new_block), (map_descr, new_map)] {
        let index = field_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[obj, stored], field_descr);
        ctx.heapcache_setfield_cached(obj, index, stored);
    }
}

/// Unboxed twin of [`emit_mapdict_add_attr_inline`]: the fresh-slot unboxed
/// int attribute stores a one-element longlong list — the same int GcArray
/// `erase_unboxed` allocates — in the new slot, emitted inline so
/// OptVirtualize removes it with the instance when nothing escapes.
pub fn emit_mapdict_add_unboxed_attr_inline(
    ctx: &mut TraceCtx,
    obj: OpRef,
    old_len: usize,
    new_map: OpRef,
    raw: OpRef,
    map_descr: majit_ir::DescrRef,
    storage_descr: majit_ir::DescrRef,
) {
    let one = ctx.const_int(1);
    let list_block =
        ctx.record_op_with_descr(OpCode::NewArray, &[one], crate::state::int_gcarray_descr());
    ctx.heap_cache_mut().new_array(list_block, one, true);
    let zero = ctx.const_int(0);
    crate::state::trace_int_block_setitem_value(ctx, list_block, zero, raw);
    emit_mapdict_add_attr_inline(
        ctx,
        obj,
        old_len,
        new_map,
        list_block,
        map_descr,
        storage_descr,
    );
}

/// Emit inline Object-strategy `W_ListObject` creation as traced
/// `NewArrayClear` + `SetarrayitemGc` + `NewWithVtable` + `SetfieldGc`
/// ops the optimizer can virtualize when the list never escapes — instead
/// of the opaque residual `jit_build_list` CallR.  Shared by the
/// `BUILD_LIST` decomposition (`try_walker_specialize_newlist`, Object
/// strategy). Exception `args_w` uses [`emit_rlist_inline`].
///
/// Mirrors `listobject.rs::w_list_new` for the Object strategy:
///   - `items` points at an `ItemsBlock` GcArray whose capacity is
///     `len.max(1)`, the capacity `alloc_list_items_block_gc` picks;
///     `pyobject_gcarray_descr` is byte-compatible with the runtime
///     `ItemsBlock` (both read `ITEMS_BLOCK_TOKEN`).
///   - `length` = `items.len()`.
///   - `strategy` = `Object` (0). The inactive typed blocks are cleared by the
///     GC rewriter and their lengths are stored as zero; the explicit stores
///     keep the heap cache and optimizer field model in agreement.
///
/// A `BUILD_LIST` caller must restrict to Object-strategy-eligible args
/// (non-empty AND not all-int AND not all-float), since an app-level list
/// picks its representation from the element types and the typed Integer /
/// Float and Bytes strategies use their typed storage with `items` null.
pub fn emit_object_list_inline(ctx: &mut TraceCtx, items: &[OpRef]) -> OpRef {
    use crate::descr::{
        list_ascii_items_len_descr, list_bytes_items_len_descr, list_float_items_len_descr,
        list_int_items_len_descr, list_items_descr, list_length_descr, list_strategy_descr,
        w_list_size_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    let len = items.len();
    // Step 1 — allocate the ItemsBlock GcArray. `alloc_list_items_block_gc`
    // takes `len.max(1)` as the capacity, so an empty list still owns a
    // one-slot block. Clear so the GcArray walker sees valid refs in every slot.
    let cap_ref = ctx.const_int(len.max(1) as i64);
    let len_ref = ctx.const_int(len as i64);
    let array_descr = pyobject_gcarray_descr();
    let items_block =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[cap_ref], array_descr.clone());
    ctx.heap_cache_mut().new_object(items_block);

    // Step 2 — items_block[i] = items[i].
    for (i, &item) in items.iter().enumerate() {
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[items_block, idx, item],
            array_descr.clone(),
        );
    }

    // Step 3 — allocate the W_ListObject wrapper.
    let list = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], w_list_size_descr());
    ctx.heap_cache_mut().new_object(list);

    // Step 4 — length / items / strategy SetfieldGc, mirroring the
    // Object-strategy arm of `w_list_new`.
    let length_descr = list_length_descr();
    let length_idx = length_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, len_ref], length_descr);
    ctx.heapcache_setfield_cached(list, length_idx, len_ref);

    let zero = ctx.const_int(0);
    for inactive_len_descr in [
        list_int_items_len_descr(),
        list_float_items_len_descr(),
        list_bytes_items_len_descr(),
        list_ascii_items_len_descr(),
    ] {
        let inactive_len_idx = inactive_len_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, zero], inactive_len_descr);
        ctx.heapcache_setfield_cached(list, inactive_len_idx, zero);
    }

    let items_descr = list_items_descr();
    let items_idx = items_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, items_block], items_descr);
    ctx.heapcache_setfield_cached(list, items_idx, items_block);

    let strategy_const = ctx.const_int(pyre_object::listobject::ListStrategy::Object as i64);
    let strategy_descr = list_strategy_descr();
    let strategy_idx = strategy_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, strategy_const], strategy_descr);
    ctx.heapcache_setfield_cached(list, strategy_idx, strategy_const);

    list
}

/// Emit the fixed-size list stored in `W_BaseException.args_w`.
///
/// `FixedSizeListRepr` is the item `GcArray` (`ll_fixed_newlist`): one
/// `NewArrayClear`, including length 0, then one `setarrayitem` per
/// element (`ll_fixed_setitem_fast`). `Arguments.__init__` fixes
/// `arguments_w` with `make_sure_not_resized`, and both `descr_new` and
/// `descr_setargs` store that list.
pub fn emit_rlist_inline(ctx: &mut TraceCtx, items: &[OpRef]) -> OpRef {
    let cap_ref = ctx.const_int(items.len() as i64);
    let array_descr = crate::state::pyobject_gcarray_descr();
    let items_block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[cap_ref], array_descr);
    ctx.heap_cache_mut().new_array(items_block, cap_ref, true);
    for (i, &item) in items.iter().enumerate() {
        let idx = ctx.const_int(i as i64);
        crate::state::trace_items_block_setitem_value(ctx, items_block, idx, item);
    }
    items_block
}

/// Emit inline Empty-strategy `W_ListObject` creation as a bare `NewWithVtable`
/// wrapper plus the `strategy` store, mirroring `w_list_new(vec![])` /
/// `w_list_new_with_strategy(vec![], Empty)`. Exception `args_w` is not this
/// wrapper; a zero-argument constructor stores the 0-length array from
/// [`emit_rlist_inline`].
///
/// `items` and the typed `int_items` / `float_items` / `bytes_items` blocks
/// stay null because
/// they are GC-pointer fields of the size descr, so `rewrite.py:498-504
/// clear_gc_fields` zeroes them behind the `NewWithVtable`.  `length` gets no
/// such pending zero — the recycled nursery bytes a `CALL_MALLOC_NURSERY`
/// hands back are not zero-filled (`incminimark.py:211 malloc_zero_filled =
/// False`) — so it is stored explicitly here, as `rlist.py ll_newlist` does.
/// OptVirtualize folds the whole wrapper when the list never escapes.
pub fn emit_empty_list_inline(ctx: &mut TraceCtx) -> OpRef {
    use crate::descr::{
        list_ascii_items_len_descr, list_bytes_items_len_descr, list_float_items_len_descr,
        list_int_items_len_descr, list_length_descr, list_strategy_descr, w_list_size_descr,
    };

    let list = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], w_list_size_descr());
    ctx.heap_cache_mut().new_object(list);

    let zero = ctx.const_int(0);
    let length_descr = list_length_descr();
    let length_idx = length_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, zero], length_descr);
    ctx.heapcache_setfield_cached(list, length_idx, zero);

    for inactive_len_descr in [
        list_int_items_len_descr(),
        list_float_items_len_descr(),
        list_bytes_items_len_descr(),
        list_ascii_items_len_descr(),
    ] {
        let inactive_len_idx = inactive_len_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, zero], inactive_len_descr);
        ctx.heapcache_setfield_cached(list, inactive_len_idx, zero);
    }

    let strategy_const = ctx.const_int(pyre_object::listobject::ListStrategy::Empty as i64);
    let strategy_descr = list_strategy_descr();
    let strategy_idx = strategy_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, strategy_const], strategy_descr);
    ctx.heapcache_setfield_cached(list, strategy_idx, strategy_const);

    list
}

/// Initialize the tuple hash cache on a trace-visible allocation.
/// The tuple sentinel is `TUPLE_HASH_UNSET`, so every virtual/materializable
/// tuple constructor must emit this store.
pub fn emit_tuple_hash_sentinel(ctx: &mut TraceCtx, tuple: OpRef, hash_descr: majit_ir::DescrRef) {
    let uncomputed = ctx.const_int(pyre_object::tupleobject::TUPLE_HASH_UNSET);
    let descr_index = hash_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, uncomputed], hash_descr);
    ctx.heapcache_setfield_cached(tuple, descr_index, uncomputed);
}

/// Trace-visible `W_SpecialisedTupleObject_oo` construction from two boxed
/// items — the shape `makespecialisedtuple2` builds for an arity-2 tuple whose
/// elements are not both plain ints or both plain floats.
///
/// `newtuple` picks the representation, so a fold that reproduces a `newtuple`
/// must pick the same one: emitting the array-backed shape for an arity the
/// runtime specialises leaves the trace disagreeing with its own record-time
/// concrete, and a consumer that dispatches on the concrete's layout (the
/// `except <tuple>:` match fold) then guards for a shape the trace never
/// builds.
///
/// `ob_type` is the JIT vtable and differs per variant so `GuardClass` can
/// dispatch on the inline-field layout; Python-level `type()` reads `w_class`,
/// which every variant shares at the public `tuple` typedef.
pub fn emit_specialised_tuple_oo_inline(ctx: &mut TraceCtx, value0: OpRef, value1: OpRef) -> OpRef {
    let tuple = ctx.execute_new_with_vtable(crate::descr::specialised_tuple_oo_size_descr());
    emit_tuple_hash_sentinel(ctx, tuple, crate::descr::specialised_tuple_oo_hash_descr());
    let w_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    if !w_class.is_null() {
        let w_class = ctx.const_ref(w_class as i64);
        let class_descr = crate::descr::specialised_tuple_oo_w_class_descr();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, w_class], class_descr.clone());
        ctx.heapcache_setfield_cached(tuple, class_descr.index(), w_class);
    }
    for (value, descr) in [
        (value0, crate::descr::specialised_tuple_oo_value0_descr()),
        (value1, crate::descr::specialised_tuple_oo_value1_descr()),
    ] {
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, value], descr.clone());
        ctx.heapcache_setfield_cached(tuple, descr.index(), value);
    }
    tuple
}

/// Trace-visible canonical `W_TupleObject` construction from boxed items.
/// This is the allocation half of `W_BaseException.descr_getargs`: the raw
/// `args_w` list is copied into a fresh tuple on every public attribute read.
pub fn emit_object_tuple_inline(ctx: &mut TraceCtx, items: &[OpRef]) -> OpRef {
    use crate::state::pyobject_gcarray_descr;

    let len = ctx.const_int(items.len() as i64);
    let array_descr = pyobject_gcarray_descr();
    let items_block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr.clone());
    ctx.heap_cache_mut().new_array(items_block, len, true);
    for (index, &item) in items.iter().enumerate() {
        let index = ctx.const_int(index as i64);
        crate::state::trace_items_block_setitem_value(ctx, items_block, index, item);
    }

    let tuple = ctx.execute_new_with_vtable(crate::descr::w_tuple_size_descr());
    emit_tuple_hash_sentinel(ctx, tuple, crate::descr::tuple_hash_descr());
    let w_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    let w_class = ctx.const_ref(w_class as i64);
    let class_descr = crate::descr::tuple_w_class_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[tuple, w_class], class_descr.clone());
    ctx.heapcache_setfield_cached(tuple, class_descr.index(), w_class);
    let items_descr = crate::descr::tuple_wrappeditems_descr();
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[tuple, items_block],
        items_descr.clone(),
    );
    ctx.heapcache_setfield_cached(tuple, items_descr.index(), items_block);
    tuple
}

/// Emit inline Integer / Float-strategy `W_ListObject` creation: a typed
/// length-prefixed backing block (`NewArray` + per-element `SetarrayitemGc`)
/// holding the already-unboxed machine values in `raws`, then the
/// `W_ListObject` wrapper (`NewWithVtable` + `strategy` / `int_items.len` /
/// `int_items.block` — or the `float_items` pair — `SetfieldGc`).  Mirrors the
/// Integer / Float arm of `listobject.rs::w_list_new` (`IntArray::from_vec` /
/// `FloatArray::from_vec`), so OptVirtualize can fold the whole list (wrapper +
/// block) when it never escapes — the orthodox `newlist` shape
/// (`rlist.py ll_newlist`, two mallocs).
///
/// The typed strategy keeps `length = 0` and `items = null`
/// (`w_list_new_with_strategy` non-Object arm): `length` is stored explicitly
/// and the GC rewriter clears `items`. `items_len_descr` / `items_block_descr`
/// select the `int_items` / `float_items` sub-struct fields and `array_descr`
/// the matching `int_gcarray_descr` / `float_gcarray_descr`. Caller must have
/// guarded + unboxed each element into `raws` already.
pub fn emit_typed_list_inline(
    ctx: &mut TraceCtx,
    raws: &[OpRef],
    array_descr: majit_ir::DescrRef,
    items_len_descr: majit_ir::DescrRef,
    items_block_descr: majit_ir::DescrRef,
    strategy: pyre_object::listobject::ListStrategy,
) -> OpRef {
    use crate::descr::{
        list_ascii_items_len_descr, list_bytes_items_len_descr, list_float_items_len_descr,
        list_int_items_len_descr, list_length_descr, list_strategy_descr, w_list_size_descr,
    };

    let len = raws.len();
    let len_ref = ctx.const_int(len as i64);

    // Step 1 — allocate the typed backing block (length-prefixed
    // `[capacity][i64|f64 ...]`, capacity == len).  The elements are machine
    // ints / floats (not refs), so `NewArray` (no GC-safe zeroing) matches
    // `IntArray::from_vec` / `FloatArray::from_vec`; every slot is filled
    // immediately below.
    let block = ctx.record_op_with_descr(OpCode::NewArray, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_array(block, len_ref, true);

    // Step 2 — block[i] = raws[i].
    let block_descr_idx = array_descr.index();
    for (i, &raw) in raws.iter().enumerate() {
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[block, idx, raw],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(block, idx, block_descr_idx, raw);
    }

    // Step 3 — allocate the W_ListObject wrapper.
    let list = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], w_list_size_descr());
    ctx.heap_cache_mut().new_object(list);

    // Step 4 — initialize every scalar field, then install the active typed
    // storage. The pointer fields are cleared by the GC rewriter.
    let zero = ctx.const_int(0);
    for scalar_descr in [
        list_length_descr(),
        list_int_items_len_descr(),
        list_float_items_len_descr(),
        list_bytes_items_len_descr(),
        list_ascii_items_len_descr(),
    ] {
        let scalar_idx = scalar_descr.index();
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, zero], scalar_descr);
        ctx.heapcache_setfield_cached(list, scalar_idx, zero);
    }

    let strategy_const = ctx.const_int(strategy as i64);
    let strategy_descr = list_strategy_descr();
    let strategy_idx = strategy_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, strategy_const], strategy_descr);
    ctx.heapcache_setfield_cached(list, strategy_idx, strategy_const);

    let items_len_idx = items_len_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, len_ref], items_len_descr);
    ctx.heapcache_setfield_cached(list, items_len_idx, len_ref);

    let items_block_idx = items_block_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, block], items_block_descr);
    ctx.heapcache_setfield_cached(list, items_block_idx, block);

    list
}

/// Empty->typed in-place promotion of an existing `W_ListObject` wrapper (the
/// comprehension accumulator). Mirrors `switch_to_correct_strategy`'s concrete
/// effect as field mutations on `list_op`: set the strategy and stage the
/// capacity-4 storage produced by the first RPython `_ll_list_resize_ge`.
/// `w_list_switch_to_strategy_for` applies the same concrete pre-grow before
/// this emitter runs; the following append sub-walk therefore records only
/// the length/item stores against identical concrete and symbolic shapes.
pub fn emit_promote_empty_list_inline(
    ctx: &mut TraceCtx,
    list_op: OpRef,
    strategy: pyre_object::listobject::ListStrategy,
) {
    use crate::descr::{
        list_ascii_items_block_descr, list_ascii_items_len_descr, list_float_items_block_descr,
        list_float_items_len_descr, list_int_items_block_descr, list_int_items_len_descr,
        list_items_descr, list_length_descr, list_strategy_descr,
    };
    use crate::state::{float_gcarray_descr, int_gcarray_descr, pyobject_gcarray_descr};

    let cap_ref = ctx.const_int(4);
    let zero_ref = ctx.const_int(0);

    match strategy {
        pyre_object::listobject::ListStrategy::Integer => {
            let array_descr = int_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArray, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let items_len_descr = list_int_items_len_descr();
            let items_len_idx = items_len_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], items_len_descr);
            ctx.heapcache_setfield_cached(list_op, items_len_idx, zero_ref);

            let items_block_descr = list_int_items_block_descr();
            let items_block_idx = items_block_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_block_descr);
            ctx.heapcache_setfield_cached(list_op, items_block_idx, block);
            let cap_idx = crate::descr::items_block_capacity_descr().index();
            ctx.heapcache_setfield_cached(block, cap_idx, cap_ref);
        }
        pyre_object::listobject::ListStrategy::Float => {
            let array_descr = float_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArray, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let items_len_descr = list_float_items_len_descr();
            let items_len_idx = items_len_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], items_len_descr);
            ctx.heapcache_setfield_cached(list_op, items_len_idx, zero_ref);

            let items_block_descr = list_float_items_block_descr();
            let items_block_idx = items_block_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_block_descr);
            ctx.heapcache_setfield_cached(list_op, items_block_idx, block);
            let cap_idx = crate::descr::items_block_capacity_descr().index();
            ctx.heapcache_setfield_cached(block, cap_idx, cap_ref);
        }
        pyre_object::listobject::ListStrategy::Object => {
            let array_descr = pyobject_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let length_descr = list_length_descr();
            let length_idx = length_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], length_descr);
            ctx.heapcache_setfield_cached(list_op, length_idx, zero_ref);

            let items_descr = list_items_descr();
            let items_idx = items_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_descr);
            ctx.heapcache_setfield_cached(list_op, items_idx, block);
        }
        pyre_object::listobject::ListStrategy::Ascii => {
            // Erased `STR` pointers share the object ItemsBlock shape, so the
            // block is `NewArrayClear` (null spare slots), not an unboxed
            // int/float array. Header `length` / `items` stay 0 / null; the
            // live length is `ascii_items.len`.
            let array_descr = pyobject_gcarray_descr();
            let block = ctx.record_op_with_descr(OpCode::NewArrayClear, &[cap_ref], array_descr);
            ctx.heap_cache_mut().new_array(block, cap_ref, true);

            let strategy_const = ctx.const_int(strategy as i64);
            let strategy_descr = list_strategy_descr();
            let strategy_idx = strategy_descr.index();
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[list_op, strategy_const],
                strategy_descr,
            );
            ctx.heapcache_setfield_cached(list_op, strategy_idx, strategy_const);

            let items_len_descr = list_ascii_items_len_descr();
            let items_len_idx = items_len_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, zero_ref], items_len_descr);
            ctx.heapcache_setfield_cached(list_op, items_len_idx, zero_ref);

            let items_block_descr = list_ascii_items_block_descr();
            let items_block_idx = items_block_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[list_op, block], items_block_descr);
            ctx.heapcache_setfield_cached(list_op, items_block_idx, block);
        }
        pyre_object::listobject::ListStrategy::Empty
        | pyre_object::listobject::ListStrategy::Size
        | pyre_object::listobject::ListStrategy::SimpleRange
        | pyre_object::listobject::ListStrategy::Range
        | pyre_object::listobject::ListStrategy::IntOrFloat
        | pyre_object::listobject::ListStrategy::Bytes => {
            // The specialized first-append path admits Integer, Float, Ascii,
            // or Object. Exact bytes are declined before this emitter;
            // IntOrFloat is reached later by a numeric strategy transition.
            // BaseRangeListStrategy append materialises first, so compact
            // range storage cannot enter this Empty-to-typed emitter.
            debug_assert!(matches!(
                strategy,
                pyre_object::listobject::ListStrategy::Empty
                    | pyre_object::listobject::ListStrategy::Size
                    | pyre_object::listobject::ListStrategy::SimpleRange
                    | pyre_object::listobject::ListStrategy::Range
                    | pyre_object::listobject::ListStrategy::IntOrFloat
                    | pyre_object::listobject::ListStrategy::Bytes
            ));
        }
    }
}

/// Emit inline `space.newslice(w_start, w_end, w_step)` creation
/// (NewWithVtable + 3 SetfieldGc).
///
/// `pypy/objspace/std/objspace.py` `space.newslice` returns
/// `W_SliceObject(w_start, w_end, w_step)` — a fresh allocation per
/// invocation (matching `pypy/interpreter/pyopcode.py BUILD_SLICE`).
/// `_immutable_fields_ = ['w_start', 'w_stop', 'w_step']`
/// (`sliceobject.py:13`) marks all three slots immutable, so the
/// `optimizeopt/virtualize.py optimize_NEW_WITH_VTABLE` pass can
/// virtualize the allocation when the slice never escapes — the IR
/// shape (NewWithVtable + 3 SetfieldGc) preserves the operand
/// dependencies the optimizer needs to reason about that.
///
/// `jtransform.py rewrite_op_setfield` skips the typeptr
/// setfield (the backend writes typeptr inside `new_with_vtable` per
/// `llmodel.py bh_new_with_vtable`); `rewrite.py handle_malloc_operation`
/// emits the vtable setfield via `fielddescr_vtable` during the GC
/// rewrite pass.
pub fn emit_box_slice_inline(
    ctx: &mut TraceCtx,
    w_start: OpRef,
    w_stop: OpRef,
    w_step: OpRef,
    size_descr: majit_ir::DescrRef,
    w_start_descr: majit_ir::DescrRef,
    w_stop_descr: majit_ir::DescrRef,
    w_step_descr: majit_ir::DescrRef,
) -> OpRef {
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
    ctx.heap_cache_mut().new_object(new_op);
    note_class_word_after_new(ctx, new_op, &size_descr);
    let w_start_idx = w_start_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_start], w_start_descr);
    ctx.heapcache_setfield_cached(new_op, w_start_idx, w_start);
    let w_stop_idx = w_stop_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_stop], w_stop_descr);
    ctx.heapcache_setfield_cached(new_op, w_stop_idx, w_stop);
    let w_step_idx = w_step_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_step], w_step_descr);
    ctx.heapcache_setfield_cached(new_op, w_step_idx, w_step);
    new_op
}

/// Emit inline W_Float creation (NewWithVtable + SetfieldGc).
pub fn emit_box_float_inline(
    ctx: &mut TraceCtx,
    raw_float: OpRef,
    size_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py rewrite_op_setfield parity: typeptr setfield filtered in trace.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
    ctx.heap_cache_mut().new_object(new_op);
    note_class_word_after_new(ctx, new_op, &size_descr);
    let floatval_idx = floatval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_float], floatval_descr);
    ctx.heapcache_setfield_cached(new_op, floatval_idx, raw_float);
    new_op
}

/// Emit a fresh callee `PyFrame` directly into the trace IR for the
/// self-recursive single-int-argument fast path.
///
/// Replaces the opaque `jit_create_self_recursive_callee_frame_1_raw_int`
/// CallR, which wraps the frame allocation, locals fill and raw_int
/// boxing in an opaque helper. The
/// helper is `#[dont_look_inside]` so the optimizer cannot virtualize
/// the new frame nor fold the boxing — every fib(35) iteration pays
/// the full helper trampoline (~336k calls/run, observed in
/// `phase2_3_self_recursive_call_perf_plan_2026_04_28.md`). PyPy emits
/// `direct_assembler_call` with `NewWithVtable(jitframe) + SetfieldGc(...)`
/// in trace IR (`backend/aarch64/opassembler.py:1080-1200`); this helper
/// is the closest pyre analogue given that pyre's `PyFrame` plays the
/// role both of CPython's `PyFrame` and of PyPy's separate `jitframe` /
/// virtualizable.
///
/// Restrictions held by the caller:
///   - self-recursive (callee `pycode` ≡ caller `pycode`), so the
///     caller passes `pycode` / `w_globals` / `execution_context` in
///     directly: `pycode` and `w_globals` arrive as trace-time
///     constants (the bound `PyCode` and its `function.w_globals`,
///     both immutable for the trace's lifetime), and `execution_context`
///     arrives as the loop's already-materialised `sym.execution_context`
///     OpRef (per-thread; not safe to const-fold across thread entries).
///     This mirrors PyPy aarch64 `direct_assembler_call` (`backend/aarch64/
///     opassembler.py:1080-1200`) which writes the callee jitframe's vable
///     scalars from constants known at trace-compile time.
///   - 1 raw-int argument (no boxed-arg path; caller is responsible
///     for the `trace_guarded_int_payload` unbox).
///   - no cellvars/freevars on the callee — `init_cells` is skipped.
///     The caller verifies this against the concrete `PyCode`
///     before invoking the helper.
///
/// The IR sequence mirrors `pyframe.rs::PyFrame::new_for_call_with_closure`:
///
/// 1. `emit_box_int_inline(raw_int_arg)` → `boxed` W_IntObject (one
///    nursery alloc, optimizer can fold into a virtual when the boxed
///    value never escapes through `GuardNotForced`).
/// 2. `NewArrayClear(array_size)` with `pyobject_gcarray_descr()` —
///    the locals_cells_stack_w `FixedObjectArray<PyObjectRef>`
///    backing storage. Clear so unset slots read as `PY_NULL`.
/// 3. `SetarrayitemGc(locals_array, 0, boxed)` — bind the lone
///    positional argument. Other slots stay `PY_NULL`.
/// 4. `NewWithVtable(pyframe_size_descr())` — `vtable=0` because
///    `PyFrame` is not an `rclass.OBJECT` instance (registered via
///    `TypeInfo::with_gc_ptrs`, see `pyre-jit/src/eval.rs::initialize_gc`),
///    so `handle_new` skips the vtable setfield (rewrite.py:925-933
///    `gen_new_with_vtable` early-out for `vtable == 0`).
/// 5. `SetfieldGc` ops for the constructor-visible fields. The non-zero
///    fields (`pycode`,
///    `locals_cells_stack_w`, `valuestackdepth`, `last_instr=-1`) mirror
///    `new_for_call_with_closure`; the nullable GC fields
///    (`f_generator_wref`, `w_yielding_from`, `f_backref`) are written
///    explicitly to match the same constructor shape instead of relying on
///    an implicit backend zero-fill side effect.
/// Build a VIRTUAL callee `PyFrame` for a multi-frame inline (#68) from
/// already-boxed positional argument refs.  Same field-complete frame shape as
/// [`emit_new_pyframe_inline_self_recursive`] but seeds `locals[0..nparams]`
/// from `param_boxes` (Ref boxes at the Python call boundary) instead of
/// boxing a single raw int. The cell band is placed at `cell_start..`,
/// matching `PyFrame::finish_for_call_with_globals_obj`: the caller composes it
/// as one freshly emitted cell per pure cellvar ([`emit_new_cell_inline`])
/// followed by the closure's existing cell objects. The frame is
/// the callee MIFrame's `frame` red —
/// `_opimpl_inline_call*` / `perform_call`+`setup_call` push a fresh MIFrame
/// per inlined call (`pyjitpl.py`) — the app-level frame
/// emitted here has no upstream counterpart at that point, because upstream
/// gets one from tracing the interpreter's own frame construction.  The box
/// stays virtual on the hot path (the optimizer folds
/// `NewWithVtable`+`SetfieldGc`) and is
/// materialized lazily only on guard failure.  Carries every field the frame
/// constructor assigns, so a forced materialization
/// (`materialize_virtual_from_rd`) never dereferences a field the interpreter
/// would have written; the class-level GC-pointer defaults listed at the end
/// of the body are cleared by the GC rewriter, exactly as they are initialized
/// in a frame the interpreter builds.
///
/// `None` when `w_globals` is not the namespace this frame would answer with:
/// the emitted frame carries no `debugdata`, so `get_w_globals` on it reads the
/// code object's published namespace (`pyframe.py get_w_globals`), and a callee
/// running under a different dictionary needs the override
/// `createframe_obj` allocates. Callers already decline a global-storing callee
/// (`pycode.py frame_stores_global`) before they get here; this is the frame
/// shape's own statement of what it can represent.
pub fn emit_new_pyframe_inline_with_params(
    ctx: &mut TraceCtx,
    param_boxes: &[OpRef],
    cell_slots: &[OpRef],
    stack_items: &[OpRef],
    cell_start: usize,
    array_size: usize,
    valuestackdepth: usize,
    pycode: OpRef,
    w_globals: OpRef,
) -> Option<OpRef> {
    if !inline_frame_answers_this_namespace(ctx, pycode, w_globals) {
        return None;
    }
    use crate::descr::{
        pyframe_code_descr, pyframe_failed_attr_cleanup_descr, pyframe_flags_descr,
        pyframe_locals_cells_stack_descr, pyframe_next_instr_descr, pyframe_size_descr,
        pyframe_stack_depth_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    // locals_cells_stack_w array, zero-filled so an unbound local reads PY_NULL.
    let len_ref = ctx.const_int(array_size as i64);
    let array_descr = pyobject_gcarray_descr();
    let locals_array =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    // `heapcache.py new_array` stores the length in `_heapc_deps[0]`.
    // `append_bound_entries` compares `slot < locals_w!(frame).len()`
    // through `opimpl_arraylen_vable`. A hit answers with this const.
    // `_escape_box` keeps deps[0]; a miss executes `ARRAYLEN_GC`.
    ctx.heap_cache_mut().new_array(locals_array, len_ref, true);

    // locals[i] = param_boxes[i] — the positional arguments (already boxed).
    // Register each store in the heapcache so a later nonstandard
    // `getarrayitem_vable` read of this VIRTUAL inline-callee frame forwards the
    // stored param box (with its concrete shadow) instead of recording a fresh
    // `GetarrayitemGc` whose result has no concrete — the gap that made an
    // in-callee pure comparison branch surface `GotoIfNotValueNotConcrete`.
    // Heapcache key for the locals-array elements must match the descr the
    // codewriter's `getarrayitem_vable`/`setarrayitem_vable` ops carry — the
    // virtualizable info's array descr (`info.array_descrs[0]`), NOT the
    // struct-layout `pyobject_gcarray_descr` the recorded `SetarrayitemGc` uses
    // for materialization.  The heapcache `heap_array_cache` is keyed by descr
    // FIRST, so a struct-vs-vinfo descr mismatch silently misses every forward.
    let heapcache_item_descr_index = ctx
        .virtualizable_info()
        .map(|info| info.array_item_descr(0).index())
        .unwrap_or_else(|| array_descr.index());
    for (i, &p) in param_boxes.iter().enumerate() {
        // A NONE slot (an unbound local in a reconstructed bridge-carrier
        // callee frame) keeps the NewArrayClear zero-fill (PY_NULL); only
        // bound slots are stored.  Forward-inline callees pass dense param
        // boxes, so this skips nothing on that path.
        if p.is_none() {
            continue;
        }
        let idx = ctx.const_int(i as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[locals_array, idx, p],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(locals_array, idx, heapcache_item_descr_index, p);
    }
    // PyFrame.finish_for_call_with_globals_obj: the pure cellvars take a fresh
    // cell each and a closure then contributes its existing cell objects
    // themselves, in that order. LOAD_DEREF must read the live cell, not a
    // snapshot of its contents, so the slot holds the cell.
    for (i, &cell) in cell_slots.iter().enumerate() {
        let idx = ctx.const_int((cell_start + i) as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[locals_array, idx, cell],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(locals_array, idx, heapcache_item_descr_index, cell);
    }
    // A paused frame's operand stack occupies the slots above the cells
    // (`pyframe.py` `locals_cells_stack_w`). A reconstructed bridge-carrier
    // callee resumes mid-body, so the operands live at the guard are stored
    // where the vable image keeps them; a NONE slot (the pending call result,
    // delivered by `make_result_of_lastop`) keeps the zero-fill. A forward
    // inline callee starts with an empty stack and passes nothing here.
    let stack_start = cell_start + cell_slots.len();
    for (i, &item) in stack_items.iter().enumerate() {
        if item.is_none() {
            continue;
        }
        let idx = ctx.const_int((stack_start + i) as i64);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[locals_array, idx, item],
            array_descr.clone(),
        );
        ctx.heapcache_setarrayitem(locals_array, idx, heapcache_item_descr_index, item);
    }

    let new_frame = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], pyframe_size_descr());
    ctx.heap_cache_mut().new_object(new_frame);

    let code_descr = pyframe_code_descr();
    let code_idx = code_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, pycode], code_descr);
    ctx.heapcache_setfield_cached(new_frame, code_idx, pycode);

    let locals_descr = pyframe_locals_cells_stack_descr();
    let locals_idx = locals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, locals_array], locals_descr);
    ctx.heapcache_setfield_cached(new_frame, locals_idx, locals_array);

    let vsd = ctx.const_int(valuestackdepth as i64);
    let vsd_descr = pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, vsd], vsd_descr);
    ctx.heapcache_setfield_cached(new_frame, vsd_idx, vsd);

    let neg_one = ctx.const_int(-1);
    let last_instr_descr = pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, neg_one], last_instr_descr);
    ctx.heapcache_setfield_cached(new_frame, last_instr_idx, neg_one);

    let zero = ctx.const_int(0);
    let flags_descr = pyframe_flags_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, zero], flags_descr);
    // The concrete carrier may already be marked escaped by record-time
    // traceback inspection before this allocation shape is emitted. Do not
    // seed TraceCtx's concrete sanity cache with the constructor-time zero;
    // a later mark_as_escaped read must observe the carrier's current byte.

    // `failed_attr_cleanup` is the second byte beside `flags` and is not a GC
    // reference, so `clear_gc_fields` does not reach it. incminimark's
    // `malloc_zero_filled = False` leaves the `NewWithVtable` block holding
    // recycled nursery bytes, and the byte's "run the deferred cleanup before
    // the next opcode" value is `u8::MAX` — a recycled `0xff` makes every
    // dispatch on this frame run a full non-moving old-gen collection.
    let failed_attr_descr = pyframe_failed_attr_cleanup_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, zero], failed_attr_descr);

    // pyframe.py `f_generator_wref`/`w_yielding_from`/`f_backref`
    // are class-level defaults (dead_ref/None/vref_None), never assigned in the
    // frame constructor. The trace of frame construction therefore emits no
    // setfield for them; clear_gc_fields initializes these GC-reference slots,
    // so the fields read back as PY_NULL. No explicit store here.
    //
    // `w_builtin` joins them: the frame constructor (pyframe.py) assigns
    // `self.builtin` only under `honor__builtins__`, which
    // `baseobjspace::HONOR_BUILTINS` leaves off, so the constructor writes
    // nothing and every reader goes through `get_builtin`, which answers
    // `space.builtin` for an unset slot.

    Some(new_frame)
}

/// Whether a `debugdata`-less frame for `pycode` answers `w_globals` from
/// `get_w_globals`, i.e. whether the code object has published that very
/// dictionary as its first namespace (`pycode.py frame_stores_global`).
///
/// Both operands are trace-time constants at every emission site. A namespace
/// that is not constant, or a code object whose slot is still unpublished,
/// leaves the question unanswered rather than answered `false`: the emission
/// sites establish the invariant themselves, and this check is here to stop a
/// later one that does not, not to close a shape that already works.
fn inline_frame_answers_this_namespace(ctx: &TraceCtx, pycode: OpRef, w_globals: OpRef) -> bool {
    let (Some(majit_ir::Value::Ref(code)), Some(majit_ir::Value::Ref(globals))) =
        (ctx.box_value(pycode), ctx.box_value(w_globals))
    else {
        return true;
    };
    if code.as_usize() == 0 || globals.as_usize() == 0 {
        return true;
    }
    let published = unsafe {
        pyre_interpreter::w_code_get_w_globals(code.as_usize() as pyre_object::PyObjectRef)
    };
    published.is_null() || published as usize == globals.as_usize()
}

/// `None` on the same namespace disagreement as
/// [`emit_new_pyframe_inline_with_params`], whose doc states the rule.
pub fn emit_new_pyframe_inline_self_recursive(
    ctx: &mut TraceCtx,
    arg_box: OpRef,
    array_size: usize,
    valuestackdepth: usize,
    pycode: OpRef,
    w_globals: OpRef,
) -> Option<OpRef> {
    if !inline_frame_answers_this_namespace(ctx, pycode, w_globals) {
        return None;
    }
    use crate::descr::{
        pyframe_code_descr, pyframe_failed_attr_cleanup_descr, pyframe_flags_descr,
        pyframe_locals_cells_stack_descr, pyframe_next_instr_descr, pyframe_size_descr,
        pyframe_stack_depth_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    // Step 1 — `locals[0]` receives the caller's already-boxed positional
    // argument box.  The caller supplies the shape-correct box so the callee
    // reads back the same representation it was traced against: under
    // `CAN_BE_TAGGED` a small `int` stays a tagged immediate (`ll_int_box`),
    // otherwise a heap `W_IntObject` (`w_int_new` fallback).  Re-boxing a raw
    // payload heap-side here would force a heap box even when the value fits
    // the tagged range, and the callee's speculative low-bit guard on the
    // local would then deopt on every recursion.
    let boxed = arg_box;

    // Step 2 — allocate the locals_cells_stack_w array. `NewArrayClear`
    // zeros every slot so any future LOAD_FAST on an unbound local
    // observes `PY_NULL` (UnboundLocalError parity).
    let len_ref = ctx.const_int(array_size as i64);
    let array_descr = pyobject_gcarray_descr();
    let locals_array =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    // Same `new_array` length as `emit_new_pyframe_inline_with_params`.
    ctx.heap_cache_mut().new_array(locals_array, len_ref, true);

    // Step 3 — locals[0] = boxed. The single positional argument of
    // the self-recursive call.
    let zero = ctx.const_int(0);
    ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[locals_array, zero, boxed],
        array_descr,
    );

    // Step 4 — allocate the new PyFrame. The GC tags it with
    // `PYFRAME_GC_TYPE_ID` because the size descr's parent type id is
    // registered in `pyre-jit/src/eval.rs`.
    let new_frame = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], pyframe_size_descr());
    ctx.heap_cache_mut().new_object(new_frame);

    // Step 5 — SetfieldGc for the constructor-visible fields, mirroring
    // the explicit assignments inside `new_for_call_with_closure`.
    // Order matches the field declaration so the optimizer's lazy-set
    // replace logic groups them together.
    // `pycode` arrives as a trace-time Ref Const (the bound `PyCode`).
    let code_descr = pyframe_code_descr();
    let code_idx = code_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, pycode], code_descr);
    ctx.heapcache_setfield_cached(new_frame, code_idx, pycode);

    // `locals_array` is a fresh `NewArrayClear` op result.  PyPy's
    // executor-while-trace model would have `Box.value` carry the
    // actual allocated array ref; pyre's `record_op` does not execute
    // the alloc, so the runtime ref does not exist until codegen +
    // execution.  Honest carrier: `Value::Void` (= "no Box.value
    // known"), so the downstream cache-hit sanity check skips
    // explicitly.  Fabricating a `Ref(0)` here would lie about the
    // payload.
    let locals_descr = pyframe_locals_cells_stack_descr();
    let locals_idx = locals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, locals_array], locals_descr);
    ctx.heapcache_setfield_cached(new_frame, locals_idx, locals_array);

    let vsd = ctx.const_int(valuestackdepth as i64);
    let vsd_descr = pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, vsd], vsd_descr);
    ctx.heapcache_setfield_cached(new_frame, vsd_idx, vsd);

    let neg_one = ctx.const_int(-1);
    let last_instr_descr = pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, neg_one], last_instr_descr);
    ctx.heapcache_setfield_cached(new_frame, last_instr_idx, neg_one);

    let zero = ctx.const_int(0);
    let flags_descr = pyframe_flags_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, zero], flags_descr);
    // See emit_new_pyframe_inline_with_params: the trace store is required for
    // materialization, but its constructor-time value is not the concrete
    // carrier's necessarily later record-time state.

    // `failed_attr_cleanup` is the second byte beside `flags` and is not a GC
    // reference, so `clear_gc_fields` does not reach it. incminimark's
    // `malloc_zero_filled = False` leaves the `NewWithVtable` block holding
    // recycled nursery bytes, and the byte's "run the deferred cleanup before
    // the next opcode" value is `u8::MAX` — a recycled `0xff` makes every
    // dispatch on this frame run a full non-moving old-gen collection.
    let failed_attr_descr = pyframe_failed_attr_cleanup_descr();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, zero], failed_attr_descr);

    // pyframe.py `f_generator_wref`/`w_yielding_from`/`f_backref`
    // are class-level defaults (dead_ref/None/vref_None), never assigned in the
    // frame constructor. The trace of frame construction therefore emits no
    // setfield for them; clear_gc_fields initializes these GC-reference slots,
    // so the fields read back as PY_NULL. No explicit store here.
    //
    // `w_builtin` joins them: the frame constructor (pyframe.py) assigns
    // `self.builtin` only under `honor__builtins__`, which
    // `baseobjspace::HONOR_BUILTINS` leaves off, so the constructor writes
    // nothing and every reader goes through `get_builtin`, which answers
    // `space.builtin` for an unset slot.

    Some(new_frame)
}

// ── Elidable canary helper ──────────────────────────────────────────
//
// rlib/jit.py `@jit.elidable` parity.  PyPy `intobject.py
// wrapint` in-range check parity: returns true iff `value` falls inside
// the prebuilt-int small cache range AND the cache is enabled.
//
// Deterministic for any `value`, no side effects, no raise →
// `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`).
//
// Pyre's first production-crate `#[elidable_cannot_raise]` callee.  The
// trace-side effect (`record_result_of_call_pure` patching `CallI` to
// `CallPureI`) is exercised by
// `pyre/pyre-jit-trace/tests/elidable_helper_canary_test.rs`, which
// invokes the helper through `TraceCtx::call_typed_with_effect_pure`
// with an explicit `ElidableCannotRaise` `EffectInfo`.  Production
// `emit_trace_call_int_typed` callsites still pass
// `default_effect_info()` until per-helper EI registration
// lands; this canary closes the macro-side gap so the EI side can
// proceed in a separate slice.
#[majit_macros::elidable_cannot_raise]
pub fn jit_int_in_small_cache_range(value: i64) -> bool {
    w_int_small_cached(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_interpreter::PyExecutionContext;
    use pyre_interpreter::{ConstantData, compile_exec};

    /// Verifies that `#[elidable_cannot_raise]` emits the
    /// `INT_ELIDABLE_CANNOT_RAISE = 19` policy byte (`call.py:299`
    /// parity / `call_policy_byte.rs`'s `INT_ELIDABLE_CANNOT_RAISE`) and
    /// produces non-null
    /// trace_target / concrete_target trampolines.  Lives in-crate
    /// because external integration tests cannot reach the macro's
    /// `pub(crate)` `__majit_call_policy_*` symbol from outside.
    #[test]
    fn elidable_helper_macro_emits_int_elidable_cannot_raise_byte() {
        let (policy, _, trace_target, concrete_target, _, _) =
            __majit_call_policy_jit_int_in_small_cache_range();
        assert_eq!(policy, 19u8);
        assert!(!trace_target.is_null());
        assert!(!concrete_target.is_null());
    }

    /// Confirms the helper still runs correctly under the elidable
    /// attribute and preserves the elidable invariant (same input →
    /// same output).  With the `WITHPREBUILTINT=false` default
    /// (`intobject.rs`) every input returns false.
    #[test]
    fn elidable_helper_is_deterministic_and_matches_pyre_object() {
        for &value in &[0i64, 7, -3, 42, i64::MIN, i64::MAX] {
            let r1 = jit_int_in_small_cache_range(value);
            let r2 = jit_int_in_small_cache_range(value);
            assert_eq!(r1, r2, "elidable helper must be deterministic");
            assert_eq!(
                r1,
                w_int_small_cached(value),
                "elidable wrapper must mirror pyre-object's underlying check",
            );
        }
    }

    #[test]
    fn test_callable_call_helper_dispatches_builtin_without_trace_side_branching() {
        // `pypy/interpreter/pyopcode.py:921 LOAD_GLOBAL_cached` resolves
        // builtin names through `frame.get_builtin().getdictvalue(name)`.
        // Pyre's `get_builtin()` returns the builtin Module whose
        // `w_dict` is a `W_ModuleDictObject` (per `dictmultiobject.py
        // allocate_and_init_instance(module=True)`); reach `abs` via
        // that path instead of the legacy raw storage pointer so the
        // test exercises the same dispatch the JIT trace helpers will
        // see once the remaining caller cutover lands.
        let ctx = PyExecutionContext::default();
        let w_builtin = ctx.get_builtin();
        let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        let abs = unsafe { pyre_object::w_dict_getitem_str(w_builtin_dict, "abs") }
            .expect("abs builtin must exist");
        let result = jit_call_callable_1(std::ptr::null(), abs, w_int_new(-11));
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 11);
        }
    }

    #[test]
    fn test_container_helpers_dispatch_expected_runtime_shapes() {
        let result = jit_build_tuple_2(w_int_new(3), w_int_new(5));
        let tuple = result;
        unsafe {
            assert!(is_tuple(tuple));
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 0).unwrap()), 3);
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 1).unwrap()), 5);
        }

        let list = w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let tuple = w_tuple_new(vec![w_int_new(7), w_int_new(9)]);
        unsafe {
            assert_eq!(w_int_get_value(jit_sequence_getitem(list, 1)), 4);
            assert_eq!(w_int_get_value(jit_sequence_getitem(tuple, 0)), 7);
        }

        let result = jit_build_map_2(w_int_new(1), w_int_new(10), w_int_new(2), w_int_new(20));
        let dict = result;
        unsafe {
            assert!(is_dict(dict));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 10);
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 2).unwrap()), 20);
        }
    }

    #[test]
    fn test_numeric_helpers_reuse_objspace_semantics() {
        let result = jit_binary_value_from_tag(w_int_new(9), w_int_new(4), 1);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 5);
        }

        let result = jit_compare_value_from_tag(w_int_new(2), w_int_new(7), 0);
        unsafe {
            assert!(w_bool_get_value(result as PyObjectRef));
        }

        let result = jit_unary_invert_value(w_int_new(5));
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), !5);
        }
    }

    #[test]
    fn test_make_function_helper_wraps_code_object() {
        let module = compile_exec("def f(x):\n    return x").expect("compile failed");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("expected nested function code");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let code_obj = pyre_interpreter::w_code_new(code_ptr);
        let func = jit_make_function_from_globals(PY_NULL, code_obj);

        unsafe {
            assert!(pyre_interpreter::is_function(func));
            // Function.code now stores the PyCode, not the raw CodeObject.
            assert_eq!(
                pyre_interpreter::function_get_code(func),
                code_obj as *const ()
            );
        }
    }

    #[test]
    fn test_range_iter_next_helper_uses_runtime_iterator_step() {
        let iter = w_range_iter_new(0, 2, 1);
        let first = jit_range_iter_next_or_null(iter);
        let second = jit_range_iter_next_or_null(iter);
        let done = jit_range_iter_next_or_null(iter);
        unsafe {
            assert_eq!(w_int_get_value(first), 0);
            assert_eq!(w_int_get_value(second), 1);
            assert!(done.is_null());
        }
    }
}
