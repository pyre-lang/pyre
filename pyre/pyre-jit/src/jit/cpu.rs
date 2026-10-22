//! RPython: `rpython/jit/backend/model.py` `AbstractCPU` /
//! `rpython/jit/backend/llgraph/runner.py` `LLGraphCPU`.
//!
//! pyre-side bundle of the blackhole helper function pointers used by
//! `transform_graph_to_jitcode` to resolve `bhimpl_residual_call` and
//! the per-arity / per-opcode helpers.
//!
//! In RPython the same `cpu` object is referenced from both
//! `CodeWriter.cpu` (codewriter.py) and `CallControl.cpu`
//! (call.py); pyre owns it on `CallControl` and exposes a
//! convenience accessor `CodeWriter::cpu(&self)` so the upstream
//! attribute access pattern still works.
//!
//! Note: pyre's "cpu" is much smaller than RPython's
//! `LLGraphCPU` — there is no `calldescrof`, no `setup_descrs`, no
//! vector-extension support, and no GC integration at this layer.
//! All those concerns either live in `pyre_jit_trace::state` (descrs)
//! or are handled by the metainterp directly. This struct holds only
//! the helpers that the codewriter needs at compile time to emit the
//! correct fn-pointer indices into the `JitCode` table.

/// `rpython/jit/backend/model.py` `class AbstractCPU(object)`.
///
/// pyre-side: blackhole `bhimpl_*` helper trampolines, all `extern "C"`
/// so the compiled `JitCode` can call them via raw fn-pointer slots.
///
/// Deliberately not `Debug`: the derived formatter requires
/// `core::marker::FnPtr` for every helper field, and Charon does not model
/// that built-in trait impl. Deriving it makes extraction emit two trait
/// resolution warnings per function pointer even though `Cpu` is never
/// formatted.
pub struct Cpu {
    /// `bhimpl_residual_call` general entry point.
    /// `(callable, null_or_self, arg0) → result`.
    pub call_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// Per-arity `bhimpl_residual_call_<n>` helpers
    /// (`call_fn_0(callable, null_or_self)` ...
    /// `call_fn_14(callable, null_or_self, a0..a13)`).  RPython
    /// `bhimpl_residual_call_r_r` carries no frame; the parent frame is
    /// resolved from the execution context inside `bh_call_fn_impl`.  A
    /// non-null `null_or_self` is the method receiver — the helper
    /// prepends it as arg0 (`eval.rs`'s `PyFrame::call`).  The arity ceiling is
    /// nargs=14: each helper takes `callable + null_or_self + nargs`
    /// i64s, and the backend dispatch table (`call_stub.rs::
    /// dispatch_arity_body!`, `MAX_HOST_CALL_ARITY` = 16) tops out at
    /// 16 i64 arguments.
    pub call_fn_0: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    pub call_fn_2: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_3: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_4: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_5: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_6: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_7: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_8: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_9: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_10: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_11: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_12: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_13: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_fn_14: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// `bhimpl_load_global` — namespace/code from getfield_vable_r plus live frame.
    pub load_global_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        *mut pyre_interpreter::PyFrame,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_load_from_dict_or_globals_fn(dict, code, frame, namei)` —
    /// LOAD_FROM_DICT_OR_GLOBALS: try the popped mapping then frame globals.
    pub load_from_dict_or_globals_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        *mut pyre_interpreter::PyFrame,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    /// — CALL_FUNCTION_EX: unpack `*`/`**` and dispatch.
    pub call_function_ex_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// Per-arity `bh_call_kw_<n>` helpers for CALL_KW, ABI
    /// `(callable, null_or_self, kwnames, arg0..arg{n-1})` = 3 + n i64.
    /// The kwnames slot leaves room for nargs 0..=13 within the backend's
    /// `MAX_HOST_CALL_ARITY` = 16 ceiling; CALL_KW with nargs > 13 aborts.
    pub call_kw_fn_0: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_1: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_2: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_3: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_4: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_5: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_6: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_7: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_8: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_9: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_10: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_11: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_12: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    pub call_kw_fn_13: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// LOOKUP_METHOD attribute half — `(obj, code, name_idx) → attr`.
    /// Reproduces `PyFrame::load_method`'s `getattr` for blackhole resume.
    pub load_attr_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// LOOKUP_METHOD `null_or_self` half — `(obj, attr, code, name_idx) →
    /// bound`. Pure binding decision shared with the interpreter.
    pub load_method_self_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// LOAD_SPECIAL attribute half — `(obj, method_kind) → attr`.
    pub load_special_fn: extern "C" fn(pyre_object::PyObjectRef, i64) -> pyre_object::PyObjectRef,
    /// WITH_EXCEPT_START — `(exit_func, exit_self, exception) → bool-like result`.
    pub with_except_start_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// STORE_ATTR residual — `(obj, value, code, name_idx) → void`.
    /// Resolves the name from the code object and runs generic `setattr`.
    pub store_attr_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> i64,
    /// BINARY_SLICE residual — `(obj, start, stop) → obj[start:stop]`.
    pub binary_slice_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// STORE_SLICE residual — `(obj, start, stop, value) → void`
    /// (`obj[start:stop] = value`).
    pub store_slice_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// DELETE_SUBSCR residual — `(obj, index) → void` (`del obj[index]`).
    pub delete_subscr_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// LIST_EXTEND residual — `(list, iterable) → void` (`list.extend(iterable)`,
    /// list peeked + mutated in place).
    pub list_extend_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// SET_ADD residual — `(set, value) → void` (`set.add(value)`, peeked).
    pub set_add_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// SET_UPDATE residual — `(set, iterable) → void` (`set.update`, peeked).
    pub set_update_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// DICT_UPDATE residual — `(dict, source) → void` (`dict.update`, peeked).
    pub dict_update_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// MAP_ADD residual — `(dict, key, value) → void` (`dict[key]=value`, peeked).
    pub map_add_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// DICT_MERGE residual — `(dict, source, callable) → void` (`**` merge,
    /// peeked; callable only for error-message prefixes).
    pub dict_merge_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// LIST_APPEND residual — `(list, value) → void` (`list.append(value)`,
    /// list peeked + mutated in place).  The full-body walker's #171 fold
    /// intercepts it (`RuntimeHelperKind::ListAppendValue`); this is the decline
    /// fallback, identical to the residual the retired MIFrame tracer recorded.
    pub list_append_fn: extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// DELETE_ATTR residual — `(obj, code, name_idx) → void` (`del obj.name`).
    /// Resolves the name from the code object and runs generic `delattr`.
    pub delete_attr_fn:
        extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef, i64) -> i64,
    /// FORMAT_SIMPLE residual — `value → str` (`f"{x}"`, empty spec).
    /// User `__format__` may run Python (fallible).
    pub format_simple_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// FORMAT_WITH_SPEC residual — `(value, spec) → str` (`f"{x:.2f}"`).
    /// User `__format__` may run Python (fallible).
    pub format_with_spec_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// CONVERT_VALUE residual — `(value, conv) → str` (`f"{x!r}"`).
    /// `conv` is a `runtime_ops::convert_value_code`; user `__str__` /
    /// `__repr__` may run Python (fallible).
    pub convert_value_fn: extern "C" fn(pyre_object::PyObjectRef, i64) -> pyre_object::PyObjectRef,
    /// `bh_import_from_fn(module, code, name_idx)` — IMPORT_FROM residual;
    /// resolves the attribute name from the code object and runs
    /// `importing::import_from` on the peeked module (attribute lookup, then a
    /// `sys.modules` fallback; a user `__getattribute__` / `__getattr__` may
    /// run Python → fallible).
    pub import_from_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_load_super_attr_fn(global_super, self, cls, frame, code, name_idx,
    /// is_two_arg)` — LOAD_SUPER_ATTR residual. Calls the actual global
    /// callable with zero or two args, then resolves the attribute.
    pub load_super_attr_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        *mut pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        i64,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_super_attr_unwrap_fn(raw, which)` — LOAD_SUPER_ATTR method-form
    /// unwrap (`which` 0 = func slot, 1 = self slot); pure / infallible.
    pub super_attr_unwrap_fn:
        extern "C" fn(pyre_object::PyObjectRef, i64) -> pyre_object::PyObjectRef,
    /// `bh_load_deref_value_fn(cell, code, deref_idx)` — LOAD_DEREF
    /// dereference residual (cell contents, raising the named unbound-variable
    /// `NameError` resolved via `code` + `deref_idx`).
    pub load_deref_value_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_store_deref_value_fn(cell, value)` — STORE_DEREF residual: mutate
    /// the cell's contents (returning the unchanged cell) or return the raw
    /// `value` for a non-cell slot; infallible.
    pub store_deref_value_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// `bh_make_cell_fn(current, code, slot)` — MAKE_CELL residual: wrap a raw
    /// slot value in a fresh cell of the `code` + `slot` cell family (or return
    /// an existing cell unchanged); infallible.
    pub make_cell_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `jit_make_function_from_globals(globals, code)` — MAKE_FUNCTION residual:
    /// wrap a code object into a function using the given globals object;
    /// allocates but runs no user code and never raises.
    pub make_function_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// `jit_set_function_attribute(func, attr, flag)` — SET_FUNCTION_ATTRIBUTE
    /// residual: stamp one attribute (`flag` discriminant) on `func`, returning
    /// `func`; sets a typed field but runs no user code and never raises.
    pub set_function_attribute_fn:
        extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef, i64) -> i64,
    /// `bh_get_iter_fn(obj)` — GET_ITER `iter(obj)` residual
    /// (a user `__iter__` may run Python → fallible).
    pub get_iter_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `jit_newlist_from_iter_hint(iter)` — BUILD_LIST_FROM_ARG residual:
    /// `space.newlist([], sizehint=length_hint(iter))`. A user
    /// `__length_hint__` may run Python → fallible.
    pub newlist_from_iter_hint_fn:
        extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `jit_next(iter)` — FOR_ITER `next(iter)` residual; returns the next
    /// item, PY_NULL on StopIteration exhaustion (the trailing for-iter
    /// GuardNonnull catches it), or publishes a real exception into the
    /// backend exception cells on error.
    pub for_iter_next_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// MRO-aware Python exception match used by the FOR_ITER catch arm.
    /// Returns a raw bool-as-int and cannot raise.
    pub for_iter_exception_match_fn:
        extern "C" fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> i64,
    /// interp_jit.py `jump_absolute` residual: the `bytecode_trace` slow
    /// path.  Void result (`residual_call_r_v`):
    /// always returns 0; an exception is published for `GuardNoException`.
    pub bytecode_trace_jitted_slow_fn: extern "C" fn(
        *mut pyre_interpreter::PyExecutionContext,
        *mut pyre_interpreter::PyFrame,
    ) -> i64,
    /// pyopcode.py `dispatch_bytecode` jitted arm residual:
    /// `ec.bytecode_only_trace(self)`.  Void result (`residual_call_r_v`).
    pub bytecode_only_trace_fn: extern "C" fn(
        *mut pyre_interpreter::PyExecutionContext,
        *mut pyre_interpreter::PyFrame,
    ) -> i64,
    /// `bh_unary_negative_fn(value)` — UNARY_NEGATIVE `-value` residual
    /// (a user `__neg__` may run Python → fallible).
    pub unary_negative_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_unary_invert_fn(value)` — UNARY_INVERT `~value` residual
    /// (a user `__invert__` may run Python → fallible).
    pub unary_invert_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_unary_positive_fn(value)` — UNARY_POSITIVE `+value` residual
    /// (a user `__pos__` may run Python → fallible).
    pub unary_positive_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_load_common_constant_fn(disc)` — LOAD_COMMON_CONSTANT residual
    /// resolving a `CommonConstant` discriminant to its pushed object
    /// (allocates for the `all`/`any` builtin variants → `MayForce`).
    pub load_common_constant_fn: extern "C" fn(i64) -> pyre_object::PyObjectRef,
    /// `bh_list_to_tuple_fn(value)` — CALL_INTRINSIC_1 ListToTuple residual
    /// (`list_to_tuple`, allocates a fresh tuple; non-list → TypeError).
    pub list_to_tuple_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_get_len_fn(subject)` — GET_LEN residual; pushes `len(subject)`
    /// without consuming it (runs `__len__`).
    pub get_len_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_match_sequence_fn(subject)` — MATCH_SEQUENCE residual; reads the
    /// subject type's PATMA marker (no user code).
    pub match_sequence_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_match_mapping_fn(subject)` — MATCH_MAPPING residual, mirroring
    /// `match_sequence_fn`.
    pub match_mapping_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_match_keys_fn(subject, keys)` — MATCH_KEYS residual; looks each
    /// pattern key up via `get` (user code; duplicate key → ValueError).
    pub match_keys_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// `bh_match_class_fn(subject, cls, kwd_attrs, count)` — MATCH_CLASS
    /// residual; runs `isinstance` and attribute lookups.
    pub match_class_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_unary_not_fn(value)` — UNARY_NOT `not value` residual returning a
    /// bool (a user `__bool__` / `__len__` may run Python; infallible).
    pub unary_not_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_to_bool_fn(value)` — TO_BOOL residual returning `bool(value)` as a
    /// bool object (a user `__bool__` / `__len__` may run Python).
    pub to_bool_fn: extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bh_load_fast_check_fn(value, code, name_idx)` — LOAD_FAST_CHECK
    /// unbound-local guard (returns `value`, or raises `NameError`).
    pub load_fast_check_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bh_unbound_local_error_fn(code, name_idx)` — construct and return the
    /// UnboundLocalError value for DELETE_FAST without publishing it.
    pub unbound_local_error_fn:
        extern "C" fn(pyre_object::PyObjectRef, i64) -> pyre_object::PyObjectRef,
    /// Residual-fallback entry point for the `compare_value_from_tag` graph.
    pub compare_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// Residual-fallback entry point for the `binary_value_from_tag` graph.
    pub binary_op_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bhimpl_w_int_new` — box a raw integer into a PyObject.
    pub box_int_fn: extern "C" fn(i64) -> pyre_object::PyObjectRef,
    /// `bhimpl_truth` — PyObjectRef → raw 0 or 1.
    pub truth_fn: extern "C" fn(pyre_object::PyObjectRef) -> i64,
    /// `bhimpl_load_const` — load constant from frame's code object.
    pub load_const_fn: extern "C" fn(pyre_object::PyObjectRef, i64) -> pyre_object::PyObjectRef,
    /// `bhimpl_store_subscr` — obj[key] = value.
    pub store_subscr_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// `bhimpl_getattr` — `getattr(obj, w_name)`.
    /// `(obj: Ref, w_name: Ref) → Ref` with `w_name` an interned str
    /// constant.  Blackhole/deopt lowering of `LOAD_ATTR`
    /// (`rclass.py rtype_getattr`).
    pub getattr_fn: extern "C" fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// `bhimpl_load_name` — `(frame: Ref, w_name: Ref, namei: Int) → Ref`
    /// with `w_name` an interned str constant.  Blackhole/deopt lowering
    /// of `LOAD_NAME` (`pyopcode.py`).
    pub load_name_fn: extern "C" fn(
        *mut pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `bhimpl_store_name` — `(frame: Ref, w_name: Ref, value: Ref) → Void`
    /// with `w_name` an interned str constant.  Blackhole/deopt lowering
    /// of `STORE_NAME` (`pyopcode.py`).
    pub store_name_fn: extern "C" fn(
        *mut pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// `bhimpl_store_global` — `(frame: Ref, w_name: Ref, value: Ref) →
    /// Void` with `w_name` an interned str constant.  Blackhole/deopt
    /// lowering of `STORE_GLOBAL` (`pyopcode.py`); writes directly
    /// into `w_globals`, bypassing `w_locals`.
    pub store_global_fn: extern "C" fn(
        *mut pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> i64,
    /// `bh_delete_name_fn` — `(frame: Ref, w_name: Ref) → Void`, the
    /// lowering of `DELETE_NAME`; deletes the binding from `w_locals` and
    /// raises `NameError` when it is absent.
    pub delete_name_fn:
        extern "C" fn(*mut pyre_interpreter::PyFrame, pyre_object::PyObjectRef) -> i64,
    /// `bh_delete_global_fn` — `(frame: Ref, w_name: Ref) → Void`, the
    /// lowering of `DELETE_GLOBAL`; deletes directly from `w_globals`,
    /// bypassing `w_locals`.
    pub delete_global_fn:
        extern "C" fn(*mut pyre_interpreter::PyFrame, pyre_object::PyObjectRef) -> i64,
    /// `bh_load_locals_fn` — `(frame: Ref) → Ref`, the lowering of
    /// `LOAD_LOCALS` (`pyopcode.py`); hands back the frame's own
    /// `w_locals` mapping.
    pub load_locals_fn: extern "C" fn(*mut pyre_interpreter::PyFrame) -> pyre_object::PyObjectRef,
    /// `bh_load_build_class_fn` — `(frame: Ref) → Ref`, the lowering of
    /// `LOAD_BUILD_CLASS` (`pyopcode.py`); reads `__build_class__` out of
    /// the frame's builtin mapping.
    pub load_build_class_fn:
        extern "C" fn(*mut pyre_interpreter::PyFrame) -> pyre_object::PyObjectRef,
    /// Load `builtins.__import__` for IMPORT_NAME.  Kept separate from the
    /// call itself so the generated jitcode has the same ordinary Python
    /// call boundary as PyPy's `IMPORT_NAME` implementation.
    pub load_import_fn: extern "C" fn(*mut pyre_interpreter::PyFrame) -> pyre_object::PyObjectRef,
    /// Load IMPORT_NAME's locals argument.  Separate from `load_import_fn`
    /// because `pyopcode.py`'s `IMPORT_NAME` reads two independent things off the
    /// frame before the call.
    pub load_import_locals_fn:
        extern "C" fn(*mut pyre_interpreter::PyFrame) -> pyre_object::PyObjectRef,
    /// IMPORT_NAME's globals argument — `self.get_w_globals()`.  (frame) → ref.
    pub load_import_globals_fn:
        extern "C" fn(*mut pyre_interpreter::PyFrame) -> pyre_object::PyObjectRef,
    /// IMPORT_NAME residual for opcodes that can still take the PEP 810
    /// lazy route (`eval_lazy_import_name`). Frame plus name / fromlist /
    /// level / explicit-lazy bit; `MayForce` / can-raise.
    pub eval_lazy_import_name_fn: extern "C" fn(
        *mut pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        i64,
    ) -> pyre_object::PyObjectRef,
    /// `newtuple(list_w)` (`objspace.py:332`) — (ref array) → new tuple.
    /// The array is the forced `popvalues` list; length travels inside
    /// the array, so any arity fits.
    pub newtuple_from_array_fn:
        extern "C" fn(*const pyre_object::object_array::GcTypedArray) -> pyre_object::PyObjectRef,
    /// BUILD_MAP — the forced `[k0, v0, ...]` pair array → dict.  Length
    /// travels inside the array, so any arity fits.
    pub build_map_from_array_fn:
        extern "C" fn(*const pyre_object::object_array::GcTypedArray) -> pyre_object::PyObjectRef,
    /// BUILD_SET — the forced element array → set (fallible: element
    /// hashing may run user `__hash__` / raise on a non-hashable element).
    pub build_set_from_array_fn:
        extern "C" fn(*const pyre_object::object_array::GcTypedArray) -> pyre_object::PyObjectRef,
    /// BUILD_STRING — the forced fragment array → concatenated str.
    /// Fragments are already strings (formatted first), so this runs no
    /// user code and is infallible (`Plain`).
    pub build_string_from_array_fn:
        extern "C" fn(*const pyre_object::object_array::GcTypedArray) -> pyre_object::PyObjectRef,
    /// `newlist(list_w)` (`objspace.py`) — (ref array) → new list.  The
    /// array is the forced `popvalues_mutable` list; length travels
    /// inside the array, so any arity fits.
    pub newlist_from_array_fn:
        extern "C" fn(*const pyre_object::object_array::GcTypedArray) -> pyre_object::PyObjectRef,
    /// `bhimpl_unpack_sequence` — (count, seq) → validated tuple of items.
    pub unpack_sequence_fn:
        extern "C" fn(i64, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// Read item `index` out of the validated unpack tuple — (index, seq) → item.
    pub unpack_item_fn: extern "C" fn(i64, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// UNPACK_EX residual — `(before, after, seq) → tuple` of the
    /// `before + 1 + after` slots (head items, starred list, tail items)
    /// in TOS order; read back with `unpack_item_fn`.
    pub unpack_ex_fn: extern "C" fn(i64, i64, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
    /// `bhimpl_build_slice` — (argc, start, stop, step) → new slice.
    pub build_slice_fn: extern "C" fn(
        i64,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// `RAISE_VARARGS` normalization helper used before `raise/r`.
    /// `(frame: Ref, exc: Ref, cause: Ref) → Ref` — normalization reads the
    /// current activation's execution context, as `space.getexecutioncontext()`.
    pub normalize_raise_varargs_fn: extern "C" fn(
        *const pyre_interpreter::PyFrame,
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef,
    /// Read per-thread `CURRENT_EXCEPTION` — used by a catch-covered bare
    /// `RAISE_VARARGS(0)`.
    pub get_current_exception_fn: extern "C" fn() -> pyre_object::PyObjectRef,
    /// The value `PUSH_EXC_INFO` saves: the current exception, or `None`.
    pub current_exception_or_none_fn: extern "C" fn() -> pyre_object::PyObjectRef,
    /// `raise_varargs(0)` value — the active exception, or a fresh
    /// `RuntimeError("No active exception to reraise")` when none is live.
    /// Used by a bare `RAISE_VARARGS(0)` with no static `last_exception` pair.
    pub reraise_varargs_zero_fn: extern "C" fn() -> pyre_object::PyObjectRef,
    /// Write per-thread `CURRENT_EXCEPTION` — used by `PUSH_EXC_INFO`
    /// (set to new exc) and `POP_EXCEPT` (restore saved prev).
    pub set_current_exception_fn: extern "C" fn(pyre_object::PyObjectRef),
    /// Clear the exception-propagation carrier after `PUSH_EXC_INFO` has
    /// transferred ownership to `CURRENT_EXCEPTION`.
    pub clear_in_flight_exception_fn: extern "C" fn(),
    /// `rpython/jit/backend/llgraph/runner.py:LLGraphCPU.rtyper` —
    /// upstream `flatten_graph` reaches `cpu.rtyper.exceptiondata.
    /// get_standard_ll_exc_instance_by_class(OverflowError)` at
    /// `flatten.py` (the `handling_ovf=True` arm of
    /// `make_exception_link`).  Pyre's rtyper shim
    /// ([`super::exceptiondata::Rtyper`]) exposes only that attribute
    /// chain; other rtyper machinery is intentionally absent because
    /// pyre operates on the flowspace graph directly, without a
    /// typed-low-level rewrite.
    pub rtyper: super::exceptiondata::Rtyper,
    /// Retired-family fn-idx pool — pyre-specific extension carried
    /// on `Cpu` so the canonical `flatten_graph(graph, regallocs,
    /// _include_all_exc_links, cpu)` entry can derive the
    /// dispatcher's `LoweringContext` from `cpu` alone (matching
    /// upstream's "cpu carries everything flatten needs" contract).
    /// Each `Option<u16>` is populated by `CodeWriter::transform_
    /// graph_to_jitcode` from `descrs.intern_int_method_index`.
    /// **Note**: upstream's `cpu` does not carry
    /// these because upstream's rtyper rewrites the graph to post-
    /// rtype shape before `flatten_graph` runs, so the dispatcher has
    /// no upstream analog.  These fields retire once pyre's
    /// walker stops emitting pre-rtype HLOps in favour of post-rtype
    /// residual_call SpaceOperations on the graph.
    pub lowering_ctx: parking_lot::RwLock<Option<super::flatten::LoweringContext>>,
}

/// wasm32: publish the descr-word shim (`word_publish::extern_addrN`) so
/// the table type equals the calldescr FUNC. Native keeps the raw address.
/// The stored bits are only taken as `*const ()` into jitcode constants.
macro_rules! cpu_word {
    (0, $f:path) => {
        cpu_word!(@apply extern_addr0, $f,)
    };
    (1, $f:path) => {
        cpu_word!(@apply extern_addr1, $f, a0)
    };
    (2, $f:path) => {
        cpu_word!(@apply extern_addr2, $f, a0, a1)
    };
    (3, $f:path) => {
        cpu_word!(@apply extern_addr3, $f, a0, a1, a2)
    };
    (4, $f:path) => {
        cpu_word!(@apply extern_addr4, $f, a0, a1, a2, a3)
    };
    (5, $f:path) => {
        cpu_word!(@apply extern_addr5, $f, a0, a1, a2, a3, a4)
    };
    (6, $f:path) => {
        cpu_word!(@apply extern_addr6, $f, a0, a1, a2, a3, a4, a5)
    };
    (7, $f:path) => {
        cpu_word!(@apply extern_addr7, $f, a0, a1, a2, a3, a4, a5, a6)
    };
    (8, $f:path) => {
        cpu_word!(@apply extern_addr8, $f, a0, a1, a2, a3, a4, a5, a6, a7)
    };
    (9, $f:path) => {
        cpu_word!(@apply extern_addr9, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8)
    };
    (10, $f:path) => {
        cpu_word!(@apply extern_addr10, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    };
    (11, $f:path) => {
        cpu_word!(@apply extern_addr11, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    };
    (12, $f:path) => {
        cpu_word!(@apply extern_addr12, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
    };
    (13, $f:path) => {
        cpu_word!(@apply extern_addr13, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
    };
    (14, $f:path) => {
        cpu_word!(@apply extern_addr14, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)
    };
    (15, $f:path) => {
        cpu_word!(@apply extern_addr15, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)
    };
    (16, $f:path) => {
        cpu_word!(@apply extern_addr16, $f, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)
    };
    (@apply $addr:ident, $f:path, $($a:ident),*) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            $f
        }
        #[cfg(target_arch = "wasm32")]
        {
            let p = pyre_interpreter::jit_fnaddr::word_publish::$addr(|$($a),*| $f($($a),*), $f);
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                ::core::mem::transmute(p)
            }
        }
    }};
}

impl Cpu {
    /// Default pyre `Cpu` — wires the production `bh_*` thunks from
    /// `crate::call_jit`. Matches the implicit `cpu = LLGraphCPU(...)`
    /// constructor in `warmspot.py:243` for the standard JIT.
    pub fn new() -> Self {
        let mut rtyper = super::exceptiondata::Rtyper::new();
        // Install a **lazy** resolver for standard exception instance
        // pointers — RPython's `RPythonTyper.specialize` ->
        // `ExceptionData.finish` invokes `get_standard_ll_exc_instance(
        // rtyper, clsdef)` at rtyper construction time
        // (`rpython/rtyper/exceptiondata.py`), which calls
        // `r_inst.get_reusable_prebuilt_instance()` and returns the
        // **prebuilt INSTANCE** pointer (not the class pointer).
        // Downstream `get_standard_ll_exc_instance_by_class` wraps
        // that pointer in a `Constant` and `flatten.py:165-170` emits
        // it directly as the operand of `raise/r`; feeding the
        // **class** pointer here would be semantically wrong
        // (`raise CLASS` is not the same as `raise INSTANCE`).
        //
        // Pyre's analog is `pyre_interpreter::lookup_exc_instance`
        // which materialises a process-global singleton
        // `W_BaseException` per `ExcKind` (see
        // `pyre_object::interp_exceptions::standard_exc_instance`).  We install
        // the resolver here but defer the per-`ExcKind` singleton
        // allocation to the first
        // `get_standard_ll_exc_instance_by_class` lookup; under the
        // current walker-driven pipeline the canonical
        // `flatten_graph` `_ovf` direct-raise rewrite (the sole
        // consumer of this table in production) never fires, so the
        // singletons stay unallocated.  The deferral matters because
        // the cranelift backend's trace compilation is sensitive to
        // heap layout — eagerly allocating 16 `W_BaseException`
        // singletons at `Cpu::new` time shifts subsequent heap
        // addresses enough to consistently push raise_catch_loop
        // tracing into a slow recompile path.
        rtyper.exceptiondata.set_lazy_resolver(|name| {
            let ptr = pyre_interpreter::lookup_exc_instance(name)?;
            Some(ptr as i64)
        });
        Self {
            call_fn: cpu_word!(3, crate::call_jit::bh_call_fn),
            call_fn_0: cpu_word!(2, crate::call_jit::bh_call_fn_0),
            call_fn_2: cpu_word!(4, crate::call_jit::bh_call_fn_2),
            call_fn_3: cpu_word!(5, crate::call_jit::bh_call_fn_3),
            call_fn_4: cpu_word!(6, crate::call_jit::bh_call_fn_4),
            call_fn_5: cpu_word!(7, crate::call_jit::bh_call_fn_5),
            call_fn_6: cpu_word!(8, crate::call_jit::bh_call_fn_6),
            call_fn_7: cpu_word!(9, crate::call_jit::bh_call_fn_7),
            call_fn_8: cpu_word!(10, crate::call_jit::bh_call_fn_8),
            call_fn_9: cpu_word!(11, crate::call_jit::bh_call_fn_9),
            call_fn_10: cpu_word!(12, crate::call_jit::bh_call_fn_10),
            call_fn_11: cpu_word!(13, crate::call_jit::bh_call_fn_11),
            call_fn_12: cpu_word!(14, crate::call_jit::bh_call_fn_12),
            call_fn_13: cpu_word!(15, crate::call_jit::bh_call_fn_13),
            call_fn_14: cpu_word!(16, crate::call_jit::bh_call_fn_14),
            load_global_fn: cpu_word!(4, crate::call_jit::bh_load_global_fn),
            load_from_dict_or_globals_fn: cpu_word!(
                4,
                crate::call_jit::bh_load_from_dict_or_globals_fn
            ),
            call_function_ex_fn: cpu_word!(4, crate::call_jit::bh_call_function_ex_fn),
            call_kw_fn_0: cpu_word!(3, crate::call_jit::bh_call_kw_0),
            call_kw_fn_1: cpu_word!(4, crate::call_jit::bh_call_kw_1),
            call_kw_fn_2: cpu_word!(5, crate::call_jit::bh_call_kw_2),
            call_kw_fn_3: cpu_word!(6, crate::call_jit::bh_call_kw_3),
            call_kw_fn_4: cpu_word!(7, crate::call_jit::bh_call_kw_4),
            call_kw_fn_5: cpu_word!(8, crate::call_jit::bh_call_kw_5),
            call_kw_fn_6: cpu_word!(9, crate::call_jit::bh_call_kw_6),
            call_kw_fn_7: cpu_word!(10, crate::call_jit::bh_call_kw_7),
            call_kw_fn_8: cpu_word!(11, crate::call_jit::bh_call_kw_8),
            call_kw_fn_9: cpu_word!(12, crate::call_jit::bh_call_kw_9),
            call_kw_fn_10: cpu_word!(13, crate::call_jit::bh_call_kw_10),
            call_kw_fn_11: cpu_word!(14, crate::call_jit::bh_call_kw_11),
            call_kw_fn_12: cpu_word!(15, crate::call_jit::bh_call_kw_12),
            call_kw_fn_13: cpu_word!(16, crate::call_jit::bh_call_kw_13),
            load_attr_fn: cpu_word!(3, crate::call_jit::bh_load_attr_fn),
            load_method_self_fn: cpu_word!(4, crate::call_jit::bh_load_method_self_fn),
            load_special_fn: cpu_word!(2, crate::call_jit::bh_load_special_fn),
            with_except_start_fn: cpu_word!(3, crate::call_jit::bh_with_except_start_fn),
            store_attr_fn: cpu_word!(4, crate::call_jit::bh_store_attr_fn),
            binary_slice_fn: cpu_word!(3, crate::call_jit::bh_binary_slice_fn),
            store_slice_fn: cpu_word!(4, crate::call_jit::bh_store_slice_fn),
            delete_subscr_fn: cpu_word!(2, crate::call_jit::bh_delete_subscr_fn),
            delete_attr_fn: cpu_word!(3, crate::call_jit::bh_delete_attr_fn),
            list_extend_fn: cpu_word!(2, crate::call_jit::bh_list_extend_fn),
            set_add_fn: cpu_word!(2, crate::call_jit::bh_set_add_fn),
            set_update_fn: cpu_word!(2, crate::call_jit::bh_set_update_fn),
            dict_update_fn: cpu_word!(2, crate::call_jit::bh_dict_update_fn),
            map_add_fn: cpu_word!(3, crate::call_jit::bh_map_add_fn),
            dict_merge_fn: cpu_word!(3, crate::call_jit::bh_dict_merge_fn),
            list_append_fn: cpu_word!(2, pyre_object::listobject::jit_list_append),
            format_simple_fn: cpu_word!(1, crate::call_jit::bh_format_simple_fn),
            format_with_spec_fn: cpu_word!(2, crate::call_jit::bh_format_with_spec_fn),
            convert_value_fn: cpu_word!(2, crate::call_jit::bh_convert_value_fn),
            import_from_fn: cpu_word!(3, crate::call_jit::bh_import_from_fn),
            load_super_attr_fn: cpu_word!(7, crate::call_jit::bh_load_super_attr_fn),
            super_attr_unwrap_fn: cpu_word!(2, crate::call_jit::bh_super_attr_unwrap_fn),
            load_deref_value_fn: cpu_word!(3, crate::call_jit::bh_load_deref_value_fn),
            store_deref_value_fn: cpu_word!(2, crate::call_jit::bh_store_deref_value_fn),
            make_cell_fn: cpu_word!(3, crate::call_jit::bh_make_cell_fn),
            make_function_fn: cpu_word!(
                2,
                pyre_interpreter::runtime_ops::jit_make_function_from_globals
            ),
            set_function_attribute_fn: cpu_word!(
                3,
                pyre_interpreter::runtime_ops::jit_set_function_attribute
            ),
            get_iter_fn: cpu_word!(1, crate::call_jit::bh_get_iter_fn),
            newlist_from_iter_hint_fn: cpu_word!(
                1,
                pyre_interpreter::opcode_ops::jit_newlist_from_iter_hint
            ),
            for_iter_next_fn: cpu_word!(1, pyre_interpreter::runtime_ops::jit_next),
            for_iter_exception_match_fn: cpu_word!(
                2,
                pyre_interpreter::runtime_ops::jit_exception_match
            ),
            unary_negative_fn: cpu_word!(1, crate::call_jit::bh_unary_negative_fn),
            unary_invert_fn: cpu_word!(1, crate::call_jit::bh_unary_invert_fn),
            unary_positive_fn: cpu_word!(1, crate::call_jit::bh_unary_positive_fn),
            load_common_constant_fn: cpu_word!(1, crate::call_jit::bh_load_common_constant_fn),
            list_to_tuple_fn: cpu_word!(1, crate::call_jit::bh_list_to_tuple_fn),
            get_len_fn: cpu_word!(1, crate::call_jit::bh_get_len_fn),
            match_sequence_fn: cpu_word!(1, crate::call_jit::bh_match_sequence_fn),
            match_mapping_fn: cpu_word!(1, crate::call_jit::bh_match_mapping_fn),
            match_keys_fn: cpu_word!(2, crate::call_jit::bh_match_keys_fn),
            match_class_fn: cpu_word!(4, crate::call_jit::bh_match_class_fn),
            unary_not_fn: cpu_word!(1, crate::call_jit::bh_unary_not_fn),
            to_bool_fn: cpu_word!(1, crate::call_jit::bh_to_bool_fn),
            load_fast_check_fn: cpu_word!(3, crate::call_jit::bh_load_fast_check_fn),
            unbound_local_error_fn: cpu_word!(2, crate::call_jit::bh_unbound_local_error_fn),
            compare_fn: cpu_word!(3, pyre_interpreter::opcode_ops::jit_compare_value_from_tag),
            binary_op_fn: cpu_word!(3, pyre_interpreter::opcode_ops::jit_binary_value_from_tag),
            box_int_fn: cpu_word!(1, crate::call_jit::bh_box_int_fn),
            truth_fn: cpu_word!(1, crate::call_jit::bh_truth_fn),
            load_const_fn: cpu_word!(2, crate::call_jit::bh_load_const_fn),
            store_subscr_fn: cpu_word!(3, pyre_interpreter::opcode_ops::bh_store_subscr_fn),
            getattr_fn: cpu_word!(2, crate::call_jit::bh_getattr_fn),
            load_name_fn: cpu_word!(3, crate::call_jit::bh_load_name_fn),
            store_name_fn: cpu_word!(3, crate::call_jit::bh_store_name_fn),
            store_global_fn: cpu_word!(3, crate::call_jit::bh_store_global_fn),
            delete_name_fn: cpu_word!(2, crate::call_jit::bh_delete_name_fn),
            delete_global_fn: cpu_word!(2, crate::call_jit::bh_delete_global_fn),
            load_locals_fn: cpu_word!(1, crate::call_jit::bh_load_locals_fn),
            load_build_class_fn: cpu_word!(1, crate::call_jit::bh_load_build_class_fn),
            load_import_fn: cpu_word!(1, crate::call_jit::bh_load_import_fn),
            load_import_locals_fn: cpu_word!(1, crate::call_jit::bh_load_import_locals_fn),
            load_import_globals_fn: cpu_word!(1, crate::call_jit::bh_load_import_globals_fn),
            eval_lazy_import_name_fn: cpu_word!(
                5,
                pyre_interpreter::lazyimport::eval_lazy_import_name_jit_abi
            ),
            newtuple_from_array_fn: cpu_word!(1, crate::call_jit::bh_newtuple_from_array),
            build_map_from_array_fn: cpu_word!(1, crate::call_jit::bh_build_map_from_array),
            build_set_from_array_fn: cpu_word!(1, crate::call_jit::bh_build_set_from_array),
            build_string_from_array_fn: cpu_word!(1, crate::call_jit::bh_build_string_from_array),
            newlist_from_array_fn: cpu_word!(1, crate::call_jit::bh_newlist_from_array),
            unpack_sequence_fn: cpu_word!(2, crate::call_jit::bh_unpack_sequence_fn),
            unpack_item_fn: cpu_word!(2, crate::call_jit::bh_unpack_item_fn),
            unpack_ex_fn: cpu_word!(3, crate::call_jit::bh_unpack_ex_fn),
            build_slice_fn: cpu_word!(4, crate::call_jit::bh_build_slice_fn),
            normalize_raise_varargs_fn: cpu_word!(
                3,
                crate::call_jit::bh_normalize_raise_varargs_with_frame
            ),
            get_current_exception_fn: cpu_word!(0, crate::call_jit::bh_get_current_exception),
            current_exception_or_none_fn: cpu_word!(
                0,
                crate::call_jit::bh_current_exception_or_none
            ),
            reraise_varargs_zero_fn: cpu_word!(0, crate::call_jit::bh_reraise_varargs_zero),
            set_current_exception_fn: cpu_word!(1, crate::call_jit::bh_set_current_exception),
            clear_in_flight_exception_fn: cpu_word!(
                0,
                crate::call_jit::bh_clear_in_flight_exception
            ),
            bytecode_trace_jitted_slow_fn: cpu_word!(
                2,
                crate::call_jit::bh_bytecode_trace_jitted_slow
            ),
            bytecode_only_trace_fn: cpu_word!(2, crate::call_jit::bh_bytecode_only_trace),
            rtyper,
            lowering_ctx: parking_lot::RwLock::new(None),
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}
